from sentence_transformers import SentenceTransformer, util
import torch
import json
import os
from ordered_set import OrderedSet

p = "cointegrated/rubert-tiny2"

class EmbeddinfRetriver:
    pass






class HFModel_FindTables_base:
    def __init__(self, huggingface_model):
        self.top_k = 1
        self.huggingface_model= huggingface_model
        """Загрузка XML -> JSON файла для NLP ретривера,
                   загрузка ембеддингов"""

        file_path = "nlp_models/LaBse_model/XMLtoDict.json"
        with open(file_path, 'r', encoding='utf-8') as file:
            self.XMLtoDict = json.load(file)

        self.XmlToDict_Keys = []
        for key, value in self.XMLtoDict.items():
            self.XmlToDict_Keys.append(key)

        self.embedder = SentenceTransformer(huggingface_model, device='cpu')
        self.corpus = self.XmlToDict_Keys

        embeddings_file = 'corpus_embeddings.pt'
        if os.path.exists(embeddings_file):
            self.corpus_embeddings = torch.load(embeddings_file, map_location='cpu')
        else:
            raise Exception("Нет файла corpus_embeddings.pt")

    def set_add_pipe_divider(self,found_tables_set):
        new_string = "||"
        while new_string in found_tables_set:
            new_string += '|'
        found_tables_set.add(new_string)

    def findtables(self, dbfields: list, top_k=None):
        if top_k is None:
            top_k = self.top_k
        """
        Принимает как аргумент список предполагаемых таблиц, самые схожие таблицы из БД
        """
        found_tables_string = ""
        useful_fields = ""
        found_tables_set = OrderedSet()
        for dbfield in dbfields:
            # Check if dbfield has less than 4 characters AND more than 2 spaces
            if len(dbfield) < 4 or dbfield.strip().count(' ') > 2:
                continue
            top_k = min(top_k, len(self.corpus))
            query_embedding = self.embedder.encode(str(dbfield), convert_to_tensor=True)

            # We use cosine-similarity and torch.topk to find the highest scores
            cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)

            for score, idx in zip(top_results[0], top_results[1]):
                table_name = str(self.corpus[idx])
                parts = table_name.split('.')
                if len(parts) >= 2:
                    table_name = '.'.join(parts[:2])
                found_tables_set.add(table_name)
            self.set_add_pipe_divider(found_tables_set) # Корявое решение для разделения по таблицам из Suggest_Tables

        """Временный, и, возможно, не оптимальный способ, добавить табличные части ('подтаблицы' типа РегистрНакопления.ВыручкаИСебестоимостьПродаж.Обороты)"""

        with_subtables_found_tables_set = OrderedSet()
        for table in found_tables_set:
            if table.startswith("||"):
                found_tables_string = found_tables_string + "\n"
                continue

            if table not in with_subtables_found_tables_set:
                found_tables_string += table + "\n"
            with_subtables_found_tables_set.add(table)


            for db_table in self.XMLtoDict:
                if str(db_table).startswith(table+'.'): #ищем только поддтаблицы, не только такой же текст
                    if db_table not in with_subtables_found_tables_set:
                        found_tables_string += db_table + "\n"
                    with_subtables_found_tables_set.add(db_table)



        for table in with_subtables_found_tables_set:
            print(with_subtables_found_tables_set)
            useful_fields += "#### " + table + "\n"  # Add '## ' for Markdown header
            fields = str(self.XMLtoDict[table])
            useful_fields += fields + "\n\n"  # Add a newline to separate tables



        return found_tables_string.strip(), useful_fields.strip()
p = HFModel_FindTables_base('cointegrated/rubert-tiny2')
print(p.findtables(['клиенты']))