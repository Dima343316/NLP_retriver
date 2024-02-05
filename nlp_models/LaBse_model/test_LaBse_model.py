from sentence_transformers import SentenceTransformer, util
import torch
import json
import os
huggingface_model = "cointegrated/rubert-tiny2"

if __name__ == "__main__":
    file_path = "XMLtoDict.json"
    XMLtoDict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        XMLtoDict = json.load(file)

    XmlToDict_Keys = []
    for key,value in XMLtoDict.items():
        XmlToDict_Keys.append(key)


    embedder = SentenceTransformer(huggingface_model)


    corpus = XmlToDict_Keys

    embeddings_file = 'corpus_embeddings.pt'
    if os.path.exists(embeddings_file):
        print("Уже создано! Пезезаписать? Пo Enter")
        input()

    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True,show_progress_bar=True)
    torch.save(corpus_embeddings, 'corpus_embeddings.pt')