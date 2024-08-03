import os
import requests
import pandas as pd
from chromadb import Client
from dotenv import load_dotenv
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

load_dotenv()

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

HUGGINGFACE_TOKEN = os.getenv("TOKEN")
DATA_URL = "https://olinda.bcb.gov.br/olinda/servico/mecir_dinheiro_em_circulacao/versao/v1/odata/informacoes_diarias_com_categoria?$top=500&$format=json"

def get_embeddings(sentences):
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')
    batch_dict = tokenizer(sentences, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask']) # type: ignore
    embeddings = F.normalize(embeddings, p=2, dim=1)  # Normalize embeddings
    return embeddings

def fetch_data():
    response = requests.get(DATA_URL)
    data = response.json()
    df = pd.DataFrame(data['value'])
    columns = ['Data', 'Valor', 'Categoria', 'Denominacao']
    df = df[columns]
    sentences = df.apply(create_sentences, axis=1).tolist()
    return sentences

def init_chromadb():
    client = Client()
    collection = client.create_collection(name="economic_data")
    return collection

def add_embeddings_to_chromadb(collection, sentences, embeddings):
    for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
        embedding_list = embedding.tolist()
        collection.add(documents=[sentence], embeddings=[embedding_list], ids=[f"id_{i}"])

def create_sentences(row):
    return f"query: Data: {row['Data']}, Valor: {row['Valor']}, Categoria: {row['Categoria']}, Denominacao: {row['Denominacao']}"
 
def main():
    sentences = fetch_data()

    query_sentence = "query: Futebol"
    
    embeddings = get_embeddings(sentences)
    
    collection = init_chromadb()
    
    add_embeddings_to_chromadb(collection, sentences, embeddings)
    
    query_embedding = get_embeddings([query_sentence]).tolist()[0]  # Gerar embedding da consulta e converter para lista
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    if results and 'documents' in results and results['documents']:
        for i, document in enumerate(results['documents'][0]):
            print(f"{i+1}°", document)
    else:
        print("Nenhum documento encontrado nos resultados.")
        
if __name__ == "__main__":
    main()
