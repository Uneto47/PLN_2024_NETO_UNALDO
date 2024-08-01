import requests
import pandas as pd
from chromadb import Client

API_URL = "https://api-inference.huggingface.co/models/intfloat/multilingual-e5-small"
HUGGINGFACE_TOKEN = ""
DATA_URL = "https://olinda.bcb.gov.br/olinda/servico/mecir_dinheiro_em_circulacao/versao/v1/odata/informacoes_diarias_com_categoria?$top=500&$format=json"

def get_embeddings(sentences, source):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
    payload = {"inputs": {"source_sentence": source, "sentences": sentences}}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def fetch_data():
    response = requests.get(DATA_URL)
    data = response.json()
    df = pd.DataFrame(data['value'])
    columns = ['Data', 'Valor', 'Quantidade', 'Categoria', 'Denominacao', 'Especie']
    df = df[columns]
    sentences = df.apply(create_sentences, axis=1).tolist()
    return sentences

def init_chromadb():
    client = Client()
    collection = client.create_collection(name="economic_data")
    return collection

def add_embeddings_to_chromadb(collection, sentences, embeddings):
    for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
        collection.add(documents=[sentence], embeddings=[embedding], ids=[f"id_{i}"])

def create_sentences(row):
    return f"Data: {row['Data']}, Valor: {row['Valor']}, Quantidade: {row['Quantidade']}, Categoria: {row['Categoria']}, Denominacao: {row['Denominacao']}, Especie: {row['Especie']}"
 
def main():
    sentences = fetch_data()
    query_sentence = "'Categoria: Moedas - 1a. família'"
    embeddings = get_embeddings(sentences, query_sentence)
    
    collection = init_chromadb()
    add_embeddings_to_chromadb(collection, sentences, embeddings)
    
    results = collection.query(
        query_embeddings=[1],
        n_results=3
    )
    
    if results and 'documents' in results and results['documents']:
        for i, document in enumerate(results['documents'][0]):
            print(f"{i+1}°", document)
    else:
        print("Nenhum documento encontrado nos resultados.")
        
if __name__ == "__main__":
    main()
