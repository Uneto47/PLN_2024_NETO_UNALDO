import os
from dotenv import load_dotenv
from data_preprocessing import fetch_data
from embedding_utils import get_embeddings
from db_utils import init_chromadb, add_embeddings_to_chromadb

load_dotenv()

def main(reload_data=False):
    # Define o caminho para o banco de dados persistente
    db_path = "/path/to/save/to"
    collection_name = "economic_data"
    
    # Inicialize a conexão com o banco de dados persistente
    collection = init_chromadb(db_path, collection_name)
    
    # Checa se precisa recarregar os dados
    if reload_data:
        print("Carregando dados...")
        sentences = fetch_data()
        embeddings = get_embeddings(sentences)
        add_embeddings_to_chromadb(collection, sentences, embeddings)
    else:
        print("A coleção já contém dados recentes. Você pode pular a etapa de carregamento de dados.")
    
    # Defina a consulta e gere o embedding da consulta
    query_sentence = "Categoria: Moedas Futebol"
    query_embedding = get_embeddings([query_sentence]).tolist()[0]
    
    # Execute a consulta no banco de dados
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    print(results)
    
    # if results and 'documents' in results and results['documents']:
    #     for i, document in enumerate(results['documents'][0]):
    #         print(f"{i+1}°", document)
    # else:
    #     print("Nenhum documento encontrado nos resultados.")

if __name__ == "__main__":
    main(reload_data=False)  # Defina como `True` se quiser forçar o recarregamento dos dados
