from dotenv import load_dotenv
from data_preprocessing import fetch_data
from embedding_utils import get_embeddings
from db_utils import init_chromadb, add_embeddings_to_chromadb

load_dotenv()

def main(reload_data=False):
    
    db_path = "/path/to/save/to"
    collection_name = "banco"
    
    collection = init_chromadb(db_path, collection_name)
    
    if reload_data:
        print("Carregando dados...")
        sentences = fetch_data()
        print("Gerando embeddings")
        embeddings = get_embeddings(sentences)
        add_embeddings_to_chromadb(collection, sentences, embeddings)
    else:
        print("A coleção já contém dados recentes. Você pode pular a etapa de carregamento de dados.")
    
    query_sentence = "Futebol"
    query_embedding_list = get_embeddings([query_sentence])[0]  # Acessa o primeiro tensor na lista
    query_embedding = query_embedding_list.squeeze().tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    # print(results)
        
    if results and 'documents' and 'distances' in results and results['distances'] and results['documents']:
        for i, (document, distance) in enumerate(zip(results['documents'][0], results["distances"][0])):
            print(f"|{i+1}°| Distância: {distance:.4f} |{document}")
    else:
        print("Nenhum documento encontrado nos resultados.")

if __name__ == "__main__":
    main(reload_data=False) 
