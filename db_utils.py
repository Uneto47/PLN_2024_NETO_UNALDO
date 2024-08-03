from chromadb import PersistentClient

def init_chromadb(db_path, collection_name):
    client = PersistentClient(path=db_path)
    try:
        collection = client.get_collection(name=collection_name)
    except:
        collection = client.create_collection(name=collection_name)
    return collection

def add_embeddings_to_chromadb(collection, sentences, embeddings):
    for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
        embedding_list = embedding.tolist()
        collection.add(documents=[sentence], embeddings=[embedding_list], ids=[f"id_{i}"])
