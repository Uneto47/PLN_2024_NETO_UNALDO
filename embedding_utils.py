from sentence_transformers import SentenceTransformer

def get_embeddings(sentences):
    model = SentenceTransformer('intfloat/multilingual-e5-small')
    return model.encode(sentences, convert_to_tensor=False)