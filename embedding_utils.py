import torch.nn.functional as F
from torch import Tensor
from sentence_transformers import SentenceTransformer

# Carregar o modelo
def get_embeddings(sentences):
    model = SentenceTransformer('intfloat/multilingual-e5-small')
    return model.encode(sentences, convert_to_tensor=False)