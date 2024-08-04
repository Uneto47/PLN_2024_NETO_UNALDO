import json
import pandas as pd

LOCAL_JSON_PATH = "inform_10000.json"

def fetch_data():
    with open(LOCAL_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data['value'])
    columns = ['Data', 'Categoria', 'Quantidade', 'Denominacao']
    df = df[columns]
    sentences = df.apply(create_sentences, axis=1).tolist()
    return sentences

def create_sentences(row):
    return f" Categoria: {row['Categoria']}, Data: {row['Data']}, Denominacao: {row['Denominacao']}, Quantidade: {row['Quantidade']}"