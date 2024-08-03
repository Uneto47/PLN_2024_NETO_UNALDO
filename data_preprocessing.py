import json
import pandas as pd

LOCAL_JSON_PATH = "inform_500.json"

def fetch_data():
    with open(LOCAL_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data['value'])
    columns = ['Data', 'Valor', 'Categoria', 'Denominacao']
    df = df[columns]
    sentences = df.apply(create_sentences, axis=1).tolist()
    return sentences

def create_sentences(row):
    return f"Data: {row['Data']}, Valor: {row['Valor']}, Categoria: {row['Categoria']}, Denominacao: {row['Denominacao']}"