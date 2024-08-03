import requests
import pandas as pd

DATA_URL = "https://olinda.bcb.gov.br/olinda/servico/mecir_dinheiro_em_circulacao/versao/v1/odata/informacoes_diarias_com_categoria?$top=1000&$format=json"

def fetch_data():
    response = requests.get(DATA_URL)
    data = response.json()
    df = pd.DataFrame(data['value'])
    columns = ['Data', 'Valor', 'Categoria', 'Denominacao']
    df = df[columns]
    sentences = df.apply(create_sentences, axis=1).tolist()
    return sentences

def create_sentences(row):
    return f"Data: {row['Data']}, Valor: {row['Valor']}, Categoria: {row['Categoria']}, Denominacao: {row['Denominacao']}"
