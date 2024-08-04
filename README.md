# Projeto de Busca Semântica com Embeddings

Este projeto utiliza embeddings de frases para buscar documentos em um banco de dados vetorial. A seguir, você encontrará instruções sobre como instalar e usar o projeto.

## Instalação

1. **Clone o repositório**

   ```
   git clone https://github.com/seu-usuario/seu-repositorio.git](https://github.com/Uneto47/PLN_2024_NETO_UNALDO.gi
   ```

2. **Instale as dependências**

   Certifique-se de ter o `pip` atualizado e, em seguida, instale as bibliotecas necessárias:

   ```
   pip install chromadb sentence-transformers pandas torch torchvision torchaudio
   ```

3. **Preparação dos Dados**

   - Certifique-se de que o arquivo JSON (`inform_10000.json`) esteja presente no mesmo diretório que o script `data_preprocessing.py`.

## Uso

1. **Inicialização do Banco de Dados**

   No arquivo `main.py`, defina o caminho para o banco de dados e o nome da coleção. O caminho deve ser alterado para onde você deseja armazenar os dados.

   ```
   db_path = "/path/to/save/to"
   collection_name = "banco"
   ```

2. **Executando o Script**

   Você pode executar o script principal para iniciar o processo de carregamento de dados, geração de embeddings e consulta:

   ```
   python main.py
   ```

   - Se você deseja carregar dados e gerar embeddings, defina `reload_data=True` na chamada de `main()`.

   ```
   main(reload_data=True)
   ```

   - Se você já carregou os dados e deseja apenas realizar a consulta, defina `reload_data=False`.

3. **Consultas**

   O script `main.py` realiza uma consulta no banco de dados usando um exemplo de sentença ("Futebol"). Os resultados são exibidos no console com a distância e o documento correspondente.

## Arquivos do Projeto

- `db_utils.py`: Funções para inicializar e adicionar embeddings ao banco de dados.
- `data_preprocessing.py`: Funções para carregar e preparar os dados.
- `embedding_utils.py`: Funções para gerar embeddings usando um modelo de `sentence-transformers`.
- `main.py`: Script principal para executar o processo completo.
