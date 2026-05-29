import pandas as pd
from utils import normalizar_min_max
from metricas import imprimir_relatorio
from knn import KNNClassificador

def main():
    caminho_treino = 'data/nba_treino.csv'
    caminho_teste = 'data/nba_teste.csv'

    try:
        df_treino = pd.read_csv(caminho_treino)
        df_teste = pd.read_csv(caminho_teste)

        df_treino_norm = normalizar_min_max(df_treino)
        df_teste_norm = normalizar_min_max(df_teste)

        X_treino = df_treino_norm.drop('TARGET_5Yrs', axis=1)
        y_treino = df_treino_norm['TARGET_5Yrs']

        X_teste = df_teste_norm.drop('TARGET_5Yrs', axis=1)
        y_teste = df_teste_norm['TARGET_5Yrs']

        valores_k = [2, 10, 50, 15]

        print("Treinando e avaliando o modelo KNN para diferentes valores de K...")

        for k in valores_k:
            print(f"Executando para K={k}...")
            modelo_knn = KNNClassificador(k=k)
            modelo_knn.fit(X_treino, y_treino)
            y_previsto = modelo_knn.predict(X_teste)

            imprimir_relatorio(y_teste, y_previsto, k_usado=k)

    except FileNotFoundError:
        print("Erro: Arquivos CSV não encontrados. Execute o script da raiz do repositório.")

if __name__ == "__main__":
    main()       