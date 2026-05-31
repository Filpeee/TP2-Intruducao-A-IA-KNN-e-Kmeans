import pandas as pd
import numpy as np

from utils import normalizar_min_max
from metricas import imprimir_relatorio_knn, imprimir_relatorio_kmeans
from knn import KNNClassificador
from kmeans import KmeansClusterizador

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

        valores_k_knn = [2, 10, 50, 15]
        
        print("="*50)
        print("PARTE 1: KNN (APRENDIZADO SUPERVISIONADO)")
        print("="*50)
        
        for k in valores_k_knn:
            print(f"Executando para K={k}...")
            modelo_knn = KNNClassificador(k=k)
            modelo_knn.fit(X_treino, y_treino)
            y_previsto = modelo_knn.predict(X_teste)

            imprimir_relatorio_knn(y_teste, y_previsto, k_usado=k)

        print("\n" + "="*50)
        print("PARTE 2: K-MEANS (APRENDIZADO NÃO-SUPERVISIONADO)")
        print("="*50)

        df_completo = pd.concat([df_treino_norm, df_teste_norm], ignore_index=True)
        df_completo_norm = normalizar_min_max(df_completo)

        colunas_features = df_completo_norm.drop('TARGET_5Yrs', axis=1).columns
        X_completo = df_completo_norm[colunas_features]
        y_verdadeiro_completo = df_completo_norm['TARGET_5Yrs'].values

        valores_k_kmeans = [2, 3]

        for k in valores_k_kmeans:
            print(f"Executando K-Means para K={k}...")
            modelo_kmeans = KmeansClusterizador(k=k)
            modelo_kmeans.fit(X_completo)
            clusters_atribuidos = modelo_kmeans.predict(X_completo)

            imprimir_relatorio_kmeans(modelo_kmeans, clusters_atribuidos, y_verdadeiro_completo, colunas_features)
    
    except FileNotFoundError:
        print("Erro: Arquivos CSV não encontrados.")

if __name__ == "__main__":
    main()       