import numpy as np
import pandas as pd


def matriz_confusao(y_true, y_pred):
    """
    Calcula os valores da Matriz de Confusão.
    Retorna: Verdadeiro Positivo, Verdadeiro Negativo, Falso Positivo, Falso Negativo
    """
    y_verdadeiro = np.array(y_true)
    y_previsto = np.array(y_pred)
    
    vn = np.sum((y_verdadeiro == 0) & (y_previsto == 0))  # Verdadeiros Negativos
    fp = np.sum((y_verdadeiro == 0) & (y_previsto == 1))  # Falsos Positivos
    fn = np.sum((y_verdadeiro == 1) & (y_previsto == 0))  # Falsos Negativos
    vp = np.sum((y_verdadeiro == 1) & (y_previsto == 1))  # Verdadeiros Positivos
    
    return vp, vn, fp, fn

def calcular_metricas(y_verdadeiro, y_previsto):
    """
    Calcula Acurácia, Precisão, Recall e F1-Score com base na Matriz de Confusão.
    """
    vp, vn, fp, fn = matriz_confusao(y_verdadeiro, y_previsto)

    total = vp + vn + fp + fn
    
    acuracia = (vp + vn) / total if total > 0 else 0
    precisao = vp / (vp + fp) if (vp + fp) > 0 else 0
    recall = vp / (vp + fn) if (vp + fn) > 0 else 0
    f1_score = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0
    
    return {
        "Acurácia": acuracia,
        "Precisão": precisao,
        "Recall": recall,
        "F1-Score": f1_score,
        "Matriz": [[vn, fp], 
                   [fn, vp]]
    }

def imprimir_relatorio_knn(y_verdadeiro, y_previsto, k_usado):
    """
    Função auxiliar para imprimir os resultados no terminal.
    """
    resultados = calcular_metricas(y_verdadeiro, y_previsto)

    print(f"\n{'='*40}")
    print(f"RESULTADOS PARA O MODELO KNN (k={k_usado})")
    print(f"{'='*40}")
    print("Matriz de Confusão:")
    print(f"[{resultados['Matriz'][0][0]:4d}] (VN)  [{resultados['Matriz'][0][1]:4d}] (FP)")
    print(f"[{resultados['Matriz'][1][0]:4d}] (FN)  [{resultados['Matriz'][1][1]:4d}] (VP)")
    print("-" * 40)
    print(f"Acurácia:  {resultados['Acurácia']:.4f}")
    print(f"Precisão:  {resultados['Precisão']:.4f}")
    print(f"Recall:    {resultados['Recall']:.4f}")
    print(f"F1-Score:  {resultados['F1-Score']:.4f}")
    print(f"{'='*40}\n")

def imprimir_relatorio_kmeans(modelo_kmeans, clusters_atribuidos, y_verdadeiro, colunas_features):
    """
    Imprime as estatísticas dos centroides e a tabela cruzada de acertos do k-Means.
    """
    print("\nPerfil dos Centroides (Top 5 estatísticas normalizadas):")
    for i, centroide in enumerate(modelo_kmeans.centroides):
        print(f"\n  [Cluster {i}]")
        estatisticas_centroide = pd.Series(centroide, index=colunas_features)
        top_5 = estatisticas_centroide.sort_values(ascending=False).head(5)
        for col, val in top_5.items():
            print(f"  - {col}: {val:.4f}")

    print("\nRelação entre os Clusters criados e a Realidade (TARGET_5Yrs):")
    df_relacao = pd.DataFrame({'Cluster_KMeans': clusters_atribuidos, 'Realidade': y_verdadeiro})
    
    tabela_cruzada = pd.crosstab(
        df_relacao['Cluster_KMeans'], 
        df_relacao['Realidade'], 
        rownames=['K-Means agrupou como:'], 
        colnames=['Durou 5 Anos? (0=Não, 1=Sim)']
    )
    print("-" * 40)
    print(tabela_cruzada)
    print("-" * 40 + "\n")