import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    caminho_treino = 'data/nba_treino.csv'
    caminho_teste = 'data/nba_teste.csv'

    try:
        df_treino = pd.read_csv(caminho_treino)
        df_teste = pd.read_csv(caminho_teste)
    except FileNotFoundError:
        print("Erro: Arquivos CSV não encontrados.")
        return
        
    # Normalização dos dados usando Min-Max Scaling
    scaler = MinMaxScaler()
        
    X_treino_bruto = df_treino.drop('TARGET_5Yrs', axis=1)
    X_treino = scaler.fit_transform(X_treino_bruto)
    y_treino = df_treino['TARGET_5Yrs']

    X_teste_bruto = df_teste.drop('TARGET_5Yrs', axis=1)
    X_teste = scaler.transform(X_teste_bruto)
    y_teste = df_teste['TARGET_5Yrs']

    print("="*50)
    print("COMPARAÇÃO SCIKIT-LEARN: KNN")
    print("="*50)

        
    valores_k_knn = [2, 10, 50, 15]
    for k in valores_k_knn:
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(X_treino, y_treino)
        y_previsto = knn.predict(X_teste)

        acc = accuracy_score(y_teste, y_previsto)
        prec = precision_score(y_teste, y_previsto, zero_division=0)
        rec = recall_score(y_teste, y_previsto, zero_division=0)
        f1 = f1_score(y_teste, y_previsto, zero_division=0)
        matriz = confusion_matrix(y_teste, y_previsto)

        print(f"\nResultados KNN Scikit-Learn (K={k}):")
        print(f"Matriz de Confusão:\n{matriz}")
        print(f"Acurácia: {acc:.4f} | Precisão: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}")


    print("\n" + "="*50)
    print("COMPARAÇÃO SCIKIT-LEARN: K-MEANS")
    print("="*50)

        
    df_completo = pd.concat([df_treino, df_teste], ignore_index=True)
    X_completo_bruto = df_completo.drop('TARGET_5Yrs', axis=1)
    X_completo = scaler.fit_transform(X_completo_bruto)

    y_verdadeiro_completo = df_completo['TARGET_5Yrs']

    valores_k_kmeans = [2, 3]
    for k in valores_k_kmeans:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        clusters_atribuidos = kmeans.fit_predict(X_completo)

        print(f"\nRelação Clusters Scikit-Learn vs Realidade (K={k}):")
        df_relacao = pd.DataFrame({'Cluster_SKLearn': clusters_atribuidos, 'Realidade': y_verdadeiro_completo})
        tabela_cruzada = pd.crosstab(
            df_relacao['Cluster_SKLearn'], 
            df_relacao['Realidade'], 
            rownames=['Agrupamento SKLearn'], 
            colnames=['Durou 5 Anos?']
        )
        print("-" * 30)
        print(tabela_cruzada)
        print("-" * 30)

if __name__ == "__main__":
    main() 