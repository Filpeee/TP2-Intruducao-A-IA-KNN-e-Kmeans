import numpy as np

class KmeansClusterizador:
    def __init__(self, k=2, max_iteracoes=100):
        """
        Inicializa o k-Means.
        k: Número de clusters (agrupamentos) desejados.
        max_iteracoes: Limite de repetições para evitar loops infinitos caso não convirja.
        """
        self.k = k
        self.max_iter = max_iteracoes
        self.centroides = None

    def _calcular_distancia(self, x1, x2):
        """
        Calcula a distância euclidiana entre dois pontos.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def predict(self, X):
        """
        Dado um conjunto de jogadores e os centroides já treinados, 
        diz a qual grupo cada jogador pertence.
        """
        X = np.array(X)
        clusters_atribuidos = []

        for jogador in X:
            # Calcula a distância desse jogador para TODOS os 'k' centroides
            distancias = [self._calcular_distancia(jogador, centroide) for centroide in self.centroides]
            
            # np.argmin devolve a POSIÇÃO do menor valor (ex: se o centroide 0 estiver mais perto, devolve 0)
            cluster_mais_proximo = np.argmin(distancias)
            clusters_atribuidos.append(cluster_mais_proximo)

        return np.array(clusters_atribuidos)

    def fit(self, X):
        """
        Encontra a posição ideal dos centroides.
        X: Matriz com as estatísticas dos jogadores. (Sem o TARGET_5Yrs)
        """
        X = np.array(X)
        num_amostras, num_caracteristicas = X.shape

        # Escolhe 'k' números aleatórios entre 0 e o total de jogadores (sem repetição)
        indices_aleatorios = np.random.choice(num_amostras, self.k, replace=False)
        self.centroides = X[indices_aleatorios]

        for iteracao in range(self.max_iteracoes):
            # Atribuir cada amostra ao cluster mais próximo
            clusters = self.predict(X)

            centtroides_antigos = np.copy(self.centroides)

            # Atualizar os centroides
            for i in range(self.k):
                pontos_dos_clusters = X[clusters == i]
                
                # Se o grupo não estiver vazio, calculamos a média de todas as colunas para atualizar o centroide
                if len(pontos_dos_clusters) > 0:
                    self.centroides[i] = np.mean(pontos_dos_clusters, axis=0)

            # Calcula a distância total percorrida pelos centroides para verificar se houve convergência
            distancia_percorrida = np.sum([self._calcular_distancia(centtroides_antigos[i], self.centroides[i]) for i in range(self.k)])    

            # Se a distância percorrida for zero, significa que os centroides não se moveram, ou seja, o algoritmo convergiu
            if distancia_percorrida == 0:
                print(f"k-Means (k={self.k}) convergiu na iteração {iteracao + 1}")
                break
