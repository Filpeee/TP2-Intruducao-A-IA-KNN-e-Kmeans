import numpy as np

class KNNClassificador:
    def __init__(self, k=3):
        """
        Inicializa o classificador com o número de vizinhos desejado.
        """
        self.k = k
        self.X_treino = None
        self.y_treino = None

    def fit(self, X, y):
        """
        Armazena os dados de treino para uso posterior na classificação.
        X: Características de treino (estatísticas dos jogadores)
        y: Rótulos de treino (TARGET_5Yrs)
        """
        self.X_treino = np.array(X)
        self.y_treino = np.array(y).astype(int)  # Garantir que os rótulos sejam inteiros (0 ou 1)

    def _calcular_distancia(self, x1, x2):
        """
        Calcula a distância euclidiana entre dois pontos.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X_teste):
        """
        Recebe uma matriz de teste inteira e faz a previsão para cada linha.
        """
        X_teste = np.array(X_teste)
        previsoes = [self._prever_um_ponto(x) for x in X_teste]

        return np.array(previsoes)
    
    def _prever_um_ponto(self, x_teste):
        """
        Função interna que faz a previsão para um único jogador de teste.
        """
        # 1. Calcula a distância desse jogador de teste para TODOS os jogadores de treino
        distancias = [self._calcular_distancia(x_teste, x_treino) for x_treino in self.X_treino]

        # 2. Ordena as distâncias e pega os índices dos K mais próximos
        k_indices = np.argsort(distancias)[:self.k]

        # 3. Pega os rótulos dos K vizinhos mais próximos
        k_rotulos = [self.y_treino[i] for i in k_indices]

        # 4. Retorna a classe mais comum entre os vizinhos
        votos = np.bincount(k_rotulos)
        classe_vencedora = votos.argmax()  # Retorna a classe mais votada (0 ou 1)
        
        return classe_vencedora  # Retorna a classe mais votada (0 ou 1)

