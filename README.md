# Trabalho Prático II: Aprendizado de Máquina 

**Disciplina:** Introdução à Inteligência Artificial  
**Autores:** Felipe Pereira e Mateus Rabelo

## Objetivo do Projeto
Este repositório contém a implementação do zero dos algoritmos de aprendizado de máquina **k-Nearest Neighbors (kNN)** e **k-Means**, desenvolvidos como requisito para o Trabalho Prático II. O objetivo central é prever o sucesso de jogadores novatos da NBA (carreiras maiores que 5 anos) e agrupá-los por similaridade estatística, sem o uso de bibliotecas prontas na construção lógica dos algoritmos.

## Estrutura do Repositório

* `data/`: Contém os conjuntos de dados utilizados (`nba_treino.csv` e `nba_teste.csv`).
* `src/`: Contém todo o código-fonte desenvolvido.
  * `knn.py`: Implementação manual do classificador k-Nearest Neighbors (Distância Euclidiana e votação majoritária).
  * `kmeans.py`: Implementação manual do clusterizador k-Means (Inicialização, Atribuição, Atualização e Convergência).
  * `utils.py`: Funções auxiliares, incluindo a normalização Min-Max manual.
  * `metricas.py`: Funções matemáticas para cálculo de Acurácia, Precisão, Recall, F1-Score e Matriz de Confusão.
  * `main.py`: Orquestrador principal dos experimentos manuais.
  * `comparacao_sklearn.py`: Script para validação dos resultados utilizando a biblioteca `scikit-learn`.
* `relatorio/`: Contém os arquivos referentes ao relatório acadêmico final.

## Tecnologias Utilizadas
* **Python 3**
* **NumPy** (Para operações de álgebra linear e vetorização)
* **Pandas** (Para manipulação dos conjuntos de dados)
* **Scikit-Learn** (Apenas para a etapa 3: validação e comparação de resultados)

## Como Executar

1. Abra um terminal na raiz do repositório.

2. Instale as dependências:
```bash
pip install -r requeriments.txt
```

3. Execute a implementação manual (kNN e k-Means):
```bash
python src/main.py
```

4. Execute o script de comparação (Scikit-Learn):
```bash
python src/comparacao_sklearn.py
```