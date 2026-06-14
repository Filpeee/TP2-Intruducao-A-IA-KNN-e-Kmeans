TRABALHO PRATICO II - APRENDIZADO DE MAQUINA (KNN E K-MEANS)
Disciplina: Introducao a Inteligencia Artificial
Autores: Felipe Pereira e Mateus Rabelo

Este documento contem as instrucoes para a execucao dos codigos desenvolvidos para o Trabalho Pratico II.

1. REQUISITOS DO SISTEMA
Certifique-se de ter o Python 3 instalado em sua maquina. 
As dependencias do projeto estao listadas no arquivo requirements.txt. Para instala-las, execute o comando na raiz do projeto:
pip install -r requirements.txt

2. ESTRUTURA DE PASTAS
O projeto espera que os arquivos de dados (nba_treino.csv e nba_teste.csv) estejam localizados dentro de uma pasta chamada "data/" sentro da pasta do projeto.
Os codigos fonte estao localizados na pasta "src/".

3. COMO EXECUTAR A IMPLEMENTACAO MANUAL (DO ZERO)
Para rodar a implementacao manual do kNN e do k-Means (com normalizacao, treino, previsao e exibicao das metricas), abra o terminal na raiz do projeto e execute:
python src/main.py

4. COMO EXECUTAR A COMPARACAO COM O SCIKIT-LEARN
Para rodar os testes utilizando a biblioteca scikit-learn e validar os resultados obtidos na implementacao manual, execute no terminal:
python src/comparacao_sklearn.py