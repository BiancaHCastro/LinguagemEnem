"""KNN (KNeighborsRegressor) no conjunto C07.

Mesma configuração inicial e mesmo espaço de busca de src/modelagem/pca_balanceamento_ablacao.py (PCA incremental com 12
componentes + 50 vizinhos ponderados pela distância, árvore kd), com uma única mudança: usa as 23 features do
C07, como os demais modelos da comparação inicial (o pca_balanceamento_ablacao.py usava 15). A preparação é a da família
"linear" de src/conjuntos/preprocessamento.py: níveis ordinais padronizados, "não sei" -> mediana do treino +
indicador, one-hot nas categóricas, log1p na renda e sem as colunas redundantes (que pesariam em dobro na
distância). Tudo é ajustado só no treino de cada dobra.

Uso (a partir da raiz do projeto):
    python -m src.modelagem.knn                    # validação cruzada da configuração inicial
    python -m src.modelagem.knn --busca            # + RandomizedSearchCV (só se estiver entre os 3 melhores; demorada)
    python -m src.modelagem.knn --amostra 30000    # teste rápido do script

Saídas: resultados/experimentos.csv (LC-C07-KNN, LC-C07-KNN-BUSCA), resultados/previsoes_cv/<id>.parquet
e resultados/busca_LC-C07-KNN-BUSCA.csv.
"""

import sys

from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

from .comum import argumentos, avaliar, buscar, log, preparar_execucao
from .pca_balanceamento_ablacao import SparseIncrementalPCA

FAMILIA = "linear"
PREPARACAO = ("níveis ordinais padronizados; 'não sei'/indeterminada -> mediana do treino + indicador; "
              "one-hot sem a 1ª categoria; log1p na renda per capita; sem colunas redundantes; "
              "PCA incremental (12 componentes) ajustado no treino da dobra antes do KNN")
# PCA + KNN como no pca_balanceamento_ablacao.py: poucos componentes mantêm a árvore kd eficiente
# (em dezenas de dimensões ela degenera em força bruta). variance_threshold=1.0 deixa n_components mandar.
REGRESSOR = Pipeline([
    ("pca", SparseIncrementalPCA(n_components=12, variance_threshold=1.0)),
    ("knn", KNeighborsRegressor(n_neighbors=50, weights="distance", algorithm="kd_tree", leaf_size=60, n_jobs=-1)),
])
HIPOTESE = ("KNN: participantes com perfil parecido (renda, escolaridade, escola, UF) têm notas parecidas; "
            "a média de 50 vizinhos no espaço do PCA estima a nota sem supor forma funcional.")
ESPACO = {
    "modelo__pca__n_components": [6, 8, 10, 12, 16],
    "modelo__knn__n_neighbors": [50, 100, 200, 300],
    "modelo__knn__weights": ["uniform", "distance"],
    "modelo__knn__p": [1, 2],
    "modelo__knn__leaf_size": [30, 60, 100],
}


def main(argv=None):
    args = argumentos("KNN (KNeighborsRegressor) no conjunto C07.", argv)
    if args.teste:
        raise SystemExit("O teste é usado só para a solução final (ver boosting.py --teste).")
    controle, X, y, dobras, rotulo, sufixo = preparar_execucao(args)
    if not args.so_busca:
        log(f"LC-C07-KNN{sufixo}: PCA(12) + KNeighborsRegressor(50, distance), configuração inicial do pca_balanceamento_ablacao.py")
        avaliar("LC-C07-KNN" + sufixo, HIPOTESE, FAMILIA, REGRESSOR, controle, X, y, dobras, rotulo, PREPARACAO,
                args.resultados)
    if args.busca or args.so_busca:
        log("Busca do KNN (RandomizedSearchCV, 12 configurações x 3 dobras) - pode levar muitas horas")
        buscar("LC-C07-KNN-BUSCA" + sufixo,
               "Busca do KNN: nº de componentes, de vizinhos, ponderação e distância que minimizam o RMSE.",
               FAMILIA, REGRESSOR, ESPACO, X, y, dobras, rotulo, PREPARACAO, n_iter=12, pasta=args.resultados)
    return 0


if __name__ == "__main__":
    sys.exit(main())
