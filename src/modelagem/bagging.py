"""Bagging (BaggingRegressor com bootstrap=True) de árvores de decisão no conjunto C07.

Uso (a partir da raiz do projeto):
    python -m src.modelagem.bagging                    # validação cruzada da configuração inicial
    python -m src.modelagem.bagging --busca            # + RandomizedSearchCV (10 configurações)
    python -m src.modelagem.bagging --teste            # AVALIAÇÃO FINAL (melhor configuração da busca, se houver)
    python -m src.modelagem.bagging --n-jobs 4         # menos núcleos = menos memória
    python -m src.modelagem.bagging --amostra 30000    # teste rápido do script

Saídas: resultados/experimentos.csv (LC-C07-BAG, LC-C07-BAG-BUSCA), resultados/previsoes_cv/<id>.parquet
e resultados/busca_LC-C07-BAG-BUSCA.csv.
"""

import sys

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

from ..conjuntos import configuracao as cfg
from .comum import (argumentos, avaliar, avaliar_teste, avisar_teste, buscar, log, melhor_da_busca,
                    preparar_execucao)

FAMILIA = "arvore"
PREPARACAO = ("códigos ordinais (especiais 'não sei'/indeterminada depois do último nível); categóricas como "
              "códigos inteiros; NaN mantido (as árvores tratam ausência); sem padronização")

# Árvore-base profunda (bagging reduz variância, então não limitamos a profundidade), com folhas de pelo menos
# 20 participantes para não decorar notas individuais e manter as árvores em tamanho viável na memória.
ARVORE_BASE = DecisionTreeRegressor(min_samples_leaf=20, random_state=cfg.SEMENTE)
# Bagging clássico: cada árvore recebe n linhas sorteadas COM reposição (max_samples = 1,0), ou seja,
# ~63% de participantes distintos, alguns repetidos.
REGRESSOR = BaggingRegressor(estimator=ARVORE_BASE, n_estimators=20, max_samples=1.0, bootstrap=True,
                             random_state=cfg.SEMENTE)
HIPOTESE = ("Bagging: a média de 20 árvores treinadas em amostras com reposição reduz a variância de uma árvore "
            "única e capta interações entre renda, escolaridade, escola e idade que a regressão linear não capta.")
# Nº de árvores, fração de linhas e de colunas por árvore e tamanho mínimo da folha (complexidade da base).
ESPACO = {
    "modelo__n_estimators": [10, 20, 40],
    "modelo__max_samples": [0.3, 0.5, 0.75, 1.0],
    "modelo__max_features": [0.6, 0.8, 1.0],
    "modelo__estimator__min_samples_leaf": [5, 20, 50, 100],
}


def main(argv=None):
    args = argumentos("Bagging de árvores (bootstrap=True) no conjunto C07.", argv, paralelo=True)
    controle, X, y, dobras, rotulo, sufixo = preparar_execucao(args)
    regressor = REGRESSOR.set_params(n_jobs=args.n_jobs)
    if args.teste:
        avisar_teste()
        final, origem = melhor_da_busca(regressor, "LC-C07-BAG-BUSCA", args.resultados)
        log(f"Bagging no teste com a {origem}")
        avaliar_teste("LC-C07-BAG-TESTE", HIPOTESE, FAMILIA, final, controle, X, y, PREPARACAO, args.arquivo,
                      args.resultados)
        return 0
    if not args.so_busca:
        log(f"LC-C07-BAG{sufixo}: 20 árvores, max_samples=1.0, bootstrap=True, n_jobs={args.n_jobs}")
        avaliar("LC-C07-BAG" + sufixo, HIPOTESE, FAMILIA, regressor, controle, X, y, dobras, rotulo, PREPARACAO,
                args.resultados)
    if args.busca or args.so_busca:
        log("Busca do bagging (RandomizedSearchCV, 10 configurações x 3 dobras)")
        buscar("LC-C07-BAG-BUSCA" + sufixo,
               "Busca do bagging: quantas árvores, que fração de linhas/colunas e que folha mínima minimizam o RMSE.",
               FAMILIA, regressor, ESPACO, X, y, dobras, rotulo, PREPARACAO, n_iter=10, pasta=args.resultados)
    return 0


if __name__ == "__main__":
    sys.exit(main())
