"""Pasting (BaggingRegressor com bootstrap=False e max_samples < 1) de árvores de decisão no conjunto C07.

Uso (a partir da raiz do projeto):
    python -m src.modelagem.pasting                    # validação cruzada da configuração inicial
    python -m src.modelagem.pasting --busca            # + RandomizedSearchCV (10 configurações)
    python -m src.modelagem.pasting --teste            # AVALIAÇÃO FINAL (melhor configuração da busca, se houver)
    python -m src.modelagem.pasting --n-jobs 4         # menos núcleos = menos memória
    python -m src.modelagem.pasting --amostra 30000    # teste rápido do script

Saídas: resultados/experimentos.csv (LC-C07-PAST, LC-C07-PAST-BUSCA), resultados/previsoes_cv/<id>.parquet
e resultados/busca_LC-C07-PAST-BUSCA.csv.
"""

import sys

from sklearn.ensemble import BaggingRegressor

from ..conjuntos import configuracao as cfg
from .bagging import ARVORE_BASE, FAMILIA, PREPARACAO
from .comum import (argumentos, avaliar, avaliar_teste, avisar_teste, buscar, log, melhor_da_busca,
                    preparar_execucao)

# Mesma árvore-base e mesmo nº de árvores do bagging; a única diferença é o sorteio SEM reposição.
# max_samples = 0,63 dá a cada árvore a mesma fração de participantes DISTINTOS que o bootstrap com
# max_samples = 1,0 (1 - 1/e ≈ 63%), então a comparação isola o efeito das repetições do bootstrap.
REGRESSOR = BaggingRegressor(estimator=ARVORE_BASE, n_estimators=20, max_samples=0.63, bootstrap=False,
                             random_state=cfg.SEMENTE)
HIPOTESE = ("Pasting: árvores treinadas em 63% dos participantes sorteados sem reposição atingem desempenho "
            "parecido com o do bagging (mesma fração de participantes distintos), sem as linhas repetidas.")
ESPACO = {
    "modelo__n_estimators": [10, 20, 40],
    "modelo__max_samples": [0.2, 0.4, 0.63, 0.8],  # < 1: com 1,0 todas as árvores veriam as mesmas linhas
    "modelo__max_features": [0.6, 0.8, 1.0],
    "modelo__estimator__min_samples_leaf": [5, 20, 50, 100],
}


def main(argv=None):
    args = argumentos("Pasting de árvores (bootstrap=False, max_samples < 1) no conjunto C07.", argv, paralelo=True)
    controle, X, y, dobras, rotulo, sufixo = preparar_execucao(args)
    regressor = REGRESSOR.set_params(n_jobs=args.n_jobs)
    if args.teste:
        avisar_teste()
        final, origem = melhor_da_busca(regressor, "LC-C07-PAST-BUSCA", args.resultados)
        log(f"Pasting no teste com a {origem}")
        avaliar_teste("LC-C07-PAST-TESTE", HIPOTESE, FAMILIA, final, controle, X, y, PREPARACAO, args.arquivo,
                      args.resultados)
        return 0
    if not args.so_busca:
        log(f"LC-C07-PAST{sufixo}: 20 árvores, max_samples=0.63, bootstrap=False, n_jobs={args.n_jobs}")
        avaliar("LC-C07-PAST" + sufixo, HIPOTESE, FAMILIA, regressor, controle, X, y, dobras, rotulo, PREPARACAO,
                args.resultados)
    if args.busca or args.so_busca:
        log("Busca do pasting (RandomizedSearchCV, 10 configurações x 3 dobras)")
        buscar("LC-C07-PAST-BUSCA" + sufixo,
               "Busca do pasting: quantas árvores, que fração de linhas/colunas e que folha mínima minimizam o RMSE.",
               FAMILIA, regressor, ESPACO, X, y, dobras, rotulo, PREPARACAO, n_iter=10, pasta=args.resultados)
    return 0


if __name__ == "__main__":
    sys.exit(main())
