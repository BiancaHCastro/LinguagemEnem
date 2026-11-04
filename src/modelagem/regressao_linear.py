"""Baseline, regressão linear sem regularização (LinearRegression) e com regularização L2 (Ridge) no conjunto C07.

Uso (a partir da raiz do projeto):
    python -m src.modelagem.regressao_linear                  # validação cruzada: baseline, OLS e Ridge
    python -m src.modelagem.regressao_linear --busca          # + GridSearchCV do alpha do Ridge (12 valores)
    python -m src.modelagem.regressao_linear --teste          # AVALIAÇÃO FINAL: baseline, OLS e Ridge no teste
    python -m src.modelagem.regressao_linear --amostra 30000  # teste rápido do script

Saídas: resultados/experimentos.csv (LC-C07-BASE, LC-C07-LIN-OLS, LC-C07-LIN-RIDGE, LC-C07-LIN-RIDGE-BUSCA,
…-TESTE), resultados/previsoes_cv/<id>.parquet, resultados/busca_LC-C07-LIN-RIDGE-BUSCA.csv e
resultados/previsoes_teste.csv.
"""

import sys

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge

from .comum import (argumentos, avaliar, avaliar_teste, avisar_teste, buscar, log, melhor_da_busca,
                    preparar_execucao)

FAMILIA = "linear"
PREPARACAO = ("níveis ordinais padronizados; 'não sei'/indeterminada -> mediana do treino + indicador; "
              "one-hot sem a 1ª categoria (UF, situação, faixa etária, tipo de escola, perfil digital); "
              "log1p na renda per capita; sem IN_TREINEIRO, Q025, TP_ESCOLA, CAPITAL_CULTURAL_RESP e N_RESP_NAO_SEI "
              "(redundantes); colunas com colinearidade exata removidas no treino da dobra")

CONFIGURACOES = {
    "LC-C07-BASE": (
        DummyRegressor(strategy="mean"),
        "Baseline obrigatório: prevê a média da nota no treino de cada dobra para todos, sem usar features.",
    ),
    "LC-C07-LIN-OLS": (
        LinearRegression(),
        "Regressão linear sem regularização: relações aditivas entre perfil e nota explicam parte da variância; "
        "referência obrigatória do enunciado.",
    ),
    "LC-C07-LIN-RIDGE": (
        Ridge(alpha=1.0),
        "Ridge (L2, alpha = 1): com ~77 colunas e milhões de linhas deve mudar pouco o erro, mas estabiliza os "
        "coeficientes quase colineares (tipo de escola × situação de conclusão).",
    ),
}
# alpha de 0,01 a 10.000 em escala log: cobre de quase-OLS a encolhimento forte dos coeficientes.
ESPACO_RIDGE = {"modelo__alpha": np.logspace(-2, 4, 12)}


def main(argv=None):
    args = argumentos("Baseline e regressão linear (OLS e Ridge) no conjunto C07.", argv)
    controle, X, y, dobras, rotulo, sufixo = preparar_execucao(args)

    if args.teste:
        avisar_teste()
        for id_experimento in ("LC-C07-BASE", "LC-C07-LIN-OLS"):
            regressor, hipotese = CONFIGURACOES[id_experimento]
            avaliar_teste(f"{id_experimento}-TESTE", hipotese, FAMILIA, regressor, controle, X, y, PREPARACAO,
                          args.arquivo, args.resultados)
        ridge, origem = melhor_da_busca(CONFIGURACOES["LC-C07-LIN-RIDGE"][0], "LC-C07-LIN-RIDGE-BUSCA", args.resultados)
        log(f"Ridge no teste com a {origem}")
        avaliar_teste("LC-C07-LIN-RIDGE-TESTE", CONFIGURACOES["LC-C07-LIN-RIDGE"][1], FAMILIA, ridge, controle, X, y,
                      PREPARACAO, args.arquivo, args.resultados)
        return 0

    if not args.so_busca:
        for id_experimento, (regressor, hipotese) in CONFIGURACOES.items():
            log(f"{id_experimento}{sufixo}: {type(regressor).__name__}")
            avaliar(id_experimento + sufixo, hipotese, FAMILIA, regressor, controle, X, y, dobras, rotulo,
                    PREPARACAO, args.resultados)
    if args.busca or args.so_busca:
        log("Busca do alpha do Ridge (GridSearchCV, 12 configurações x 3 dobras)")
        buscar("LC-C07-LIN-RIDGE-BUSCA" + sufixo,
               "Busca do alpha do Ridge: verifica se algum grau de regularização reduz o RMSE da validação.",
               FAMILIA, Ridge(), ESPACO_RIDGE, X, y, dobras, rotulo, PREPARACAO, pasta=args.resultados)
    return 0


if __name__ == "__main__":
    sys.exit(main())
