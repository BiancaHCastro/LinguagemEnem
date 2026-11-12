"""Árvore de decisão (DecisionTreeRegressor) no conjunto C07.

Mesma configuração inicial do notebook ENEM_2023_Linguagens_DecisionTree_Stacking.ipynb
(DecisionTreeRegressor(random_state=42), sem limite de profundidade), agora no protocolo comum dos scripts:
validação cruzada nas 3 dobras fixas, preparação ajustada só no treino de cada dobra e registro em
resultados/experimentos.csv. O teste não é usado aqui (só para a solução final).

Uso (a partir da raiz do projeto):
    python -m src.modelagem.arvore_decisao                    # validação cruzada da configuração inicial
    python -m src.modelagem.arvore_decisao --busca            # + RandomizedSearchCV (só se estiver entre os 3 melhores)
    python -m src.modelagem.arvore_decisao --amostra 30000    # teste rápido do script

Saídas: resultados/experimentos.csv (LC-C07-ARV, LC-C07-ARV-BUSCA), resultados/previsoes_cv/<id>.parquet
e resultados/busca_LC-C07-ARV-BUSCA.csv.
"""

import sys

from sklearn.tree import DecisionTreeRegressor

from ..conjuntos import configuracao as cfg
from .comum import argumentos, avaliar, buscar, log, preparar_execucao

FAMILIA = "arvore"
PREPARACAO = ("códigos ordinais (especiais 'não sei'/indeterminada depois do último nível); categóricas como "
              "códigos inteiros; NaN mantido (a árvore trata ausência); sem padronização")
# Configuração inicial = a do notebook: árvore sem restrição de crescimento.
REGRESSOR = DecisionTreeRegressor(random_state=cfg.SEMENTE)
HIPOTESE = ("Árvore de decisão sem restrições: capta interações entre renda, escolaridade, escola e idade, mas, "
            "crescendo até folhas de 1 participante, tende a decorar o treino (sobreajuste).")
# Controle de complexidade: profundidade, tamanho mínimo da folha e de divisão e poda por custo-complexidade.
ESPACO = {
    "modelo__max_depth": [6, 8, 10, 12, 16, None],
    "modelo__min_samples_leaf": [20, 50, 100, 200, 500],
    "modelo__min_samples_split": [2, 100, 1000],
    "modelo__max_features": [None, 0.8, 0.6],
    "modelo__ccp_alpha": [0.0, 0.1, 1.0],
}


def main(argv=None):
    args = argumentos("Árvore de decisão (DecisionTreeRegressor) no conjunto C07.", argv)
    if args.teste:
        raise SystemExit("O teste é usado só para a solução final (ver boosting.py --teste).")
    controle, X, y, dobras, rotulo, sufixo = preparar_execucao(args)
    if not args.so_busca:
        log(f"LC-C07-ARV{sufixo}: DecisionTreeRegressor(random_state=42), configuração inicial do notebook")
        avaliar("LC-C07-ARV" + sufixo, HIPOTESE, FAMILIA, REGRESSOR, controle, X, y, dobras, rotulo, PREPARACAO,
                args.resultados)
    if args.busca or args.so_busca:
        log("Busca da árvore (RandomizedSearchCV, 12 configurações x 3 dobras)")
        buscar("LC-C07-ARV-BUSCA" + sufixo,
               "Busca da árvore: limitar a complexidade (profundidade, folha mínima, poda) reduz o sobreajuste "
               "e o RMSE da validação.",
               FAMILIA, REGRESSOR, ESPACO, X, y, dobras, rotulo, PREPARACAO, n_iter=12, pasta=args.resultados)
    return 0


if __name__ == "__main__":
    sys.exit(main())
