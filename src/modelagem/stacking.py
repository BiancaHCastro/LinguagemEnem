"""Stacking (StackingRegressor) no conjunto C07.

Mesma combinação do notebook ENEM_2023_Linguagens_DecisionTree_Stacking.ipynb: árvore de decisão
(max_depth=8, min_samples_leaf=100) + Ridge (alpha=10) como modelos de base e Ridge (alpha=1) como modelo
final. Ajustes mínimos para atender ao enunciado:
- cada modelo de base tem a preparação adequada à sua família, ajustada dentro da dobra: a árvore usa códigos
  ordinais; o Ridge usa one-hot e padronização (no notebook o Ridge recebia UF e tipo de escola como números);
- o combinador é treinado com previsões fora da dobra: a validação cruzada interna do StackingRegressor
  (KFold de 3 partes, embaralhado, semente 42) roda só dentro do treino de cada dobra externa, então cada
  previsão usada pelo modelo final vem de um modelo de base que não treinou naquele participante;
- protocolo comum (3 dobras fixas, registro em resultados/experimentos.csv); o teste não é usado aqui.

Uso (a partir da raiz do projeto):
    python -m src.modelagem.stacking                    # validação cruzada da configuração inicial
    python -m src.modelagem.stacking --amostra 30000    # teste rápido do script

Saídas: resultados/experimentos.csv (LC-C07-STACK) e resultados/previsoes_cv/LC-C07-STACK.parquet.
"""

import sys

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

from ..conjuntos import configuracao as cfg
from ..conjuntos.preprocessamento import montar_pipeline
from .comum import CONJUNTO, argumentos, avaliar, log, preparar_execucao

FAMILIA = "arvore"  # só identifica as 23 colunas de entrada no registro; cada base tem a sua preparação
PREPARACAO = ("árvore-base: códigos ordinais, categóricas como códigos inteiros, NaN mantido; Ridge-base: níveis "
              "ordinais padronizados, 'não sei' -> mediana + indicador, one-hot, log1p na renda, sem redundâncias; "
              "cada preparação ajustada no treino da dobra; combinador treinado com previsões fora da dobra "
              "(KFold interno de 3 partes)")
REGRESSOR = StackingRegressor(
    estimators=[
        ("arvore", montar_pipeline(CONJUNTO, "arvore",
                                   DecisionTreeRegressor(max_depth=8, min_samples_leaf=100, random_state=cfg.SEMENTE))),
        ("ridge", montar_pipeline(CONJUNTO, "linear", Ridge(alpha=10.0))),
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=KFold(3, shuffle=True, random_state=cfg.SEMENTE),
    n_jobs=1,
)
HIPOTESE = ("Stacking: a árvore capta interações e o Ridge capta efeitos aditivos; um modelo final treinado com "
            "previsões fora da dobra combina os dois e erra menos que cada um isolado.")


def main(argv=None):
    args = argumentos("Stacking (árvore + Ridge -> Ridge) no conjunto C07.", argv)
    if args.teste or args.busca or args.so_busca:
        raise SystemExit("Stacking: só a validação cruzada da configuração inicial (fora dos 3 melhores; "
                         "o teste é usado só para a solução final).")
    controle, X, y, dobras, rotulo, sufixo = preparar_execucao(args)
    log(f"LC-C07-STACK{sufixo}: [árvore(max_depth=8, folha 100), Ridge(10)] -> Ridge(1), CV interna de 3 partes")
    avaliar("LC-C07-STACK" + sufixo, HIPOTESE, FAMILIA, REGRESSOR, controle, X, y, dobras, rotulo, PREPARACAO,
            args.resultados, preparar=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
