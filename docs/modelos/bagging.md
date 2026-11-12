# Bagging: `BaggingRegressor(bootstrap=True)` de árvores de decisão

**Resumo:** a média de 20 árvores, cada uma treinada numa amostra **com reposição**. Ficou em **2º lugar (RMSE
65,26, R² 0,253)** e seguiu para a busca de hiperparâmetros.

## Arquitetura

1. Do treino da dobra (~1,5 milhão), sorteia **1,5 milhão de linhas com reposição** (`max_samples = 1.0`). Cada
   amostra tem ~63% de participantes distintos, alguns repetidos.
2. Treina **uma árvore de decisão** em cada amostra, com **folhas de pelo menos 20 participantes**
   (`min_samples_leaf = 20`) e sem limite de profundidade.
3. Repete **20 vezes** e prevê a **média das 20 árvores**.

| Hiperparâmetro | Valor |
|---|---|
| `n_estimators` | 20 |
| `bootstrap` | `True` (com reposição) |
| `max_samples` | 1,0 (amostra do tamanho do treino) |
| `max_features` | 1,0 (todas as features em cada árvore) |
| árvore-base | `DecisionTreeRegressor(min_samples_leaf=20)` |

## Preparação dos dados

Família "árvore": códigos ordinais, categóricas como códigos inteiros e ausências mantidas. As features foram
convertidas para números sem padronização.

## Estratégia

- **O problema da árvore única:** ela decora o treino (alta variância, ver
  [arvore_decisao.md](arvore_decisao.md)).
- **A solução do bagging:** treinar árvores em amostras diferentes e tirar a média. Os erros de cada árvore se
  compensam, e a média fica estável.
- **Por que folha mínima 20:** impede folhas de 1 pessoa e mantém o tamanho das árvores viável na memória.

## Resultados

| | Dobra 1 | Dobra 2 | Dobra 3 | Média ± DP | Treino |
|---|---:|---:|---:|---:|---:|
| RMSE | 65,18 | 65,27 | 65,33 | **65,26 ± 0,08** | 61,17 |
| MAE | 50,21 | 50,28 | 50,24 | 50,24 | 47,04 |
| R² | 0,252 | 0,253 | 0,252 | **0,253** | 0,343 |

Tempo: 30 s de ajuste por dobra (8 núcleos em paralelo), 117 s no total.

## Diagnóstico

- **A média corrige a árvore única:** de 92,34 para 65,26.
- **Ainda há sobreajuste moderado:** treino 61,2 contra validação 65,3, uma diferença de 4 pontos. As árvores ainda
  decoram parte do treino.
- **Melhor que linear e stacking**, mas 0,8 ponto atrás do Boosting.

## Hiperparâmetros completos

Todos os parâmetros do estimador usado, lidos do código. **Negrito** = valor escolhido pela equipe; os demais são o padrão do scikit-learn 1.9.

`n_jobs = 8` é definido pelo script (opção `--n-jobs`) e não muda o resultado.

**BaggingRegressor**

| Parâmetro | Valor | Padrão do scikit-learn? | Significado |
|---|---|---|---|
| `bootstrap` | True | sim | sorteio com reposição (True) ou sem (False) |
| `bootstrap_features` | False | sim | sorteia features com reposição |
| `max_features` | 1,0 | sim | fração de features consideradas |
| **`max_samples`** | 1,0 | não (escolhido) | fração de linhas sorteadas por árvore |
| **`n_estimators`** | 20 | não (escolhido) | número de árvores |
| `n_jobs` | None | sim | núcleos de CPU em paralelo |
| `oob_score` | False | sim | calcula erro fora da amostra |
| **`random_state`** | 42 | não (escolhido) | semente da aleatoriedade |
| `verbose` | 0 | sim | nível de mensagens |
| `warm_start` | False | sim | reaproveita ajuste anterior |

**Árvore-base (`DecisionTreeRegressor`)**

| Parâmetro | Valor | Padrão do scikit-learn? | Significado |
|---|---|---|---|
| `ccp_alpha` | 0,0 | sim | poda por custo-complexidade (0 = sem poda) |
| `criterion` | squared_error | sim | medida de erro para escolher divisões |
| `max_depth` | None | sim | profundidade máxima (None = sem limite) |
| `max_features` | None | sim | fração de features consideradas |
| `max_leaf_nodes` | None | sim | máximo de folhas (None = sem limite) |
| `min_impurity_decrease` | 0,0 | sim | redução mínima de erro para dividir |
| **`min_samples_leaf`** | 20 | não (escolhido) | mínimo de participantes por folha |
| `min_samples_split` | 2 | sim | mínimo de participantes para dividir um nó |
| `min_weight_fraction_leaf` | 0,0 | sim | fração mínima do peso total por folha |
| `monotonic_cst` | None | sim | restrições monotônicas (None = nenhuma) |
| **`random_state`** | 42 | não (escolhido) | semente da aleatoriedade |
| `splitter` | best | sim | como escolher a divisão (melhor ou aleatória) |

## Para o pitch

- **A ideia do ensemble em uma frase:** muitas árvores instáveis viram um modelo estável.
- **Do pior para o 2º melhor:** a mesma árvore passa de último lugar (92) para 2º (65) só com amostragem e média.

**Reproduzir:** `python -m src.modelagem.bagging`. ID: `LC-C07-BAG`.
