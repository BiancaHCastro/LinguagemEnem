# Pasting: `BaggingRegressor(bootstrap=False, max_samples=0.63)` de árvores de decisão

**Resumo:** igual ao bagging, mas cada árvore recebe uma amostra **sem reposição**. Ficou em **3º lugar (RMSE 65,27,
R² 0,252)**, praticamente empatado com o bagging, e seguiu para a busca de hiperparâmetros.

## Arquitetura

1. Do treino da dobra, sorteia **63% das linhas sem reposição**: cada participante aparece no máximo uma vez por
   árvore. Amostras de árvores diferentes podem compartilhar participantes.
2. Treina uma árvore de decisão com **folhas de pelo menos 20 participantes**.
3. Repete **20 vezes** e prevê a **média**.

| Hiperparâmetro | Valor |
|---|---|
| `n_estimators` | 20 |
| `bootstrap` | `False` (sem reposição) |
| `max_samples` | **0,63** |
| `max_features` | 1,0 |
| árvore-base | `DecisionTreeRegressor(min_samples_leaf=20)` (a mesma do bagging) |

## Preparação dos dados

Família "árvore", idêntica à do bagging.

## Estratégia

- **Comparação justa com o bagging:** o bootstrap com amostra do tamanho do treino dá a cada árvore ~63%
  (1 − 1/e) de participantes **distintos**. Com `max_samples = 0,63`, o pasting dá a mesma quantidade de
  participantes distintos, só que **sem repetições**. A única diferença entre os dois é a reposição.
- **Por que `max_samples` menor que 1:** com 100%, todas as árvores veriam as mesmas linhas e não haveria diversidade.

## Resultados

| | Dobra 1 | Dobra 2 | Dobra 3 | Média ± DP | Treino |
|---|---:|---:|---:|---:|---:|
| RMSE | 65,17 | 65,29 | 65,36 | **65,27 ± 0,09** | 61,16 |
| MAE | 50,21 | 50,29 | 50,27 | 50,26 | 47,04 |
| R² | 0,253 | 0,253 | 0,251 | **0,252** | 0,344 |

Tempo: 29 s de ajuste por dobra, 108 s no total (um pouco mais rápido que o bagging, porque usa amostras menores).

## Diagnóstico

- **Empate técnico com o bagging:** a diferença é de 0,01 ponto, menor que o desvio entre as dobras. **Sortear com ou
  sem reposição não muda o resultado** quando a quantidade de participantes distintos é a mesma.
- **Mesmo sobreajuste moderado do bagging:** treino 61,2 contra validação 65,3.

## Hiperparâmetros completos

Todos os parâmetros do estimador usado, lidos do código. **Negrito** = valor escolhido pela equipe; os demais são o padrão do scikit-learn 1.9.

`n_jobs = 8` é definido pelo script (opção `--n-jobs`) e não muda o resultado.

**BaggingRegressor**

| Parâmetro | Valor | Padrão do scikit-learn? | Significado |
|---|---|---|---|
| **`bootstrap`** | False | não (escolhido) | sorteio com reposição (True) ou sem (False) |
| `bootstrap_features` | False | sim | sorteia features com reposição |
| `max_features` | 1,0 | sim | fração de features consideradas |
| **`max_samples`** | 0,63 | não (escolhido) | fração de linhas sorteadas por árvore |
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

- **Experimento controlado:** isolamos uma única variável (com × sem reposição) e mostramos que ela não importa
  aqui.
- Bagging e Pasting são famílias diferentes no enunciado, embora usem a mesma classe do scikit-learn.

**Reproduzir:** `python -m src.modelagem.pasting`. ID: `LC-C07-PAST`.
