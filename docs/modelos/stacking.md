# Stacking: `StackingRegressor` (árvore + Ridge → Ridge)

**Resumo:** combina uma árvore de decisão e uma regressão Ridge com um terceiro modelo, que aprende quanto confiar em
cada um. Ficou em **4º lugar (RMSE 65,30, R² 0,252)**. Supera os dois modelos de base, mas fica atrás de Bagging,
Pasting e Boosting.

## Arquitetura

```
                ┌─ árvore de decisão (profundidade 8, folha ≥ 100) ─┐
participante ───┤                                                    ├─► Ridge final (alpha=1) ─► nota
                └─ Ridge (alpha=10) ────────────────────────────────┘
```

| Peça | Configuração | Preparação própria |
|---|---|---|
| Base 1: árvore | `DecisionTreeRegressor(max_depth=8, min_samples_leaf=100)` | família "árvore": códigos ordinais, NaN mantido |
| Base 2: Ridge | `Ridge(alpha=10)` | família "linear": one-hot, padronização, imputação + indicador, `log1p` na renda |
| Combinador | `Ridge(alpha=1)` | recebe as 2 previsões |
| Validação interna | `KFold(3, shuffle=True, random_state=42)` dentro do treino de cada dobra | — |

**Pesos aprendidos pelo combinador** (dobra 1): **Ridge 0,66** e **árvore 0,38**, com intercepto −19,4.

## Como o combinador é treinado sem vazamento

1. **Dentro do treino de cada dobra externa**, o treino é dividido em 3 partes.
2. Cada modelo de base treina em 2 partes e prevê a 3ª. Assim **cada previsão vem de um modelo que não viu aquele
   participante** (previsões fora da dobra, como exige o enunciado).
3. O combinador aprende os pesos com essas previsões.
4. Os modelos de base são reajustados no treino inteiro da dobra e o conjunto é avaliado na validação, que não
   participou de nada.

## Estratégia

- **Combinar dois tipos de acerto:** a árvore capta **interações** e o Ridge capta **efeitos aditivos suaves**.
- **Configuração do notebook original da equipe.** Mudou só a preparação do Ridge de base: antes ele recebia UF e
  tipo de escola como números, e agora usa one-hot.

## Resultados

| | Dobra 1 | Dobra 2 | Dobra 3 | Média ± DP | Treino |
|---|---:|---:|---:|---:|---:|
| RMSE | 65,19 | 65,31 | 65,39 | **65,30 ± 0,10** | 65,27 |
| MAE | 50,25 | 50,35 | 50,33 | 50,31 | 50,29 |
| R² | 0,252 | 0,252 | 0,251 | **0,252** | 0,252 |

**Comparação com os modelos de base** (mesmas dobras):

| Modelo | RMSE |
|---|---:|
| Árvore (profundidade 8) sozinha | 66,01 |
| Ridge sozinho | 65,54 |
| **Stacking** | **65,30** |

Tempo: 40 s de ajuste por dobra (cada base treina 4 vezes), 142 s no total.

## Diagnóstico

- **A combinação funciona:** o stacking erra menos que cada base sozinha (−0,7 ponto contra a árvore, −0,24 contra
  o Ridge).
- **Sem sobreajuste:** treino 65,27 contra validação 65,30.
- **Bases fracas limitam o resultado:** o stacking não alcança o Boosting.

## Hiperparâmetros completos

Todos os parâmetros do estimador usado, lidos do código. **Negrito** = valor escolhido pela equipe; os demais são o padrão do scikit-learn 1.9.

**StackingRegressor**

| Parâmetro | Valor | Padrão do scikit-learn? | Significado |
|---|---|---|---|
| **`cv`** | KFold(n_splits=3, random_state=42, shuffle=True) | não (escolhido) | validação interna que gera as previsões fora da dobra |
| **`n_jobs`** | 1 | não (escolhido) | núcleos de CPU em paralelo |
| `passthrough` | False | sim | passa as features originais ao combinador |
| `verbose` | 0 | sim | nível de mensagens |

**Base 1: árvore (`DecisionTreeRegressor`)**

| Parâmetro | Valor | Padrão do scikit-learn? | Significado |
|---|---|---|---|
| `ccp_alpha` | 0,0 | sim | poda por custo-complexidade (0 = sem poda) |
| `criterion` | squared_error | sim | medida de erro para escolher divisões |
| **`max_depth`** | 8 | não (escolhido) | profundidade máxima (None = sem limite) |
| `max_features` | None | sim | fração de features consideradas |
| `max_leaf_nodes` | None | sim | máximo de folhas (None = sem limite) |
| `min_impurity_decrease` | 0,0 | sim | redução mínima de erro para dividir |
| **`min_samples_leaf`** | 100 | não (escolhido) | mínimo de participantes por folha |
| `min_samples_split` | 2 | sim | mínimo de participantes para dividir um nó |
| `min_weight_fraction_leaf` | 0,0 | sim | fração mínima do peso total por folha |
| `monotonic_cst` | None | sim | restrições monotônicas (None = nenhuma) |
| **`random_state`** | 42 | não (escolhido) | semente da aleatoriedade |
| `splitter` | best | sim | como escolher a divisão (melhor ou aleatória) |

**Base 2: `Ridge`**

| Parâmetro | Valor | Padrão do scikit-learn? | Significado |
|---|---|---|---|
| **`alpha`** | 10,0 | não (escolhido) | força da penalidade L2 (maior = coeficientes menores) |
| `copy_X` | True | sim | copia X antes de ajustar |
| `fit_intercept` | True | sim | ajusta o intercepto |
| `max_iter` | None | sim | máximo de iterações |
| `positive` | False | sim | força coeficientes ≥ 0 |
| `random_state` | None | sim | semente da aleatoriedade |
| `solver` | auto | sim | algoritmo de otimização |
| `tol` | 0,0001 | sim | tolerância de convergência |

**Combinador: `Ridge`**

| Parâmetro | Valor | Padrão do scikit-learn? | Significado |
|---|---|---|---|
| `alpha` | 1,0 | sim | força da penalidade L2 (maior = coeficientes menores) |
| `copy_X` | True | sim | copia X antes de ajustar |
| `fit_intercept` | True | sim | ajusta o intercepto |
| `max_iter` | None | sim | máximo de iterações |
| `positive` | False | sim | força coeficientes ≥ 0 |
| `random_state` | None | sim | semente da aleatoriedade |
| `solver` | auto | sim | algoritmo de otimização |
| `tol` | 0,0001 | sim | tolerância de convergência |

## Para o pitch

- **Treino do combinador sem vazamento:** as previsões vêm de modelos que não viram o participante.
- **O combinador confia mais no Ridge** (66% do peso). Os efeitos aditivos explicam mais que as interações de uma
  árvore rasa.

**Reproduzir:** `python -m src.modelagem.stacking`. ID: `LC-C07-STACK`.
