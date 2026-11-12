# Baseline: `DummyRegressor(strategy="mean")`

**Resumo:** prevê a mesma nota para todos, a média do treino (~518). Erra **75,48 pontos (RMSE)** e é a régua para
medir os outros modelos.

## Arquitetura

Não usa nenhuma feature. No treino de cada dobra calcula a média das notas e prevê esse valor para todos os
participantes da validação. É obrigatório pelo enunciado.

## Estratégia

Mostrar quanto erraríamos **sem saber nada** do participante. O ganho de qualquer modelo é medido contra esse erro.
A média é recalculada no treino de cada dobra (e nos 80% do desenvolvimento no teste final), nunca com a validação ou
com o teste.

## Resultados

| | Dobra 1 | Dobra 2 | Dobra 3 | Média ± DP |
|---|---:|---:|---:|---:|
| RMSE | 75,39 | 75,52 | 75,54 | **75,48 ± 0,08** |
| MAE | 59,26 | 59,42 | 59,36 | 59,35 |
| R² | 0,000 | 0,000 | 0,000 | 0,000 |

- **RMSE de treino:** 75,48.
- **Tempo:** 30 s no total.
- **Teste reservado:** RMSE 75,33, MAE 59,19.

## Hiperparâmetros completos

Todos os parâmetros do estimador usado, lidos do código. **Negrito** = valor escolhido pela equipe; os demais são o padrão do scikit-learn 1.9.

| Parâmetro | Valor | Padrão do scikit-learn? | Significado |
|---|---|---|---|
| `constant` | None | sim | valor fixo (só para strategy='constant') |
| `quantile` | None | sim | quantil (só para strategy='quantile' ou perda quantílica) |
| `strategy` | mean | sim | o que prever (média do treino) |

## Para o pitch

- O RMSE do baseline (75,5) é praticamente o **desvio padrão das notas** (75,5): sem features, o erro é a própria
  dispersão das notas.
- O melhor modelo reduz esse erro em **~11 pontos (−15%)**.

**Reproduzir:** `python -m src.modelagem.regressao_linear`. ID no `experimentos.csv`: `LC-C07-BASE`.
