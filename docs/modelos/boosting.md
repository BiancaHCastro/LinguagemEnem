# Boosting: `HistGradientBoostingRegressor`

**Resumo:** árvores pequenas treinadas **em sequência**, cada uma corrigindo o erro das anteriores. **1º lugar na
comparação inicial (RMSE 64,50, R² 0,270)**, com treino e validação quase iguais. Seguiu para a busca e virou a
**solução final**.

## Arquitetura

1. Começa prevendo a média.
2. Treina uma **árvore pequena (até 31 folhas)** para prever o **erro** atual (resíduo) de cada participante.
3. Soma à previsão **10% (taxa de aprendizado 0,1)** do que essa árvore indica.
4. Repete até **200 árvores**, ou até o erro parar de cair numa parte separada do treino (*early stopping*).
5. **"Hist"** de histograma: antes de treinar, cada feature é dividida em até **255 faixas**. Isso torna o treino
   rápido em milhões de linhas.
6. **Categorias nativas:** UF, tipo de escola, situação escolar, faixa etária e perfil digital são divididos como
   **grupos de categorias**, sem one-hot.

| Hiperparâmetro | Valor |
|---|---|
| `learning_rate` | 0,1 |
| `max_iter` | 200 |
| `max_leaf_nodes` | 31 |
| `min_samples_leaf` | 20 |
| `l2_regularization` | 0 |
| `early_stopping` | sim: 10% **do treino da dobra** (`validation_fraction = 0,1`), paciência de 10 iterações |
| `categorical_features` | as 6 categóricas nominais |
| `random_state` | 42 |

**Iterações usadas:** 200, 200 e 200. O *early stopping* nunca parou o treino, o que indica que mais árvores com taxa
menor ainda poderiam ajudar. Isso motivou a busca.

## Preparação dos dados (família "boosting")

| Etapa | Como |
|---|---|
| Ordinais | código na ordem do dicionário; "não sei" depois do último nível |
| Categóricas nominais | marcadas como `category`, para as divisões nativas do HistGB |
| Ausências | mantidas como NaN; o HistGB aprende para que lado mandar quem não tem o valor |
| Escala | nenhuma |

## Estratégia

- **Diferença para o bagging:** em vez de muitas árvores independentes tiradas em média, o boosting faz árvores
  **dependentes**, cada uma focada no que as anteriores erraram.
- **Árvores pequenas e taxa baixa:** captam interações (renda × escola × idade) com **pouco sobreajuste**.
- **Por que o HistGB:** é a versão do scikit-learn feita para bases grandes, com categorias e ausências nativas.
- **Origem da configuração:** a mesma do script original da equipe, agora com as 23 features do C07.

## Resultados

| | Dobra 1 | Dobra 2 | Dobra 3 | Média ± DP | Treino |
|---|---:|---:|---:|---:|---:|
| RMSE | 64,41 | 64,52 | 64,59 | **64,50 ± 0,09** | 64,30 |
| MAE | 49,55 | 49,65 | 49,63 | **49,61** | 49,46 |
| R² | 0,270 | 0,270 | 0,269 | **0,270** | 0,274 |

Tempo: **9,5 s de ajuste por dobra**, 51 s no total. Foi o melhor modelo e um dos mais rápidos.

## Diagnóstico

- **Sem sobreajuste:** treino 64,30 contra validação 64,50, uma diferença de só 0,2 ponto (no bagging são 4).
- **Melhor que todos:** 0,8 ponto à frente do bagging e 1,0 da linear. A diferença para a linear é o ganho das
  **interações**.
- **Os erros são altos e próximos entre treino e validação.** Isso indica **falta de informação nas features**, não
  falta de complexidade, e a curva de aprendizado do modelo final confirma.

## Depois da busca (resumo; detalhes no README principal)

- A busca com 10 estratégias levou a validação de 64,50 para **64,43**.
- No **teste reservado**, o modelo final fez **RMSE 64,32, MAE 49,42 e R² 0,271**.

## Hiperparâmetros completos

Todos os parâmetros do estimador usado, lidos do código. **Negrito** = valor escolhido pela equipe; os demais são o padrão do scikit-learn 1.9.

| Parâmetro | Valor | Padrão do scikit-learn? | Significado |
|---|---|---|---|
| `categorical_features` | from_dtype | sim | como identificar as categóricas |
| **`early_stopping`** | True | não (escolhido) | para quando a validação interna não melhora |
| `interaction_cst` | None | sim | restrições de interação (None = nenhuma) |
| `l2_regularization` | 0,0 | sim | penalidade L2 nas folhas |
| `learning_rate` | 0,1 | sim | taxa de aprendizado (peso de cada árvore) |
| `loss` | squared_error | sim | função de perda |
| `max_bins` | 255 | sim | faixas do histograma por feature |
| `max_depth` | None | sim | profundidade máxima (None = sem limite) |
| `max_features` | 1,0 | sim | fração de features consideradas |
| **`max_iter`** | 200 | não (escolhido) | máximo de iterações |
| `max_leaf_nodes` | 31 | sim | máximo de folhas (None = sem limite) |
| `min_samples_leaf` | 20 | sim | mínimo de participantes por folha |
| `monotonic_cst` | None | sim | restrições monotônicas (None = nenhuma) |
| `n_iter_no_change` | 10 | sim | paciência do early stopping (iterações) |
| `quantile` | None | sim | quantil (só para strategy='quantile' ou perda quantílica) |
| **`random_state`** | 42 | não (escolhido) | semente da aleatoriedade |
| `scoring` | loss | sim | métrica do early stopping |
| `tol` | 1e-07 | sim | tolerância de convergência |
| `validation_fraction` | 0,1 | sim | fração do treino da dobra separada para o early stopping |
| `verbose` | 0 | sim | nível de mensagens |
| `warm_start` | False | sim | reaproveita ajuste anterior |

## Para o pitch

- **O melhor modelo é também o que menos sobreajusta:** treino e validação praticamente iguais.
- **Rápido em milhões de linhas:** histogramas, categorias e ausências nativas.
- **Indício de margem para melhorar:** usou todas as 200 iterações, e isso justificou a busca de hiperparâmetros.

**Reproduzir:** `python -m src.modelagem.boosting`. ID: `LC-C07-HGB`.
