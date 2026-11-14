# Random Forest: `RandomForestRegressor`

**Resumo:** muitas árvores, cada uma treinada numa amostra com reposição e **sorteando features em cada divisão**.
Há duas versões:
- a **usada na escolha do top 3** (50 árvores, profundidade 12) fez **65,36** e ficou atrás do Pasting;
- a **do script atual** parte da configuração do Bagging otimizado e fez **65,24 ± 0,09 (R² 0,253)**, ainda **0,23
  pior que o Bagging otimizado (65,01)**.

Script do outro integrante: `src/modelagem/random_forest.py`.

## Arquitetura

1. Do treino da dobra, sorteia **75% das linhas com reposição** para cada árvore (`max_samples = 0,75`).
2. Em **cada divisão** de cada árvore, sorteia **√23 ≈ 4 features** (`max_features = "sqrt"`) e escolhe a melhor
   pergunta só entre elas.
3. As árvores crescem até **profundidade 24**, com folhas de pelo menos **20 participantes**.
4. Repete para **40 árvores** e prevê a **média**.

**A diferença para o Bagging** (o enunciado pede essa comparação):

| | Bagging otimizado | Random Forest (script atual) |
|---|---|---|
| Árvores | 40 | 40 |
| Linhas por árvore | 75%, com reposição | 75%, com reposição |
| Folha mínima | 20 | 20 |
| **Sorteio de features** | **80% por árvore** (cada árvore usa 18 das 23 em todas as divisões) | **√23 ≈ 4 em cada divisão** |
| Profundidade máxima | sem limite | 24 |
| **RMSE validação** | **65,01** | 65,24 |

**A ideia do sorteio em cada divisão:** sem ele, quase todas as árvores começam pelas mesmas features fortes (idade,
renda) e ficam parecidas. Sorteando em cada divisão, as árvores ficam mais diferentes, e a média deveria errar menos.

## Preparação dos dados (dentro do Pipeline, ajustada no treino de cada dobra)

| Colunas | Tratamento |
|---|---|
| 8 numéricas | vazios → valor constante **−999** (a árvore separa esse grupo sozinha) |
| 15 códigos | `OrdinalEncoder`: cada código vira um número, na ordem alfabética do código; categoria não vista → −1 |

Sem padronização: árvores não dependem de escala.

## Resultados (validação cruzada, mesmas 3 dobras)

**Script atual** (`resultados/resultados_random_forest.csv`):

| | Dobra 1 | Dobra 2 | Dobra 3 | Média ± DP |
|---|---:|---:|---:|---:|
| RMSE | 65,14 | 65,25 | 65,31 | **65,24 ± 0,09** |
| MAE | 50,18 | 50,27 | 50,23 | 50,22 |
| R² | 0,253 | 0,254 | 0,253 | **0,253** |

- **Tempo:** 79 s de ajuste e 10 s de previsão por dobra.
- **Baseline nas mesmas dobras:** 75,48, igual ao `experimentos.csv` (mesmas dobras).
- **Comparação RF − Bagging otimizado:** RMSE **+0,23**, MAE +0,18, R² −0,005.
- **Figura:** `resultados/figuras/desempenho_random_forest_vs_bagging.png`.

**Versão usada na escolha do top 3** (50 árvores, profundidade 12, demais parâmetros no padrão): RMSE **65,36**,
MAE 50,33, R² 0,250. O notebook foi substituído pelo script atual e continua no histórico do Git (commit `3c76778`,
`src/modelagem/random_forest.ipynb`).

## Diagnóstico

- **Fora do top 3:** na escolha, o RF fez 65,36, atrás de Boosting (64,50), Bagging (65,26) e Pasting (65,27).
- **Contra o Bagging otimizado, o RF perdeu por 0,23.** A comparação muda duas coisas ao mesmo tempo (o sorteio
  passa a ser por divisão e a profundidade é limitada a 24), então não isola o efeito de cada uma.
  - Uma explicação plausível: sortear só 4 de 23 features em cada divisão é agressivo. Muitas divisões ficam sem as
    features fortes.
  - No Boosting, sortear 40–60% por divisão ajudou (`docs/top3/boosting.md`), o que sugere que a fração importa
    mais que o tipo de sorteio.
- **Por que a versão atual não entra no ranking inicial:** ela já parte da configuração do Bagging otimizado, que
  saiu de uma busca. Compará-la às configurações iniciais dos outros modelos daria vantagem a ela. A comparação justa
  é com o Bagging otimizado, que é a que o script faz.
- **Sem RMSE de treino registrado:** não dá para comparar treino × validação.

## Hiperparâmetros completos (script atual)

**Negrito** = valor escolhido no script; os demais são o padrão do scikit-learn 1.9.

| Parâmetro | Valor | Padrão do scikit-learn? | Significado |
|---|---|---|---|
| **`n_estimators`** | 40 | não (padrão 100) | número de árvores |
| `bootstrap` | True | sim | amostras com reposição |
| **`max_samples`** | 0,75 | não (padrão: 100%) | fração de linhas por árvore |
| **`max_features`** | "sqrt" (≈ 4 de 23) | não (padrão 1,0) | features sorteadas **em cada divisão** |
| **`max_depth`** | 24 | não (padrão: sem limite) | profundidade máxima |
| **`min_samples_leaf`** | 20 | não (padrão 1) | mínimo de participantes por folha |
| `min_samples_split` | 2 | sim | mínimo para dividir um nó |
| `criterion` | squared_error | sim | erro usado para escolher divisões |
| `ccp_alpha` | 0,0 | sim | poda (0 = sem poda) |
| `oob_score` | False | sim | erro fora da amostra |
| **`n_jobs`** | −1 | não | todos os núcleos (não muda o resultado) |
| **`random_state`** | 42 | não (escolhido) | semente |

## Observações sobre o script

- **Caminhos do Google Colab:** `CAMINHO_PARQUET = "/content/enem_2023_lc_c07.parquet"` e
  `PASTA_SAIDA = "/content/resultados_random_forest"`. Para rodar no repositório, troque por
  `dados/processed/enem_2023_lc_c07.parquet` e `resultados/`.
- **Os números do Bagging otimizado estão escritos no código** (copiados de `busca_LC-C07-BAG-BUSCA.csv`); o
  Bagging não é treinado de novo.
- **O script também avalia no teste** (RMSE 65,07, R² 0,254). Pelo protocolo, só o baseline, a regressão linear e a
  solução final vão ao teste. Esse número não participou de nenhuma escolha e não deve ser usado para comparar
  modelos.
- **Ordem alfabética dos códigos:** alguns códigos numéricos guardados como texto (ex.: faixa etária "1", "10", "2"…)
  ficam fora da ordem natural. A árvore ainda consegue separá-los, mas com mais divisões.
- **Ainda não registrado** no `experimentos.csv` nem no ranking de `comparacao_inicial.csv`.

## Para o pitch

- **O que o enunciado pede:** comparar RF e Bagging. Mesma configuração, só mudando o sorteio para "em cada
  divisão": o RF fez 65,24 contra 65,01 do Bagging otimizado.
- **A lição:** sortear features só ajuda se a fração for adequada. Com 4 de 23 por divisão, o RF perdeu informação
  demais.

**Reproduzir:** `python src/modelagem/random_forest.py`, depois de ajustar os caminhos.
