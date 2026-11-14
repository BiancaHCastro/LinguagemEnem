# Modelos: como cada um foi treinado e o resultado

Um arquivo por modelo, com arquitetura, configuração, preparação dos dados, estratégia, resultados, **tabela com
todos os hiperparâmetros** (o que a equipe escolheu × o padrão do scikit-learn) e pontos para o pitch. Todos os números são da **configuração inicial** (antes da busca de hiperparâmetros), na **validação
cruzada de 3 dobras** do desenvolvimento.

## Protocolo comum a todos os modelos

| Item | Como |
|---|---|
| Dados | conjunto **C07** (23 features), **2.258.114 participantes** do desenvolvimento (80%); o teste (20%) não foi usado aqui |
| Validação | `KFold(3, shuffle=True, random_state=42)`: cada dobra treina em ~1,5 milhão e valida em ~750 mil; **mesmas dobras para todos** |
| Sem vazamento | a preparação (codificação, imputação, escala, PCA) fica dentro de um `Pipeline` e é ajustada **só no treino de cada dobra** |
| Métricas | **RMSE** (principal) e MAE, em pontos da nota, menor é melhor; **R²** é a fração da variação das notas explicada |
| Diagnóstico | RMSE de treino × validação: diferença grande indica sobreajuste |

## Ranking da comparação inicial

| # | Modelo | RMSE | MAE | R² | RMSE treino | Tempo total | Detalhes |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | **Boosting (HistGradientBoosting)** | **64,50 ± 0,09** | **49,61** | **0,270** | 64,30 | 51 s | [boosting.md](boosting.md) |
| 2 | Bagging | 65,26 ± 0,08 | 50,24 | 0,253 | 61,17 | 117 s | [bagging.md](bagging.md) |
| 3 | Pasting | 65,27 ± 0,09 | 50,26 | 0,252 | 61,16 | 108 s | [pasting.md](pasting.md) |
| 4 | Stacking (árvore + Ridge) | 65,30 ± 0,10 | 50,31 | 0,252 | 65,27 | 142 s | [stacking.md](stacking.md) |
| 5 | Ridge | 65,54 ± 0,11 | 50,54 | 0,246 | 65,53 | 32 s | [regressao_linear.md](regressao_linear.md) |
| 6 | Regressão linear | 65,54 ± 0,11 | 50,54 | 0,246 | 65,53 | 39 s | [regressao_linear.md](regressao_linear.md) |
| 7 | KNN | 70,86 ± 0,05 | 54,51 | 0,119 | 23,42 | 764 s | [knn.md](knn.md) |
| — | Baseline (média) | 75,48 ± 0,08 | 59,35 | 0,000 | 75,48 | 30 s | [baseline.md](baseline.md) |
| 8 | Árvore de decisão | 92,34 ± 0,03 | 71,16 | −0,497 | 23,32 | 75 s | [arvore_decisao.md](arvore_decisao.md) |

**Random Forest e SVM** (outro integrante): mesmas dobras e mesmo conjunto, mas em scripts próprios. Por isso
ainda não estão no `experimentos.csv` nem no ranking acima, e não há RMSE de treino registrado.

| Modelo | RMSE | MAE | R² | Ajuste/dobra | Detalhes |
|---|---:|---:|---:|---:|---|
| SVM (`LinearSVR`) | 65,40 ± 0,11 | 50,42 | 0,249 | 25 s | [svm.md](svm.md) |
| Random Forest: versão usada na escolha do top 3 (50 árvores, profundidade 12) | 65,36 | 50,33 | 0,250 | — | [random_forest.md](random_forest.md) |
| Random Forest: script atual (config. do Bagging otimizado + sorteio por divisão) | 65,24 ± 0,09 | 50,22 | 0,253 | 79 s | [random_forest.md](random_forest.md) |

- **O RF e o SVM ficaram fora do top 3:** na escolha, o RF fez 65,36 e o SVM 65,40, os dois atrás do Pasting (65,27).
- **A versão atual do RF** parte da configuração do Bagging otimizado. Ela serve para comparar RF × Bagging (65,24 ×
  65,01), e não para entrar no ranking das configurações iniciais.
- **Os 3 melhores** (Boosting, Bagging e Pasting) seguiram para a busca de hiperparâmetros: estratégias, todas as
  configurações testadas e resultados em [`docs/top3/`](../top3/README.md).

## Três mensagens para o pitch

1. **Boosting venceu e generaliza bem:** menor erro e treino quase igual à validação (64,3 × 64,5).
2. **Mostramos sub e sobreajuste com números:**
   - a árvore sem limite decora o treino (23 × 92) e fica pior que prever a média. Com 500 participantes por folha
     (estudo de complexidade, `LC-C07-ARV-BUSCA`), cai para 65,59, com treino em 65,25;
   - Bagging e Pasting decoram parte do treino (61 × 65).
3. **Todos os bons modelos param em R² ~0,25–0,27:** o limite está na informação do perfil socioeconômico, não no
   algoritmo.
