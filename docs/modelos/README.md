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

- **Random Forest e SVM** são do outro integrante. O RF fez **65,36** na validação, entre o Stacking e o Ridge, e
  ficou fora do top 3.
- **Os 3 melhores** (Boosting, Bagging e Pasting) seguiram para a busca de hiperparâmetros: estratégias, todas as
  configurações testadas e resultados em [`docs/top3/`](../top3/README.md).

## Três mensagens para o pitch

1. **Boosting venceu e generaliza bem:** menor erro e treino quase igual à validação (64,3 × 64,5).
2. **Mostramos sub e sobreajuste com números:**
   - a árvore sem limite decora o treino (23 × 92) e fica pior que prever a média;
   - Bagging e Pasting decoram parte do treino (61 × 65).
3. **Todos os bons modelos param em R² ~0,25–0,27:** o limite está na informação do perfil socioeconômico, não no
   algoritmo.
