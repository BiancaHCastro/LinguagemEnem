# Boosting (HistGradientBoosting): busca, aprofundamento e modelo final

**Resumo:** foi o 1º colocado na comparação inicial (RMSE 64,50). Testamos **10 estratégias de hiperparâmetros**, cada
uma com hipótese própria, e **5 variações de treino** para os erros nas notas extremas. A melhor estratégia (**HGB-6**)
chegou a **RMSE 64,43 na validação** e **64,32 no teste**, e é a **solução final** do trabalho.

## 1. Ponto de partida

- **Configuração inicial:** taxa 0,1, até 200 árvores de 31 folhas, *early stopping* (ver
  [docs/modelos/boosting.md](../modelos/boosting.md)).
- **RMSE inicial: 64,50 ± 0,09**, com treino em 64,30.
- **Por que valia buscar:** o modelo usou **todas as 200 árvores** nas 3 dobras, porque o *early stopping* nunca
  parou o treino. Mais árvores com uma taxa menor poderiam ajudar.

## 2. Busca de hiperparâmetros: `GridSearchCV` com 10 estratégias

- **Formato:** em vez de sortear valores ao acaso, cada uma das 10 configurações testa **uma ideia diferente**. Todas
  rodaram no `GridSearchCV`, nas mesmas 3 dobras.
- **Comum a todas:**
  - perda quadrática;
  - *early stopping* com 10% **do treino da dobra**;
  - categóricas nativas;
  - semente 42.
- **O que varia:** os 8 hiperparâmetros da tabela abaixo.

| Hiperparâmetro | O que controla | Valores testados |
|---|---|---|
| `learning_rate` | quanto cada árvore corrige | 0,02 · 0,04 · 0,0605 · 0,1 |
| `max_iter` | máximo de árvores (o *early stopping* decide quantas usar) | 400 a 4.000 |
| `max_leaf_nodes` | tamanho de cada árvore | 15 · 63 · 127 · 255 |
| `max_depth` | profundidade máxima | sem limite · 4 |
| `min_samples_leaf` | mínimo de participantes por folha | 20 · 50 · 200 · 300 · 1.000 |
| `l2_regularization` | penalidade nos valores das folhas | 0 · 0,1 · 1 · 10 |
| `max_features` | fração de features sorteadas **em cada divisão** | 0,4 · 0,5 · 0,6 · 1,0 |
| `n_iter_no_change` | paciência do *early stopping* | 10 · 30 · 50 |

### As 10 estratégias e os resultados (validação cruzada)

| # | ID | Estratégia | Hipótese | Taxa | Árvores (máx.) | Folhas | Folha mín. | L2 | Features/divisão | RMSE | MAE | R² | RMSE treino | Ajuste/dobra |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **HGB-6** | **lento + sorteio de 50% das features** | as duas melhores ideias somadas superam cada uma | 0,02 | 4.000 | 127 | 300 | 1 | **0,5** | **64,433** | **49,55** | **0,2714** | 63,82 | 54 s |
| 2 | HGB-7 | sorteio agressivo | sortear menos features descorrelaciona mais | 0,04 | 1.500 | 127 | 300 | 1 | 0,4 | 64,441 | 49,55 | 0,2712 | 63,83 | 29 s |
| 3 | HGB-4 | sorteio de 60% das features | sorteio por divisão (como no Random Forest) | 0,04 | 1.500 | 127 | 300 | 1 | 0,6 | 64,444 | 49,56 | 0,2711 | 63,79 | 29 s |
| 4 | HGB-10 | rasas + sorteio | o sorteio também ajuda árvores rasas | 0,1 | 2.000 | 15 (prof. 4) | 50 | 0 | 0,6 | 64,453 | 49,57 | 0,2709 | 64,19 | 31 s |
| 5 | HGB-2 | lento e longo | passos pequenos generalizam melhor | 0,02 | 3.000 | 63 | 200 | 1 | 1,0 | 64,454 | 49,57 | 0,2709 | 63,96 | 54 s |
| 6 | HGB-8 | grandes, regularizadas + sorteio | sorteio complementa a regularização | 0,04 | 1.500 | 255 | 1.000 | 10 | 0,6 | 64,455 | 49,56 | 0,2709 | 63,76 | 28 s |
| 7 | HGB-5 | rasas e muitos estágios | se o sinal é quase aditivo, árvores rasas bastam | 0,1 | 2.000 | 15 (prof. 4) | 50 | 0 | 1,0 | 64,456 | 49,57 | 0,2708 | 64,17 | 29 s |
| 8 | HGB-1 | melhor da busca anterior (15 features) | as 8 features novas do C07 reduzem o erro | 0,0605 | 400 | 127 | 200 | 0,1 | 1,0 | 64,466 | 49,57 | 0,2706 | 63,96 | 15 s |
| 9 | HGB-9 | pouca regularização | com 1,5 mi de linhas, folhas pequenas não sobreajustam | 0,04 | 1.500 | 127 | 20 | 0 | 1,0 | 64,467 | 49,57 | 0,2706 | 63,82 | 23 s |
| 10 | HGB-3 | grandes e muito regularizadas | muitas folhas + L2 alto captam interações sem ruído | 0,04 | 1.500 | 255 | 1.000 | 10 | 1,0 | 64,476 | 49,58 | 0,2704 | 63,74 | 26 s |

Tempo total da busca: 24 minutos. RMSE do vencedor por dobra: 64,34 · 64,44 · 64,52 (DP 0,08).

### O que a busca mostrou

- **Ganho pequeno sobre a configuração inicial:** de 64,50 para 64,43.
- **As 10 estratégias empatam na prática:** ficam a menos de **0,05 ponto** entre si, abaixo do desvio entre as
  dobras (0,08). Tamanho das árvores, regularização e taxa quase não importam aqui.
- **Só uma ideia teve efeito consistente:** **sortear features em cada divisão** (`max_features < 1`) aparece nas 3
  primeiras posições. Sortear 40–60% das features evita que todas as árvores dependam das mesmas variáveis, como a
  renda.
- **Sem sobreajuste em nenhuma estratégia:** treino e validação ficam a no máximo 0,74 ponto em todas.
- **Alternativa mais simples:** o enunciado permite escolher um modelo mais simples com desempenho semelhante. O
  **HGB-4** fica a 0,01 ponto do HGB-6 e **treina na metade do tempo**. Mantivemos o HGB-6 pelo critério de menor RMSE.

## 3. Aprofundamento: os erros nas notas extremas podem ser reduzidos?

**Pergunta:** o modelo erra muito nas notas raras: superestima quem tirou menos de 400 e subestima quem tirou mais
de 700. Testamos 5 mudanças **na forma de treinar**, partindo do HGB-6, nas mesmas dobras.

| ID | Variação | Hipótese | RMSE | MAE | RMSE notas 0–400 | RMSE notas 700+ | Conclusão |
|---|---|---|---:|---:|---:|---:|---|
| V0 | referência (HGB-6) | — | 64,43 | 49,55 | 134,3 | 136,8 | — |
| V1 | treino sem as 1.736 notas zero (retiradas só do treino de cada dobra) | os zeros puxam as previsões para baixo | 64,43 | 49,51 | 134,6 | 136,2 | sem efeito |
| V2 | perda absoluta (`absolute_error`) | menos sensível às notas extremas | 64,72 | **49,34** | 139,5 | **130,9** | melhora as notas altas e piora as baixas |
| V3 | alvo transformado (quantis, `QuantileTransformer`) | abrir as caudas leva o modelo a prever extremos | 64,53 | 49,37 | 137,7 | 133,9 | mesmo padrão da V2, mais fraco |
| V4 | restrições monotônicas (renda, renda per capita, computadores) | um modelo mais interpretável sem perder precisão | 64,49 | 49,59 | 134,3 | 136,5 | perda pequena (+0,05) |
| V5 | árvores de 31 folhas, em vez de 127 | um modelo menor mantém o erro | 64,44 | 49,55 | 134,3 | 137,0 | mesmo erro, mas 3,5× mais lento |

- **Nenhuma variação reduziu os erros nos extremos sem piorar o restante.**
- **Perda absoluta e alvo transformado** trocam erro de um lado por erro do outro: melhoram as notas altas e pioram as
  baixas.
- **Conclusão:** o problema dos extremos é **falta de informação nas features**, não a forma de treinar. As variações
  ficam como resultado do aprofundamento, e a solução final segue o HGB-6 sem mudanças.
- **Arquivos:** `resultados/variacoes_hgb.csv` e `figuras/variacoes_hgb_erro_por_faixa.png`.

## 4. Modelo final: curva de aprendizado e teste reservado

**Curva de aprendizado** (HGB-6, 10% a 100% do treino de cada dobra; `figuras/curva_aprendizado_hgb_final.png`):

| Treino da dobra | RMSE treino | RMSE validação |
|---:|---:|---:|
| 150 mil | 62,63 | 64,81 |
| 753 mil | 63,44 | 64,50 |
| 1,51 mi | 63,81 | **64,44** |

- **A diferença entre treino e validação cai de 2,2 para 0,6 ponto:** não há sobreajuste.
- **Mais dados quase não ajudam:** dez vezes mais treino melhoram a validação em só 0,4 ponto.

**Teste reservado** (564.529 participantes, avaliados uma única vez; `resultados/metricas_teste_final.csv`):

| Modelo | RMSE | MAE | R² |
|---|---:|---:|---:|
| Baseline (média) | 75,33 | 59,19 | 0,000 |
| Regressão linear sem regularização | 65,46 | 50,45 | 0,245 |
| **HistGradientBoosting HGB-6 (final)** | **64,32** | **49,42** | **0,271** |

**Erro por faixa de nota no teste:**

| Faixa | RMSE | Viés |
|---|---:|---|
| 500–600 (a faixa mais comum) | 43,8 | — |
| abaixo de 400 | 134,7 | superestima em 127 |
| acima de 700 | 135,1 | subestima em 131 |

**Por tipo de escola:** privada 61,7, pública 64,0 (`figuras/teste_hgb_final_*.png`).

## Hiperparâmetros do modelo final (HGB-6)

| Parâmetro | Valor | Mudou em relação à configuração inicial? |
|---|---|---|
| `learning_rate` | 0,02 | sim (era 0,1) |
| `max_iter` | 4.000 (o *early stopping* decide quantas usar) | sim (era 200) |
| `max_leaf_nodes` | 127 | sim (era 31) |
| `max_depth` | None | não |
| `min_samples_leaf` | 300 | sim (era 20) |
| `l2_regularization` | 1,0 | sim (era 0) |
| `max_features` | **0,5** | sim (era 1,0) |
| `early_stopping` / `validation_fraction` / `n_iter_no_change` | True / 0,1 / 50 | paciência: sim (era 10) |
| `loss` / `categorical_features` / `random_state` | squared_error / from_dtype / 42 | não |

## Para o pitch

- **10 estratégias com hipótese, não um sorteio cego.** Todas empataram: o Boosting já estava no limite.
- **Sortear features em cada divisão** foi o único ajuste com efeito consistente.
- **O aprofundamento mostrou um *trade-off*:** perda absoluta e alvo transformado melhoram as notas altas, mas pioram
  as baixas e o RMSE.
- **Validação 64,43 × teste 64,32:** o resultado se confirmou em dados nunca vistos.

**Reproduzir:**
```bash
python -m src.modelagem.boosting --so-busca                # busca (GridSearchCV, 10 estratégias)
python -m src.modelagem.aprofundamento_notas_extremas      # 5 variações
python -m src.modelagem.boosting --curva                   # curva de aprendizado
python -m src.modelagem.regressao_linear --teste           # baseline e linear no teste
python -m src.modelagem.boosting --teste                   # modelo final no teste + gráficos
```
