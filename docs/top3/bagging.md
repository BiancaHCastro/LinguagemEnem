# Bagging: busca de hiperparâmetros

**Resumo:** foi o 2º colocado na comparação inicial (RMSE 65,26). Uma busca aleatória com **10 configurações** levou
o erro a **65,01**, um ganho de 0,25 ponto. A melhor configuração usa **40 árvores**, **75% das linhas** e **80% das
features** por árvore. Mesmo assim ficou 0,6 ponto atrás do Boosting.

## 1. Ponto de partida

- **Configuração inicial:** 20 árvores, amostras do tamanho do treino com reposição, folha mínima 20 (ver
  [docs/modelos/bagging.md](../modelos/bagging.md)).
- **RMSE inicial: 65,26 ± 0,08**, com treino em 61,17.
- **Por que buscar:** o treino estava 4 pontos abaixo da validação, sinal de sobreajuste. A pergunta era se mais
  árvores, amostras menores ou folhas maiores reduziriam esse sobreajuste e o erro.

## 2. Busca: `RandomizedSearchCV`, 10 configurações

| Hiperparâmetro | O que controla | Valores possíveis | Por que testar |
|---|---|---|---|
| `n_estimators` | número de árvores na média | 10 · 20 · 40 | mais árvores = média mais estável |
| `max_samples` | fração de linhas sorteadas por árvore | 0,3 · 0,5 · 0,75 · 1,0 | amostras menores = árvores mais diferentes entre si |
| `max_features` | fração de features sorteadas **por árvore** | 0,6 · 0,8 · 1,0 | outra forma de diversidade |
| `min_samples_leaf` (árvore) | mínimo de participantes por folha | 5 · 20 · 50 · 100 | controla o tamanho das árvores e o sobreajuste |

- **Fixos:** `bootstrap = True` (com reposição), árvore sem limite de profundidade e semente 42.
- **Tamanho do espaço:** 144 combinações possíveis, das quais **10 foram sorteadas**.

### Resultados das 10 configurações (validação cruzada)

| # | Árvores | Linhas/árvore | Features/árvore | Folha mín. | RMSE | MAE | R² | RMSE treino | Ajuste/dobra |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | **40** | **0,75** | **0,8** | **20** | **65,008** | **50,04** | **0,2583** | 61,99 | 31 s |
| 2 | 40 | 0,5 | 1,0 | 20 | 65,039 | 50,06 | 0,2576 | 62,40 | 29 s |
| 3 | 20 | 0,3 | 1,0 | 50 | 65,074 | 50,09 | 0,2568 | 64,29 | 12 s |
| 4 | 20 | 1,0 | 0,6 | 50 | 65,151 | 50,18 | 0,2550 | 64,05 | 15 s |
| 5 | 10 | 0,3 | 1,0 | 100 | 65,220 | 50,21 | 0,2534 | 64,81 | 8 s |
| 6 | 10 | 0,75 | 1,0 | 20 | 65,374 | 50,34 | 0,2499 | 61,92 | 15 s |
| 7 | 10 | 1,0 | 0,6 | 100 | 65,394 | 50,39 | 0,2494 | 64,84 | 11 s |
| 8 | 20 | 0,75 | 0,8 | 5 | 65,725 | 50,65 | 0,2418 | 56,21 | 20 s |
| 9 | 10 | 0,3 | 0,8 | 5 | 66,074 | 50,94 | 0,2338 | 61,01 | 8 s |
| 10 | 10 | 0,75 | 0,8 | 5 | 66,242 | 51,09 | 0,2299 | 56,81 | 13 s |

- **Tempo total da busca:** 11 minutos.
- **RMSE da melhor por dobra:** 64,91 · 65,02 · 65,09 (DP 0,07).

### O que a busca mostrou

- **Mais árvores ajudam:** as duas melhores usam 40 árvores. Com a mesma folha mínima, mais árvores sempre erram
  menos:
  - folha 20: 10 árvores fazem 65,37 e 40 árvores fazem 65,01;
  - folha 5: 10 árvores fazem 66,07–66,24 e 20 árvores fazem 65,73.
- **Folha de 5 participantes é a pior escolha:** as 3 piores configurações usam folha 5. O treino cai para 56, a
  validação sobe para 66: **mais sobreajuste**.
- **Folha de 20 é o ponto certo** quando há árvores suficientes. Com folhas de 50 ou 100 o modelo fica estável, mas
  perde detalhe.
- **Amostras menores e menos features por árvore** (0,75 / 0,8) dão um pouco mais de diversidade e aparecem na
  melhor configuração.

## 3. Antes × depois

| | Árvores | Linhas/árvore | Features/árvore | Folha mín. | RMSE | MAE | R² | RMSE treino |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Inicial | 20 | 1,0 | 1,0 | 20 | 65,26 | 50,24 | 0,253 | 61,17 |
| **Após a busca** | **40** | **0,75** | **0,8** | **20** | **65,01** | **50,04** | **0,258** | 61,99 |

- **O ganho de 0,25 ponto veio de mais árvores e mais diversidade.** A diferença entre treino e validação caiu de 4,1
  para 3,0 pontos.
- **Ainda 0,6 ponto atrás do Boosting (64,43):** o bagging não entrou na solução final.

## Para o pitch

- **O tamanho da folha é o hiperparâmetro que mais pesa:** folha 5 decora o treino, e folha 20 com 40 árvores é o
  melhor equilíbrio.
- **Bagging melhorou mais com a busca que o Boosting** (0,25 contra 0,07 ponto), porque partia de uma configuração
  mais distante do ideal.

**Reproduzir:** `python -m src.modelagem.bagging --so-busca`. IDs: `LC-C07-BAG-BUSCA` e
`resultados/busca_LC-C07-BAG-BUSCA.csv`.
