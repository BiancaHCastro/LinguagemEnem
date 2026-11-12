# Pasting: busca de hiperparâmetros

**Resumo:** foi o 3º colocado na comparação inicial (RMSE 65,27). Uma busca aleatória com **10 configurações** levou
o erro a **65,03**, um ganho de 0,24 ponto. A melhor configuração usa **40 árvores**, **63% das linhas sem reposição**
e **80% das features**. Ficou empatada com o bagging e 0,6 ponto atrás do Boosting.

## 1. Ponto de partida

- **Configuração inicial:** 20 árvores, 63% das linhas **sem reposição**, folha mínima 20 (ver
  [docs/modelos/pasting.md](../modelos/pasting.md)).
- **RMSE inicial: 65,27 ± 0,09**, com treino em 61,16.
- **Por que buscar:** a mesma pergunta do bagging. O treino estava 4 pontos abaixo da validação, sinal de
  sobreajuste.

## 2. Busca: `RandomizedSearchCV`, 10 configurações

| Hiperparâmetro | O que controla | Valores possíveis | Por que testar |
|---|---|---|---|
| `n_estimators` | número de árvores | 10 · 20 · 40 | mais árvores = média mais estável |
| `max_samples` | fração de linhas por árvore, **sem reposição** | 0,2 · 0,4 · 0,63 · 0,8 | **sempre menor que 1**: com 100%, todas as árvores veriam as mesmas linhas |
| `max_features` | fração de features por árvore | 0,6 · 0,8 · 1,0 | diversidade entre árvores |
| `min_samples_leaf` (árvore) | mínimo de participantes por folha | 5 · 20 · 50 · 100 | tamanho das árvores e sobreajuste |

- **Fixos:** `bootstrap = False` (sem reposição), árvore sem limite de profundidade e semente 42.
- **Tamanho do espaço:** 144 combinações, das quais **10 foram sorteadas**.

### Resultados das 10 configurações (validação cruzada)

| # | Árvores | Linhas/árvore | Features/árvore | Folha mín. | RMSE | MAE | R² | RMSE treino | Ajuste/dobra |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | **40** | **0,63** | **0,8** | **20** | **65,029** | **50,06** | **0,2578** | 61,46 | 131 s |
| 2 | 40 | 0,4 | 1,0 | 20 | 65,050 | 50,07 | 0,2573 | 62,35 | 138 s |
| 3 | 20 | 0,2 | 1,0 | 50 | 65,106 | 50,12 | 0,2560 | 64,49 | 29 s |
| 4 | 20 | 0,8 | 0,6 | 50 | 65,128 | 50,16 | 0,2556 | 63,78 | 46 s |
| 5 | 10 | 0,2 | 1,0 | 100 | 65,264 | 50,25 | 0,2524 | 64,94 | 17 s |
| 6 | 10 | 0,8 | 0,6 | 100 | 65,369 | 50,37 | 0,2500 | 64,68 | 13 s |
| 7 | 10 | 0,63 | 1,0 | 20 | 65,417 | 50,38 | 0,2489 | 61,30 | 53 s |
| 8 | 20 | 0,63 | 0,8 | 5 | 65,802 | 50,71 | 0,2401 | 54,53 | 24 s |
| 9 | 10 | 0,2 | 0,8 | 5 | 66,022 | 50,90 | 0,2350 | 61,98 | 17 s |
| 10 | 10 | 0,63 | 0,8 | 5 | 66,255 | 51,09 | 0,2296 | 55,04 | 50 s |

- **Tempo total da busca:** 33 minutos.
- **RMSE da melhor por dobra:** 64,93 · 65,04 · 65,11 (DP 0,07).
- **Sobre os tempos:** variaram bastante entre execuções (carga da máquina). Para comparar custo, use o bagging, que
  rodou nas mesmas condições.

### O que a busca mostrou

- **O mesmo padrão do bagging:**
  - **40 árvores** nas duas melhores posições;
  - **folha 5 nas 3 piores**, com o treino caindo para 54–55: sobreajuste;
  - **folha 20 é o melhor tamanho** quando há árvores suficientes.
- **A fração de linhas importa pouco:** de 20% a 80% sem reposição, todas aparecem entre as melhores. O número de
  árvores e o tamanho da folha decidem.
- **Com ou sem reposição dá no mesmo:** a melhor configuração do pasting (65,03) empata com a do bagging (65,01),
  dentro do desvio entre as dobras (0,07).

## 3. Antes × depois

| | Árvores | Linhas/árvore | Features/árvore | Folha mín. | RMSE | MAE | R² | RMSE treino |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Inicial | 20 | 0,63 | 1,0 | 20 | 65,27 | 50,26 | 0,252 | 61,16 |
| **Após a busca** | **40** | **0,63** | **0,8** | **20** | **65,03** | **50,06** | **0,258** | 61,46 |

- **O ganho de 0,24 ponto veio de dobrar as árvores e sortear 80% das features.**
- **Ainda 0,6 ponto atrás do Boosting (64,43):** o pasting não entrou na solução final.

## Para o pitch

- **Bagging e Pasting chegaram ao mesmo lugar** (65,01 contra 65,03). Sortear com ou sem reposição não muda o
  resultado nem antes nem depois da busca.
- **Os dois ensinam a mesma lição:** mais árvores e folhas de tamanho moderado. Folhas muito pequenas levam ao
  sobreajuste.

**Reproduzir:** `python -m src.modelagem.pasting --so-busca`. IDs: `LC-C07-PAST-BUSCA` e
`resultados/busca_LC-C07-PAST-BUSCA.csv`.
