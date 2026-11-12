# Modelo final: HistGradientBoostingRegressor (estratégia HGB-6)

**Resumo:** o melhor modelo do trabalho prevê a nota de Linguagens do ENEM 2023 com **erro médio de 64,3 pontos (RMSE)**
e explica **27,1% da variação das notas (R²)** em 564.529 participantes que ele nunca viu. Prever a média para todos erra
75,3 pontos. Ele usa só o perfil socioeconômico, escolar e demográfico (23 features, nenhuma nota da prova).

## Ficha técnica

| Item | Valor |
|---|---|
| Algoritmo | `HistGradientBoostingRegressor` (scikit-learn 1.9.1) |
| Estratégia | **HGB-6**: aprendizado lento (taxa 0,02) + sorteio de 50% das features em cada divisão |
| Alvo | `NU_NOTA_LC`, nota de Linguagens, Códigos e suas Tecnologias |
| Features | conjunto **C07**: 23 colunas (14 originais do Inep + 9 criadas pela equipe) |
| Treino final | 2.258.114 participantes (100% do desenvolvimento) |
| Árvores usadas | **1.145** (o *early stopping* parou antes do teto de 4.000) |
| Tamanho das árvores | até 127 folhas, cada folha com pelo menos 300 participantes |
| Validação cruzada (3 dobras) | **RMSE 64,43 ± 0,08 · MAE 49,55 · R² 0,2714** |
| **Teste reservado** | **RMSE 64,32 · MAE 49,42 · R² 0,2711** |
| Tempo de treino | ~55 s por dobra; ~1,5 min no ajuste final (CPU) |

---

## 1. Como chegamos nele

```
24 conjuntos de features ─► C07 (2º em todos os modelos, 23 colunas)
            │
9 modelos na mesma validação ─► top 3: Boosting 64,50 · Bagging 65,26 · Pasting 65,27
            │
busca de hiperparâmetros (30 configurações) ─► Boosting HGB-6 64,43 · Bagging 65,01 · Pasting 65,03
            │
aprofundamento (5 variações de treino) ─► nenhuma superou o HGB-6
            │
teste reservado (uma única vez) ─► RMSE 64,32 · R² 0,271
```

| Etapa | O que decidiu | Detalhes |
|---|---|---|
| Features | o C07 ficou em 2º nos 3 modelos da triagem, com menos da metade das colunas do completo | [resumo_conjuntos_features.md](resumo_conjuntos_features.md) |
| Modelo | o Boosting foi o 1º na comparação inicial (64,50), com treino e validação quase iguais | [modelos/](modelos/README.md) |
| Configuração | o HGB-6 teve o menor RMSE entre 10 estratégias | [top3/boosting.md](top3/boosting.md) |
| Teste | avaliado uma vez, junto com o baseline e a regressão linear, nos mesmos participantes | seção 6 |

---

## 2. Dados e features

- **Participantes:** 2.822.643 com nota de Linguagens.
  - Inclui treineiros, todas as idades e as 2.169 notas zero.
  - **Desenvolvimento (80%):** 2.258.114, usados para treinar e escolher.
  - **Teste (20%):** 564.529, usados só no fim.
- **Vazamento:** nenhuma nota da prova, resposta, gabarito ou presença entra no modelo.

| Tipo | Features | Como o modelo recebe |
|---|---|---|
| Ordinais | renda familiar (`Q006`), escolaridade do pai/mãe (`Q001`, `Q002`), ocupação do pai/mãe (`Q003`, `Q004`), computadores (`Q024`), internet (`Q025`), `ESCOLARIDADE_MAX_RESP`, `OCUPACAO_MAX_RESP` | código na ordem do dicionário (A = 0, B = 1, …). "Não sei" recebe um código depois do último nível |
| Categóricas | tipo de escola (`TP_ESCOLA`), situação no ensino médio (`TP_ST_CONCLUSAO`), faixa etária (`TP_FAIXA_ETARIA`), UF da prova (`SG_UF_PROVA`), `PERFIL_ACESSO_DIGITAL`, `TIPO_ESCOLA_DETALHADO` | **categorias nativas** do HistGB (divide por grupos de categorias, sem one-hot) |
| Numéricas | moradores (`Q005`), `RENDA_PER_CAPITA_APROX`, `CAPITAL_CULTURAL_RESP`, `N_RESP_NAO_SEI`, `ANOS_DESDE_CONCLUSAO` | valor original; **ausências mantidas** (o modelo aprende para que lado mandar quem não tem o valor) |
| Binárias | treineiro (`IN_TREINEIRO`), língua estrangeira (`TP_LINGUA`), `ATRASO_ESCOLAR` | 0/1 |

- **Sem padronização nem imputação:** o modelo não precisa delas.
- **Sem vazamento na preparação:** toda a preparação fica num `Pipeline` e é ajustada só no treino
  (`src/conjuntos/preprocessamento.py`, família "boosting").

---

## 3. Como o modelo funciona

1. Começa prevendo a nota média.
2. Treina uma **árvore de até 127 folhas** para prever o **erro atual** de cada participante. Cada divisão da árvore
   olha só um **sorteio de 50% das features**.
3. Soma à previsão **2% (taxa 0,02)** do que essa árvore indica: passos pequenos e cuidadosos.
4. Repete. A cada árvore mede o erro em **10% do treino separados para isso** (*early stopping*) e **para quando o
   erro não melhora por 50 árvores seguidas**. No modelo final isso aconteceu na **árvore 1.145**.
5. **Histogramas:** cada feature é dividida em até 255 faixas antes do treino, o que torna o treino rápido em milhões
   de linhas.

**Por que essa estratégia venceu:**

- **Sorteio de features:** como cada divisão vê só metade das features, as árvores não dependem sempre das mesmas
  variáveis (renda, idade). Isso deixa o conjunto mais robusto. Foi a única ideia com efeito consistente na busca: as
  3 melhores estratégias sorteiam features.
- **Taxa baixa com muitas árvores:** passos pequenos generalizam um pouco melhor que poucos passos grandes.
- **Folhas de pelo menos 300 participantes e L2 = 1:** impedem que uma folha se ajuste a poucos participantes, ou
  seja, a ruído.

---

## 4. Hiperparâmetros completos

| Parâmetro | Valor | Configuração inicial | Significado |
|---|---|---|---|
| **`learning_rate`** | **0,02** | 0,1 | quanto cada árvore corrige |
| **`max_iter`** | **4.000** (usou 1.145) | 200 | teto de árvores; o *early stopping* decide quantas usar |
| **`max_leaf_nodes`** | **127** | 31 | tamanho máximo de cada árvore |
| `max_depth` | None | None | sem limite de profundidade (o nº de folhas limita) |
| **`min_samples_leaf`** | **300** | 20 | mínimo de participantes por folha |
| **`l2_regularization`** | **1,0** | 0 | penalidade nos valores das folhas |
| **`max_features`** | **0,5** | 1,0 | fração de features sorteadas em cada divisão |
| `early_stopping` | True | True | para quando a validação interna não melhora |
| `validation_fraction` | 0,1 | 0,1 | 10% **do treino** separados para o *early stopping* |
| **`n_iter_no_change`** | **50** | 10 | paciência do *early stopping* |
| `scoring` | loss | loss | métrica do *early stopping* (erro quadrático) |
| `loss` | squared_error | squared_error | função de perda: erro quadrático, que é o que o RMSE mede |
| `categorical_features` | from_dtype | from_dtype | as 6 categóricas como categorias nativas |
| `max_bins` | 255 | 255 | faixas do histograma por feature |
| `tol` | 1e-07 | 1e-07 | tolerância do *early stopping* |
| `random_state` | 42 | 42 | semente (sorteio de features e da parte do *early stopping*) |
| `interaction_cst` / `monotonic_cst` | None / None | None / None | sem restrições |
| `warm_start` / `verbose` / `quantile` | False / 0 / None | iguais | não usados |

Em **negrito**, o que a busca mudou em relação à configuração inicial.

---

## 5. Validação cruzada (desenvolvimento)

| | Dobra 1 | Dobra 2 | Dobra 3 | Média ± DP | Treino |
|---|---:|---:|---:|---:|---:|
| RMSE | 64,34 | 64,44 | 64,52 | **64,43 ± 0,08** | 63,82 |
| MAE | — | — | — | **49,55** | — |
| R² | — | — | — | **0,2714** | — |
| Árvores usadas | 860 | 959 | 977 | — | — |

**Evolução no desenvolvimento:**

| Etapa | RMSE |
|---|---:|
| Baseline | 75,48 |
| Configuração inicial do Boosting | 64,50 |
| **HGB-6** | **64,43** |

---

## 6. Teste reservado (resultado oficial)

Os três modelos foram reajustados nos 2.258.114 participantes do desenvolvimento e avaliados **uma única vez** nos
564.529 do teste, nos mesmos participantes.

| Modelo | RMSE | MAE | R² |
|---|---:|---:|---:|
| Baseline (prevê a média) | 75,33 | 59,19 | 0,000 |
| Regressão linear sem regularização | 65,46 | 50,45 | 0,245 |
| **HistGradientBoosting HGB-6** | **64,32** | **49,42** | **0,271** |

- **Contra o baseline:** −11,0 pontos de RMSE (−14,6%) e −9,8 pontos de MAE.
- **Contra a regressão linear:** −1,1 ponto de RMSE. Esse é o ganho de captar **interações** entre fatores.
- **Teste igual à validação** (64,32 × 64,43 ± 0,08): o resultado **se confirma em dados novos**, e a escolha não foi
  otimista.
- **Arquivos:** `resultados/metricas_teste_final.csv` e `resultados/previsoes_teste.csv`, com a previsão de cada
  participante.

---

## 7. Curva de aprendizado (sub e sobreajuste)

Reajuste do modelo com 10%, 25%, 50%, 75% e 100% do treino de cada dobra, em subconjuntos aninhados com semente 42 e
a validação fixa (`figuras/curva_aprendizado_hgb_final.png`).

| Treino da dobra | RMSE treino | RMSE validação | Diferença |
|---:|---:|---:|---:|
| 150 mil (10%) | 62,63 ± 0,22 | 64,81 ± 0,08 | 2,2 |
| 376 mil (25%) | 63,25 ± 0,12 | 64,61 ± 0,09 | 1,4 |
| 753 mil (50%) | 63,44 ± 0,04 | 64,50 ± 0,09 | 1,1 |
| 1,13 mi (75%) | 63,70 ± 0,11 | 64,46 ± 0,09 | 0,8 |
| 1,51 mi (100%) | 63,81 ± 0,18 | 64,44 ± 0,09 | 0,6 |

- **Não há sobreajuste:** com todos os dados, treino e validação ficam a 0,6 ponto.
- **Mais dados quase não ajudam:** dez vezes mais treino melhoram a validação em só 0,4 ponto, e a curva já está
  plana.
- **O limite não é falta de complexidade:** os erros são altos e próximos, e modelos mais complexos (busca, variações)
  não mudaram o resultado. O que falta é **informação**: o perfil socioeconômico não explica ~73% da variação da nota
  (motivação, preparo, condições do dia da prova etc.).

---

## 8. Onde o modelo acerta e onde erra (teste)

### Por faixa de nota (`figuras/teste_hgb_final_erro_por_faixa.png`)

| Faixa | Participantes | RMSE | Viés (observada − prevista) |
|---|---:|---:|---:|
| zero | 433 | 512,8 | −510,9 |
| 0–400 | 36.217 | 134,7 | −127,3 |
| 400–500 | 176.481 | 55,0 | −38,7 |
| **500–600** | **278.114** | **43,8** | +22,9 |
| 600–700 | 71.758 | 80,8 | +72,2 |
| 700+ | 1.526 | 135,1 | +130,9 |

**O modelo puxa as previsões para a média:**

- **As previsões são estreitas:** vão de **382 a 663**, com desvio padrão de **39 pontos**. As notas reais têm desvio
  de **75**.
- **Consequência:** acerta bem o centro (500–600, metade dos participantes), **superestima quem tirou pouco** e
  **subestima quem tirou muito**.
- **Na média o modelo não erra para nenhum lado:** o resíduo médio é de +0,17.

### Por grupo de perfil

| Grupo | Melhor RMSE | Pior RMSE |
|---|---|---|
| Tipo de escola | privada 61,7 | não respondeu 64,8 (pública: 64,0) |
| Renda familiar (`Q006`) | faixa N 60,1 | faixa B (até R$ 1.320) 65,6 |
| UF da prova | SC 60,7 · RS 61,1 · SP 61,5 | CE 67,8 · PI 67,3 · RR 67,0 |

- **Sem viés sistemático por grupo:** o viés fica entre −1,5 e +2,7 pontos em todos. O maior é o do Acre (+2,6,
  com 3.255 participantes), pequeno perto do RMSE de 64.
- **O erro é maior onde as notas variam mais** entre pessoas de perfil parecido (menor renda e alguns estados do
  Nordeste e Norte).
- **Tabela completa:** `resultados/erros_teste_LC-C07-HGB.csv`.

---

## 9. Quais features mais pesam

Importância por permutação: embaralhamos uma feature por vez e medimos **quanto o RMSE piora**. Feito na validação da
dobra 1 (100 mil participantes, 5 repetições), **sem usar o teste** (`resultados/importancia_features_hgb_final.csv`).

| # | Feature | Aumento do RMSE |
|---:|---|---:|
| 1 | `ANOS_DESDE_CONCLUSAO` (anos desde o fim do ensino médio, feature criada) | 2,40 |
| 2 | `TP_LINGUA` (inglês ou espanhol) | 1,82 |
| 3 | `TP_FAIXA_ETARIA` (faixa etária) | 1,76 |
| 4 | `RENDA_PER_CAPITA_APROX` (renda por pessoa, feature criada) | 1,15 |
| 5 | `SG_UF_PROVA` (UF da prova) | 1,06 |
| 6 | `TIPO_ESCOLA_DETALHADO` (rede da escola, feature criada) | 0,63 |
| 7 | `TP_ST_CONCLUSAO` (situação no ensino médio) | 0,36 |
| 8–10 | computadores (`Q024`), ocupação do pai (`Q003`), escolaridade do pai (`Q001`) | 0,29–0,32 |

**Como ler:**

- **Três das seis mais importantes foram criadas pela equipe:** tempo desde a conclusão, renda por pessoa e rede da
  escola.
- **Trajetória escolar e idade pesam mais que renda:** quem terminou o ensino médio há muito tempo ou está fora da
  idade esperada tende a ter nota mais baixa.
- **Atenção a features correlacionadas:** a permutação **subestima** cada uma delas, porque as outras compensam.
  - A renda familiar (`Q006`, 0,15) aparece baixa porque a renda per capita carrega a mesma informação.
  - Por isso a **ablação por blocos** (retirar renda, escolaridade etc. inteiros) completa esta leitura (README
    principal, seção 11).

---

## 10. O que testamos e não adotamos

| Tentativa | Resultado | Por que não entrou |
|---|---|---|
| HGB-4 (sorteio de 60%, taxa 0,04) | RMSE 64,44 | empata com o HGB-6 e treina na **metade do tempo**. É uma alternativa mais simples defensável; mantivemos o critério de menor RMSE |
| Treino sem as notas zero | 64,43 | sem efeito |
| Perda absoluta | 64,72 | melhora notas altas e MAE, piora notas baixas e RMSE |
| Alvo transformado (quantis) | 64,53 | o mesmo *trade-off*, mais fraco |
| Restrições monotônicas | 64,49 | mais interpretável, mas +0,05 de erro |
| Árvores de 31 folhas | 64,44 | mesmo erro e 3,5× mais lento |
| PCA antes do boosting (teste com 15 features) | piorou 0,7 ponto | o HistGB já trata categorias; o PCA perde informação |
| Balanceamento das faixas de nota (teste com 15 features) | piorou 0,5 a 15 pontos | melhorar os extremos custa caro no centro, onde está a maioria |

---

## 11. Limitações

- **Teto de R² ~0,27:** o perfil socioeconômico explica pouco mais de 1/4 da nota. O resto depende de fatores que não
  estão nos dados.
- **Notas extremas:** o modelo não distingue bem quem vai muito bem ou muito mal dentro do mesmo perfil.
- **Escopo da generalização:** a avaliação vale para **novos participantes do ENEM 2023**, não para outros anos,
  escolas ou municípios inéditos.
- **Associação, não causa:** o modelo descreve padrões entre perfil e nota. Não mede a capacidade de ninguém nem
  explica por que a nota é alta ou baixa.

---

## 12. Reproduzir

```bash
python -m src.modelagem.boosting                     # configuração inicial
python -m src.modelagem.boosting --so-busca          # busca: escolhe o HGB-6
python -m src.modelagem.boosting --curva             # curva de aprendizado
python -m src.modelagem.regressao_linear --teste     # baseline e regressão linear no teste
python -m src.modelagem.boosting --teste             # modelo final no teste + gráficos e erros
```

| Arquivo | Conteúdo |
|---|---|
| `resultados/experimentos.csv` | linhas `LC-C07-HGB`, `LC-C07-HGB-BUSCA` e `LC-C07-HGB-TESTE` |
| `resultados/busca_LC-C07-HGB-BUSCA.csv` | as 10 estratégias |
| `resultados/metricas_teste_final.csv` · `previsoes_teste.csv` | teste reservado |
| `resultados/curva_aprendizado_LC-C07-HGB.csv` | curva de aprendizado |
| `resultados/erros_teste_LC-C07-HGB.csv` | erros por faixa, UF, escola e renda |
| `resultados/importancia_features_hgb_final.csv` | importância por permutação |
| `resultados/figuras/teste_hgb_final_*.png`, `curva_aprendizado_hgb_final.png` | gráficos do artigo |

---

## Para o pitch: o modelo final em 5 frases

1. **HistGradientBoosting, com sorteio de features e aprendizado lento: erra 64 pontos e explica 27% da nota.**
2. **Ganha do baseline em 11 pontos e da regressão linear em 1 ponto.** A diferença para a linear vem das interações
   entre fatores.
3. **Validação 64,43 e teste 64,32:** o resultado se confirma em dados novos.
4. **Sem sobreajuste, e mais dados não ajudam:** o limite está na informação do perfil, não no modelo.
5. **Puxa as notas para o centro:** acerta a maioria (500–600) e erra os extremos. Trajetória escolar, idade, língua e
   renda são o que mais pesa.
