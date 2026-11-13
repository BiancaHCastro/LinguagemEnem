# Aprendizado de Máquina com Dados do ENEM 2023: nota de Linguagens

Projeto 2 da disciplina de Ciência de Dados (Unidade II). Investiga **quanto o perfil socioeconômico, escolar e
demográfico permite estimar a nota de Linguagens, Códigos e suas Tecnologias (`NU_NOTA_LC`)** do ENEM 2023, sem usar
nenhuma outra nota da prova.

**Resultado:** o melhor modelo foi o **HistGradientBoostingRegressor**. No teste reservado (564.529 participantes)
ele errou em média **64,3 pontos (RMSE)** e explicou **27,1% da variação das notas (R²)**. O baseline, que prevê a
média para todos, erra 75,3 pontos.

| Teste reservado (mesmos participantes) | RMSE | MAE | R² |
|---|---:|---:|---:|
| Baseline (média do treino) | 75,33 | 59,19 | 0,000 |
| Regressão linear sem regularização | 65,46 | 50,45 | 0,245 |
| **HistGradientBoosting (solução final)** | **64,32** | **49,42** | **0,271** |

Ficha completa do modelo final (hiperparâmetros, curva, erros por grupo e importância das features):
[`docs/modelo_final.md`](docs/modelo_final.md).

O teto de ~0,27 de R² se repetiu em todas as famílias de modelos, buscas e variações. O limite está na informação que
o perfil carrega sobre a nota, e não no algoritmo (ver [Diagnóstico](#10-avaliação-final-curva-de-aprendizado-e-erros)).

---

## Sumário

1. [Onde está cada coisa](#1-onde-está-cada-coisa)
2. [Ambiente](#2-ambiente)
3. [Reprodução em uma tela](#3-reprodução-em-uma-tela)
4. [Obter os dados brutos](#4-obter-os-dados-brutos)
5. [Tratamento da base](#5-tratamento-da-base)
6. [Features: criação, triagem e escolha do conjunto C07](#6-features-criação-triagem-e-escolha-do-conjunto-c07)
7. [Base de modelagem e protocolo de avaliação](#7-base-de-modelagem-e-protocolo-de-avaliação)
8. [Comparação inicial dos modelos](#8-comparação-inicial-dos-modelos)
9. [Busca de hiperparâmetros nos 3 melhores](#9-busca-de-hiperparâmetros-nos-3-melhores)
10. [Avaliação final, curva de aprendizado e erros](#10-avaliação-final-curva-de-aprendizado-e-erros)
11. [PCA, balanceamento e ablação](#11-pca-balanceamento-e-ablação)
12. [Aprofundamento: os erros nas notas extremas podem ser reduzidos?](#12-aprofundamento-os-erros-nas-notas-extremas-podem-ser-reduzidos)
13. [Arquivos de resultados e figuras](#13-arquivos-de-resultados-e-figuras)
14. [Limitações e observações](#14-limitações-e-observações)

---

## 1. Onde está cada coisa

```
LinguagemEnem/
├── README.md                  este guia
├── requirements.txt           dependências
├── dados/
│   ├── raw/                   dicionário oficial do Inep (os microdados ficam fora do Git, ver seção 4)
│   ├── processed/             bases tratadas (.parquet, locais) + documentação
│   │   ├── README_BASE_TRATADA.md   regras do tratamento, colunas e qualidade
│   │   ├── README_C07.md            as 23 features do C07, ausências e preparação para cada família de modelo
│   │   └── relatorios/              auditoria do tratamento (totais, hash do CSV, qualidade por coluna)
│   └── splits/
│       └── particoes_lc.parquet     desenvolvimento/teste e dobra de cada participante (versionado)
├── docs/
│   ├── enunciado_projeto2.pdf       enunciado da disciplina
│   ├── acompanhamento.pdf           slides do acompanhamento
│   └── conjuntos_features_lc.md     justificativa de cada feature e dos 24 conjuntos testados
├── src/
│   ├── dados/                 etapa 1: leitura, validação e tratamento dos microdados
│   ├── conjuntos/             etapa 2: engenharia de features, conjuntos candidatos e pré-processamento (Pipeline)
│   └── modelagem/             etapa 3: um arquivo por modelo + comparação + experimentos complementares
│       ├── comum.py                    protocolo comum: dobras, métricas, registro, busca e teste
│       ├── regressao_linear.py         baseline, regressão linear e Ridge
│       ├── arvore_decisao.py           árvore de decisão
│       ├── knn.py                      KNN
│       ├── bagging.py / pasting.py     bagging e pasting de árvores
│       ├── boosting.py                 HistGradientBoosting: inicial, busca, curva, teste e gráficos finais
│       ├── stacking.py                 stacking (árvore + Ridge)
│       ├── random_forest.ipynb / svm.ipynb   Random Forest e SVM (outro integrante; ver seção 14)
│       ├── comparacao_inicial.py       ranking, 3 melhores, antes × depois da busca e figura
│       ├── triagem_conjuntos.py        triagem dos 24 conjuntos de features
│       ├── pca_balanceamento_ablacao.py   PCA, balanceamento e ablação por blocos
│       └── aprofundamento_notas_extremas.py   5 variações do modelo final
└── resultados/
    ├── experimentos.csv       registro de TODOS os experimentos (hipótese, features, preparação, parâmetros,
    │                          métricas por dobra, média, desvio, tempo e conclusão)
    ├── previsoes_teste.csv    nota observada + previsão do baseline, da linear e do modelo final no teste
    ├── figuras/               todos os gráficos do artigo e do pitch
    └── ...                    tabelas de cada etapa (lista completa na seção 13)
```

---

## 2. Ambiente

- Python 3.10 ou mais recente (testado com 3.13.7).
- Instalação: `pip install -r requirements.txt`.
- Versões testadas: pandas 3.0.2, numpy 2.4.1, scikit-learn 1.9.1, matplotlib 3.10.8, scipy 1.18.1 e pyarrow 26.0.0.
- Todas as operações aleatórias usam a **semente 42**. Tudo roda em CPU, com ~4 GB de RAM livre, e na base nacional
  completa, sem amostragem.
- Todos os comandos rodam **na raiz do projeto**. Os scripts de cada modelo aceitam `--amostra 30000`, para um
  teste rápido que não substitui a base completa, e gravam cada experimento em `resultados/experimentos.csv` assim que
  ele termina.

---

## 3. Reprodução em uma tela

Com o CSV do Inep no lugar (seção 4), estes comandos refazem o trabalho inteiro, na ordem:

```bash
# Tratamento e features
python -m src.dados.executar_tratamento               # seção 5: base dos 2.822.643 elegíveis
python -m src.conjuntos.gerar_conjuntos               # seção 6: 24 conjuntos candidatos (amostra de triagem)
python -m src.modelagem.triagem_conjuntos             # seção 6: triagem -> escolha do C07
python -m src.conjuntos.exportar_conjunto C07         # seção 7: base de modelagem + partições

# Comparação inicial: um modelo por comando (seção 8)
python -m src.modelagem.regressao_linear              # baseline, linear e Ridge
python -m src.modelagem.arvore_decisao
python -m src.modelagem.knn
python -m src.modelagem.bagging
python -m src.modelagem.pasting
python -m src.modelagem.boosting
python -m src.modelagem.stacking
python -m src.modelagem.comparacao_inicial            # ranking e 3 melhores

# Busca nos 3 melhores (seção 9)
python -m src.modelagem.boosting --so-busca
python -m src.modelagem.bagging --so-busca
python -m src.modelagem.pasting --so-busca
python -m src.modelagem.comparacao_inicial            # refaz a figura com antes x depois

# Avaliação final (seção 10), só depois de fixar a solução
python -m src.modelagem.boosting --curva
python -m src.modelagem.regressao_linear --teste
python -m src.modelagem.boosting --teste

# Experimentos complementares (seções 11 e 12)
python -m src.modelagem.pca_balanceamento_ablacao completo
python -m src.modelagem.aprofundamento_notas_extremas
python -m src.modelagem.arvore_decisao --so-busca     # estudo de complexidade da árvore (seção 8)
```

Para encadear vários comandos numa linha: no Prompt de Comando (cmd) use `&&`; no PowerShell use `;`.

---

## 4. Obter os dados brutos

Os microdados (1,8 GB) e as bases derivadas não vão para o GitHub.

1. Baixe os **Microdados do ENEM 2023** no portal do Inep:
   <https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem>.
2. Extraia e coloque o arquivo em `microdados_enem_2023/DADOS/MICRODADOS_ENEM_2023.csv`, na raiz do projeto. O
   caminho é definido em `src/dados/configuracao.py`, e outro caminho pode ser passado com `--entrada`.
3. O tratamento confere o arquivo: **3.933.955 linhas**, SHA-256 começando com `c8b1fae0a97b6ac8`, registrado em
   `dados/processed/relatorios/resumo_processamento.json`.

A divisão desenvolvimento/teste, por ser pequena e obrigatória, está versionada em `dados/splits/particoes_lc.parquet`.

---

## 5. Tratamento da base

```bash
python -m src.dados.executar_tratamento
```

**O que faz:** lê o CSV em blocos e lê apenas as colunas permitidas. Valida todos os códigos contra o dicionário do
Inep e gera duas bases em `dados/processed/`:

| Arquivo | Linhas | Uso |
|---|---:|---|
| `enem_2023_lc_diagnostico.parquet` | 3.933.955 | todos os inscritos, para o diagnóstico descritivo |
| `enem_2023_lc_elegiveis.parquet` | 2.822.643 | participantes com nota de LC: a base de modelagem |

**Regras e por quê:**

| Regra | Por quê |
|---|---|
| Entram só os 2.822.643 com nota de LC observada; os 1.111.312 sem nota ficam só no diagnóstico | Sem nota não há valor a prever. Nota ausente não é imputada nem vira zero, como manda o enunciado. |
| As **2.169 notas zero são mantidas** | Zero é uma nota obtida, diferente de nota ausente |
| Treineiros (19,4%) e todas as faixas etárias são mantidos | Excluí-los mudaria a população estudada |
| Nenhuma linha é removida por falta de feature | As ausências carregam informação e são tratadas na modelagem |
| Ficam **fora** de X: notas de outras áreas e da redação, respostas, gabaritos, códigos de prova, presença e situação da redação | Vazamento: são produzidos pela prova ou pela correção |
| `NU_INSCRICAO` só identifica linhas e partições | Não é informação de perfil |

Resultado da auditoria: 0 códigos inválidos, 0 inscrições duplicadas e nenhuma imputação. Detalhes de cada coluna
estão em [`dados/processed/README_BASE_TRATADA.md`](dados/processed/README_BASE_TRATADA.md).

---

## 6. Features: criação, triagem e escolha do conjunto C07

```bash
python -m src.conjuntos.gerar_conjuntos      # monta os 24 conjuntos numa amostra de 300 mil do desenvolvimento
python -m src.modelagem.triagem_conjuntos    # compara os conjuntos com 3 regressores, nas mesmas 3 dobras
```

**Features criadas** (`src/conjuntos/engenharia.py`). As 9 do conjunto final são calculadas só com a própria linha,
sem olhar a nota:

| Feature | Como é calculada | Hipótese |
|---|---|---|
| `RENDA_PER_CAPITA_APROX` | ponto médio da faixa de renda (`Q006`) ÷ moradores (`Q005`); vazio na faixa sem teto | a mesma renda vale menos dividida por mais pessoas |
| `ESCOLARIDADE_MAX_RESP` | maior escolaridade entre pai e mãe; `INDETERMINADA` se algum é "não sei" | um responsável com ensino superior já muda o ambiente |
| `PERFIL_ACESSO_DIGITAL` | computador × internet em 4 perfis | "só internet" costuma ser acesso só pelo celular |
| `OCUPACAO_MAX_RESP` | maior grupo de ocupação entre pai e mãe | ocupação resume renda e capital cultural |
| `CAPITAL_CULTURAL_RESP` | nível de escolaridade + nível de ocupação | as duas juntas descrevem melhor a família |
| `N_RESP_NAO_SEI` | nº de "não sei" nas perguntas sobre os pais | desconhecer os pais sinaliza ausência de responsável |
| `TIPO_ESCOLA_DETALHADO` | federal/estadual/municipal/privada quando informado | separa a rede federal da estadual |
| `ATRASO_ESCOLAR` | conclui o ensino médio em 2023 com 19 anos ou mais | atraso escolar é um marcador forte |
| `ANOS_DESDE_CONCLUSAO` | anos desde a conclusão do ensino médio | tempo longe da escola |

**Triagem (seleção de features):** 24 conjuntos, incluindo originais, novas features, seleção automática
(`SelectKBest`), PCA, clusters e ablações. Todos foram comparados com regressão linear, árvore e boosting na mesma
validação cruzada (`resultados/triagem_conjuntos_lc.csv`). Resultado, em RMSE na validação:

| Conjunto | Colunas | Linear | Árvore | Boosting |
|---|---:|---:|---:|---:|
| C02: 12 originais + 3 derivadas | 15 | 66,32 | 66,70 | 65,61 |
| **C07: C02 + ocupação dos pais + trajetória escolar** | **23** | **65,68** | **66,31** | **64,87** |
| C11: completo | 54 | 64,87 | 66,28 | 64,10 |
| C12: seleção automática (`SelectKBest`, 20 do C11) | 20 | 67,17 | 67,59 | 66,73 |
| C13: PCA global | 24 | 66,44 | 67,93 | 65,94 |

**Por que o C07:**

- Ficou em **2º lugar nos três modelos**.
- Usa menos da metade das colunas do C11 e perde só ~0,8 ponto para ele.
- A seleção automática e o PCA ficaram **piores** que o conjunto escolhido pela equipe.

Resumo de uma página, para a apresentação: [`docs/resumo_conjuntos_features.md`](docs/resumo_conjuntos_features.md).
As 23 features e a preparação de cada uma estão em [`dados/processed/README_C07.md`](dados/processed/README_C07.md).
A justificativa completa está em [`docs/conjuntos_features_lc.md`](docs/conjuntos_features_lc.md).

---

## 7. Base de modelagem e protocolo de avaliação

```bash
python -m src.conjuntos.exportar_conjunto C07
```

Gera `dados/processed/enem_2023_lc_c07.parquet`, com os 2.822.643 participantes, as 23 features, o alvo e as colunas
de controle `CONJUNTO` e `DOBRA`. Também cria `dados/splits/particoes_lc.parquet`, se ele ainda não existir.

**Protocolo** (enunciado, Seção 4), implementado em `src/modelagem/comum.py` e seguido por todos os scripts:

| Etapa | Como |
|---|---|
| Divisão | `train_test_split(test_size=0.2, random_state=42)`: **2.258.114 no desenvolvimento** e **564.529 no teste** |
| Validação cruzada | `KFold(n_splits=3, shuffle=True, random_state=42)` só no desenvolvimento; **as mesmas 3 dobras em todas as comparações** |
| Sem vazamento | codificação, imputação, escala, PCA e seleção ficam num `Pipeline`, ajustado **só no treino de cada dobra**; o early stopping separa 10% do próprio treino da dobra |
| Preparação por família | linear/KNN: níveis ordinais padronizados, "não sei" → mediana + indicador, one-hot, `log1p` na renda e sem colunas redundantes. Árvores/boosting: códigos ordinais, ausências mantidas e categorias nativas no HistGB (`src/conjuntos/preprocessamento.py`) |
| Métricas | RMSE (principal), MAE e R², na escala original, com média e desvio padrão entre as dobras, mais erro de treino e tempo |
| Registro | cada experimento vira uma linha de `resultados/experimentos.csv` |
| Teste | usado **uma única vez**, depois de fixar a solução: baseline, regressão linear sem regularização e modelo final, nos mesmos participantes |

---

## 8. Comparação inicial dos modelos

Cada modelo tem o seu arquivo em `src/modelagem/`. Todos rodam na base completa, com o mesmo protocolo e a sua
configuração inicial registrada antes de qualquer busca. Arquitetura, estratégia e resultados por dobra de cada
modelo: [`docs/modelos/`](docs/modelos/README.md).

```bash
python -m src.modelagem.regressao_linear     # LC-C07-BASE, LC-C07-LIN-OLS, LC-C07-LIN-RIDGE
python -m src.modelagem.arvore_decisao       # LC-C07-ARV
python -m src.modelagem.knn                  # LC-C07-KNN
python -m src.modelagem.bagging              # LC-C07-BAG
python -m src.modelagem.pasting              # LC-C07-PAST
python -m src.modelagem.boosting             # LC-C07-HGB
python -m src.modelagem.stacking             # LC-C07-STACK
python -m src.modelagem.comparacao_inicial   # ranking -> resultados/comparacao_inicial.csv e figuras/comparacao_modelos.png
```

| # | Modelo | Configuração inicial | Por quê | RMSE validação | R² |
|---:|---|---|---|---:|---:|
| 1 | **Boosting** (`HistGradientBoostingRegressor`) | taxa 0,1, até 200 árvores de 31 folhas, early stopping | árvores em sequência corrigem o erro das anteriores e captam interações | **64,50 ± 0,09** | 0,270 |
| 2 | **Bagging** (`BaggingRegressor`, `bootstrap=True`) | 20 árvores, folha mínima 20, amostras do tamanho do treino com reposição | a média de árvores profundas reduz a variância de uma árvore única | 65,26 ± 0,08 | 0,253 |
| 3 | **Pasting** (`BaggingRegressor`, `bootstrap=False`) | 20 árvores, 63% das linhas sem reposição | a mesma fração de participantes distintos do bootstrap, sem repetições | 65,27 ± 0,09 | 0,252 |
| 4 | Stacking (árvore + Ridge → Ridge) | árvore com profundidade 8 + Ridge(10); combinador Ridge(1) treinado com previsões fora da dobra | une interações (árvore) e efeitos aditivos (Ridge) | 65,30 ± 0,10 | 0,252 |
| 5 | Ridge | `alpha=1` | regularização L2 da regressão linear | 65,54 ± 0,11 | 0,246 |
| 6 | Regressão linear | sem regularização | referência obrigatória de modelo aditivo | 65,54 ± 0,11 | 0,246 |
| 7 | KNN | PCA incremental (12 componentes) + 50 vizinhos ponderados pela distância | perfis parecidos têm notas parecidas; o PCA mantém a busca de vizinhos viável em 2,26 mi de linhas | 70,86 ± 0,05 | 0,119 |
| — | Baseline (`DummyRegressor(mean)`) | média do treino de cada dobra | referência obrigatória | 75,48 ± 0,08 | 0,000 |
| 8 | Árvore de decisão | sem limite de crescimento | captar interações sem supor forma funcional | 92,34 ± 0,03 | −0,497 |
| — | Random Forest (outro integrante) | 50 árvores, profundidade 12 | comparação com o bagging | 65,36 | 0,250 |

Leitura dos resultados:

- **A árvore única decora o treino:** RMSE de treino 23,3 contra 92,3 na validação. Fica pior que o baseline e é o
  exemplo claro de sobreajuste.
  - **Estudo de complexidade** (`python -m src.modelagem.arvore_decisao --so-busca`, `LC-C07-ARV-BUSCA`): 15
    estratégias com `GridSearchCV` nas mesmas dobras. Exigir **500 participantes por folha** (ARV-F500) leva a árvore
    a **65,59 (R² 0,245)**, com treino em 65,25.
  - **Isso não muda a comparação inicial:** a configuração inicial continua registrada, como pede o enunciado. Mesmo
    controlada, a árvore fica fora do top 3.
  - Detalhes e figura `arvore_complexidade.png`: [`docs/modelos/arvore_decisao.md`](docs/modelos/arvore_decisao.md).
- **Bagging e Pasting também decoram parte do treino:** 61,2 no treino contra 65,3 na validação.
- **O Boosting tem treino e validação próximos:** 64,3 contra 64,5.
- **Ridge empata com a regressão linear:** com 2,26 milhões de linhas, a regularização não muda nada.
- **Os 3 melhores para a busca são Boosting, Bagging e Pasting.** O Random Forest (65,36) ficou atrás do Pasting e
  fora do top 3.

---

## 9. Busca de hiperparâmetros nos 3 melhores

Todas as buscas usam as mesmas 3 dobras, critério RMSE médio e pelo menos 10 configurações válidas por modelo. A
tabela completa de cada busca fica em `resultados/busca_LC-C07-<modelo>-BUSCA.csv`. Explicação de cada estratégia e
de todas as configurações testadas: [`docs/top3/`](docs/top3/README.md).

```bash
python -m src.modelagem.boosting --so-busca   # GridSearchCV, 10 estratégias
python -m src.modelagem.bagging --so-busca    # RandomizedSearchCV, 10 configurações
python -m src.modelagem.pasting --so-busca    # RandomizedSearchCV, 10 configurações
```

| Modelo | Espaço de busca | Melhor configuração | RMSE inicial → após a busca |
|---|---|---|---|
| **Boosting** | 10 estratégias, cada uma com hipótese própria (abaixo) | **HGB-6:** taxa 0,02, até 4.000 árvores, 127 folhas, folha mínima 300, L2 = 1, **50% das features sorteadas em cada divisão** | **64,50 → 64,43** |
| Bagging | nº de árvores {10, 20, 40}, fração de linhas {0,3; 0,5; 0,75; 1}, fração de features {0,6; 0,8; 1}, folha mínima {5, 20, 50, 100} | 40 árvores, 75% das linhas, 80% das features, folha mínima 20 | 65,26 → 65,01 |
| Pasting | igual ao bagging, com fração de linhas {0,2; 0,4; 0,63; 0,8} | 40 árvores, 63% das linhas, 80% das features, folha mínima 20 | 65,27 → 65,03 |

**As 10 estratégias do Boosting:**

| ID | Estratégia | RMSE |
|---|---|---:|
| HGB-6 | aprendizado lento (taxa 0,02) + sorteio de 50% das features | **64,433** |
| HGB-7 | sorteio agressivo (40% das features) | 64,441 |
| HGB-4 | sorteio de 60% das features por divisão | 64,444 |
| HGB-10 | árvores rasas + sorteio de 60% | 64,453 |
| HGB-2 | aprendizado lento e longo (taxa 0,02, até 3.000 árvores) | 64,454 |
| HGB-8 | árvores grandes regularizadas + sorteio de 60% | 64,455 |
| HGB-5 | árvores rasas (profundidade 4) e muitos estágios | 64,456 |
| HGB-1 | melhor configuração de uma busca anterior com 15 features | 64,466 |
| HGB-9 | pouca regularização (folha mínima 20, L2 = 0) | 64,467 |
| HGB-3 | árvores grandes e muito regularizadas | 64,476 |

**Escolha da solução final:** o **HGB-6**, pelo menor RMSE na validação cruzada.

- As 10 estratégias ficaram a menos de 0,05 ponto entre si, abaixo do desvio entre as dobras (~0,08). Ajustar
  hiperparâmetros chegou ao limite.
- Só o sorteio de features teve efeito consistente: as 3 primeiras posições usam essa estratégia.
- O Boosting segue 0,6 ponto à frente do Bagging e do Pasting mesmo depois da busca deles.

---

## 10. Avaliação final, curva de aprendizado e erros

```bash
python -m src.modelagem.boosting --curva             # curva de aprendizado (só desenvolvimento)
python -m src.modelagem.regressao_linear --teste     # baseline e regressão linear no teste
python -m src.modelagem.boosting --teste             # modelo final no teste + gráficos e erros por faixa/grupo
```

**Teste reservado:** a tabela do início deste README, com o arquivo `resultados/metricas_teste_final.csv`. O modelo
final foi reajustado nos 2.258.114 participantes do desenvolvimento e avaliado uma vez. O RMSE no teste (64,32) ficou
igual ao da validação cruzada (64,43 ± 0,09), então a escolha não foi otimista.

**Curva de aprendizado do modelo final:** 10%, 25%, 50%, 75% e 100% do treino de cada dobra, em subconjuntos
aninhados com semente 42 (`curva_aprendizado_LC-C07-HGB.csv` e `figuras/curva_aprendizado_hgb_final.png`).

| Treino da dobra | RMSE treino | RMSE validação |
|---:|---:|---:|
| 150 mil (10%) | 62,63 | 64,81 |
| 376 mil (25%) | 63,25 | 64,61 |
| 753 mil (50%) | 63,44 | 64,50 |
| 1,13 mi (75%) | 63,70 | 64,46 |
| 1,51 mi (100%) | 63,81 | 64,44 |

**Diagnóstico:**

- **Não há sobreajuste:** a diferença entre treino e validação cai de 2,2 para 0,6 ponto.
- **Mais dados quase não ajudam:** decuplicar o treino melhora a validação em só 0,4 ponto.
- **O limite não é falta de complexidade:** os erros são altos e próximos, e modelos mais complexos (busca,
  ensembles) não mudaram o resultado. O que falta é informação: o perfil socioeconômico não explica ~73% da
  variação da nota.

**Erros por faixa de nota no teste** (`erros_teste_LC-C07-HGB.csv` e `figuras/teste_hgb_final_*.png`):

| Faixa | Participantes | RMSE | Viés (observada − prevista) |
|---|---:|---:|---:|
| zero | 433 | 512,8 | −510,9 |
| 0–400 | 36.217 | 134,7 | −127,3 |
| 400–500 | 176.481 | 55,0 | −38,7 |
| 500–600 | 278.114 | 43,8 | +22,9 |
| 600–700 | 71.758 | 80,8 | +72,2 |
| 700+ | 1.526 | 135,1 | +130,9 |

O modelo **puxa as previsões para a média**: superestima as notas baixas e subestima as altas. Por tipo de escola o
erro é parecido (privada 61,7; pública 64,0) e sem viés sistemático. Os erros por UF e por faixa de renda (`Q006`)
estão na mesma tabela.

---

## 11. PCA, balanceamento e ablação

```bash
python -m src.modelagem.pca_balanceamento_ablacao completo
```

Saídas em `resultados/pca_balanceamento_ablacao/` e figuras `pca_variancia_acumulada.png`, `ablacao_*.png` e
`erro_por_faixa_desenvolvimento.png`. **Atenção:** estes experimentos rodaram com as **15 features do conjunto C02**,
antes da escolha do C07, e cada comparação é feita dentro desse mesmo conjunto.

**PCA em 2 regressores.**

- **Preparação:** o PCA é ajustado no treino de cada dobra, depois do one-hot e da padronização (93 colunas).
- **Número de componentes:** **18 componentes** preservam **80,2% da variância**, o critério sugerido no enunciado.
- **Resultado:** os componentes substituem as features e **pioram os dois modelos**:

| Modelo | Sem PCA | Com PCA |
|---|---:|---:|
| Boosting | 65,33 | 65,99 |
| Regressão linear | 66,15 | 67,01 |

**Balanceamento.**

- **Faixas:** 10 faixas de nota definidas só com o treino da dobra.
- **Estratégias:** pesos maiores nas faixas raras, undersampling das faixas frequentes ou oversampling das raras.
  A validação mantém a distribuição original.
- **Resultado:** todas as estratégias **pioraram** o erro geral, porque melhorar os extremos custa caro no centro,
  onde está a maioria:

| Modelo | Sem balanceamento | Pesos | Undersampling | Oversampling |
|---|---:|---:|---:|---:|
| Boosting | 65,33 | 80,45 | 71,86 | 65,86 |
| KNN | 73,36 | — | 84,05 | 75,71 |

**Ablação por blocos** (boosting, RMSE sem o bloco contra 65,33 com todos):

| Bloco retirado | RMSE |
|---|---:|
| perfil escolar | 66,70 |
| renda | 66,07 |
| língua estrangeira | 66,03 |
| escolaridade dos pais | 65,82 |
| localização (UF) | 65,78 |
| acesso digital | 65,72 |

**Os blocos que mais contribuem** são a situação escolar e a renda.

---

## 12. Aprofundamento: os erros nas notas extremas podem ser reduzidos?

```bash
python -m src.modelagem.aprofundamento_notas_extremas
```

O script parte do modelo final (HGB-6) e testa 5 variações nas mesmas dobras. Saídas: `variacoes_hgb.csv`,
`variacoes_hgb_por_faixa.csv` e `figuras/variacoes_hgb_erro_por_faixa.png`.

| ID | Variação | RMSE | MAE | Conclusão |
|---|---|---:|---:|---|
| V0 | referência (HGB-6) | 64,43 | 49,55 | — |
| V1 | treino sem as notas zero (retiradas só do treino de cada dobra) | 64,43 | 49,51 | sem efeito |
| V2 | perda absoluta | 64,72 | 49,34 | melhora as notas altas e o MAE, piora as baixas e o RMSE |
| V3 | alvo transformado (quantis) | 64,53 | 49,37 | mesmo padrão da V2, mais fraco |
| V4 | restrições monotônicas (renda, computadores) | 64,49 | 49,59 | modelo mais interpretável, perda pequena |
| V5 | árvores de 31 folhas | 64,44 | 49,55 | mesmo erro com árvores menores, mas mais lento |

**Conclusão:** nenhuma variação reduziu o erro dos extremos sem piorar o resto. Os erros nas notas extremas vêm de
falta de informação nas features, e não da forma de treinar.

---

## 13. Arquivos de resultados e figuras

| Arquivo em `resultados/` | Conteúdo |
|---|---|
| `experimentos.csv` | registro de todos os experimentos, com conclusão; validação cruzada e teste em linhas separadas |
| `previsoes_teste.csv` | `NU_INSCRICAO`, nota observada e previsão do baseline, da linear, do Ridge e do modelo final |
| `comparacao_inicial.csv` | ranking das configurações iniciais |
| `antes_depois_busca.csv` | os 3 melhores antes e depois da busca |
| `busca_LC-C07-{HGB,BAG,PAST}-BUSCA.csv` | todas as configurações de cada busca |
| `busca_LC-C07-ARV-BUSCA.csv` | estudo de complexidade da árvore: 15 estratégias (seção 8) |
| `metricas_teste_final.csv` | teste: baseline × linear × modelo final |
| `curva_aprendizado_LC-C07-HGB.csv` | curva de aprendizado por dobra |
| `erros_teste_LC-C07-HGB.csv` | erros no teste por faixa de nota, UF, tipo de escola e renda |
| `variacoes_hgb*.csv` | aprofundamento (seção 12) |
| `triagem_conjuntos_lc.csv`, `experimentos_triagem.csv` | triagem de features (seção 6) |
| `pca_balanceamento_ablacao/` | log completo de PCA, balanceamento e ablação (seção 11) |
| `previsoes_cv/` | previsões fora da dobra (~35 MB cada; locais, fora do Git) |

| Figura obrigatória do artigo | Arquivo em `resultados/figuras/` |
|---|---|
| Distribuição da nota no desenvolvimento (com zeros) | `distribuicao_nota_alvo_desenvolvimento.png` |
| Comparação dos modelos com o baseline, antes e depois da busca | `comparacao_modelos.png` |
| Variância explicada acumulada do PCA | `pca_variancia_acumulada.png` |
| Curva de aprendizado do modelo final | `curva_aprendizado_hgb_final.png` |
| Observado × previsto e resíduos no teste | `teste_hgb_final_observado_vs_previsto.png`, `teste_hgb_final_residuos.png` |
| Erros no teste por faixa e por grupo | `teste_hgb_final_erro_por_faixa.png`, `teste_hgb_final_erro_por_TP_ESCOLA.png`, `teste_hgb_final_erro_por_Q006.png` |
| Triagem e ablação de features | `lc_triagem_conjuntos.png`, `lc_triagem_ablacao.png`, `ablacao_boosting.png`, `ablacao_knn.png` |
| Aprofundamento | `variacoes_hgb_erro_por_faixa.png` |
| Sobreajuste × complexidade da árvore (complementar) | `arvore_complexidade.png` |

---

## 14. Limitações e observações

- **Random Forest e SVM** estão em notebooks (`src/modelagem/random_forest.ipynb` e `svm.ipynb`) e ainda serão
  registrados no `experimentos.csv` por outro integrante. O RF ficou fora do top 3 (65,36). O SVR com kernel RBF não
  escala para 1,5 milhão de linhas por dobra; a versão final deve rodar na base completa.
- **PCA, balanceamento e ablação** usam 15 features (C02), não o C07. Cada comparação é válida dentro do próprio
  conjunto.
- **Bagging e Pasting** usam a mesma classe (`BaggingRegressor`), mas são famílias distintas no enunciado: amostragem
  com e sem reposição.
- **KNN:** com ponderação pela distância, o RMSE de treino é baixo por construção, porque cada participante é o
  próprio vizinho mais próximo. Isso não indica sobreajuste.
- **A divisão avalia novos participantes da mesma edição.** Ela não demonstra generalização para outros anos,
  escolas ou municípios.
- **Associações não são causalidade:** o modelo descreve padrões entre perfil e nota e não mede a capacidade
  individual de ninguém.
- **`SG_UF_PROVA` e `TP_LINGUA`:** a UF é o local da prova, não a residência; `TP_LINGUA` é a língua escolhida, não a
  proficiência.
