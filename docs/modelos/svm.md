# SVM: `LinearSVR`

**Resumo:** uma máquina de vetores de suporte **linear**, a versão de SVM que escala para os 2,26 milhões de
participantes do desenvolvimento. Fez **RMSE 65,40 ± 0,11 (R² 0,249)**: praticamente o mesmo patamar da regressão
linear (65,54) e fora do top 3. Script do outro integrante: `src/modelagem/svm_svrlinear.py`.

## Arquitetura

- **O que o SVR faz:** ajusta uma função linear das features, como a regressão linear, mas com outra função de
  perda e com regularização controlada por `C`.
- **Perda `squared_epsilon_insensitive`:** erros menores que `epsilon` (0,1 ponto) não custam nada; acima disso, o
  custo cresce com o quadrado do erro. Com notas em escala de centenas, a faixa de 0,1 é desprezível, e na prática o
  modelo fica muito próximo de uma regressão linear com regularização L2.
- **`dual=False`:** resolve o problema na forma primal. É a forma recomendada quando há muito mais linhas que colunas
  (~1,5 milhão × ~140).

**Por que linear e não kernel RBF:** um SVR com kernel monta uma matriz de similaridade entre todos os pares de
participantes. Com 1,5 milhão de linhas por dobra, isso é inviável em memória e tempo. O enunciado permite o
`LinearSVR` para esse caso. Uma versão preliminar usava SVR com kernel RBF numa amostra de 20 mil linhas e foi
abandonada, porque o enunciado não permite trocar a base por uma amostra.

## Preparação dos dados (dentro do Pipeline, ajustada no treino de cada dobra)

| Colunas | Tratamento |
|---|---|
| 8 numéricas (`Q005`, `IN_TREINEIRO`, `TP_LINGUA`, renda per capita, capital cultural, nº de "não sei", atraso escolar, anos desde a conclusão) | vazios → **mediana do treino**; depois **padronização** (`StandardScaler`) |
| 15 códigos (renda, escolaridade, ocupação, escola, situação, faixa etária, computadores, internet, UF, perfil digital, rede da escola…) | **one-hot** (~130 colunas, matriz esparsa); categoria não vista → ignorada |

**Diferença em relação à preparação "linear" do projeto:** ordinais como renda e escolaridade entram como one-hot (sem
usar a ordem dos níveis), não há coluna indicadora de vazio e não há `log1p` na renda. Nenhuma das diferenças gera
vazamento.

## Resultados (validação cruzada, mesmas 3 dobras)

| | Dobra 1 | Dobra 2 | Dobra 3 | Média ± DP |
|---|---:|---:|---:|---:|
| RMSE | 65,29 | 65,42 | 65,50 | **65,40 ± 0,11** |
| MAE | 50,35 | 50,45 | 50,44 | 50,42 |
| R² | 0,250 | 0,250 | 0,248 | **0,249** |

- **Tempo:** 25 s de ajuste e 3 s de previsão por dobra.
- **Baseline nas mesmas dobras:** 75,48, igual ao registrado no `experimentos.csv`, o que confirma que as dobras são
  as mesmas.
- **Onde estão os números:** `resultados/resultados_svm.csv` e `resultados/figuras/desempenho_svm.png`.

## Diagnóstico

- **Fica no patamar dos modelos lineares:** 65,40 contra 65,54 da regressão linear. A diferença de 0,14 vem
  provavelmente da preparação (one-hot de todos os códigos) e da perda, não do SVM em si.
- **Fora do top 3:** atrás do Pasting (65,27). Por isso não entrou na busca de hiperparâmetros.
- **Sem RMSE de treino registrado:** não dá para comparar treino × validação. Por ser linear e ter milhões de linhas
  para ~140 colunas, sobreajuste não é esperado.

## Hiperparâmetros completos

**Negrito** = valor escolhido no script; os demais são o padrão do scikit-learn 1.9.

| Parâmetro | Valor | Padrão do scikit-learn? | Significado |
|---|---|---|---|
| `C` | 1,0 | sim | inverso da força da regularização (maior = menos regularização) |
| **`epsilon`** | 0,1 | não (padrão 0) | largura da faixa de erro sem custo |
| **`loss`** | squared_epsilon_insensitive | não (padrão epsilon_insensitive) | erro acima de `epsilon` custa o quadrado |
| **`dual`** | False | não (padrão "auto") | resolve na forma primal (muitas linhas, poucas colunas) |
| `fit_intercept` | True | sim | ajusta o intercepto |
| `intercept_scaling` | 1,0 | sim | escala do intercepto |
| `tol` | 0,0001 | sim | tolerância de convergência |
| **`max_iter`** | 2.000 | não (padrão 1.000) | máximo de iterações |
| **`random_state`** | 42 | não (escolhido) | semente |

## Observações sobre o script

- **Caminhos do Google Colab:** `CAMINHO_PARQUET = "/content/enem_2023_lc_c07.parquet"` e
  `PASTA_SAIDA = "/content/resultados_svm"`. Para rodar no repositório, troque por
  `dados/processed/enem_2023_lc_c07.parquet` e `resultados/`.
- **O script também avalia no teste** (RMSE 65,32, R² 0,248). Pelo protocolo, só o baseline, a regressão linear e a
  solução final vão ao teste. Esse número não participou de nenhuma escolha e não deve ser usado para comparar
  modelos.
- **Ainda não registrado** no `experimentos.csv` nem no ranking de `comparacao_inicial.csv`.

## Para o pitch

- **Por que LinearSVR:** é a forma de SVM que cabe em 2,26 milhões de linhas. Kernels exigiriam comparar todos os
  pares de participantes.
- **Resultado:** 65,40, no mesmo patamar da regressão linear. Mais uma família de modelos parando em R² ~0,25.

**Reproduzir:** `python src/modelagem/svm_svrlinear.py`, depois de ajustar os caminhos.
