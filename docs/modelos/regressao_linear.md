# Regressão linear: sem regularização (`LinearRegression`) e com regularização (`Ridge`)

**Resumo:** as duas versões empatam em **RMSE 65,54 e R² 0,246**. Um modelo de soma simples, sem interações, já
explica 1/4 da variação das notas. É a referência obrigatória e fica em **5º e 6º lugar**.

## Arquitetura

- **Regressão linear (OLS):** nota = intercepto + soma de (peso × feature). Os pesos minimizam o erro quadrático.
- **Ridge:** o mesmo modelo com penalidade L2 sobre os pesos (`alpha = 1`), que encolhe coeficientes para reduzir a
  variância.

## Preparação dos dados (família "linear")

| Etapa | Como | Por quê |
|---|---|---|
| Ordinais (renda, escolaridade, ocupação, computadores) | posição na ordem (A = 0, B = 1, …), padronizada | preserva a ordem com uma coluna só |
| "Não sei" / `INDETERMINADA` | vira vazio → mediana do treino + coluna indicadora | "não sei" não tem posição na escala, e o indicador guarda a informação |
| Categóricas (UF, situação escolar, faixa etária, tipo de escola, perfil digital) | one-hot sem a 1ª categoria | categorias sem ordem |
| Renda per capita | `log1p` + padronização | muito assimétrica (assimetria 3,8) |
| Colunas redundantes | retiradas 5 que são função exata de outras (`IN_TREINEIRO`, `Q025`, `TP_ESCOLA`, `CAPITAL_CULTURAL_RESP`, `N_RESP_NAO_SEI`) | sem isso a OLS não tem solução única |
| Colinearidade restante | removida no treino da dobra | idem |

Resultado da preparação: **18 features → 77 colunas numéricas.**

## Estratégia

Testar se relações **aditivas** (cada fator soma ou subtrai pontos) explicam a nota, e se a regularização ajuda.
O enunciado exige as duas versões.

## Resultados

| | Dobra 1 | Dobra 2 | Dobra 3 | Média ± DP | Treino |
|---|---:|---:|---:|---:|---:|
| RMSE OLS | 65,43 | 65,55 | 65,63 | **65,54 ± 0,11** | 65,53 |
| RMSE Ridge | 65,42 | 65,55 | 65,63 | **65,54 ± 0,11** | 65,53 |
| MAE (ambos) | 50,48 | 50,58 | 50,57 | 50,54 | 50,54 |
| R² (ambos) | 0,247 | 0,247 | 0,245 | 0,246 | 0,246 |

- **Tempo total:** 39 s (OLS) e 32 s (Ridge).
- **Teste reservado (OLS):** RMSE 65,46, MAE 50,45, R² 0,245.

## Diagnóstico

- **Sem sobreajuste:** o erro de treino é igual ao de validação.
- **Por que o Ridge não muda nada:** com 2,26 milhões de linhas para 77 colunas, não há variância para a
  regularização reduzir.
- **O limite é a forma do modelo:** a distância para o Boosting (~1 ponto) mede o ganho das interações entre fatores,
  que a linear não capta.

## Hiperparâmetros completos

Todos os parâmetros do estimador usado, lidos do código. **Negrito** = valor escolhido pela equipe; os demais são o padrão do scikit-learn 1.9.

No Ridge, `alpha = 1` é o padrão e foi mantido de propósito como configuração inicial.

**LinearRegression (sem regularização)**

| Parâmetro | Valor | Padrão do scikit-learn? | Significado |
|---|---|---|---|
| `copy_X` | True | sim | copia X antes de ajustar |
| `fit_intercept` | True | sim | ajusta o intercepto |
| `n_jobs` | None | sim | núcleos de CPU em paralelo |
| `positive` | False | sim | força coeficientes ≥ 0 |
| `tol` | 1e-06 | sim | tolerância de convergência |

**Ridge (com regularização)**

| Parâmetro | Valor | Padrão do scikit-learn? | Significado |
|---|---|---|---|
| `alpha` | 1,0 | sim | força da penalidade L2 (maior = coeficientes menores) |
| `copy_X` | True | sim | copia X antes de ajustar |
| `fit_intercept` | True | sim | ajusta o intercepto |
| `max_iter` | None | sim | máximo de iterações |
| `positive` | False | sim | força coeficientes ≥ 0 |
| `random_state` | None | sim | semente da aleatoriedade |
| `solver` | auto | sim | algoritmo de otimização |
| `tol` | 0,0001 | sim | tolerância de convergência |

## Para o pitch

- **Um modelo simples já chega perto:** a linear fica só ~1 ponto atrás do melhor modelo.
- **Os coeficientes são interpretáveis**, porque cada feature tem um peso próprio.
- **A regularização não ajudou**, e a explicação é o volume de dados.

**Reproduzir:** `python -m src.modelagem.regressao_linear`. IDs: `LC-C07-LIN-OLS` e `LC-C07-LIN-RIDGE`.
