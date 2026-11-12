# KNN: `KNeighborsRegressor`

**Resumo:** prevê a nota pela média dos 50 participantes de perfil mais parecido. Ficou em **7º lugar (RMSE 70,86,
R² 0,12)**, bem atrás dos modelos lineares e de árvores. Também foi o mais lento (13 minutos).

## Arquitetura

1. **Preparação "linear":** ordinais padronizados, one-hot, imputação + indicador e `log1p` na renda. Resultado:
   **77 colunas**.
2. **PCA incremental** com **12 componentes**, ajustado em lotes de 20 mil linhas no treino da dobra. Preserva
   **82% da variância** das 77 colunas.
3. **KNN:** para cada participante da validação, busca os **50 vizinhos** mais próximos (distância euclidiana) no
   treino e faz a **média ponderada pela distância** (vizinho mais perto pesa mais). A busca usa uma árvore kd
   (`leaf_size = 60`).

| Hiperparâmetro | Valor |
|---|---|
| `n_neighbors` | 50 |
| `weights` | `"distance"` |
| `algorithm` / `leaf_size` | `"kd_tree"` / 60 |
| métrica | euclidiana (`p = 2`) |
| PCA | 12 componentes |

## Estratégia

- **A hipótese:** participantes com perfil parecido têm notas parecidas. O KNN estima a nota sem supor nenhuma
  forma funcional.
- **Por que o PCA:** em 77 dimensões a árvore kd vira força bruta, o que é inviável com 1,5 milhão de pontos de
  treino e 750 mil consultas por dobra. Com 12 componentes a busca fica viável na **base completa**.
- **Mesmas escolhas do script original da equipe:** a única mudança foi passar de 15 para as 23 features do C07.

## Resultados

| | Dobra 1 | Dobra 2 | Dobra 3 | Média ± DP | Treino |
|---|---:|---:|---:|---:|---:|
| RMSE | 70,80 | 70,87 | 70,91 | **70,86 ± 0,05** | 23,42* |
| MAE | 54,47 | 54,56 | 54,50 | 54,51 | 8,39* |
| R² | 0,118 | 0,119 | 0,119 | **0,119** | 0,904* |

**Tempo:** 13 s de ajuste e **81 s de previsão** por dobra, 764 s no total. Prever é o custo do KNN, porque ele
precisa procurar vizinhos para cada pessoa.

\*O erro de treino não serve para diagnosticar sobreajuste aqui. Com ponderação pela distância, cada participante do
treino é o próprio vizinho mais próximo (distância 0), então o modelo "acerta" o treino por construção.

## Diagnóstico

- **Pior que a regressão linear (65,5)**, que é bem mais simples.
- **Causas prováveis:**
  - **As features são quase todas categorias.** A distância trata "UF = CE" × "UF = SP" e diferenças de renda com a
    mesma régua, depois do one-hot e do PCA.
  - **O PCA descarta 18% da variância.**
  - **Distância ponderada com 50 vizinhos dá estimativas ruidosas** num alvo muito disperso.
- **Mais vizinhos ajudariam:** numa busca anterior (15 features), 300 vizinhos com média simples chegaram a RMSE
  66,3. Mas o KNN ficou fora do top 3 e não entrou na busca oficial.

## Hiperparâmetros completos

Todos os parâmetros do estimador usado, lidos do código. **Negrito** = valor escolhido pela equipe; os demais são o padrão do scikit-learn 1.9.

O PCA é uma classe própria da equipe (`src/modelagem/pca_balanceamento_ablacao.py`), sem "padrão do scikit-learn": a coluna compara com os padrões da própria classe.

**PCA incremental (`SparseIncrementalPCA`, da equipe)**

| Parâmetro | Valor | Padrão do scikit-learn? | Significado |
|---|---|---|---|
| `batch_size` | 20000 | sim | linhas por lote do PCA incremental |
| **`n_components`** | 12 | não (escolhido) | nº de componentes do PCA |
| **`variance_threshold`** | 1,0 | não (escolhido) | variância mínima (1,0 = n_components manda) |

**KNeighborsRegressor**

| Parâmetro | Valor | Padrão do scikit-learn? | Significado |
|---|---|---|---|
| **`algorithm`** | kd_tree | não (escolhido) | estrutura de busca de vizinhos |
| **`leaf_size`** | 60 | não (escolhido) | tamanho da folha da árvore de busca |
| `metric` | minkowski | sim | métrica de distância |
| `metric_params` | None | sim | parâmetros extras da métrica |
| **`n_jobs`** | -1 | não (escolhido) | núcleos de CPU em paralelo |
| **`n_neighbors`** | 50 | não (escolhido) | número de vizinhos |
| `p` | 2 | sim | tipo de distância (2 = euclidiana) |
| **`weights`** | distance | não (escolhido) | peso dos vizinhos ('distance' = mais perto pesa mais) |

## Para o pitch

- **Exemplo de modelo inadequado ao tipo de dado:** distâncias funcionam mal com perfis categóricos.
- **Trade-off computacional:** foi preciso PCA para rodar na base completa, e mesmo assim foi o modelo mais lento.

**Reproduzir:** `python -m src.modelagem.knn`. ID: `LC-C07-KNN`.
