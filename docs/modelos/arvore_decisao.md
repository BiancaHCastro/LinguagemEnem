# Árvore de decisão: `DecisionTreeRegressor`

**Resumo:** sem limite de crescimento, a árvore **decora o treino**. Erra 23 pontos no treino e **92,34 na
validação**, **pior que prever a média** (R² −0,50). É o exemplo de sobreajuste do trabalho e fica em último lugar.

## Arquitetura

- **Como divide:** a árvore separa os participantes em grupos com perguntas do tipo "renda ≤ faixa F?" e "escola =
  federal?". Em cada divisão escolhe a pergunta que mais reduz o erro quadrático.
- **Como prevê:** a média das notas do grupo final (folha).
- **Configuração inicial:** `DecisionTreeRegressor(random_state=42)` sem restrições. A árvore cresce até cada folha
  ficar "pura".
- **Tamanho resultante** (dobra 1): **1.295.186 folhas** e **profundidade 50**, para ~1,5 milhão de participantes no
  treino. Quase **uma folha por pessoa**.

## Preparação dos dados (família "árvore")

| Etapa | Como |
|---|---|
| Ordinais | código na ordem do dicionário; "não sei" recebe um código depois do último nível, e a árvore separa sozinha |
| Categóricas | códigos inteiros |
| Ausências | mantidas como NaN; a árvore trata nativamente |
| Escala | nenhuma (árvores não dependem de escala) |

## Estratégia

Captar **interações** entre renda, escolaridade, escola e idade sem supor forma funcional. A configuração sem limites
é a do notebook original e foi mantida como **configuração inicial**, para mostrar o efeito da complexidade.

## Resultados

| | Dobra 1 | Dobra 2 | Dobra 3 | Média ± DP | Treino |
|---|---:|---:|---:|---:|---:|
| RMSE | 92,37 | 92,34 | 92,31 | **92,34 ± 0,03** | **23,32** |
| MAE | 71,24 | 71,14 | 71,09 | 71,16 | 8,32 |
| R² | −0,501 | −0,495 | −0,493 | **−0,497** | 0,905 |

Tempo: 18 s de ajuste por dobra, 75 s no total.

## Diagnóstico: sobreajuste extremo

- **Treino com R² 0,905, validação com R² −0,497.** A árvore aprende o ruído de cada participante, inclusive notas
  muito diferentes de pessoas com o mesmo perfil, e não generaliza.
- **R² negativo:** errar mais que a média significa que as previsões de "perfis decorados" são piores que não
  prever nada.
- **Limitar o crescimento resolve:** a mesma árvore com profundidade 8 e folhas de pelo menos 100 participantes,
  usada no stacking, cai para **RMSE 66,01**.

## Hiperparâmetros completos

Todos os parâmetros do estimador usado, lidos do código. **Negrito** = valor escolhido pela equipe; os demais são o padrão do scikit-learn 1.9.

Só `random_state` foi definido: a configuração inicial é a árvore sem nenhuma restrição.

| Parâmetro | Valor | Padrão do scikit-learn? | Significado |
|---|---|---|---|
| `ccp_alpha` | 0,0 | sim | poda por custo-complexidade (0 = sem poda) |
| `criterion` | squared_error | sim | medida de erro para escolher divisões |
| `max_depth` | None | sim | profundidade máxima (None = sem limite) |
| `max_features` | None | sim | fração de features consideradas |
| `max_leaf_nodes` | None | sim | máximo de folhas (None = sem limite) |
| `min_impurity_decrease` | 0,0 | sim | redução mínima de erro para dividir |
| `min_samples_leaf` | 1 | sim | mínimo de participantes por folha |
| `min_samples_split` | 2 | sim | mínimo de participantes para dividir um nó |
| `min_weight_fraction_leaf` | 0,0 | sim | fração mínima do peso total por folha |
| `monotonic_cst` | None | sim | restrições monotônicas (None = nenhuma) |
| **`random_state`** | 42 | não (escolhido) | semente da aleatoriedade |
| `splitter` | best | sim | como escolher a divisão (melhor ou aleatória) |

## Para o pitch

- **O contraste mais visual de sobreajuste do trabalho:** 23 pontos no treino contra 92 na validação.
- **Uma folha por participante:** é a explicação intuitiva do problema.
- **Os ensembles corrigem isso:** Bagging, Pasting, Boosting e Random Forest usam muitas árvores controladas e estão
  todos entre os melhores.

**Reproduzir:** `python -m src.modelagem.arvore_decisao`. ID: `LC-C07-ARV`.
