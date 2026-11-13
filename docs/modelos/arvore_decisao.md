# Árvore de decisão: `DecisionTreeRegressor`

**Resumo:** sem limite de crescimento, a árvore **decora o treino**. Erra 23 pontos no treino e **92,34 na
validação**, **pior que prever a média** (R² −0,50). É o exemplo de sobreajuste do trabalho e fica em último lugar
na comparação inicial. Um **estudo de complexidade** com 15 estratégias mostrou que, exigindo **500 participantes por
folha** (estratégia **ARV-F500**), a mesma árvore cai para **RMSE 65,59 (R² 0,245)**, com treino e validação quase
iguais.

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
- **Limitar o crescimento resolve:** veja o estudo de complexidade abaixo.

## Estudo de complexidade: como controlar o sobreajuste

**Por que fazer, e dentro das regras:** a árvore não está entre os 3 melhores, então a busca obrigatória não se
aplica a ela. O enunciado, porém, pede para discutir sobreajuste relacionando-o à complexidade. A **configuração
inicial continua registrada como está** (`LC-C07-ARV`), e a comparação inicial não muda. O estudo é um registro à
parte (`LC-C07-ARV-BUSCA`), feito com `GridSearchCV` nas **mesmas 3 dobras** e no mesmo Pipeline, com critério de
**menor RMSE médio**. O teste não foi usado.

**15 estratégias em 4 grupos**, cada uma com hipótese (`ESTRATEGIAS` em `src/modelagem/arvore_decisao.py`):

| # | ID | Estratégia | RMSE validação | MAE | R² | RMSE treino | Treino − validação | Ajuste/dobra |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | **ARV-F500** | **folha mínima 500, sem limite de profundidade** | **65,59** | **50,52** | **0,245** | 65,25 | 0,34 | 10 s |
| 2 | ARV-P12-F500 | profundidade 12 + folha 500 | 65,65 | 50,58 | 0,243 | 65,43 | 0,23 | 9 s |
| 3 | ARV-P15-F1000 | profundidade 15 + folha 1.000 | 65,71 | 50,63 | 0,242 | 65,52 | 0,19 | 9 s |
| 4 | ARV-P10-F200 | profundidade 10 + folha 200 | 65,73 | 50,66 | 0,242 | 65,57 | 0,17 | 8 s |
| 5 | ARV-F100 | folha mínima 100 | 65,76 | 50,66 | 0,241 | 64,30 | 1,47 | 10 s |
| 6 | ARV-P10 | profundidade 10 | 65,78 | 50,68 | 0,241 | 65,49 | 0,29 | 8 s |
| 7 | ARV-F2000 | folha mínima 2.000 | 65,84 | 50,74 | 0,239 | 65,74 | 0,10 | 9 s |
| 8 | ARV-P12 | profundidade 12 | 65,86 | 50,67 | 0,239 | 64,91 | 0,95 | 9 s |
| 9 | ARV-P08 | profundidade 8 | 66,02 | 50,91 | 0,235 | 65,95 | 0,07 | 7 s |
| 10 | ARV-PODA | poda `ccp_alpha` = 1 + folha 20 | 66,37 | 51,22 | 0,227 | 66,35 | 0,02 | 21 s |
| 11 | ARV-P06 | profundidade 6 | 66,55 | 51,38 | 0,223 | 66,54 | 0,01 | 6 s |
| 12 | ARV-P15 | profundidade 15 | 67,07 | 51,33 | 0,210 | 63,28 | 3,79 | 10 s |
| 13 | ARV-P04 | profundidade 4 | 67,65 | 52,33 | 0,197 | 67,64 | 0,01 | 5 s |
| 14 | ARV-F20 | folha mínima 20 | 67,86 | 52,43 | 0,192 | 61,53 | 6,33 | 12 s |
| 15 | ARV-P20 | profundidade 20 | 73,03 | 55,19 | 0,064 | 56,79 | 16,25 | 12 s |
| — | *inicial* | *sem limite (folha 1)* | *92,34* | *71,16* | *−0,497* | *23,32* | *69,02* | *18 s* |

Tempo total do estudo: 10 minutos. RMSE da escolhida por dobra: 65,48 · 65,61 · 65,69.
Arquivos: `resultados/busca_LC-C07-ARV-BUSCA.csv` e `resultados/figuras/arvore_complexidade.png`.

**O que o estudo mostra:**

- **A curva clássica de complexidade** (`arvore_complexidade.png`):
  - até a profundidade 10, treino e validação andam juntos e o erro cai;
  - a partir de 12–15, o treino continua caindo e a validação sobe;
  - na árvore sem limite, a distância chega a 69 pontos.
- **Profundidade 4–6 é simples demais (subajuste):** treino e validação iguais, mas altos (66,5–67,6).
- **O que mais importa é o tamanho mínimo da folha:**
  - com 500 participantes por folha, a árvore prevê a **média de grupos**, não a nota de indivíduos;
  - com 20 por folha ainda sobreajusta (diferença de 6,3 pontos).
- **Os 9 melhores ficam a 0,4 ponto entre si:** qualquer limite razoável resolve o sobreajuste.
- **Mesmo controlada, a árvore única não alcança os comitês.** 65,59 empata com a regressão linear (65,54) e fica
  atrás de Bagging (65,26), Pasting (65,27) e Boosting (64,50). Ela continua fora do top 3.

**Configuração escolhida** (`ESCOLHIDA` e `REGRESSOR_ESCOLHIDO` no script):
`DecisionTreeRegressor(min_samples_leaf=500, random_state=42)`, com todos os outros parâmetros no padrão do
scikit-learn.

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
- **A correção:** exigir 500 participantes por folha leva o erro de 92,34 para 65,59, com treino e validação colados
  (65,25 × 65,59). A figura `arvore_complexidade.png` mostra a curva clássica de sobreajuste.
- **Os ensembles vão além:** Bagging, Pasting, Boosting e Random Forest usam muitas árvores controladas e ficam
  todos à frente da melhor árvore única.

**Reproduzir:**
```bash
python -m src.modelagem.arvore_decisao              # configuração inicial (LC-C07-ARV)
python -m src.modelagem.arvore_decisao --so-busca   # estudo de complexidade (LC-C07-ARV-BUSCA, 15 estratégias)
```
