# Conjuntos de features: o que testamos e o que escolhemos

## Como foi o processo

1. **Ponto de partida:** as 12 colunas originais do Inep (renda, escolaridade dos pais, tipo de escola, idade, UF,
   língua etc.).
2. **Criação de features:** 18 features novas, como renda per capita, escolaridade e ocupação máximas dos pais,
   perfil de acesso digital, tipo de escola detalhado e atraso escolar.
3. **Montagem dos conjuntos:** combinamos as features em **16 conjuntos** (e mais 8 ablações), cada um testando uma
   hipótese.
4. **Triagem:** cada conjunto foi testado com **3 modelos** (regressão linear, árvore de decisão e HistGradientBoosting),
   nas mesmas 3 dobras, numa amostra de 300 mil participantes do desenvolvimento.

## Resultado (RMSE na validação; menor é melhor; baseline pela média = 75,6)

| Conjunto | O que testa | Colunas | Linear | Árvore | Boosting |
|---|---|---:|---:|---:|---:|
| C01 Originais | só as 12 colunas do Inep | 12 | 66,52 | 66,77 | 65,62 |
| C02 Atual | + renda per capita, escolaridade máx. dos pais, acesso digital | 15 | 66,32 | 66,70 | 65,61 |
| C03 Enxuto | só 7 colunas: dá para simplificar? | 7 | 67,36 | 67,46 | 66,80 |
| C04 Família ampliada | C02 + ocupação dos pais | 20 | 66,15 | 66,62 | 65,36 |
| C05 Bens brutos | C04 + 17 perguntas sobre bens da casa | 37 | 65,81 | 66,61 | 64,97 |
| C06 Bens resumidos | C04 + índice de bens | 26 | 66,10 | 66,70 | 65,33 |
| **C07 Trajetória escolar** | **C04 + rede da escola, atraso escolar, anos desde a conclusão** | **23** | **65,68** | **66,31** | **64,87** |
| C08 Localização fina | C04 + município da prova | 22 | 65,95 | 66,56 | 65,21 |
| C08B Região | região no lugar da UF | 20 | 66,28 | 66,53 | 65,55 |
| C09 Demográfico | C04 + sexo, cor/raça, estado civil | 24 | 65,85 | 66,46 | 65,01 |
| C10 Interações | C04 + renda × língua, renda × escolaridade | 23 | 66,11 | 66,55 | 65,36 |
| C11 Completo | todas as colunas permitidas | 54 | **64,87** | **66,28** | **64,10** |
| C12 Seleção automática | as 20 melhores do C11 (`SelectKBest`) | 20 | 67,17 | 67,59 | 66,73 |
| C13 PCA global | C11 resumido em componentes (80% da variância) | 24 | 66,44 | 67,93 | 65,94 |
| C14 PCA dos bens | PCA só do bloco de bens | 31 | 65,95 | 66,97 | 65,16 |
| C15 Clusters | C04 + distância a perfis típicos (k-means) | 26 | 66,10 | 66,72 | 65,37 |

**Ablação** (tirar um bloco do C11 por vez): os blocos que mais fazem falta são **situação escolar** (+0,9 ponto) e
**língua estrangeira** (+0,6 ponto).

## O que aprendemos

- **O modelo pesa mais que as features:** o **Boosting** foi o melhor modelo em todos os 16 conjuntos.
- **O completo (C11) foi o melhor**, mas usa **54 colunas**.
- **O C07 ficou em 2º lugar nos 3 modelos** com **23 colunas**, perdendo só ~0,8 ponto para o completo.
- **Seleção automática e PCA pioraram:** o C12 e o C13 ficaram atrás das features escolhidas pela equipe.
- **Mais informação ajuda pouco:** do pior conjunto ao melhor, a diferença no Boosting é de só 2,7 pontos.

## Escolha final

**Conjunto C07 (23 features).** Fica em 2º lugar com menos da metade das colunas do C11. Por ser mais leve, permitiu
rodar todos os modelos (KNN, bagging, stacking etc.) na **base completa de 2,8 milhões de participantes**. A hipótese
confirmada: **a ocupação dos pais e a trajetória escolar acrescentam informação** ao perfil socioeconômico.

**Com o C07, o melhor modelo final foi o HistGradientBoosting:** R² 0,271 e RMSE 64,3 pontos no teste, contra 75,3 do
baseline.
