# Top 3: o que fizemos para melhorar os 3 melhores modelos

Depois da [comparação inicial](../modelos/README.md), os 3 regressores com menor RMSE seguiram para a **busca de
hiperparâmetros**, como pede o enunciado (Seção 4, passo 3). O vencedor também passou por um **aprofundamento** e pela
**avaliação final** no teste.

| Modelo | RMSE inicial | Método de busca | Configurações | RMSE após a busca | Ganho | Detalhes |
|---|---:|---|---:|---:|---:|---|
| **Boosting (HistGB)** | 64,50 | `GridSearchCV`, 10 estratégias com hipótese | 10 | **64,43** | −0,07 | [boosting.md](boosting.md) |
| Bagging | 65,26 | `RandomizedSearchCV` | 10 | 65,01 | −0,25 | [bagging.md](bagging.md) |
| Pasting | 65,27 | `RandomizedSearchCV` | 10 | 65,03 | −0,24 | [pasting.md](pasting.md) |

## Regras comuns às 3 buscas

- **Mesma validação da comparação inicial:** mesmas 3 dobras do desenvolvimento e mesmo `Pipeline`, com a preparação
  ajustada só no treino de cada dobra.
- **Critério de escolha:** **RMSE médio** nas 3 dobras. MAE, R², erro de treino e tempo também foram registrados.
- **Pelo menos 10 configurações válidas por modelo,** com semente 42.
- **O teste não participou de nenhuma escolha.** Só o vencedor final foi avaliado nele, uma única vez.
- **Onde ficam os resultados:**
  - `resultados/busca_LC-C07-<modelo>-BUSCA.csv`: todas as configurações;
  - `resultados/antes_depois_busca.csv` e `resultados/figuras/comparacao_modelos.png`: antes × depois.

## Resultado final

1. **O Boosting continua na frente:** depois da busca de todos, ele fica **0,6 ponto à frente** do Bagging e do Pasting.
2. **Bagging e Pasting ganharam mais com a busca** (0,25 ponto), porque partiam de uma configuração pior. O
   Boosting ganhou pouco (0,07): já estava perto do limite.
3. **Solução final:** a estratégia **HGB-6** do Boosting. No **teste reservado** fez **RMSE 64,32, MAE 49,42 e R² 0,271**,
   igual à validação (64,43). A escolha não foi otimista.
4. **O que limita o resultado:** 30 configurações, 3 famílias de modelos e 5 variações de treino pararam no mesmo
   patamar de R² ~0,27. O limite está na informação das features, não no modelo.

## Para o pitch

- **"Buscamos 30 configurações nos 3 melhores modelos; o ganho máximo foi de 0,25 ponto."** O ajuste fino tem pouco
  efeito quando o modelo já está no limite da informação.
- **No Bagging e no Pasting, o que mais importou foi o número de árvores e o tamanho mínimo da folha.**
  - **40 árvores** e **folha de 20 participantes** foram os melhores valores.
  - **Folhas de 5 participantes** decoram o treino e foram a pior escolha.
- **No Boosting, a única estratégia com efeito consistente foi sortear features em cada divisão.**
