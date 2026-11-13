"""Árvore de decisão (DecisionTreeRegressor) no conjunto C07.

Mesma configuração inicial do notebook ENEM_2023_Linguagens_DecisionTree_Stacking.ipynb
(DecisionTreeRegressor(random_state=42), sem limite de profundidade), agora no protocolo comum dos scripts:
validação cruzada nas 3 dobras fixas, preparação ajustada só no treino de cada dobra e registro em
resultados/experimentos.csv. O teste não é usado aqui (só para a solução final).

A configuração inicial fica registrada como está (o enunciado pede registrar as configurações iniciais antes das
buscas). Como ela decora o treino (sobreajuste extremo), --busca roda um estudo de complexidade: 15 estratégias com
hipótese (profundidade, folha mínima, as duas juntas e poda), em GridSearchCV nas mesmas 3 dobras. A árvore não está
entre os 3 melhores; este estudo é diagnóstico de sobreajuste x complexidade e não muda a comparação inicial.

Uso (a partir da raiz do projeto):
    python -m src.modelagem.arvore_decisao                    # validação cruzada da configuração inicial
    python -m src.modelagem.arvore_decisao --so-busca         # só o estudo de complexidade (15 estratégias)
    python -m src.modelagem.arvore_decisao --busca            # as duas coisas
    python -m src.modelagem.arvore_decisao --amostra 30000    # teste rápido do script

Saídas: resultados/experimentos.csv (LC-C07-ARV, LC-C07-ARV-BUSCA), resultados/previsoes_cv/<id>.parquet,
resultados/busca_LC-C07-ARV-BUSCA.csv e resultados/figuras/arvore_complexidade.png.
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from ..conjuntos import configuracao as cfg
from .comum import argumentos, avaliar, buscar, log, preparar_execucao

FAMILIA = "arvore"
PREPARACAO = ("códigos ordinais (especiais 'não sei'/indeterminada depois do último nível); categóricas como "
              "códigos inteiros; NaN mantido (a árvore trata ausência); sem padronização")
# Configuração inicial = a do notebook: árvore sem restrição de crescimento.
REGRESSOR = DecisionTreeRegressor(random_state=cfg.SEMENTE)
HIPOTESE = ("Árvore de decisão sem restrições: capta interações entre renda, escolaridade, escola e idade, mas, "
            "crescendo até folhas de 1 participante, tende a decorar o treino (sobreajuste).")
ID_BUSCA = "LC-C07-ARV-BUSCA"

# Estudo de complexidade: cada estratégia testa uma forma de impedir que a árvore decore o treino.
_CHAVES = ("max_depth", "min_samples_leaf", "ccp_alpha")
ESTRATEGIAS = [
    # id, estratégia, hipótese, (profundidade máx., folha mínima, poda)
    ("ARV-P04", "Profundidade 4", "Árvore rasa: pouco sobreajuste, mas pode ficar simples demais (subajuste).",
     (4, 1, 0.0)),
    ("ARV-P06", "Profundidade 6", "Limitar a profundidade impede folhas de 1 participante.", (6, 1, 0.0)),
    ("ARV-P08", "Profundidade 8", "Mais níveis captam mais interações sem decorar o treino.", (8, 1, 0.0)),
    ("ARV-P10", "Profundidade 10", "Ponto intermediário da curva de complexidade.", (10, 1, 0.0)),
    ("ARV-P12", "Profundidade 12", "A partir daqui o sobreajuste deve começar a crescer.", (12, 1, 0.0)),
    ("ARV-P15", "Profundidade 15", "Árvore profunda: espera-se treino bem abaixo da validação.", (15, 1, 0.0)),
    ("ARV-P20", "Profundidade 20", "Perto da árvore sem limite: sobreajuste forte.", (20, 1, 0.0)),
    ("ARV-F20", "Folha mínima 20 (sem limite de profundidade)",
     "Exigir 20 participantes por folha faz a árvore prever médias de grupos, não de indivíduos.", (None, 20, 0.0)),
    ("ARV-F100", "Folha mínima 100", "Folhas maiores = médias mais estáveis.", (None, 100, 0.0)),
    ("ARV-F500", "Folha mínima 500", "Com 1,5 milhão de linhas, folhas de 500 ainda permitem muitos grupos.",
     (None, 500, 0.0)),
    ("ARV-F2000", "Folha mínima 2.000", "Folhas muito grandes podem perder detalhe (subajuste).", (None, 2000, 0.0)),
    ("ARV-P10-F200", "Profundidade 10 + folha 200", "Os dois limites juntos controlam altura e tamanho dos grupos.",
     (10, 200, 0.0)),
    ("ARV-P12-F500", "Profundidade 12 + folha 500", "Mais profundidade compensada por folhas maiores.",
     (12, 500, 0.0)),
    ("ARV-P15-F1000", "Profundidade 15 + folha 1.000", "Árvore alta, mas só com grupos grandes.", (15, 1000, 0.0)),
    ("ARV-PODA", "Poda por custo-complexidade (ccp_alpha = 1) + folha 20",
     "Cortar ramos que reduzem pouco o erro remove os que só decoram ruído.", (None, 20, 1.0)),
]
ESPACO = [{f"modelo__{k}": [v] for k, v in zip(_CHAVES, valores)} for *_, valores in ESTRATEGIAS]

# Escolhida pelo menor RMSE médio nas 3 dobras (resultados/busca_LC-C07-ARV-BUSCA.csv, execução de 12/11/2026):
# ARV-F500 = folha mínima 500, sem limite de profundidade. RMSE 65,59 ± 0,10 | MAE 50,52 | R² 0,245 | treino 65,25
# (a configuração inicial, sem limite, fez 92,34 na validação e 23,32 no treino). O teste não foi usado na escolha.
ESCOLHIDA = "ARV-F500"
REGRESSOR_ESCOLHIDO = DecisionTreeRegressor(min_samples_leaf=500, random_state=cfg.SEMENTE)


def _marcar_estrategias(pasta, id_busca=ID_BUSCA):
    """Acrescenta id, estratégia e hipótese a cada linha de busca_LC-C07-ARV-BUSCA.csv."""
    arquivo = pasta / f"busca_{id_busca}.csv"
    resultados = pd.read_csv(arquivo)
    por_parametros = {json.dumps({f"modelo__{k}": v for k, v in zip(_CHAVES, valores)}, sort_keys=True): (i, e, h)
                      for i, e, h, valores in ESTRATEGIAS}
    rotulos = [por_parametros.get(json.dumps(json.loads(p), sort_keys=True), ("?", "?", "?"))
               for p in resultados["params_json"]]
    for posicao, coluna in enumerate(["id_estrategia", "estrategia", "hipotese_estrategia"]):
        resultados[coluna] = [r[posicao] for r in rotulos]
    resultados.to_csv(arquivo, index=False)
    colunas = ["posicao_rmse", "id_estrategia", "mean_test_rmse", "std_test_rmse", "mean_test_mae", "mean_test_r2",
               "mean_train_rmse", "mean_fit_time", "estrategia"]
    print(resultados[colunas].to_string(index=False), flush=True)
    return resultados


def grafico_complexidade(resultados, pasta, id_inicial="LC-C07-ARV"):
    """RMSE de treino e validação por profundidade máxima (folha mínima 1) + a árvore sem limite, se registrada."""
    prof = resultados[resultados["id_estrategia"].str.fullmatch(r"ARV-P\d+")].copy()
    prof["profundidade"] = prof["id_estrategia"].str.removeprefix("ARV-P").astype(int)
    prof = prof.sort_values("profundidade")
    eixo_x = [str(p) for p in prof["profundidade"]]
    treino, validacao = list(prof["mean_train_rmse"]), list(prof["mean_test_rmse"])
    dp = list(prof["std_test_rmse"])

    experimentos = pasta / "experimentos.csv"
    if experimentos.exists():
        inicial = pd.read_csv(experimentos).query("id_experimento == @id_inicial")
        if len(inicial):
            eixo_x.append("sem limite")
            treino.append(float(inicial["rmse_treino_media"].iloc[0]))
            validacao.append(float(inicial["rmse_media"].iloc[0]))
            dp.append(float(inicial["rmse_dp"].iloc[0]))

    melhor = resultados.iloc[0]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(eixo_x, validacao, marker="o", color="#e0813b", label="validação (média de 3 dobras)")
    ax.plot(eixo_x, treino, marker="o", color="#3b78c3", label="treino")
    ax.axhline(melhor["mean_test_rmse"], color="#2a9d55", ls="--", lw=1,
               label=f"melhor do estudo: {melhor['id_estrategia']} ({melhor['mean_test_rmse']:.2f})")
    ax.set_xlabel("Profundidade máxima da árvore (folha mínima = 1)")
    ax.set_ylabel("RMSE (pontos)")
    ax.set_title("Árvore de decisão: complexidade x sobreajuste (C07, desenvolvimento)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figuras = pasta / "figuras"
    figuras.mkdir(parents=True, exist_ok=True)
    fig.savefig(figuras / "arvore_complexidade.png", dpi=150)
    plt.close(fig)
    log(f"Figura: {figuras / 'arvore_complexidade.png'}")


def main(argv=None):
    args = argumentos("Árvore de decisão (DecisionTreeRegressor) no conjunto C07.", argv)
    if args.teste:
        raise SystemExit("O teste é usado só para a solução final (ver boosting.py --teste).")
    controle, X, y, dobras, rotulo, sufixo = preparar_execucao(args)
    if not args.so_busca:
        log(f"LC-C07-ARV{sufixo}: DecisionTreeRegressor(random_state=42), configuração inicial do notebook")
        avaliar("LC-C07-ARV" + sufixo, HIPOTESE, FAMILIA, REGRESSOR, controle, X, y, dobras, rotulo, PREPARACAO,
                args.resultados)
    if args.busca or args.so_busca:
        log(f"Estudo de complexidade da árvore (GridSearchCV, {len(ESPACO)} estratégias x 3 dobras)")
        buscar(ID_BUSCA + sufixo,
               "Estudo de complexidade da árvore: limitar profundidade, tamanho da folha ou podar reduz o "
               "sobreajuste da configuração inicial e o RMSE da validação.",
               FAMILIA, REGRESSOR, ESPACO, X, y, dobras, rotulo, PREPARACAO, pasta=args.resultados)
        resultados = _marcar_estrategias(args.resultados, ID_BUSCA + sufixo)
        grafico_complexidade(resultados, args.resultados, "LC-C07-ARV" + sufixo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
