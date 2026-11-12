"""Boosting (HistGradientBoostingRegressor) no conjunto C07: configuração inicial, busca, teste final, curva de
aprendizado e gráficos do modelo final.

Configuração inicial = a de src/modelagem/pca_balanceamento_ablacao.py (taxa 0,1, até 200 árvores de 31 folhas, early
stopping), agora com as 23 features do C07, como os demais modelos da comparação inicial (o pca_balanceamento_ablacao.py
usava 15). Preparação da família "boosting" de src/conjuntos/preprocessamento.py: códigos ordinais, NaN
mantido e categóricas nominais tratadas como categorias nativas do HistGB. O early stopping separa 10% do
PRÓPRIO treino da dobra; a validação da dobra nunca participa da parada.

Busca (enunciado, Seção 4): GridSearchCV nas mesmas 3 dobras com 10 configurações válidas, cada uma uma
estratégia com hipótese própria (as 5 de src/modelagem/otimizacao_top3.py e mais 5). Critério: RMSE médio.

Uso (a partir da raiz do projeto), nesta ordem:
    python -m src.modelagem.boosting                 # 1) validação cruzada da configuração inicial
    python -m src.modelagem.boosting --so-busca      # 2) GridSearchCV com as 10 estratégias
    python -m src.modelagem.boosting --curva         # 3) curva de aprendizado da melhor configuração (só desenvolvimento)
    python -m src.modelagem.boosting --teste         # 4) AVALIAÇÃO FINAL no teste + gráficos e erros por faixa/grupo
    python -m src.modelagem.boosting --graficos      #    refaz só os gráficos do teste a partir de previsoes_teste.csv
    python -m src.modelagem.boosting --amostra 30000 #    teste rápido do script

O passo 4 é a avaliação final: rodar depois de fixar a solução pela validação cruzada (comparação entre os 3
melhores) e junto com `python -m src.modelagem.regressao_linear --teste` (baseline e regressão linear sem
regularização nos mesmos participantes).

Saídas: resultados/experimentos.csv (LC-C07-HGB, LC-C07-HGB-BUSCA, LC-C07-HGB-TESTE),
resultados/busca_LC-C07-HGB-BUSCA.csv (com a estratégia de cada configuração),
resultados/curva_aprendizado_LC-C07-HGB.csv, resultados/erros_teste_LC-C07-HGB.csv,
resultados/metricas_teste_final.csv, resultados/previsoes_teste.csv e figuras em resultados/figuras/.
"""

import json
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from ..conjuntos import configuracao as cfg
from ..conjuntos.preprocessamento import montar_pipeline
from .comum import (ARQUIVO_C07, CONJUNTO, PASTA_RESULTADOS, argumentos, avaliar, avaliar_teste, avisar_teste,
                    buscar, carregar_teste, log, melhor_da_busca, preparar_execucao)
from .pca_balanceamento_ablacao import TAMANHOS_CURVA, grafico_faixas, tabelas_completas

FAMILIA = "boosting"
ID = "LC-C07-HGB"
ID_BUSCA = f"{ID}-BUSCA"
ID_TESTE = f"{ID}-TESTE"
PREPARACAO = ("códigos ordinais (especiais 'não sei'/indeterminada depois do último nível); categóricas nominais "
              "como categorias nativas do HistGB; NaN mantido; early stopping com 10% do treino da dobra")

# Configuração inicial: a mesma do build_boosting() de pca_balanceamento_ablacao.py.
REGRESSOR = HistGradientBoostingRegressor(
    learning_rate=0.1, max_iter=200, max_leaf_nodes=31, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=10, scoring="loss", categorical_features="from_dtype", random_state=cfg.SEMENTE)
HIPOTESE = ("Boosting: árvores rasas em sequência, cada uma corrigindo o erro das anteriores, captam interações "
            "entre renda, escolaridade, escola e idade com pouco sobreajuste.")

# --- Busca: 10 estratégias (GridSearchCV com uma configuração por estratégia) --------------------------------
_CHAVES = ["learning_rate", "max_iter", "max_leaf_nodes", "max_depth", "min_samples_leaf", "l2_regularization",
           "max_features", "n_iter_no_change"]
ESTRATEGIAS = [
    # id, estratégia, hipótese, (taxa, máx. iterações, folhas, profundidade, folha mín., L2, features/divisão, paciência)
    ("HGB-1", "Melhor configuração da busca anterior (pca_balanceamento_ablacao.py), agora com as 23 features",
     "As 8 features do C07 que faltavam (ocupação dos pais, trajetória escolar) reduzem o erro.",
     (0.0605, 400, 127, None, 200, 0.1, 1.0, 10)),
    ("HGB-2", "Aprendizado lento e longo",
     "Passos menores com mais estágios generalizam melhor que poucos passos grandes.",
     (0.02, 3000, 63, None, 200, 1.0, 1.0, 50)),
    ("HGB-3", "Árvores grandes e muito regularizadas",
     "Muitas folhas captam interações de ordem alta; folha grande e L2 alto evitam ajustar ruído.",
     (0.04, 1500, 255, None, 1000, 10.0, 1.0, 30)),
    ("HGB-4", "Sorteio de 60% das features em cada divisão",
     "Sortear features em cada divisão (como no Random Forest) reduz a correlação entre os estágios.",
     (0.04, 1500, 127, None, 300, 1.0, 0.6, 30)),
    ("HGB-5", "Árvores rasas (profundidade 4) e muitos estágios",
     "Se o sinal for quase aditivo (interações fracas), árvores rasas generalizam melhor.",
     (0.1, 2000, 15, 4, 50, 0.0, 1.0, 30)),
    ("HGB-6", "Aprendizado lento com sorteio de 50% das features",
     "As duas melhores ideias (taxa baixa e sorteio de features) somadas superam cada uma isolada.",
     (0.02, 4000, 127, None, 300, 1.0, 0.5, 50)),
    ("HGB-7", "Sorteio agressivo (40% das features por divisão)",
     "Se o sorteio ajuda por descorrelacionar os estágios, sortear menos features ajuda mais, até perder sinal.",
     (0.04, 1500, 127, None, 300, 1.0, 0.4, 30)),
    ("HGB-8", "Árvores grandes regularizadas com sorteio de 60% das features",
     "O sorteio de features complementa a regularização das árvores grandes do HGB-3.",
     (0.04, 1500, 255, None, 1000, 10.0, 0.6, 30)),
    ("HGB-9", "Pouca regularização (folha mínima 20, L2 = 0)",
     "Com 1,5 milhão de linhas por dobra, folhas pequenas não sobreajustam e captam grupos raros.",
     (0.04, 1500, 127, None, 20, 0.0, 1.0, 30)),
    ("HGB-10", "Árvores rasas com sorteio de 60% das features",
     "O sorteio de features também beneficia o boosting raso do HGB-5.",
     (0.1, 2000, 15, 4, 50, 0.0, 0.6, 30)),
]
ESPACO = [{f"modelo__{k}": [v] for k, v in zip(_CHAVES, valores)} for *_, valores in ESTRATEGIAS]


def _marcar_estrategias(pasta, id_busca=ID_BUSCA):
    """Acrescenta id e estratégia a cada linha de busca_LC-C07-HGB-BUSCA.csv."""
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


# --- Curva de aprendizado ----------------------------------------------------------------------------------
def curva_aprendizado(X, y, dobras, regressor, origem, pasta):
    """10/25/50/75/100% do treino de cada dobra (subconjuntos aleatórios e aninhados, semente 42); validação fixa."""
    log(f"Curva de aprendizado com a {origem}")
    FIG = pasta / "figuras"
    linhas = []
    for k, (idx_treino, idx_validacao) in enumerate(dobras, start=1):
        ordem = np.random.default_rng(cfg.SEMENTE).permutation(idx_treino)
        for fracao in TAMANHOS_CURVA:
            sub = ordem[: int(round(fracao * len(idx_treino)))]
            pipeline = montar_pipeline(CONJUNTO, FAMILIA, clone(regressor))
            t0 = time.perf_counter()
            pipeline.fit(X.iloc[sub], y[sub])
            tempo = time.perf_counter() - t0
            rmse_treino = root_mean_squared_error(y[sub], pipeline.predict(X.iloc[sub]))
            rmse_validacao = root_mean_squared_error(y[idx_validacao], pipeline.predict(X.iloc[idx_validacao]))
            linhas.append({"dobra": k, "fracao": fracao, "n_treino": len(sub), "rmse_treino": rmse_treino,
                           "rmse_validacao": rmse_validacao, "tempo_ajuste_s": tempo})
            log(f"  dobra {k} | {fracao:>4.0%} (n={len(sub):,}) | treino {rmse_treino:.3f} | "
                f"validação {rmse_validacao:.3f} | {tempo:.0f} s")
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(pasta / f"curva_aprendizado_{ID}.csv", index=False)
    g = tabela.groupby("fracao").agg(n=("n_treino", "mean"), tr_m=("rmse_treino", "mean"),
                                     tr_s=("rmse_treino", "std"), va_m=("rmse_validacao", "mean"),
                                     va_s=("rmse_validacao", "std"))
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for pref, rotulo, cor in [("tr", "Treino", "#3b78c3"), ("va", "Validacao", "#e0813b")]:
        ax.plot(g["n"], g[f"{pref}_m"], marker="o", label=rotulo, color=cor)
        ax.fill_between(g["n"], g[f"{pref}_m"] - g[f"{pref}_s"], g[f"{pref}_m"] + g[f"{pref}_s"], alpha=0.2, color=cor)
    ax.set_xlabel("Exemplos no treino da dobra")
    ax.set_ylabel("RMSE (pontos)")
    ax.set_title("Curva de aprendizado - modelo final HGB (media e DP em 3 dobras)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.text(0.01, 0.01, "Desenvolvimento; teste nao utilizado. Fonte: Inep, ENEM 2023.", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "curva_aprendizado_hgb_final.png", dpi=150)
    plt.close(fig)
    print("Resumo da curva de aprendizado:\n" + g.to_string(), flush=True)
    log(f"Curva salva em {FIG / 'curva_aprendizado_hgb_final.png'}")


# --- Gráficos e tabelas do teste ----------------------------------------------------------------------------
def _metricas(y, previsto):
    return {"rmse": root_mean_squared_error(y, previsto), "mae": mean_absolute_error(y, previsto),
            "r2": r2_score(y, previsto)}


def graficos_teste(arquivo=ARQUIVO_C07, pasta=PASTA_RESULTADOS):
    """Gráficos e erros por faixa/grupo do modelo final no teste, a partir de previsoes_teste.csv."""
    FIG = pasta / "figuras"
    previsoes = pd.read_csv(pasta / "previsoes_teste.csv", dtype={cfg.COLUNA_ID: str})
    coluna = f"PREVISTO_{ID_TESTE}"
    if coluna not in previsoes:
        raise SystemExit(f"{coluna} não está em previsoes_teste.csv. Rode antes: python -m src.modelagem.boosting --teste")
    controle, X_teste, y_teste = carregar_teste(arquivo)
    if list(previsoes[cfg.COLUNA_ID]) != list(controle[cfg.COLUNA_ID].astype(str)):
        raise RuntimeError("previsoes_teste.csv não corresponde aos participantes do teste.")
    previsto = previsoes[coluna].to_numpy()

    # Tabela comparativa do teste: baseline, linear sem regularização e solução final, nos mesmos participantes.
    linhas = []
    for id_modelo, rotulo in [("LC-C07-BASE-TESTE", "Baseline (média)"),
                              ("LC-C07-LIN-OLS-TESTE", "Regressão linear sem regularização"),
                              (ID_TESTE, "Solução final (HistGradientBoosting)")]:
        if f"PREVISTO_{id_modelo}" in previsoes:
            linhas.append({"modelo": rotulo, "id_experimento": id_modelo, "n_teste": len(y_teste),
                           **_metricas(y_teste, previsoes[f"PREVISTO_{id_modelo}"].to_numpy())})
        else:
            log(f"AVISO: {id_modelo} ainda não avaliado no teste (python -m src.modelagem.regressao_linear --teste)")
    pd.DataFrame(linhas).to_csv(pasta / "metricas_teste_final.csv", index=False)
    print("Métricas no teste:\n" + pd.DataFrame(linhas).to_string(index=False), flush=True)

    tabelas = tabelas_completas(X_teste, y_teste, previsto)
    tabelas.to_csv(pasta / f"erros_teste_{ID}.csv", index=False)
    print("Erros por faixa e grupo no teste:\n" + tabelas.to_string(index=False), flush=True)

    nome = "hgb_final"
    FIG.mkdir(parents=True, exist_ok=True)
    residuo = y_teste - previsto
    limite = [0, max(y_teste.max(), previsto.max())]
    fig, ax = plt.subplots(figsize=(6, 5.5))
    h = ax.hexbin(previsto, y_teste, gridsize=70, bins="log", cmap="viridis", mincnt=1)
    ax.plot(limite, limite, "r--", lw=1, label="previsao perfeita")
    ax.set_xlabel("Nota prevista")
    ax.set_ylabel("Nota observada")
    ax.set_title("Observado x previsto - teste (HGB final)")
    fig.colorbar(h, label="participantes (log)")
    ax.legend()
    fig.text(0.01, 0.01, "Fonte: Inep, ENEM 2023. Celulas hexagonais agregam todos os registros do teste.", fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / f"teste_{nome}_observado_vs_previsto.png", dpi=150)
    plt.close(fig)

    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.5))
    eixos[0].hexbin(previsto, residuo, gridsize=70, bins="log", cmap="viridis", mincnt=1)
    eixos[0].axhline(0, color="r", ls="--", lw=1)
    eixos[0].set_xlabel("Nota prevista")
    eixos[0].set_ylabel("Residuo (observada - prevista)")
    eixos[0].set_title("Residuos x previsao")
    eixos[1].hist(residuo, bins=80, color="#3b78c3")
    eixos[1].axvline(0, color="r", ls="--", lw=1)
    eixos[1].set_xlabel("Residuo")
    eixos[1].set_ylabel("Participantes")
    eixos[1].set_title("Distribuicao dos residuos")
    fig.text(0.01, 0.01, "Teste reservado. Fonte: Inep, ENEM 2023.", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / f"teste_{nome}_residuos.png", dpi=150)
    plt.close(fig)

    grafico_faixas({"HGB final": tabelas[tabelas["agrupamento"] == "faixa_nota"].reset_index(drop=True)},
                   FIG / f"teste_{nome}_erro_por_faixa.png", "RMSE por faixa de nota - teste")
    for agrupamento in ["TP_ESCOLA", "Q006"]:
        t = tabelas[tabelas["agrupamento"] == agrupamento]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(t["grupo"], t["rmse"], color="#3b78c3")
        for i, n in enumerate(t["n"]):
            ax.text(i, 0, f"n={int(n):,}".replace(",", "."), ha="center", va="bottom", fontsize=7, color="w",
                    rotation=90)
        ax.set_ylabel("RMSE (pontos)")
        ax.set_title(f"RMSE por {agrupamento} - teste (HGB final)")
        fig.tight_layout()
        fig.savefig(FIG / f"teste_{nome}_erro_por_{agrupamento}.png", dpi=150)
        plt.close(fig)
    log(f"Gráficos do teste salvos em {FIG} (teste_{nome}_*.png)")


def main(argv=None):
    import argparse
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--curva", action="store_true")
    pre.add_argument("--graficos", action="store_true")
    extras, resto = pre.parse_known_args(argv)
    args = argumentos("HistGradientBoosting no conjunto C07 (inicial, busca, curva e teste final). "
                      "Extras: --curva (curva de aprendizado) e --graficos (só os gráficos do teste).", resto)
    for fluxo in (sys.stdout, sys.stderr):
        if hasattr(fluxo, "reconfigure"):
            fluxo.reconfigure(encoding="utf-8", errors="replace")

    if extras.graficos:
        graficos_teste(args.arquivo, args.resultados)
        return 0

    controle, X, y, dobras, rotulo, sufixo = preparar_execucao(args)
    final, origem = melhor_da_busca(REGRESSOR, ID_BUSCA, args.resultados)

    if args.teste:
        avisar_teste()
        log(f"Solução final no teste com a {origem}")
        avaliar_teste(ID_TESTE, HIPOTESE, FAMILIA, final, controle, X, y, PREPARACAO, args.arquivo, args.resultados)
        graficos_teste(args.arquivo, args.resultados)
        return 0
    if extras.curva:
        curva_aprendizado(X, y, dobras, final, origem, args.resultados)
        return 0
    if not args.so_busca:
        log(f"{ID}{sufixo}: configuração inicial do pca_balanceamento_ablacao.py (taxa 0,1, 200 árvores, 31 folhas)")
        avaliar(ID + sufixo, HIPOTESE, FAMILIA, REGRESSOR, controle, X, y, dobras, rotulo, PREPARACAO,
                args.resultados)
    if args.busca or args.so_busca:
        log(f"Busca do boosting (GridSearchCV, {len(ESPACO)} estratégias x 3 dobras)")
        buscar(ID_BUSCA + sufixo,
               "Busca do boosting: 10 estratégias (taxa, tamanho e regularização das árvores, sorteio de features) "
               "comparadas pelo RMSE médio nas 3 dobras.",
               FAMILIA, REGRESSOR, ESPACO, X, y, dobras, rotulo, PREPARACAO, pasta=args.resultados)
        _marcar_estrategias(args.resultados, ID_BUSCA + sufixo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
