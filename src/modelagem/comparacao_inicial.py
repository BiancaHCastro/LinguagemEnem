"""Comparação dos modelos (enunciado, Seção 4, passo 3, e figura obrigatória do artigo).

1. Configurações iniciais de todos os regressores pelo RMSE médio nas mesmas 3 dobras, a partir de
   resultados/experimentos.csv, e indicação dos 3 melhores regressores distintos (sem contar o baseline).
2. Os 3 melhores antes e depois da busca de hiperparâmetros (linhas <id>-BUSCA).
3. Figura com as duas comparações.

Uso (a partir da raiz do projeto):
    python -m src.modelagem.comparacao_inicial

Saídas: resultados/comparacao_inicial.csv, resultados/antes_depois_busca.csv e
resultados/figuras/comparacao_modelos.png
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .comum import PASTA_RESULTADOS

# Fora da comparação inicial: buscas, testes, amostras e variações de aprofundamento (aprofundamento_notas_extremas.py).
EXCLUIR = r"BUSCA|TESTE|-V\d+$|-A\d+$"
NOMES = {
    "LC-C07-BASE": "Baseline (média)", "LC-C07-LIN-OLS": "Regressão linear", "LC-C07-LIN-RIDGE": "Ridge",
    "LC-C07-ARV": "Árvore de decisão", "LC-C07-KNN": "KNN", "LC-C07-BAG": "Bagging", "LC-C07-PAST": "Pasting",
    "LC-C07-HGB": "Boosting (HistGB)", "LC-C07-STACK": "Stacking",
}


def main(argv=None):
    for fluxo in (sys.stdout, sys.stderr):
        if hasattr(fluxo, "reconfigure"):
            fluxo.reconfigure(encoding="utf-8", errors="replace")
    experimentos = pd.read_csv(PASTA_RESULTADOS / "experimentos.csv")
    cv = experimentos[experimentos["conjunto_avaliado"] == "validacao_cruzada_desenvolvimento_3_dobras"]

    # 1) comparação inicial
    colunas = ["id_experimento", "modelo", "rmse_media", "rmse_dp", "mae_media", "mae_dp", "r2_media", "r2_dp",
               "rmse_treino_media", "tempo_total_s"]
    tabela = cv[~cv["id_experimento"].str.contains(EXCLUIR, regex=True)][colunas]
    tabela = tabela.sort_values("rmse_media").reset_index(drop=True)
    tabela.insert(1, "nome", tabela["id_experimento"].map(NOMES).fillna(tabela["modelo"]))
    tabela.insert(0, "posicao", range(1, len(tabela) + 1))
    tabela.to_csv(PASTA_RESULTADOS / "comparacao_inicial.csv", index=False)
    print("Comparação inicial (validação cruzada, 3 dobras):")
    print(tabela.drop(columns="modelo").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    top3 = tabela[tabela["modelo"] != "DummyRegressor"].head(3)
    print("\n3 melhores para a busca de hiperparâmetros: " + ", ".join(top3["nome"]))

    # 2) antes x depois da busca
    linhas = []
    for r in top3.itertuples():
        busca = cv[cv["id_experimento"] == f"{r.id_experimento}-BUSCA"]
        if busca.empty:
            print(f"AVISO: {r.id_experimento}-BUSCA ainda não existe")
            continue
        b = busca.iloc[0]
        linhas.append({"nome": r.nome, "rmse_inicial": r.rmse_media, "rmse_apos_busca": b["rmse_media"],
                       "rmse_dp_apos_busca": b["rmse_dp"], "mae_inicial": r.mae_media, "mae_apos_busca": b["mae_media"],
                       "r2_inicial": r.r2_media, "r2_apos_busca": b["r2_media"],
                       "ganho_rmse": r.rmse_media - b["rmse_media"], "tempo_busca_s": b["tempo_total_s"]})
    antes_depois = pd.DataFrame(linhas)
    antes_depois.to_csv(PASTA_RESULTADOS / "antes_depois_busca.csv", index=False)
    if len(antes_depois):
        print("\nAntes x depois da busca:")
        print(antes_depois.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # 3) figura
    fig, (esq, dir_) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.4, 1]})
    ordem = tabela.iloc[::-1]
    cores = ["#9aa5b1" if m == "DummyRegressor" else ("#e0813b" if i in set(top3["id_experimento"]) else "#3b78c3")
             for m, i in zip(ordem["modelo"], ordem["id_experimento"])]
    esq.barh(ordem["nome"], ordem["rmse_media"], xerr=ordem["rmse_dp"], color=cores)
    for y, v in enumerate(ordem["rmse_media"]):
        esq.text(v + 0.5, y, f"{v:.2f}", va="center", fontsize=8)
    esq.set_xlim(60, max(ordem["rmse_media"]) + 6)
    esq.set_xlabel("RMSE médio na validação (pontos; menor é melhor)")
    esq.set_title("Configurações iniciais (laranja: 3 melhores)")
    esq.grid(axis="x", alpha=0.3)
    if len(antes_depois):
        x = range(len(antes_depois))
        dir_.bar([i - 0.2 for i in x], antes_depois["rmse_inicial"], 0.4, label="inicial", color="#9aa5b1")
        dir_.bar([i + 0.2 for i in x], antes_depois["rmse_apos_busca"], 0.4, label="após a busca", color="#e0813b")
        for i, (a, b) in enumerate(zip(antes_depois["rmse_inicial"], antes_depois["rmse_apos_busca"])):
            dir_.text(i - 0.2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
            dir_.text(i + 0.2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
        dir_.set_xticks(list(x))
        dir_.set_xticklabels(antes_depois["nome"])
        baixo = min(antes_depois["rmse_apos_busca"].min(), antes_depois["rmse_inicial"].min())
        dir_.set_ylim(baixo - 0.6, max(antes_depois["rmse_inicial"].max(), antes_depois["rmse_apos_busca"].max()) + 0.4)
        dir_.set_ylabel("RMSE médio na validação (pontos)")
        dir_.set_title("3 melhores: antes e depois da busca")
        dir_.legend()
        dir_.grid(axis="y", alpha=0.3)
    fig.text(0.01, 0.01, "Desenvolvimento, validação cruzada de 3 dobras (barras de erro: DP entre dobras). "
             "Fonte: Inep, ENEM 2023.", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    (PASTA_RESULTADOS / "figuras").mkdir(parents=True, exist_ok=True)
    fig.savefig(PASTA_RESULTADOS / "figuras" / "comparacao_modelos.png", dpi=150)
    plt.close(fig)
    print("\nFigura: resultados/figuras/comparacao_modelos.png")
    print("Obs.: Random Forest e SVM entram aqui quando forem registrados em experimentos.csv.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
