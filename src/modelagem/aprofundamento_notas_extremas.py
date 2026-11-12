"""Aprofundamento: os erros nas notas extremas podem ser reduzidos? (enunciado, "Possibilidades de aprofundamento")

Parte da melhor configuração da busca do boosting (resultados/busca_LC-C07-HGB-BUSCA.csv) e testa 5 variações,
cada uma com uma hipótese, nas mesmas 3 dobras e com o mesmo protocolo (tudo ajustado só no treino da dobra):

    V0  referência: melhor configuração da busca, sem mudanças
    V1  treino sem as notas zero (retiradas só do treino de cada dobra; a validação mantém os zeros)
    V2  função de perda absoluta (absolute_error) em vez da quadrática
    V3  alvo transformado (QuantileTransformer normal, ajustado no treino da dobra; métricas na escala original)
    V4  restrições monotônicas: a nota prevista não diminui quando renda, renda per capita ou computadores aumentam
    V5  solução mais simples: árvores de 31 folhas em vez de 127

Para cada variação: RMSE, MAE e R² (média e DP nas dobras), erro de treino, tempo e erros por faixa de nota
(fora da dobra), comparados com a V0.

Uso (a partir da raiz do projeto):
    python -m src.modelagem.aprofundamento_notas_extremas                  # as 6 rodadas (pula as já feitas)
    python -m src.modelagem.aprofundamento_notas_extremas --apenas V2 V3   # só algumas
    python -m src.modelagem.aprofundamento_notas_extremas --refazer        # roda de novo
    python -m src.modelagem.aprofundamento_notas_extremas --resumo         # só refaz tabelas e gráfico
    python -m src.modelagem.aprofundamento_notas_extremas --amostra 30000  # teste rápido do script

Saídas: linhas LC-C07-HGB-V0..V5 em resultados/experimentos.csv, resultados/previsoes_cv/LC-C07-HGB-V*.parquet,
resultados/variacoes_hgb.csv, resultados/variacoes_hgb_por_faixa.csv e
resultados/figuras/variacoes_hgb_erro_por_faixa.png.
"""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer

from ..conjuntos import configuracao as cfg
from .boosting import FAMILIA, ID_BUSCA, PREPARACAO, REGRESSOR
from .comum import (ARQUIVO_C07, PASTA_RESULTADOS, avaliar, carregar_desenvolvimento, log, melhor_da_busca,
                    rotulo_avaliacao)
from .pca_balanceamento_ablacao import faixa_nota, grafico_faixas, tabela_erros


class SemZerosNoTreino(RegressorMixin, BaseEstimator):
    """Ajusta o regressor sem as linhas de nota zero; prevê para todos (inclusive quem tirou zero).

    Fica dentro do pipeline, então a exclusão acontece só no treino de cada dobra; a validação e o teste
    continuam com os zeros, como pede o enunciado.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        manter = np.asarray(y) != 0
        self.n_zeros_removidos_ = int((~manter).sum())
        self.estimator_ = clone(self.estimator).fit(X[manter], np.asarray(y)[manter])
        self.n_features_in_ = getattr(self.estimator_, "n_features_in_", X.shape[1])
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


def variacoes(base):
    """(id, nome, hipótese, regressor) das 6 rodadas, a partir da melhor configuração da busca."""
    monotonico = {"ord__Q006": 1, "ord__Q024": 1, "num__RENDA_PER_CAPITA_APROX": 1}
    return [
        ("V0", "Referência: melhor configuração da busca",
         "Ponto de comparação das variações, nas mesmas dobras.", clone(base)),
        ("V1", "Treino sem as notas zero",
         "Os 1.736 zeros do desenvolvimento puxam as previsões para baixo; sem eles no treino o erro das demais "
         "notas cai, mas o erro de quem tirou zero aumenta.", SemZerosNoTreino(clone(base))),
        ("V2", "Perda absoluta (absolute_error)",
         "A perda absoluta é menos sensível às notas extremas: reduz o MAE, possivelmente à custa do RMSE e das "
         "faixas extremas.", clone(base).set_params(loss="absolute_error")),
        ("V3", "Alvo transformado (QuantileTransformer normal)",
         "Abrir as caudas da distribuição do alvo no ajuste faz o modelo prever notas mais extremas e reduz o "
         "erro nas faixas raras.",
         TransformedTargetRegressor(regressor=clone(base), transformer=QuantileTransformer(
             output_distribution="normal", n_quantiles=1000, subsample=200_000, random_state=cfg.SEMENTE))),
        ("V4", "Restrições monotônicas (renda, renda per capita, computadores)",
         "Impor que a nota prevista não diminua com renda e computadores torna o modelo mais simples de "
         "interpretar sem perder precisão.", clone(base).set_params(monotonic_cst=monotonico)),
        ("V5", "Solução mais simples (31 folhas por árvore)",
         "Árvores com 4 vezes menos folhas mantêm o desempenho com um modelo menor.",
         clone(base).set_params(max_leaf_nodes=31)),
    ]


def resumo(pasta, sufixo=""):
    experimentos = pd.read_csv(pasta / "experimentos.csv")
    linhas, faixas = [], {}
    for id_variacao, nome, hipotese, _ in variacoes(REGRESSOR):
        id_exp = f"LC-C07-HGB-{id_variacao}{sufixo}"
        reg = experimentos[experimentos["id_experimento"] == id_exp]
        arquivo_oof = pasta / "previsoes_cv" / f"{id_exp}.parquet"
        if reg.empty or not arquivo_oof.exists():
            continue
        r = reg.iloc[0]
        oof = pd.read_parquet(arquivo_oof)
        y, previsto = oof[cfg.COLUNA_ALVO].to_numpy(), oof["PREVISTO"].to_numpy()
        tabela = tabela_erros(y, previsto, faixa_nota(y), "faixa_nota")
        tabela.insert(0, "variacao", id_variacao)
        faixas[f"{id_variacao}"] = tabela
        linhas.append({"variacao": id_variacao, "id_experimento": id_exp, "descricao": nome, "hipotese": hipotese,
                       **{k: r[k] for k in ["rmse_media", "rmse_dp", "mae_media", "mae_dp", "r2_media", "r2_dp",
                                            "rmse_treino_media", "tempo_ajuste_medio_s", "tempo_total_s"]}})
    if not linhas:
        log("Nenhuma variação avaliada ainda.")
        return
    tabela = pd.DataFrame(linhas)
    if "V0" in set(tabela["variacao"]):
        ref = tabela.set_index("variacao").loc["V0"]
        tabela["delta_rmse_vs_V0"] = tabela["rmse_media"] - ref["rmse_media"]
        tabela["delta_mae_vs_V0"] = tabela["mae_media"] - ref["mae_media"]
        tabela["delta_r2_vs_V0"] = tabela["r2_media"] - ref["r2_media"]
    tabela.to_csv(pasta / "variacoes_hgb.csv", index=False)
    por_faixa = pd.concat(faixas.values(), ignore_index=True)
    por_faixa.to_csv(pasta / "variacoes_hgb_por_faixa.csv", index=False)

    colunas = ["variacao", "rmse_media", "rmse_dp", "mae_media", "r2_media", "rmse_treino_media",
               *[c for c in tabela if c.startswith("delta_")], "tempo_total_s", "descricao"]
    print("\nVariações do HGB (validação cruzada, 3 dobras):")
    print(tabela[colunas].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nRMSE por faixa de nota (previsões fora da dobra):")
    print(por_faixa.pivot(index="grupo", columns="variacao", values="rmse")
          .reindex(faixas[next(iter(faixas))]["grupo"]).to_string(float_format=lambda v: f"{v:.1f}"))
    (pasta / "figuras").mkdir(parents=True, exist_ok=True)
    grafico_faixas({k: v.reset_index(drop=True) for k, v in faixas.items()},
                   pasta / "figuras" / "variacoes_hgb_erro_por_faixa.png",
                   "RMSE por faixa de nota - variacoes do HGB (desenvolvimento, fora da dobra)")
    log(f"Tabelas em {pasta / 'variacoes_hgb.csv'} e gráfico em {pasta / 'figuras'}")


def main(argv=None):
    for fluxo in (sys.stdout, sys.stderr):
        if hasattr(fluxo, "reconfigure"):
            fluxo.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Variações do HGB para os erros nas notas extremas (C07).")
    parser.add_argument("--apenas", nargs="+", choices=["V0", "V1", "V2", "V3", "V4", "V5"])
    parser.add_argument("--refazer", action="store_true")
    parser.add_argument("--resumo", action="store_true", help="só refaz tabelas e gráfico")
    parser.add_argument("--amostra", type=int, default=None)
    parser.add_argument("--arquivo", type=__import__("pathlib").Path, default=ARQUIVO_C07)
    parser.add_argument("--resultados", type=__import__("pathlib").Path, default=PASTA_RESULTADOS)
    args = parser.parse_args(argv)
    sufixo = ""
    if args.resumo:
        resumo(args.resultados)
        return 0

    base, origem = melhor_da_busca(REGRESSOR, ID_BUSCA, args.resultados)
    log(f"Base das variações: {origem}")
    controle, X, y, dobras = carregar_desenvolvimento(args.arquivo, args.amostra)
    sufixo = f"-A{len(y)}" if args.amostra else ""
    rotulo = rotulo_avaliacao(len(y), args.amostra)
    feitos = set(pd.read_csv(args.resultados / "experimentos.csv")["id_experimento"]) \
        if (args.resultados / "experimentos.csv").exists() else set()
    for id_variacao, nome, hipotese, regressor in variacoes(base):
        if args.apenas and id_variacao not in args.apenas:
            continue
        id_exp = f"LC-C07-HGB-{id_variacao}{sufixo}"
        if id_exp in feitos and (args.resultados / "previsoes_cv" / f"{id_exp}.parquet").exists() and not args.refazer:
            log(f"{id_exp} já registrado (use --refazer para repetir)")
            continue
        log(f"{id_exp}: {nome}")
        avaliar(id_exp, hipotese, FAMILIA, regressor, controle, X, y, dobras, rotulo,
                PREPARACAO + f"; variação: {nome}", args.resultados)
    resumo(args.resultados, sufixo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
