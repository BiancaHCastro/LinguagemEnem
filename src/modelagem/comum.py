"""Partes comuns dos scripts de modelagem do conjunto C07 (regressao_linear.py, bagging.py, pasting.py).

Protocolo (Seção 4 do enunciado), com TODOS os 2.822.643 participantes elegíveis (nota de LC observada):
- Validação cruzada: os 2.258.114 do desenvolvimento (80%), nas 3 dobras fixas da coluna DOBRA
  (KFold(3, shuffle=True, random_state=42)); cada dobra treina em 2/3 e valida no 1/3 restante.
- Sem vazamento: o pipeline completo (preparação + regressor) é clonado e ajustado só no treino de cada dobra;
  nenhuma estatística é calculada com a validação ou com o teste.
- Métricas: RMSE (principal), MAE e R² na escala original, por dobra, com média, desvio padrão e erro de treino.
- Teste final (--teste): reajuste nos 80% completos e avaliação única nos 564.529 do teste (20%).
- Registro: resultados/experimentos.csv (mesmas colunas do notebook), previsões fora da dobra em
  resultados/previsoes_cv/<id>.parquet e previsões do teste em resultados/previsoes_teste.csv.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from ..conjuntos import configuracao as cfg
from ..conjuntos.preprocessamento import colunas_para_familia, dobras_da_coluna, montar_pipeline
from ..dados import configuracao as cfg_dados

CONJUNTO = "C07"
ARQUIVO_C07 = cfg_dados.PASTA_SAIDA_PADRAO / "enem_2023_lc_c07.parquet"
PASTA_RESULTADOS = cfg.RAIZ_PROJETO / "resultados"
METRICAS = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
_FUNCOES_METRICAS = {"rmse": root_mean_squared_error, "mae": mean_absolute_error, "r2": r2_score}
CONTROLE = [cfg.COLUNA_ID, "CONJUNTO", cfg.COLUNA_DOBRA, cfg.COLUNA_ALVO]


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def argumentos(descricao, argv=None, paralelo=False):
    """Opções comuns aos três scripts."""
    parser = argparse.ArgumentParser(description=descricao)
    parser.add_argument("--busca", action="store_true",
                        help="roda também a busca de hiperparâmetros (≥ 10 configurações, 3 dobras)")
    parser.add_argument("--so-busca", action="store_true", help="pula a configuração inicial e roda só a busca")
    parser.add_argument("--teste", action="store_true",
                        help="AVALIAÇÃO FINAL: reajusta nos 80%% e avalia no teste (só depois de fixar a solução)")
    parser.add_argument("--amostra", type=int, default=None,
                        help="usa N linhas do desenvolvimento (1/3 por dobra) só para testar o script")
    parser.add_argument("--arquivo", type=Path, default=ARQUIVO_C07, help="parquet do C07")
    parser.add_argument("--resultados", type=Path, default=PASTA_RESULTADOS, help="pasta de resultados")
    if paralelo:
        parser.add_argument("--n-jobs", type=int, default=8,
                            help="núcleos usados pelo ensemble (padrão 8; -1 = todos). Diminua se faltar memória")
    return parser.parse_args(argv)


def _milhar(n):
    return f"{n:,}".replace(",", ".")


def contar_partes(arquivo=ARQUIVO_C07):
    """Nº de participantes por parte (desenvolvimento/teste) no arquivo, lendo só a coluna CONJUNTO."""
    return pd.read_parquet(arquivo, columns=["CONJUNTO"])["CONJUNTO"].value_counts().to_dict()


def preparar_execucao(args):
    """Carrega os dados e devolve (controle, X, y, dobras, rótulo do conjunto avaliado, sufixo dos ids).

    Sem --amostra, confere que TODOS os participantes do desenvolvimento entraram na validação cruzada.
    """
    if args.teste and args.amostra:
        raise SystemExit("--teste usa todo o desenvolvimento para o reajuste final; não combine com --amostra.")
    partes = contar_partes(args.arquivo)
    log(f"{args.arquivo.name}: {_milhar(sum(partes.values()))} elegíveis = {_milhar(partes['desenvolvimento'])} "
        f"desenvolvimento + {_milhar(partes.get('teste', 0))} teste (reservado)")
    controle, X, y, dobras = carregar_desenvolvimento(args.arquivo, args.amostra)
    if not args.amostra and len(y) != partes["desenvolvimento"]:
        raise RuntimeError(f"Esperados {partes['desenvolvimento']} participantes do desenvolvimento, lidos {len(y)}.")
    log(f"Validação cruzada com {_milhar(len(y))} participantes"
        f"{' (AMOSTRA de teste do script)' if args.amostra else ' (todo o desenvolvimento)'}, {X.shape[1]} features; "
        f"validação por dobra: {' / '.join(_milhar(len(v)) for _, v in dobras)}")
    sufixo = f"-A{len(y)}" if args.amostra else ""
    return controle, X, y, dobras, rotulo_avaliacao(len(y), args.amostra), sufixo


# --- Dados -----------------------------------------------------------------------------------------
def carregar_desenvolvimento(arquivo=ARQUIVO_C07, amostra=None, semente=cfg.SEMENTE):
    """Desenvolvimento do C07: (controle, X, y, dobras). `amostra` = nº de linhas para um teste rápido (1/3 por dobra)."""
    if not arquivo.exists():
        raise FileNotFoundError(f"{arquivo} não existe. Rode: python -m src.conjuntos.exportar_conjunto C07")
    df = pd.read_parquet(arquivo)
    dev = df[df["CONJUNTO"] == "desenvolvimento"]
    del df
    if amostra and amostra < len(dev):
        rng = np.random.default_rng(semente)
        dobra = dev[cfg.COLUNA_DOBRA].to_numpy()
        tamanhos = np.diff(np.linspace(0, amostra, 4).round().astype(int))
        posicoes = [rng.choice(np.flatnonzero(dobra == k), size=n, replace=False) for k, n in zip((1, 2, 3), tamanhos)]
        dev = dev.iloc[np.sort(np.concatenate(posicoes))]
    dev = dev.reset_index(drop=True)
    X = dev.drop(columns=CONTROLE)
    return dev[CONTROLE], X, dev[cfg.COLUNA_ALVO].to_numpy(dtype=float), dobras_da_coluna(dev[cfg.COLUNA_DOBRA])


def rotulo_avaliacao(n_linhas, amostra):
    if amostra:
        return f"validacao_cruzada_desenvolvimento_amostra_{n_linhas}_3_dobras"
    return "validacao_cruzada_desenvolvimento_3_dobras"


# --- Registro --------------------------------------------------------------------------------------
def salvar_experimento(registro, pasta=PASTA_RESULTADOS):
    """Acrescenta (ou substitui, pelo id) uma linha em experimentos.csv."""
    arquivo = pasta / "experimentos.csv"
    tabela = pd.read_csv(arquivo, dtype={"conclusao": "object"}) if arquivo.exists() else pd.DataFrame()
    if len(tabela):
        tabela = tabela[tabela["id_experimento"] != registro["id_experimento"]]
    tabela = pd.concat([tabela, pd.DataFrame([registro])], ignore_index=True)
    pasta.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(arquivo, index=False)


def _registro(id_experimento, hipotese, familia, regressor, parametros, conjunto_avaliado, n_linhas, preparacao):
    # O baseline prevê a média do treino: não usa nenhuma feature.
    colunas = [] if isinstance(regressor, DummyRegressor) else colunas_para_familia(CONJUNTO, familia)
    return {
        "id_experimento": id_experimento, "data_hora": datetime.now().isoformat(timespec="seconds"),
        "alvo": cfg.COLUNA_ALVO, "hipotese": hipotese, "features": "|".join(colunas),
        "n_features_entrada": len(colunas), "preparacao": preparacao, "modelo": type(regressor).__name__,
        "parametros": json.dumps(parametros, default=str), "conjunto_avaliado": conjunto_avaliado,
        "semente": cfg.SEMENTE, "id_conjunto": CONJUNTO, "familia": familia, "linhas_avaliadas": n_linhas,
    }


def _metricas_no_registro(registro, validacao, treino):
    """validacao/treino: {métrica: array com o valor de cada dobra}."""
    for metrica in METRICAS:
        for k, valor in enumerate(validacao[metrica], start=1):
            registro[f"{metrica}_dobra{k}"] = float(valor)
        registro[f"{metrica}_media"] = float(np.mean(validacao[metrica]))
        registro[f"{metrica}_dp"] = float(np.std(validacao[metrica], ddof=1))
        registro[f"{metrica}_treino_media"] = float(np.mean(treino[metrica]))


def _imprimir(registro, tempo_total):
    log(f"{registro['id_experimento']}: RMSE = {registro['rmse_media']:.2f} ± {registro['rmse_dp']:.2f} | "
        f"MAE = {registro['mae_media']:.2f} | R² = {registro['r2_media']:.4f} | "
        f"RMSE treino = {registro['rmse_treino_media']:.2f} | {tempo_total:.0f} s")


# --- Validação cruzada da configuração inicial --------------------------------------------------------
def avaliar(id_experimento, hipotese, familia, regressor, controle, X, y, dobras, conjunto_avaliado,
            preparacao, pasta=PASTA_RESULTADOS):
    """Treina e valida nas 3 dobras; grava métricas, tempos e as previsões fora da dobra."""
    inicio = time.perf_counter()
    validacao = {m: [] for m in METRICAS}
    treino = {m: [] for m in METRICAS}
    tempos_ajuste, tempos_previsao, n_saida = [], [], []
    previsto = np.full(len(y), np.nan)
    for k, (idx_treino, idx_validacao) in enumerate(dobras, start=1):
        pipeline = montar_pipeline(CONJUNTO, familia, clone(regressor))
        t0 = time.perf_counter()
        pipeline.fit(X.iloc[idx_treino], y[idx_treino])
        tempos_ajuste.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        previsto[idx_validacao] = pipeline.predict(X.iloc[idx_validacao])
        tempos_previsao.append(time.perf_counter() - t0)
        previsto_treino = pipeline.predict(X.iloc[idx_treino])
        for metrica, funcao in _FUNCOES_METRICAS.items():
            validacao[metrica].append(funcao(y[idx_validacao], previsto[idx_validacao]))
            treino[metrica].append(funcao(y[idx_treino], previsto_treino))
        n_saida.append(pipeline.named_steps["modelo"].n_features_in_)
        log(f"  dobra {k}: RMSE validação {validacao['rmse'][-1]:.2f} | treino {treino['rmse'][-1]:.2f} | "
            f"ajuste {tempos_ajuste[-1]:.0f} s")
        del pipeline, previsto_treino

    registro = _registro(id_experimento, hipotese, familia, regressor, regressor.get_params(), conjunto_avaliado,
                         len(y), preparacao)
    registro["n_features_saida"] = int(np.median(n_saida))
    _metricas_no_registro(registro, validacao, treino)
    registro["tempo_ajuste_medio_s"] = float(np.mean(tempos_ajuste))
    registro["tempo_previsao_medio_s"] = float(np.mean(tempos_previsao))
    registro["tempo_total_s"] = time.perf_counter() - inicio
    registro["conclusao"] = ""
    salvar_experimento(registro, pasta)

    pasta_previsoes = pasta / "previsoes_cv"
    pasta_previsoes.mkdir(parents=True, exist_ok=True)
    controle.assign(PREVISTO=previsto).drop(columns="CONJUNTO").to_parquet(
        pasta_previsoes / f"{id_experimento}.parquet", index=False)
    _imprimir(registro, registro["tempo_total_s"])
    return registro


# --- Avaliação final no teste --------------------------------------------------------------------------
def carregar_teste(arquivo=ARQUIVO_C07):
    """(controle, X, y) dos participantes do teste. Só para a avaliação final."""
    df = pd.read_parquet(arquivo)
    teste = df[df["CONJUNTO"] == "teste"].reset_index(drop=True)
    return teste[CONTROLE], teste.drop(columns=CONTROLE), teste[cfg.COLUNA_ALVO].to_numpy(dtype=float)


def melhor_da_busca(regressor, id_busca, pasta=PASTA_RESULTADOS):
    """Regressor com os hiperparâmetros da 1ª posição de resultados/busca_<id_busca>.csv, se a busca já rodou."""
    arquivo = pasta / f"busca_{id_busca}.csv"
    if not arquivo.exists():
        return clone(regressor), "configuração inicial (busca ainda não rodada)"
    # params_json preserva o tipo (max_samples=1.0 é 100% das linhas; max_samples=1 seria UMA linha).
    melhor = json.loads(pd.read_csv(arquivo).iloc[0]["params_json"])
    parametros = {k.removeprefix("modelo__"): v for k, v in melhor.items()}
    return clone(regressor).set_params(**parametros), f"melhor configuração de {arquivo.name}: {parametros}"


def avaliar_teste(id_experimento, hipotese, familia, regressor, controle_dev, X_dev, y_dev, preparacao,
                  arquivo=ARQUIVO_C07, pasta=PASTA_RESULTADOS):
    """Reajusta o pipeline nos 80% completos (desenvolvimento) e avalia UMA vez nos 20% do teste."""
    controle_teste, X_teste, y_teste = carregar_teste(arquivo)
    pipeline = montar_pipeline(CONJUNTO, familia, clone(regressor))
    inicio = time.perf_counter()
    pipeline.fit(X_dev, y_dev)
    tempo_ajuste = time.perf_counter() - inicio
    previsto = pipeline.predict(X_teste)
    previsto_treino = pipeline.predict(X_dev)

    registro = _registro(id_experimento, hipotese, familia, regressor, regressor.get_params(),
                         "teste_final_ajuste_em_100pct_do_desenvolvimento", len(y_teste), preparacao)
    registro["linhas_treino"] = len(y_dev)
    registro["n_features_saida"] = int(pipeline.named_steps["modelo"].n_features_in_)
    for metrica, funcao in _FUNCOES_METRICAS.items():
        registro[f"{metrica}_teste"] = float(funcao(y_teste, previsto))
        registro[f"{metrica}_treino_media"] = float(funcao(y_dev, previsto_treino))
    registro["tempo_ajuste_medio_s"] = tempo_ajuste
    registro["tempo_total_s"] = time.perf_counter() - inicio
    registro["conclusao"] = ""
    salvar_experimento(registro, pasta)

    # previsoes_teste.csv: identificador, nota observada e uma coluna de previsão por modelo avaliado no teste.
    arquivo_previsoes = pasta / "previsoes_teste.csv"
    previsoes = (pd.read_csv(arquivo_previsoes, dtype={cfg.COLUNA_ID: str}) if arquivo_previsoes.exists()
                 else controle_teste[[cfg.COLUNA_ID, cfg.COLUNA_ALVO]].astype({cfg.COLUNA_ID: str}))
    if list(previsoes[cfg.COLUNA_ID]) != list(controle_teste[cfg.COLUNA_ID].astype(str)):
        raise RuntimeError(f"{arquivo_previsoes} não corresponde aos participantes do teste atual.")
    previsoes[f"PREVISTO_{id_experimento}"] = previsto
    previsoes.to_csv(arquivo_previsoes, index=False)
    log(f"{id_experimento}: TESTE RMSE = {registro['rmse_teste']:.2f} | MAE = {registro['mae_teste']:.2f} | "
        f"R² = {registro['r2_teste']:.4f} | treino em {_milhar(len(y_dev))}, teste em {_milhar(len(y_teste))} | "
        f"{registro['tempo_total_s']:.0f} s")
    return registro


def avisar_teste():
    log("ATENÇÃO: avaliação no TESTE. Pelo enunciado, só depois de fixar a solução final pela validação cruzada; "
        "não volte a escolher features ou hiperparâmetros com base nestes números.")


# --- Busca de hiperparâmetros ------------------------------------------------------------------------
def buscar(id_experimento, hipotese, familia, regressor, espaco, X, y, dobras, conjunto_avaliado, preparacao,
           n_iter=None, pasta=PASTA_RESULTADOS):
    """GridSearchCV (n_iter=None) ou RandomizedSearchCV nas mesmas 3 dobras, critério = RMSE médio.

    Grava todas as configurações em resultados/busca_<id>.csv e a melhor em experimentos.csv.
    O reajuste nos 80% completos (refit) fica para a etapa final, depois de escolher a solução.
    """
    pipeline = montar_pipeline(CONJUNTO, familia, regressor)
    opcoes = dict(scoring=METRICAS, refit=False, cv=dobras, return_train_score=True, n_jobs=1, error_score="raise",
                  verbose=1)
    if n_iter is None:
        busca = GridSearchCV(pipeline, espaco, **opcoes)
    else:
        busca = RandomizedSearchCV(pipeline, espaco, n_iter=n_iter, random_state=cfg.SEMENTE, **opcoes)
    inicio = time.perf_counter()
    busca.fit(X, y)
    tempo_total = time.perf_counter() - inicio

    resultados = pd.DataFrame(busca.cv_results_)
    resultados["params_json"] = [json.dumps(p, default=float) for p in resultados.pop("params")]
    for metrica in METRICAS:
        sinal = 1 if metrica == "r2" else -1
        for coluna in [c for c in resultados.columns if c.endswith(f"test_{metrica}") or c.endswith(f"train_{metrica}")]:
            if not coluna.startswith("std_") and not coluna.startswith("rank_"):
                resultados[coluna] = sinal * resultados[coluna]
    resultados = resultados.sort_values("mean_test_rmse").reset_index(drop=True)
    resultados.insert(0, "posicao_rmse", np.arange(1, len(resultados) + 1))
    pasta.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(pasta / f"busca_{id_experimento}.csv", index=False)

    melhor = resultados.iloc[0]
    parametros = {**regressor.get_params(), **{k.removeprefix("param_modelo__"): melhor[k]
                                               for k in resultados.columns if k.startswith("param_modelo__")}}
    registro = _registro(id_experimento, hipotese, familia, regressor, parametros, conjunto_avaliado, len(y),
                         preparacao)
    _metricas_no_registro(
        registro,
        {m: [melhor[f"split{k}_test_{m}"] for k in range(3)] for m in METRICAS},
        {m: [melhor[f"split{k}_train_{m}"] for k in range(3)] for m in METRICAS},
    )
    registro["n_configuracoes_busca"] = len(resultados)
    registro["tempo_ajuste_medio_s"] = float(melhor["mean_fit_time"])
    registro["tempo_total_s"] = tempo_total
    registro["conclusao"] = ""
    salvar_experimento(registro, pasta)
    _imprimir(registro, tempo_total)
    colunas = ["posicao_rmse", *[c for c in resultados.columns if c.startswith("param_")], "mean_test_rmse",
               "std_test_rmse", "mean_train_rmse", "mean_fit_time"]
    print(resultados[colunas].to_string(index=False))
    return resultados
