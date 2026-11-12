import argparse
import csv
import json
import math
import re
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import IncrementalPCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "dados" / "processed" / "enem_2023_lc_c07.parquet"
SPLITS_PATH = ROOT / "dados" / "splits" / "particoes_lc.parquet"
OUT = ROOT / "resultados" / "knn_boosting"
FIG = ROOT / "resultados" / "figuras"
SAIDA = ROOT / "resultados" / "pca_balanceamento_ablacao"
RESUMO_EXECUCAO = SAIDA / "resultados_execucao.txt"
CSV_EXECUCAO = SAIDA / "resultados_execucao.csv"
SEED = 42
ID_COL = "NU_INSCRICAO"
TARGET = "NU_NOTA_LC"
FEATURES = [
    "Q006",
    "Q001",
    "Q002",
    "TP_ESCOLA",
    "TP_ST_CONCLUSAO",
    "TP_FAIXA_ETARIA",
    "Q005",
    "Q024",
    "Q025",
    "IN_TREINEIRO",
    "SG_UF_PROVA",
    "TP_LINGUA",
    "RENDA_PER_CAPITA_APROX",
    "ESCOLARIDADE_MAX_RESP",
    "PERFIL_ACESSO_DIGITAL",
]
NUMERIC_FEATURES = ["TP_FAIXA_ETARIA", "Q005", "RENDA_PER_CAPITA_APROX"]
TEXT_CATEGORICAL_FEATURES = [
    "Q006",
    "Q001",
    "Q002",
    "Q024",
    "Q025",
    "SG_UF_PROVA",
    "ESCOLARIDADE_MAX_RESP",
    "PERFIL_ACESSO_DIGITAL",
]
CODE_CATEGORICAL_FEATURES = [
    "TP_ESCOLA",
    "TP_ST_CONCLUSAO",
    "IN_TREINEIRO",
    "TP_LINGUA",
]
N_FOLDS = 3
CV_RESULTS = {}
ERROR_TABLES = {}
BEST_PARAMS = {}

# knn/boosting: escopo original; linear (sem regularizacao) e ridge (regularizada)
# cobrem a exigencia do PDF de testar as duas versoes da regressao linear.
MODELOS = ["knn", "boosting", "linear", "ridge"]
MODELOS_COM_PESO = {"boosting", "linear", "ridge"}  # aceitam sample_weight
ARQUIVOS_RESULTADO_MODELO = {
    modelo: {
        "txt": SAIDA / f"resultados_{modelo}.txt",
        "csv": SAIDA / f"resultados_{modelo}.csv",
    }
    for modelo in MODELOS
}

# Faixas de nota fixadas no desenvolvimento e mantidas no teste final.
FAIXAS_LIMITES = [400, 500, 600, 700]
FAIXAS_ROTULOS = ["zero", "0-400", "400-500", "500-600", "600-700", "700+"]
TAMANHOS_CURVA = [0.10, 0.25, 0.50, 0.75, 1.00]


class Tee:
    def __init__(self, terminal: TextIO, arquivo: TextIO, lock: threading.Lock):
        self.terminal = terminal
        self.arquivo = arquivo
        self.lock = lock

    def write(self, texto: str) -> int:
        with self.lock:
            self.terminal.write(texto)
            self.arquivo.write(texto)
            self.arquivo.flush()
        return len(texto)

    def flush(self) -> None:
        with self.lock:
            self.terminal.flush()
            self.arquivo.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()

    @property
    def encoding(self) -> str | None:
        return self.terminal.encoding

    @property
    def errors(self) -> str | None:
        return self.terminal.errors


@contextmanager
def registrar_saida():
    RESUMO_EXECUCAO.parent.mkdir(parents=True, exist_ok=True)
    stdout_original, stderr_original = sys.stdout, sys.stderr
    with RESUMO_EXECUCAO.open("w", encoding="utf-8", buffering=1) as arquivo:
        lock = threading.Lock()
        sys.stdout = Tee(stdout_original, arquivo, lock)
        sys.stderr = Tee(stderr_original, arquivo, lock)
        try:
            print(f"RESULTADOS DA EXECUCAO - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Comando: {' '.join(sys.argv)}")
            print()
            yield
        finally:
            sys.stdout = stdout_original
            sys.stderr = stderr_original
            exportar_log_csv()


def exportar_log_csv():
    colunas = [
        "linha",
        "secao",
        "tipo_registro",
        "experimento",
        "dobra",
        "fracao",
        "n",
        "metrica",
        "valor",
        "rmse",
        "mae",
        "r2",
        "tempo_treino_s",
        "tempo_predicao_s",
        "media",
        "desvio_padrao",
        "agrupamento",
        "grupo",
        "componente",
        "variancia",
        "variancia_acumulada",
        "parametros",
        "texto_original",
        "rmse_treino",
        "dp_rmse_treino",
        "rmse_validacao",
        "dp_rmse_validacao",
    ]
    texto = RESUMO_EXECUCAO.read_text(encoding="utf-8")
    secao = ""
    with CSV_EXECUCAO.open("w", encoding="utf-8-sig", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=colunas)
        escritor.writeheader()
        for numero, linha in enumerate(texto.splitlines(), start=1):
            registro = {
                "linha": numero,
                "tipo_registro": "linha_vazia" if not linha.strip() else "texto",
                "texto_original": linha,
            }
            secao_match = re.match(r"\[(\d+/\d+)\]\s*(.*)", linha)
            if secao_match:
                secao = f"{secao_match.group(1)} {secao_match.group(2)}".strip()
                registro["tipo_registro"] = "etapa"
            registro["secao"] = secao
            escritor.writerow(registro)
    exportar_resultados_por_modelo(texto, colunas)


def exportar_resultados_por_modelo(texto: str, colunas: list[str]):
    padroes = {
        modelo: re.compile(rf"(?<![A-Za-z0-9]){re.escape(modelo)}(?![A-Za-z0-9])", re.IGNORECASE)
        for modelo in MODELOS
    }
    linhas_por_modelo = {modelo: [] for modelo in MODELOS}
    registros_por_modelo = {modelo: [] for modelo in MODELOS}
    secao = ""
    modelo_atual = None

    for numero, linha in enumerate(texto.splitlines(), start=1):
        hits = {
            modelo for modelo, padrao in padroes.items()
            if padrao.search(linha)
        }
        secao_match = re.match(r"\[(\d+/\d+)\]\s*(.*)", linha)
        if secao_match:
            secao = f"{secao_match.group(1)} {secao_match.group(2)}".strip()
            if not hits:
                modelo_atual = None

        if len(hits) == 1:
            modelo_atual = next(iter(hits))
            destinatarios = hits
        elif hits:
            modelo_atual = None
            destinatarios = hits
        else:
            destinatarios = {modelo_atual} if modelo_atual else set()

        for modelo in destinatarios:
            linhas_por_modelo[modelo].append(linha)
            registro = {
                "linha": numero,
                "secao": secao,
                "tipo_registro": "linha_vazia" if not linha.strip() else "texto",
                "texto_original": linha,
                "modelo": modelo,
            }
            if secao_match:
                registro["tipo_registro"] = "etapa"
            registros_por_modelo[modelo].append(registro)

    for modelo, caminhos in ARQUIVOS_RESULTADO_MODELO.items():
        caminhos["txt"].parent.mkdir(parents=True, exist_ok=True)
        cabecalho = texto.splitlines()[:2]
        linhas = [*cabecalho, "", *linhas_por_modelo[modelo]]
        caminhos["txt"].write_text("\n".join(linhas) + "\n", encoding="utf-8")
        with caminhos["csv"].open("w", encoding="utf-8-sig", newline="") as arquivo:
            escritor = csv.DictWriter(arquivo, fieldnames=["modelo", *colunas])
            escritor.writeheader()
            for registro in registros_por_modelo[modelo]:
                escritor.writerow(registro)

BLOCOS = {
    "renda": ["Q006", "Q005", "RENDA_PER_CAPITA_APROX"],
    "escolaridade_responsaveis": ["Q001", "Q002", "ESCOLARIDADE_MAX_RESP"],
    "acesso_digital": ["Q024", "Q025", "PERFIL_ACESSO_DIGITAL"],
    "localizacao": ["SG_UF_PROVA"],
    "perfil_escolar": ["TP_ESCOLA", "TP_ST_CONCLUSAO", "IN_TREINEIRO", "TP_FAIXA_ETARIA"],
    "lingua": ["TP_LINGUA"],
}

ESPACOS_BUSCA = {
    "boosting": {
        "model__learning_rate": loguniform(0.03, 0.3),
        "model__max_iter": [200, 400, 600],
        "model__max_leaf_nodes": [15, 31, 63, 127],
        "model__min_samples_leaf": [20, 50, 100, 200],
        "model__l2_regularization": [0.0, 0.1, 1.0, 10.0],
        "model__max_depth": [None, 6, 10],
        "model__n_iter_no_change": [10, 20],
    },
    "ridge": {
        "model__alpha": loguniform(1e-3, 1e3),
        "model__fit_intercept": [True],
        "model__solver": ["auto", "sparse_cg"],
    },
    "knn": {
        # Poucos componentes mantem a arvore kd eficiente (em ~40 dimensoes ela
        # degenera em forca bruta). variance_threshold=1.0 deixa n_components mandar.
        "pca__n_components": [6, 8, 10, 12, 16],
        "model__n_neighbors": [50, 100, 200, 300],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
        "model__leaf_size": [30, 60, 100],
    },
}

HIPOTESES = {
    "base": "Referencia inicial com hiperparametros padrao e as 15 features do grupo.",
    "pca-variancia": "Ha redundancia suficiente no one-hot para resumir 80% da variancia em poucos componentes.",
    "boosting_pca": "Trocar as features por componentes do PCA piora o boosting, que ja lida com categorias de forma nativa.",
    "knn_sem_pca": "Sem PCA o KNN ficaria mais fiel as categorias, mas o custo de busca de vizinhos cresce muito.",
    "linear_pca": "Para a regressao linear, trocar o one-hot por componentes do PCA descarta variancia e piora o erro em relacao a versao sem reducao.",
    "linear": "Se o sinal for quase aditivo, a regressao linear sem regularizacao fica proxima do boosting; a diferenca mede o ganho das interacoes.",
    "ridge": "A regularizacao L2 ajuda pouco: com ~100 colunas e 2 milhoes de linhas, o sobreajuste da regressao linear e pequeno.",
    "balanceamento": "Notas extremas (faixas raras) concentram erro; dar mais peso a elas reduz o erro ali, a custa do centro.",
    "ablacao": "Retirar um bloco de informacao aumenta o RMSE em proporcao a sua contribuicao para a previsao.",
    "busca": "Ajustar hiperparametros reduz o RMSE em relacao a configuracao inicial.",
}

# Dados, particoes e utilidades
class SparseIncrementalPCA(BaseEstimator, TransformerMixin):
    """Aplica PCA em lotes, densificando somente o lote em processamento."""

    def __init__(self, n_components=15, variance_threshold=0.8, batch_size=20000):
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.batch_size = batch_size

    @staticmethod
    def _dense_batch(X, start, end):
        batch = X[start:end]
        if hasattr(batch, "toarray"):
            batch = batch.toarray()
        return np.asarray(batch, dtype=np.float64)

    def _batch_ranges(self, n_samples):
        n_batches = math.ceil(n_samples / self.batch_size)
        batch_size = math.ceil(n_samples / n_batches)
        return (
            (start, min(start + batch_size, n_samples))
            for start in range(0, n_samples, batch_size)
        )

    def fit(self, X, y=None):
        if self.n_components < 1:
            raise ValueError("n_components deve ser maior que zero.")
        if not 0 <= self.variance_threshold <= 1:
            raise ValueError("variance_threshold deve estar entre 0 e 1.")
        if self.batch_size < 1:
            raise ValueError("batch_size deve ser maior que zero.")

        n_samples, n_features = X.shape
        component_count = min(self.n_components, n_features, n_samples)
        if component_count < 1:
            raise ValueError("A matriz de entrada nao possui dimensoes para PCA.")

        ranges = list(self._batch_ranges(n_samples))
        if any(end - start < component_count for start, end in ranges):
            raise ValueError("Cada lote do PCA precisa ter ao menos n_components linhas.")

        self.pca_ = IncrementalPCA(
            n_components=component_count,
            batch_size=self.batch_size,
        )
        for start, end in ranges:
            self.pca_.partial_fit(self._dense_batch(X, start, end))

        cumulative_variance = np.cumsum(self.pca_.explained_variance_ratio_)
        selected = np.flatnonzero(cumulative_variance >= self.variance_threshold)
        self.n_components_ = (
            int(selected[0] + 1) if selected.size else component_count
        )
        self.explained_variance_ratio_ = float(
            cumulative_variance[self.n_components_ - 1]
        )
        self.n_features_in_ = n_features
        return self

    def transform(self, X):
        check_is_fitted(self, "pca_")
        n_samples = X.shape[0]
        if n_samples == 0:
            return np.empty((0, self.n_components_), dtype=np.float32)
        transformed = np.empty((n_samples, self.n_components_), dtype=np.float32)
        for start, end in self._batch_ranges(n_samples):
            batch = self._dense_batch(X, start, end)
            transformed[start:end] = self.pca_.transform(batch)[:, : self.n_components_]
        return transformed


def build_knn():
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    preparation = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            (
                "categorical",
                categorical,
                TEXT_CATEGORICAL_FEATURES + CODE_CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=1.0,
    )
    return Pipeline(
        [
            ("preparation", preparation),
            ("pca", SparseIncrementalPCA(n_components=12, variance_threshold=1.0)),
            (
                "model",
                KNeighborsRegressor(
                    n_neighbors=50,
                    weights="distance",
                    algorithm="kd_tree",
                    leaf_size=60,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_boosting():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    text_categories = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                    encoded_missing_value=np.nan,
                    dtype=np.float64,
                ),
            ),
        ]
    )
    code_categories = Pipeline(
        [
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                    encoded_missing_value=np.nan,
                    dtype=np.float64,
                ),
            ),
        ]
    )
    preparation = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("text_categories", text_categories, TEXT_CATEGORICAL_FEATURES),
            ("code_categories", code_categories, CODE_CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0,
    )
    categorical_mask = np.array(
        [False] * len(NUMERIC_FEATURES)
        + [True] * (len(TEXT_CATEGORICAL_FEATURES) + len(CODE_CATEGORICAL_FEATURES))
    )
    return Pipeline(
        [
            ("preparation", preparation),
            (
                "model",
                HistGradientBoostingRegressor(
                    categorical_features=categorical_mask,
                    learning_rate=0.1,
                    max_iter=200,
                    max_leaf_nodes=31,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    random_state=SEED,
                ),
            ),
        ]
    )


def load_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Base ENEM nao encontrada: {DATA_PATH}. "
            "Gere o conjunto C07 com `python -m src.conjuntos.exportar_conjunto C07`."
        )

    data = pd.read_parquet(DATA_PATH, columns=[ID_COL, *FEATURES, TARGET])
    expected_columns = [ID_COL, *FEATURES, TARGET]
    missing_columns = set(expected_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Colunas ausentes na base: {sorted(missing_columns)}")
    if data[ID_COL].isna().any() or data[ID_COL].duplicated().any():
        raise ValueError(f"{ID_COL} deve estar preenchido e ser unico.")
    if data[TARGET].isna().any():
        raise ValueError(
            f"A base elegivel contem notas ausentes em {TARGET}; "
            "nao e permitido imputar o alvo."
        )
    if data[TEXT_CATEGORICAL_FEATURES + CODE_CATEGORICAL_FEATURES].isna().any().any():
        raise ValueError("A base contem valores ausentes em features categoricas.")

    for column in TEXT_CATEGORICAL_FEATURES:
        data[column] = data[column].astype("category")
    return data


def partition_frame(ids, dev_indices, test_indices, folds):
    fold_by_dev_position = np.full(len(dev_indices), -1, dtype=np.int8)
    for fold_number, (_, validation_indices) in enumerate(folds, start=1):
        fold_by_dev_position[validation_indices] = fold_number

    fold_by_original_index = np.full(len(ids), -1, dtype=np.int8)
    fold_by_original_index[dev_indices] = fold_by_dev_position
    partition = np.full(len(ids), "", dtype=object)
    partition[dev_indices] = "desenvolvimento"
    partition[test_indices] = "teste"
    fold_values = pd.array(fold_by_original_index, dtype="Int8")
    fold_values[fold_by_original_index < 0] = pd.NA

    split_data = pd.DataFrame(
        {
            ID_COL: ids,
            "conjunto": partition,
            "dobra_validacao": fold_values,
        }
    )
    return split_data


def preparar(com_teste=False):
    """Prepara os dados e valida as particoes reproduziveis."""
    data = load_data()
    ids = data[ID_COL].to_numpy()
    X_all = data[FEATURES]
    y_all = data[TARGET].to_numpy(dtype=np.float64)
    dev_idx, test_idx = train_test_split(
        np.arange(len(data)), test_size=0.2, random_state=SEED, shuffle=True
    )
    folds = list(
        KFold(N_FOLDS, shuffle=True, random_state=SEED).split(X_all.iloc[dev_idx])
    )

    if SPLITS_PATH.is_file():
        salvo = pd.read_parquet(SPLITS_PATH)
        if {"CONJUNTO", "DOBRA"}.issubset(salvo.columns):  # formato oficial (src/conjuntos/base_ampliada.py)
            dobra = salvo["DOBRA"].astype("Int8")
            salvo = pd.DataFrame({ID_COL: salvo[ID_COL], "conjunto": salvo["CONJUNTO"].to_numpy(dtype=object),
                                  "dobra_validacao": dobra.where(dobra > 0)})
        required = {ID_COL, "conjunto", "dobra_validacao"}
        if not required.issubset(salvo.columns) or salvo[ID_COL].duplicated().any():
            raise RuntimeError(f"Arquivo de particoes invalido: {SPLITS_PATH}.")
        esperado = partition_frame(ids, dev_idx, test_idx, folds)
        salvo = salvo.set_index(ID_COL).sort_index()
        esperado = esperado.set_index(ID_COL).sort_index()
        if not salvo.index.equals(esperado.index) or not salvo[
            ["conjunto", "dobra_validacao"]
        ].equals(esperado[["conjunto", "dobra_validacao"]]):
            raise RuntimeError(
                f"As particoes salvas diferem do split recalculado: {SPLITS_PATH}. "
                "Preserve o arquivo correto ou regenere-o explicitamente."
            )
    X_dev = X_all.iloc[dev_idx].reset_index(drop=True)
    y_dev = y_all[dev_idx]
    d = {
        "X_dev": X_dev,
        "y_dev": y_dev,
        "ids_dev": ids[dev_idx],
        "folds": folds,
        "n_total": len(data),
        "n_test": len(test_idx),
    }
    if com_teste:
        d["X_test"] = X_all.iloc[test_idx].reset_index(drop=True)
        d["y_test"] = y_all[test_idx]
        d["ids_test"] = ids[test_idx]
    return d


def metricas(y, pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }


def faixa_nota(y):
    y = np.asarray(y)
    codigo = np.digitize(y, FAIXAS_LIMITES) + 1
    codigo[y == 0] = 0
    return np.array(FAIXAS_ROTULOS, dtype=object)[codigo]


def construir(modelo, params_path=None):
    construtores = {"knn": build_knn, "boosting": build_boosting,
                    "linear": build_linear, "ridge": build_ridge}
    if modelo not in construtores:
        raise ValueError(f"Modelo desconhecido: {modelo}")
    estimator = construtores[modelo]()
    if params_path:
        if isinstance(params_path, dict):
            params = params_path
        else:
            params = json.loads(Path(params_path).read_text(encoding="utf-8"))["params"]
        estimator.set_params(**params)
    return estimator


def build_linear():
    """Regressao linear sem regularizacao (OLS) sobre o one-hot esparso."""
    preparo = build_knn().named_steps["preparation"]
    return Pipeline([("preparation", preparo), ("model", LinearRegression())])


def build_ridge():
    """Regressao linear com regularizacao L2 (Ridge); alpha ajustado na busca."""
    preparo = build_knn().named_steps["preparation"]
    return Pipeline([("preparation", preparo), ("model", Ridge(alpha=1.0, random_state=SEED))])


def build_linear_pca():
    """Mesmo preparo do KNN + PCA (>=80% da variancia) alimentando a regressao linear."""
    preparo = build_knn().named_steps["preparation"]
    return Pipeline(
        [
            ("preparation", preparo),
            ("pca", SparseIncrementalPCA(n_components=40, variance_threshold=0.8)),
            ("model", LinearRegression()),
        ]
    )


def build_boosting_pca():
    """Mesmo preparo do KNN (one-hot + padronizacao + PCA) alimentando o boosting."""
    preparo = build_knn().named_steps["preparation"]
    return Pipeline(
        [
            ("preparation", preparo),
            ("pca", SparseIncrementalPCA(n_components=40, variance_threshold=0.8)),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.1,
                    max_iter=200,
                    max_leaf_nodes=31,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    random_state=SEED,
                ),
            ),
        ]
    )


def build_knn_sem_pca():
    """KNN direto sobre o one-hot esparso (busca por forca bruta: muito custoso)."""
    preparo = build_knn().named_steps["preparation"]
    return Pipeline(
        [
            ("preparation", preparo),
            (
                "model",
                KNeighborsRegressor(
                    n_neighbors=50, weights="distance", algorithm="brute", n_jobs=-1
                ),
            ),
        ]
    )


@contextmanager
def sem_features(removidas):
    """Constrói um pipeline sem as features removidas."""
    global NUMERIC_FEATURES, TEXT_CATEGORICAL_FEATURES, CODE_CATEGORICAL_FEATURES

    originais = (
        NUMERIC_FEATURES,
        TEXT_CATEGORICAL_FEATURES,
        CODE_CATEGORICAL_FEATURES,
    )
    NUMERIC_FEATURES = [c for c in originais[0] if c not in removidas]
    TEXT_CATEGORICAL_FEATURES = [c for c in originais[1] if c not in removidas]
    CODE_CATEGORICAL_FEATURES = [c for c in originais[2] if c not in removidas]
    try:
        yield
    finally:
        NUMERIC_FEATURES, TEXT_CATEGORICAL_FEATURES, CODE_CATEGORICAL_FEATURES = originais

# Balanceamento (somente no treino da dobra)
def _bins_treino(ytr, n_bins=10):
    arestas = np.histogram_bin_edges(ytr, bins=n_bins)
    return np.clip(np.digitize(ytr, arestas[1:-1]), 0, n_bins - 1)


def aplicar_estrategia(estrategia, aceita_peso, Xtr, ytr, n_bins=10, peso_max=10.0):
    """Devolve (Xtr, ytr, fit_params). Faixas definidas so com o alvo do treino."""
    if estrategia is None:
        return Xtr, ytr, {}
    b = _bins_treino(ytr, n_bins)
    contagem = np.bincount(b, minlength=n_bins)

    if estrategia == "peso":
        if not aceita_peso:
            raise SystemExit("O KNN nao aceita sample_weight: use under ou over.")
        media = contagem[contagem > 0].mean()
        w_faixa = np.clip(media / np.maximum(contagem, 1), 0.0, peso_max)
        w = w_faixa[b]
        w = w * (len(w) / w.sum())
        return Xtr, ytr, {"model__sample_weight": w}

    rng = np.random.default_rng(SEED)
    teto = int(np.median(contagem[contagem > 0]))
    escolhidos = []
    for k in range(n_bins):
        idx = np.flatnonzero(b == k)
        if idx.size == 0:
            continue
        if estrategia == "under" and idx.size > teto:
            idx = rng.choice(idx, teto, replace=False)
        elif estrategia == "over" and idx.size < teto:
            extra = min(teto, 5 * idx.size) - idx.size
            idx = np.concatenate([idx, rng.choice(idx, extra, replace=True)])
        escolhidos.append(idx)
    escolhidos = np.sort(np.concatenate(escolhidos))
    return Xtr.iloc[escolhidos], ytr[escolhidos], {}


def prever_em_lotes(modelo, X, indices, tamanho_lote=10_000):
    previsoes = [
        modelo.predict(X.iloc[indices[inicio:inicio + tamanho_lote]])
        for inicio in range(0, len(indices), tamanho_lote)
    ]
    return np.concatenate(previsoes)


# Validacao cruzada com previsoes fora da dobra
def rodar_cv(nome, estimator, d, estrategia=None, aceita_peso=True, max_dobras=None,
             hipotese="", conclusao="(preencher)", modelo_rotulo=None, features=None):
    X, y, folds = d["X_dev"], d["y_dev"], d["folds"]
    if max_dobras is not None:
        if not 1 <= max_dobras <= len(folds):
            raise SystemExit(f"--dobras deve estar entre 1 e {len(folds)}.")
        folds = folds[:max_dobras]
    oof = np.full(len(y), np.nan)
    fold_id = np.zeros(len(y), dtype=np.int8)
    linhas = []
    for k, (tr, va) in enumerate(folds, start=1):
        modelo = clone(estimator)
        Xtr, ytr, fit_params = aplicar_estrategia(estrategia, aceita_peso, X.iloc[tr], y[tr])
        t0 = time.perf_counter()
        modelo.fit(Xtr, ytr, **fit_params)
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = modelo.predict(X.iloc[va])
        t_pred = time.perf_counter() - t0
        oof[va] = pred
        fold_id[va] = k
        linha = {
            "modelo": nome, "dobra": k, "n_treino": len(ytr), "n_validacao": len(va),
            **metricas(y[va], pred), "tempo_treino_s": t_fit, "tempo_predicao_s": t_pred,
        }
        if hasattr(modelo, "named_steps") and "pca" in modelo.named_steps:
            linha["pca_componentes"] = modelo.named_steps["pca"].n_components_
        linhas.append(linha)
        print(f"{nome:<28} dobra {k}: RMSE {linha['rmse']:.2f} | MAE {linha['mae']:.2f} | "
              f"R2 {linha['r2']:.4f} | treino {t_fit:.1f}s | predicao {t_pred:.1f}s", flush=True)

    resultado = pd.DataFrame(linhas)
    CV_RESULTS[nome] = resultado
    validos = fold_id > 0
    ERROR_TABLES[nome] = tabelas_completas(
        d["X_dev"].iloc[np.flatnonzero(validos)], y[validos], oof[validos]
    )
    cols = ["rmse", "mae", "r2", "tempo_treino_s", "tempo_predicao_s"]
    resumo = resultado[cols].agg(["mean", lambda v: v.std(ddof=0)])
    resumo.index = ["media", "desvio_padrao"]
    print(f"Modelo: {modelo_rotulo or nome}")
    if features is not None:
        print(f"Features: {', '.join(features)}")
    print(f"\nResumo de {nome}:\n{resumo.to_string()}")
    print(f"Hipotese: {hipotese or '(nao informada)'}")
    print(f"Conclusao: {conclusao}")
    return resultado, oof

# Erros por faixa e por grupo
def tabela_erros(y, pred, grupos, nome):
    df = pd.DataFrame({"grupo": pd.Series(grupos).astype(str).to_numpy(), "res": y - pred})
    def rmse(r): return float(np.sqrt(np.mean(np.square(r))))
    def mae(r): return float(np.mean(np.abs(r)))
    t = df.groupby("grupo", sort=False)["res"].agg(
        n="size", rmse=rmse, mae=mae, vies="mean").reset_index()
    t.insert(0, "agrupamento", nome)
    if nome == "faixa_nota":
        t["ordem"] = t["grupo"].map({r: i for i, r in enumerate(FAIXAS_ROTULOS)})
        t = t.sort_values("ordem").drop(columns="ordem")
    return t.reset_index(drop=True)


def grupos_de_perfil(X, y):
    escola = X["TP_ESCOLA"].astype(str).str.replace(r"\.0$", "", regex=True)
    escola = escola.map({"1": "nao respondeu", "2": "publica", "3": "privada"}).fillna(escola)
    return {
        "faixa_nota": faixa_nota(y),
        "SG_UF_PROVA": X["SG_UF_PROVA"].astype(str).to_numpy(),
        "TP_ESCOLA": escola.to_numpy(),
        "Q006": X["Q006"].astype(str).to_numpy(),
    }


def tabelas_completas(X, y, pred):
    return pd.concat(
        [tabela_erros(y, pred, g, nome) for nome, g in grupos_de_perfil(X, y).items()],
        ignore_index=True,
    )


def grafico_faixas(tabelas, caminho, titulo):
    """tabelas: {rotulo: DataFrame de faixa_nota}. Barras de RMSE por faixa."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    n = len(tabelas)
    largura = 0.8 / n
    base_t = next(iter(tabelas.values()))
    x = np.arange(len(base_t))
    for i, (rotulo, t) in enumerate(tabelas.items()):
        ax.bar(x + i * largura - 0.4 + largura / 2, t["rmse"], largura, label=rotulo)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n(n={int(c):,})".replace(",", ".")
                        for g, c in zip(base_t["grupo"], base_t["n"])])
    ax.set_ylabel("RMSE (pontos)")
    ax.set_title(titulo)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)

# Subcomandos
def cmd_distribuicao(args):
    d = preparar()
    y = d["y_dev"]
    FIG.mkdir(parents=True, exist_ok=True)
    zeros = int((y == 0).sum())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(y[y > 0], bins=60, color="#3b78c3", label=f"nota > 0 (n={int((y > 0).sum()):,})".replace(",", "."))
    if zeros:
        ax.bar([0], [min(zeros, ax.get_ylim()[1])], width=8, color="#d9534f",
               label=f"nota zero (n={zeros:,})".replace(",", "."))
    ax.axvline(y.mean(), color="k", ls="--", lw=1, label=f"media = {y.mean():.1f}")
    ax.set_xlabel("NU_NOTA_LC")
    ax.set_ylabel("Participantes")
    ax.set_title("Distribuicao da nota de Linguagens - desenvolvimento")
    ax.legend()
    fig.text(0.01, 0.01, "Fonte: Inep, Microdados do ENEM 2023.", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "distribuicao_nota_alvo_desenvolvimento.png", dpi=150)
    plt.close(fig)
    resumo = pd.Series(y).describe()
    resumo["zeros"] = zeros
    resumo["desvio_padrao_populacional"] = y.std()
    print("Resumo da distribuicao:\n" + resumo.to_string())


def cmd_base(args):
    d = preparar()
    rodar_cv("baseline_media", DummyRegressor(strategy="mean"), d,
             hipotese="Previsao constante pela media do treino da dobra.",
             modelo_rotulo="DummyRegressor(mean)", conclusao="Referencia: erro sem usar features.")
    for modelo in args.modelos:
        estimator = construir(modelo)
        rodar_cv(f"{modelo}_base", estimator, d, hipotese=HIPOTESES["base"],
                 modelo_rotulo=modelo)


def cmd_pca_variancia(args):
    if not 0 < args.variancia <= 1:
        raise SystemExit("--variancia deve ser maior que 0 e menor ou igual a 1.")
    d = preparar()
    tr, _ = d["folds"][0]
    preparo = build_knn().named_steps["preparation"]
    Xt = preparo.fit_transform(d["X_dev"].iloc[tr])
    pca = SparseIncrementalPCA(n_components=40, variance_threshold=args.variancia).fit(Xt)
    razao = pca.pca_.explained_variance_ratio_
    acum = np.cumsum(razao)
    tabela = pd.DataFrame({"componente": np.arange(1, len(razao) + 1),
                           "variancia": razao, "variancia_acumulada": acum})
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tabela["componente"], acum, marker="o", ms=3)
    ax.axhline(args.variancia, color="gray", ls="--", lw=1, label=f"alvo {args.variancia:.0%}")
    ax.axvline(pca.n_components_, color="#d9534f", ls="--", lw=1,
               label=f"{pca.n_components_} componentes ({pca.explained_variance_ratio_:.1%})")
    ax.set_xlabel("Numero de componentes")
    ax.set_ylabel("Variancia explicada acumulada")
    ax.set_title("PCA - treino da dobra 1 (desenvolvimento)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.text(0.01, 0.01, f"Colunas apos one-hot: {Xt.shape[1]}. Fonte: Inep, ENEM 2023.", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "pca_variancia_acumulada.png", dpi=150)
    plt.close(fig)
    print("Variancia por componente:\n" + tabela.to_string(index=False))
    print(f"Colunas apos preparo: {Xt.shape[1]} | componentes escolhidos: {pca.n_components_} | "
          f"variancia: {pca.explained_variance_ratio_:.4f}")


def cmd_pca(args):
    if args.dobras is not None and not 1 <= args.dobras <= N_FOLDS:
        raise SystemExit(f"--dobras deve estar entre 1 e {N_FOLDS}.")
    d = preparar()
    if args.variante == "boosting_pca":
        rodar_cv("boosting_pca", build_boosting_pca(), d, hipotese=HIPOTESES["boosting_pca"],
                 modelo_rotulo="HistGradientBoosting + PCA(80%)", max_dobras=args.dobras)
    elif args.variante == "linear_pca":
        rodar_cv("linear_pca", build_linear_pca(), d, hipotese=HIPOTESES["linear_pca"],
                 modelo_rotulo="Regressao linear + PCA(80%)", max_dobras=args.dobras)
    else:
        print("ATENCAO: KNN sem PCA usa busca por forca bruta em ~1,5 milhao de linhas por dobra; "
              "pode levar horas. Use --dobras 1 e registre o limite se nao for viavel.")
        rodar_cv("knn_sem_pca", build_knn_sem_pca(), d, hipotese=HIPOTESES["knn_sem_pca"],
                 modelo_rotulo="KNN sem PCA (one-hot esparso, brute)", max_dobras=args.dobras)


def validar_estrategia(modelo, estrategia):
    if modelo == "knn" and estrategia == "peso":
        raise SystemExit("O KNN nao aceita sample_weight: use --estrategia under ou over.")


def cmd_balanceamento(args):
    validar_estrategia(args.modelo, args.estrategia)
    d = preparar()
    nome = f"{args.modelo}_{args.estrategia}"
    rodar_cv(nome, construir(args.modelo), d, estrategia=args.estrategia,
             aceita_peso=(args.modelo in MODELOS_COM_PESO), hipotese=HIPOTESES["balanceamento"],
             modelo_rotulo=args.modelo)


def cmd_ablacao(args):
    d = preparar()
    linhas = []
    completo = OUT / f"{args.modelo}_base_cv_por_dobra.csv"
    ref = CV_RESULTS.get(f"{args.modelo}_base")
    if ref is None and completo.is_file():
        ref = pd.read_csv(completo)
    if ref is None:
        ref, _ = rodar_cv(
            f"{args.modelo}_base", construir(args.modelo), d,
            hipotese=HIPOTESES["base"], modelo_rotulo=args.modelo,
        )
    linhas.append({"bloco_removido": "(nenhum)", "colunas": "todas",
                   "rmse_media": ref["rmse"].mean(), "rmse_dp": ref["rmse"].std(ddof=0),
                   "r2_media": ref["r2"].mean()})
    for bloco, colunas in BLOCOS.items():
        with sem_features(set(colunas)):
            est = construir(args.modelo)
            restantes = [c for c in FEATURES if c not in colunas]
        r, _ = rodar_cv(f"{args.modelo}_sem_{bloco}", est, d, hipotese=HIPOTESES["ablacao"],
                        modelo_rotulo=args.modelo, features=restantes)
        linhas.append({"bloco_removido": bloco, "colunas": ", ".join(colunas),
                       "rmse_media": r["rmse"].mean(), "rmse_dp": r["rmse"].std(ddof=0),
                       "r2_media": r["r2"].mean()})
    t = pd.DataFrame(linhas)
    t["delta_rmse"] = t["rmse_media"] - t["rmse_media"].iloc[0]
    FIG.mkdir(parents=True, exist_ok=True)
    s = t.iloc[1:].sort_values("delta_rmse")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh(s["bloco_removido"], s["delta_rmse"], color="#3b78c3")
    ax.set_xlabel("Aumento do RMSE ao remover o bloco (pontos)")
    ax.set_title(f"Ablacao por bloco - {args.modelo} (desenvolvimento)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / f"ablacao_{args.modelo}.png", dpi=150)
    plt.close(fig)
    print("Resultados da ablacao:\n" + t.to_string(index=False))


def _para_json(valor):
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (np.floating,)):
        return float(valor)
    return valor


def cmd_busca(args):
    if args.n_iter < 10:
        raise SystemExit("--n-iter deve ser pelo menos 10 para cumprir o minimo de configuracoes.")
    if args.modelo not in ESPACOS_BUSCA:
        raise SystemExit(f"Nao ha espaco de busca para '{args.modelo}' "
                         "(a regressao linear sem regularizacao nao tem hiperparametros).")
    d = preparar()
    estimator = construir(args.modelo)
    busca = RandomizedSearchCV(
        estimator, ESPACOS_BUSCA[args.modelo], n_iter=args.n_iter,
        scoring="neg_root_mean_squared_error", cv=d["folds"], random_state=SEED,
        refit=False, n_jobs=1, verbose=2, error_score="raise",
    )
    t0 = time.perf_counter()
    busca.fit(d["X_dev"], d["y_dev"])
    total = time.perf_counter() - t0
    res = pd.DataFrame(busca.cv_results_)
    res["rmse_media"] = -res["mean_test_score"]
    res["rmse_dp"] = res["std_test_score"]
    res = res.sort_values("rmse_media")
    melhor = {k: _para_json(v) for k, v in busca.best_params_.items()}
    BEST_PARAMS[args.modelo] = melhor
    print("Melhores configuracoes:\n" +
          res[["params", "rmse_media", "rmse_dp"]].head(10).to_string(index=False))
    print(f"Melhor RMSE: {-busca.best_score_:.3f} | {melhor} | {len(res)} configuracoes | {total/60:.1f} min")

    final = clone(estimator).set_params(**busca.best_params_)
    rodar_cv(f"{args.modelo}_busca_melhor", final, d, hipotese=HIPOTESES["busca"],
             modelo_rotulo=args.modelo)


def cmd_curva(args):
    validar_estrategia(args.modelo, args.estrategia)
    if args.max_aval_treino is not None and args.max_aval_treino < 1:
        raise SystemExit("--max-aval-treino deve ser maior que zero.")
    d = preparar()
    X, y, folds = d["X_dev"], d["y_dev"], d["folds"]
    est = construir(args.modelo, args.params)
    aceita_peso = args.modelo in MODELOS_COM_PESO
    linhas = []
    for k, (tr, va) in enumerate(folds, start=1):
        rng = np.random.default_rng(SEED)
        ordem = rng.permutation(tr)  # subconjuntos aleatorios e progressivos (aninhados)
        for frac in TAMANHOS_CURVA:
            sub = ordem[: int(round(frac * len(tr)))]
            Xs, ys, fit_params = aplicar_estrategia(args.estrategia, aceita_peso, X.iloc[sub], y[sub])
            modelo = clone(est)
            t0 = time.perf_counter()
            modelo.fit(Xs, ys, **fit_params)
            t_fit = time.perf_counter() - t0
            if args.max_aval_treino and len(sub) > args.max_aval_treino:
                amostra = rng.choice(sub, args.max_aval_treino, replace=False)
            else:
                amostra = sub
            pred_tr = prever_em_lotes(modelo, X, amostra)
            pred_va = prever_em_lotes(modelo, X, va)
            rmse_tr = metricas(y[amostra], pred_tr)["rmse"]
            rmse_va = metricas(y[va], pred_va)["rmse"]
            linhas.append({"dobra": k, "fracao": frac, "n_treino": len(sub),
                           "rmse_treino": rmse_tr, "rmse_validacao": rmse_va, "tempo_treino_s": t_fit})
            print(f"dobra {k} | {frac:>4.0%} (n={len(sub):,}) | treino {rmse_tr:.2f} | "
                  f"validacao {rmse_va:.2f}", flush=True)
    t = pd.DataFrame(linhas)
    sufixo = f"{args.modelo}" + (f"_{args.estrategia}" if args.estrategia else "")
    g = t.groupby("fracao").agg(n=("n_treino", "mean"),
                                tr_m=("rmse_treino", "mean"), tr_s=("rmse_treino", lambda v: v.std(ddof=0)),
                                va_m=("rmse_validacao", "mean"), va_s=("rmse_validacao", lambda v: v.std(ddof=0)))
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for pref, rot, cor in [("tr", "Treino", "#3b78c3"), ("va", "Validacao", "#e0813b")]:
        ax.plot(g["n"], g[f"{pref}_m"], marker="o", label=rot, color=cor)
        ax.fill_between(g["n"], g[f"{pref}_m"] - g[f"{pref}_s"], g[f"{pref}_m"] + g[f"{pref}_s"],
                        alpha=0.2, color=cor)
    ax.set_xlabel("Exemplos no treino da dobra")
    ax.set_ylabel("RMSE (pontos)")
    ax.set_title(f"Curva de aprendizado - {sufixo} (media e DP em 3 dobras)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.text(0.01, 0.01, "Desenvolvimento; teste nao utilizado. Fonte: Inep, ENEM 2023.", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / f"curva_aprendizado_{sufixo}.png", dpi=150)
    plt.close(fig)
    print("Resumo da curva de aprendizado:\n" + g.to_string())


def cmd_relatorio(args):
    dados = None
    faixas = {}
    todas = []
    for nome in args.experimentos:
        t = ERROR_TABLES.get(nome)
        if t is not None:
            t = t.copy()
        if t is None:
            caminho = OUT / f"oof_{nome}.parquet"
            if not caminho.is_file():
                raise SystemExit(
                    f"Nao ha resultados em memoria ou arquivo legado para '{nome}'. "
                    "Execute `completo` para gerar o relatorio sem gravar arquivos intermediarios."
                )
            if dados is None:
                dados = preparar()
            oof = pd.read_parquet(caminho)
            oof = oof.sort_values("pos_dev")
            pos = oof["pos_dev"].to_numpy()
            t = tabelas_completas(
                dados["X_dev"].iloc[pos], dados["y_dev"][pos], oof["pred"].to_numpy()
            )
        t.insert(0, "experimento", nome)
        todas.append(t)
        faixas[nome] = t[t["agrupamento"] == "faixa_nota"].reset_index(drop=True)
    resultado = pd.concat(todas, ignore_index=True)
    FIG.mkdir(parents=True, exist_ok=True)
    grafico_faixas(faixas, FIG / "erro_por_faixa_desenvolvimento.png",
                   "RMSE por faixa de nota - desenvolvimento (fora da dobra)")
    print("Erros por faixa e grupo:\n" + resultado.to_string(index=False))


def cmd_teste(args):
    if not args.confirmo_solucao_final:
        raise SystemExit(
            "O teste reservado so pode ser usado depois de fixar a solucao final pelo "
            "desenvolvimento. Se for o caso, repita o comando com --confirmo-solucao-final."
        )
    validar_estrategia(args.modelo, args.estrategia)
    d = preparar(com_teste=True)
    X_dev, y_dev, X_te, y_te = d["X_dev"], d["y_dev"], d["X_test"], d["y_test"]
    est = construir(args.modelo, args.params)
    aceita_peso = args.modelo in MODELOS_COM_PESO
    nome = args.modelo + (f"_{args.estrategia}" if args.estrategia else "") + "_final"

    dummy = DummyRegressor(strategy="mean").fit(X_dev, y_dev)
    pred_base = dummy.predict(X_te)

    Xs, ys, fit_params = aplicar_estrategia(args.estrategia, aceita_peso, X_dev, y_dev)
    modelo = clone(est)
    t0 = time.perf_counter()
    modelo.fit(Xs, ys, **fit_params)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = modelo.predict(X_te)
    t_pred = time.perf_counter() - t0

    # O PDF pede baseline, regressao linear sem regularizacao e solucao final
    # avaliados nos mesmos participantes; o OLS e ajustado nos 80% de desenvolvimento.
    incluir_linear = nome != "linear_final"
    if incluir_linear:
        linear = build_linear()
        t0 = time.perf_counter()
        linear.fit(X_dev, y_dev)
        t_fit_lin = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred_lin = linear.predict(X_te)
        t_pred_lin = time.perf_counter() - t0

    linhas = [{"modelo": "baseline_media", **metricas(y_te, pred_base), "n_teste": len(y_te)}]
    if incluir_linear:
        linhas.append({"modelo": "linear_sem_regularizacao", **metricas(y_te, pred_lin),
                       "tempo_treino_s": t_fit_lin, "tempo_predicao_s": t_pred_lin,
                       "n_teste": len(y_te)})
    linhas.append({"modelo": nome, **metricas(y_te, pred), "tempo_treino_s": t_fit,
                   "tempo_predicao_s": t_pred, "n_teste": len(y_te)})
    print("Metricas no teste:\n" + pd.DataFrame(linhas).to_string(index=False))

    tabelas = tabelas_completas(X_te, y_te, pred)
    print("Erros por faixa e grupo no teste:\n" + tabelas.to_string(index=False))

    FIG.mkdir(parents=True, exist_ok=True)
    res = y_te - pred
    lim = [0, max(y_te.max(), pred.max())]
    fig, ax = plt.subplots(figsize=(6, 5.5))
    h = ax.hexbin(pred, y_te, gridsize=70, bins="log", cmap="viridis", mincnt=1)
    ax.plot(lim, lim, "r--", lw=1, label="previsao perfeita")
    ax.set_xlabel("Nota prevista")
    ax.set_ylabel("Nota observada")
    ax.set_title(f"Observado x previsto - teste ({nome})")
    fig.colorbar(h, label="participantes (log)")
    ax.legend()
    fig.text(0.01, 0.01, "Fonte: Inep, ENEM 2023. Celulas hexagonais agregam todos os registros do teste.", fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / f"teste_{nome}_observado_vs_previsto.png", dpi=150)
    plt.close(fig)

    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.5))
    h = eixos[0].hexbin(pred, res, gridsize=70, bins="log", cmap="viridis", mincnt=1)
    eixos[0].axhline(0, color="r", ls="--", lw=1)
    eixos[0].set_xlabel("Nota prevista")
    eixos[0].set_ylabel("Residuo (observada - prevista)")
    eixos[0].set_title("Residuos x previsao")
    eixos[1].hist(res, bins=80, color="#3b78c3")
    eixos[1].axvline(0, color="r", ls="--", lw=1)
    eixos[1].set_xlabel("Residuo")
    eixos[1].set_ylabel("Participantes")
    eixos[1].set_title("Distribuicao dos residuos")
    fig.tight_layout()
    fig.savefig(FIG / f"teste_{nome}_residuos.png", dpi=150)
    plt.close(fig)

    grafico_faixas({nome: tabelas[tabelas["agrupamento"] == "faixa_nota"].reset_index(drop=True)},
                   FIG / f"teste_{nome}_erro_por_faixa.png", "RMSE por faixa de nota - teste")
    for agrup in ["TP_ESCOLA", "Q006"]:
        t = tabelas[tabelas["agrupamento"] == agrup]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(t["grupo"], t["rmse"], color="#3b78c3")
        for i, (g, n) in enumerate(zip(t["grupo"], t["n"])):
            ax.text(i, 0, f"n={int(n):,}".replace(",", "."), ha="center", va="bottom", fontsize=7, color="w", rotation=90)
        ax.set_ylabel("RMSE (pontos)")
        ax.set_title(f"RMSE por {agrup} - teste")
        fig.tight_layout()
        fig.savefig(FIG / f"teste_{nome}_erro_por_{agrup}.png", dpi=150)
        plt.close(fig)


def cmd_completo(_args=None):
    CV_RESULTS.clear()
    ERROR_TABLES.clear()
    BEST_PARAMS.clear()
    etapas = [
        ("Distribuicao das notas", cmd_distribuicao, argparse.Namespace()),
        ("Modelos de referencia", cmd_base, argparse.Namespace(modelos=MODELOS)),
        ("Variancia do PCA", cmd_pca_variancia, argparse.Namespace(variancia=0.8)),
        ("Boosting com PCA", cmd_pca, argparse.Namespace(variante="boosting_pca", dobras=None)),
        # PCA em dois regressores (boosting e regressao linear). O KNN sem PCA (forca
        # bruta) segue disponivel em `python -m src.modelagem.pca_balanceamento_ablacao pca --variante knn_sem_pca`.
        ("Regressao linear com PCA", cmd_pca, argparse.Namespace(variante="linear_pca", dobras=None)),
    ]
    for modelo, estrategias in [
        ("knn", ["under", "over"]),
        ("boosting", ["peso", "under", "over"]),
    ]:
        etapas.extend(
            (f"Balanceamento: {modelo} ({estrategia})", cmd_balanceamento,
             argparse.Namespace(modelo=modelo, estrategia=estrategia))
            for estrategia in estrategias
        )
    etapas.extend(
        (f"Ablacao: {modelo}", cmd_ablacao, argparse.Namespace(modelo=modelo))
        for modelo in ["knn", "boosting"]
    )
    etapas.extend(
        (f"Busca de hiperparametros: {modelo}", cmd_busca,
         argparse.Namespace(modelo=modelo, n_iter=12))
        for modelo in ["knn", "boosting", "ridge"]
    )
    etapas.extend(
        (f"Curva de aprendizado: {modelo}", cmd_curva,
         argparse.Namespace(
             modelo=modelo,
         params=None,
             estrategia=None,
             max_aval_treino=None,
         ))
        for modelo in ["knn", "boosting"]
    )

    experimentos = [
        "baseline_media", "knn_base", "boosting_base", "boosting_pca", "linear_base", "ridge_base", "linear_pca",
        "knn_under", "knn_over", "boosting_peso", "boosting_under", "boosting_over",
    ]
    experimentos.extend(
        f"{modelo}_sem_{bloco}"
        for modelo in ["knn", "boosting"]
        for bloco in BLOCOS
    )
    experimentos.extend(["knn_busca_melhor", "boosting_busca_melhor", "ridge_busca_melhor"])
    etapas.append(
        ("Relatorio de erros por faixa e grupo", cmd_relatorio,
         argparse.Namespace(experimentos=experimentos))
    )

    total = len(etapas) + 1
    print(
        "Execucao completa iniciada. O KNN sem PCA e as buscas podem exigir "
        "muitas horas e bastante memoria.",
        flush=True,
    )
    for numero, (nome, comando, argumentos) in enumerate(etapas, start=1):
        print(f"\n[{numero}/{total}] {nome}", flush=True)
        if nome.startswith("Curva de aprendizado:"):
            argumentos.params = BEST_PARAMS[argumentos.modelo]
        comando(argumentos)

    candidatos = [
        ("knn_base", "knn", None, None),
        ("boosting_base", "boosting", None, None),
        ("linear_base", "linear", None, None),
        ("ridge_base", "ridge", None, None),
        ("knn_under", "knn", "under", None),
        ("knn_over", "knn", "over", None),
        ("boosting_peso", "boosting", "peso", None),
        ("boosting_under", "boosting", "under", None),
        ("boosting_over", "boosting", "over", None),
        ("knn_busca_melhor", "knn", None, BEST_PARAMS["knn"]),
        ("boosting_busca_melhor", "boosting", None, BEST_PARAMS["boosting"]),
        ("ridge_busca_melhor", "ridge", None, BEST_PARAMS["ridge"]),
    ]
    resultados = []
    for nome, modelo, estrategia, params in candidatos:
        metricas_cv = CV_RESULTS.get(nome)
        if metricas_cv is None or metricas_cv.empty or not np.isfinite(metricas_cv["rmse"]).all():
            raise RuntimeError(f"Metricas de validacao invalidas para {nome}.")
        resultados.append((metricas_cv["rmse"].mean(), nome, modelo, estrategia, params))

    rmse, nome, modelo, estrategia, params = min(resultados, key=lambda linha: linha[0])
    print(
        f"\n[{total}/{total}] Avaliacao final no teste reservado\n"
        f"Solucao selecionada somente pela validacao cruzada: {nome} "
        f"(RMSE medio {rmse:.3f}).",
        flush=True,
    )
    cmd_teste(
        argparse.Namespace(
            modelo=modelo,
            params=params,
            estrategia=estrategia,
            confirmo_solucao_final=True,
        )
    )
    print(
        "\nExecucao completa concluida. Resultados: "
        f"terminal | Figuras: {FIG}",
        flush=True,
    )


def carregar_checkpoint_execucao():
    if not RESUMO_EXECUCAO.is_file():
        raise SystemExit(
            f"Nao encontrei {RESUMO_EXECUCAO.name}. "
            "O comando retomar precisa do resumo da execucao anterior."
        )
    texto = RESUMO_EXECUCAO.read_text(encoding="utf-8")

    padroes = {
        "knn": (
            r"KNN:\s+RMSE medio da busca [\d,]+;\s*"
            r"n_components=(\d+),\s*weights=(\w+),\s*p=(\d+),\s*"
            r"n_neighbors=(\d+),\s*leaf_size=(\d+)\."
        ),
        "boosting": (
            r"Boosting:\s+RMSE medio da busca [\d,]+;\s*"
            r"l2_regularization=([\d,]+),\s*learning_rate=([\d,]+),\s*"
            r"max_depth=(None|\d+),\s*max_iter=(\d+),\s*max_leaf_nodes=(\d+),\s*"
            r"min_samples_leaf=(\d+),\s*n_iter_no_change=(\d+)\."
        ),
        "ridge": (
            r"Ridge:\s+RMSE medio da busca [\d,]+;\s*alpha=([\d,]+),\s*"
            r"fit_intercept=(True|False),\s*solver=(\w+)\."
        ),
    }
    matches = {
        nome: re.search(padrao, texto, flags=re.DOTALL)
        for nome, padrao in padroes.items()
    }
    knn_match = matches["knn"]
    boosting_match = matches["boosting"]
    ridge_match = matches["ridge"]
    if knn_match is None or boosting_match is None or ridge_match is None:
        ausentes = [
            nome for nome, match in matches.items() if match is None
        ]
        raise SystemExit(
            f"O resumo nao tem os hiperparametros necessarios para retomar: "
            f"{', '.join(ausentes)}."
        )

    knn = knn_match.groups()
    boosting = boosting_match.groups()
    ridge = ridge_match.groups()
    melhores = {
        "knn": {
            "pca__n_components": int(knn[0]),
            "model__weights": knn[1],
            "model__p": int(knn[2]),
            "model__n_neighbors": int(knn[3]),
            "model__leaf_size": int(knn[4]),
        },
        "boosting": {
            "model__l2_regularization": float(boosting[0].replace(",", ".")),
            "model__learning_rate": float(boosting[1].replace(",", ".")),
            "model__max_depth": None if boosting[2] == "None" else int(boosting[2]),
            "model__max_iter": int(boosting[3]),
            "model__max_leaf_nodes": int(boosting[4]),
            "model__min_samples_leaf": int(boosting[5]),
            "model__n_iter_no_change": int(boosting[6]),
        },
        "ridge": {
            "model__alpha": float(ridge[0].replace(",", ".")),
            "model__fit_intercept": ridge[1] == "True",
            "model__solver": ridge[2],
        },
    }

    candidatos = [
        ("knn_base", "knn", None, None),
        ("boosting_base", "boosting", None, None),
        ("linear_base", "linear", None, None),
        ("ridge_base", "ridge", None, None),
        ("knn_under", "knn", "under", None),
        ("knn_over", "knn", "over", None),
        ("boosting_peso", "boosting", "peso", None),
        ("boosting_under", "boosting", "under", None),
        ("boosting_over", "boosting", "over", None),
        ("knn_busca_melhor", "knn", None, melhores["knn"]),
        ("boosting_busca_melhor", "boosting", None, melhores["boosting"]),
        ("ridge_busca_melhor", "ridge", None, melhores["ridge"]),
    ]
    resultados = {}
    for nome, _, _, _ in candidatos:
        padrao = (
            rf"^\s*{re.escape(nome)}\s+([\d,]+)\s+[\d,]+\s+-?[\d,]+\s*$"
        )
        match = re.search(padrao, texto, flags=re.MULTILINE)
        if match is not None:
            resultados[nome] = float(match.group(1).replace(",", "."))
    faltantes = [nome for nome, _, _, _ in candidatos if nome not in resultados]
    if faltantes:
        raise SystemExit(
            "O resumo nao tem as metricas de validacao necessarias para escolher "
            f"o modelo final: {', '.join(faltantes)}."
        )

    rmse, nome, modelo, estrategia, params = min(
        (resultados[nome], nome, modelo, estrategia, params)
        for nome, modelo, estrategia, params in candidatos
    )
    return melhores, (rmse, nome, modelo, estrategia, params)


def cmd_retomar(_args=None):
    melhores, selecionado = carregar_checkpoint_execucao()
    rmse, nome, modelo, estrategia, params = selecionado
    print(
        "Retomando a execucao a partir da etapa 16/19; "
        "etapas 1 a 15 foram recuperadas do resumo existente.",
        flush=True,
    )

    print("\n[16/19] Curva de aprendizado: knn", flush=True)
    cmd_curva(argparse.Namespace(
        modelo="knn",
        params=melhores["knn"],
        estrategia=None,
        max_aval_treino=None,
    ))

    print("\n[17/19] Curva de aprendizado: boosting", flush=True)
    cmd_curva(argparse.Namespace(
        modelo="boosting",
        params=melhores["boosting"],
        estrategia=None,
        max_aval_treino=None,
    ))

    print(
        "\n[18/19] Relatorio OOF ignorado: as previsoes fora da dobra das etapas "
        "anteriores nao foram salvas e nao podem ser recuperadas do resumo.",
        flush=True,
    )

    print(
        f"\n[19/19] Avaliacao final no teste reservado\n"
        f"Solucao selecionada somente pela validacao cruzada: {nome} "
        f"(RMSE medio {rmse:.3f}).",
        flush=True,
    )
    cmd_teste(argparse.Namespace(
        modelo=modelo,
        params=params,
        estrategia=estrategia,
        confirmo_solucao_final=True,
    ))
    print(
        f"\nRetomada concluida. Resultados no terminal | Figuras: {FIG}",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("completo", help="executa toda a analise e a avaliacao final").set_defaults(f=cmd_completo)
    sub.add_parser(
        "retomar",
        help="retoma nas etapas 16 a 19 usando o resumo da execucao interrompida",
    ).set_defaults(f=cmd_retomar)

    sub.add_parser("distribuicao").set_defaults(f=cmd_distribuicao)

    sp = sub.add_parser("base")
    sp.add_argument("--modelos", nargs="+", choices=MODELOS, default=MODELOS)
    sp.set_defaults(f=cmd_base)

    sp = sub.add_parser("pca-variancia")
    sp.add_argument("--variancia", type=float, default=0.8)
    sp.set_defaults(f=cmd_pca_variancia)

    sp = sub.add_parser("pca")
    sp.add_argument("--variante", choices=["boosting_pca", "linear_pca", "knn_sem_pca"], required=True)
    sp.add_argument("--dobras", type=int, default=None, help="limita o numero de dobras (registrar!)")
    sp.set_defaults(f=cmd_pca)

    sp = sub.add_parser("balanceamento")
    sp.add_argument("--modelo", choices=MODELOS, required=True)
    sp.add_argument("--estrategia", choices=["peso", "under", "over"], required=True)
    sp.set_defaults(f=cmd_balanceamento)

    sp = sub.add_parser("ablacao")
    sp.add_argument("--modelo", choices=MODELOS, default="boosting")
    sp.set_defaults(f=cmd_ablacao)

    sp = sub.add_parser("busca")
    sp.add_argument("--modelo", choices=MODELOS, required=True)
    sp.add_argument("--n-iter", type=int, default=12)
    sp.set_defaults(f=cmd_busca)

    sp = sub.add_parser("curva")
    sp.add_argument("--modelo", choices=MODELOS, required=True)
    sp.add_argument("--params", default=None, help="arquivo JSON existente com parametros do modelo")
    sp.add_argument("--estrategia", choices=["peso", "under", "over"], default=None)
    sp.add_argument("--max-aval-treino", type=int, default=None,
                    help="amostra (semente 42) para medir o RMSE de treino; documente se usar")
    sp.set_defaults(f=cmd_curva)

    sp = sub.add_parser("relatorio")
    sp.add_argument("--experimentos", nargs="+", required=True)
    sp.set_defaults(f=cmd_relatorio)

    sp = sub.add_parser("teste")
    sp.add_argument("--modelo", choices=MODELOS, required=True)
    sp.add_argument("--params", default=None)
    sp.add_argument("--estrategia", choices=["peso", "under", "over"], default=None)
    sp.add_argument("--confirmo-solucao-final", action="store_true")
    sp.set_defaults(f=cmd_teste)

    if len(sys.argv) == 1:
        cmd_completo()
        return
    args = p.parse_args()
    args.f(args)


def executar_com_registro():
    # Ajuda ou chamada sem subcomando não executam nada: não podem sobrescrever o log da última execução.
    if len(sys.argv) == 1 or {"-h", "--help"} & set(sys.argv[1:]):
        main()
        return
    with registrar_saida():
        try:
            main()
        except SystemExit as exc:
            if isinstance(exc.code, str):
                print(exc.code, file=sys.stderr)
                raise SystemExit(1) from None
            raise
        except BaseException:
            traceback.print_exc()
            raise SystemExit(1) from None


if __name__ == "__main__":
    executar_com_registro()
