from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.special import expit, softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from app.pipeline.config import TaskConfig
from app.pipeline.data import Dataset, targets

def _preprocess_texts(texts: list[str], config: TaskConfig) -> list[str]:
    if not config.language_normalize and not config.strip_zero_width and not config.clean_text:
        return texts
    from app.pipeline.language import clean_text as do_clean_text, normalize_unicode
    out = []
    for t in texts:
        cur = t
        if config.language_normalize:
            cur = normalize_unicode(cur, form=config.language_normalize, strip_zero_width=config.strip_zero_width)
        elif config.strip_zero_width:
            cur = normalize_unicode(cur, form="NFC", strip_zero_width=True)
        if config.clean_text:
            cur = do_clean_text(cur, collapse_whitespace=True, strip_zero_width=config.strip_zero_width)
        out.append(cur)
    return out


def _build_vectorizer(config: TaskConfig, vocabulary: dict | None = None) -> TfidfVectorizer:
    kwargs: dict[str, object] = {}
    if config.subword_ngrams:
        kwargs["analyzer"] = "char_wb"
        kwargs["ngram_range"] = (3, 5)
    else:
        kwargs["token_pattern"] = r"(?u)\b\w+\b"
    if vocabulary is not None:
        kwargs["vocabulary"] = vocabulary
    return TfidfVectorizer(**kwargs)

def fit_baseline(data: Dataset, config: TaskConfig) -> tuple[dict, dict]:
    preprocessing = {"text": None, "numeric": None}
    if config.text_column:
        texts = _preprocess_texts([row[config.text_column] for row in data.rows], config)
        vectorizer = _build_vectorizer(config)
        try:
            vectorizer.fit(texts)
        except ValueError as exc:
            raise ValueError(f"cannot fit text preprocessing: {exc}") from exc
        preprocessing["text"] = {"vocabulary": {k: int(v) for k, v in vectorizer.vocabulary_.items()},
                                 "idf": vectorizer.idf_.tolist()}
    if config.feature_columns:
        numeric = np.asarray([[float(row[c]) for c in config.feature_columns] for row in data.rows])
        scaler = StandardScaler().fit(numeric)
        preprocessing["numeric"] = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    features = transform(data, config, preprocessing)
    if config.task_type == "regression":
        estimator = Ridge(alpha=1.0, solver="lsqr")
        kind = "ridge"
    else:
        estimator = LogisticRegression(random_state=config.seed, max_iter=1000)
        kind = "logistic_regression"
    estimator.fit(features, targets(data, config))
    model = {"kind": kind, "classes": estimator.classes_.tolist() if kind == "logistic_regression" else [],
             "coefficients": np.atleast_2d(estimator.coef_).tolist(),
             "intercept": np.atleast_1d(estimator.intercept_).tolist()}
    return preprocessing, model


def transform(data: Dataset, config: TaskConfig, preprocessing: dict):
    parts = []
    if config.text_column:
        text = preprocessing["text"]
        texts = _preprocess_texts([row[config.text_column] for row in data.rows], config)
        vectorizer = _build_vectorizer(config, vocabulary=text["vocabulary"])
        vectorizer.idf_ = np.asarray(text["idf"], dtype=float)
        parts.append(vectorizer.transform(texts))
    if config.feature_columns:
        numeric = preprocessing["numeric"]
        values = np.asarray([[float(row[c]) for c in config.feature_columns] for row in data.rows])
        parts.append(csr_matrix((values - numeric["mean"]) / np.asarray(numeric["scale"])))
    return hstack(parts, format="csr")


def infer(data: Dataset, config: TaskConfig, preprocessing: dict, model: dict) -> tuple[np.ndarray, np.ndarray | None]:
    features = transform(data, config, preprocessing)
    scores = np.asarray(features @ np.asarray(model["coefficients"]).T) + model["intercept"]
    if model["kind"] == "ridge":
        predictions, probabilities = scores[:, 0], None
    else:
        classes = np.asarray(model["classes"])
        if len(classes) == 2:
            positive = expit(scores[:, 0])
            probabilities = np.column_stack((1 - positive, positive))
        else:
            probabilities = softmax(scores, axis=1)
        predictions = classes[np.argmax(probabilities, axis=1)]
    if predictions.shape != (len(data.rows),) or (probabilities is not None and not np.isfinite(probabilities).all()) or (config.task_type == "regression" and not np.isfinite(predictions).all()):
        raise ValueError("model produced invalid prediction shape or nonfinite values")
    return predictions, probabilities


def metric_score(data: Dataset, config: TaskConfig, model: dict, predictions, probabilities) -> float:
    labels = targets(data, config)
    metric = config.metric
    if metric == "accuracy":
        score = accuracy_score(labels, predictions)
    elif metric in ("f1", "f1_macro"):
        score = f1_score(labels, predictions, average="binary" if metric == "f1" else "macro",
                         pos_label=config.positive_label if metric == "f1" else 1, zero_division=0)
    elif metric == "roc_auc":
        index = model["classes"].index(config.positive_label)
        score = roc_auc_score(labels == config.positive_label, probabilities[:, index])
    elif metric == "rmse":
        score = np.sqrt(mean_squared_error(labels, predictions))
    elif metric == "mae":
        score = mean_absolute_error(labels, predictions)
    else:  # Config validation limits this branch to r2.
        if len(labels) < 2:
            raise ValueError("r2 requires at least two evaluation rows")
        score = r2_score(labels, predictions)
    if not np.isfinite(score):
        raise ValueError("evaluation metric is not finite")
    return float(score)
