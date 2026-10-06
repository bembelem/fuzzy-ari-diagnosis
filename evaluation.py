"""Оценка качества модели (п. 3.5) и анализ чувствительности (п. 3.4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import model as m

ATYPICAL = -1  # код ответа «картина нетипична»


def predict(X: np.ndarray, etalons=m.ETALONS, alpha=m.ALPHA, delta=0.05, metric=m.METRIC):
    """Возвращает top-1 (или ATYPICAL), top-2 и признак «неуверенного» ответа для каждого случая."""
    top1, top2, uncertain = [], [], []
    for a in X:
        d = m.diagnose(a, etalons, alpha, delta, metric)
        top1.append(ATYPICAL if d.status == "atypical" else d.order[0])
        top2.append(d.order[:2])
        uncertain.append(d.status == "uncertain")
    return np.array(top1), np.array(top2), np.array(uncertain)


def _metrics(y, p, top2):
    k = len(m.DISEASES)
    labels = list(range(k))
    return {
        "Доля верных ответов (accuracy)": float(np.mean(p == y)),
        "Сбалансированная точность": float(balanced_accuracy_score(y, np.where(p == ATYPICAL, k, p))),
        "Macro-F1": float(f1_score(y, p, labels=labels, average="macro", zero_division=0)),
        "Каппа Коэна": float(cohen_kappa_score(y, p, labels=labels + [ATYPICAL])),
        "Top-2 accuracy": float(np.mean([y[i] in top2[i] for i in range(len(y))])),
        "Доля отказов («нетипично»)": float(np.mean(p == ATYPICAL)),
    }


def evaluate(X, y, etalons=m.ETALONS, alpha=m.ALPHA, delta=0.05, metric=m.METRIC, n_boot=500, seed=0):
    """Все метрики, 95 % доверительные интервалы (бутстрэп), матрица ошибок, чувствительность/специфичность."""
    p, top2, unc = predict(X, etalons, alpha, delta, metric)
    point = _metrics(y, p, top2)
    point["Доля ответов «нельзя исключить»"] = float(np.mean(unc))

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        boots.append(_metrics(y[idx], p[idx], top2[idx]))
    ci = {key: (float(np.percentile([b[key] for b in boots], 2.5)),
                float(np.percentile([b[key] for b in boots], 97.5))) for key in boots[0]}

    k = len(m.DISEASES)
    cm = confusion_matrix(y, p, labels=list(range(k)) + [ATYPICAL])[:k]
    cm_df = pd.DataFrame(cm, index=m.DISEASES, columns=m.DISEASES + ["нетипично"])

    per_class = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c)); fn = np.sum((p != c) & (y == c))
        fp = np.sum((p == c) & (y != c)); tn = np.sum((p != c) & (y != c))
        per_class.append({"Заболевание": m.DISEASES[c],
                          "Чувствительность": tp / (tp + fn) if tp + fn else np.nan,
                          "Специфичность": tn / (tn + fp) if tn + fp else np.nan,
                          "Случаев": int(np.sum(y == c))})
    return point, ci, cm_df, pd.DataFrame(per_class), p


def baseline_logreg(X, y, seed=0):
    """Модель сравнения: логистическая регрессия, предсказания по 5-кратной перекрёстной проверке."""
    n_splits = int(min(5, np.bincount(y).min()))
    if n_splits < 2:
        return None
    clf = LogisticRegression(max_iter=2000)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
    p = proba.argmax(1)
    top2 = np.argsort(-proba, axis=1)[:, :2]
    return _metrics(y, p, top2)


# ---------- анализ чувствительности (п. 3.4) ----------

def robustness_expert(X, etalons=m.ETALONS, eps=0.1, n_runs=100, **kw):
    """Доля случаев, у которых top-1 не меняется при случайном сдвиге экспертных значений на ±eps."""
    base, _, _ = predict(X, etalons, **kw)
    same = []
    for r in range(n_runs):
        p, _, _ = predict(X, m.perturb_expert(etalons, eps, seed=r), **kw)
        same.append(np.mean(p == base))
    return float(np.mean(same)), float(np.min(same))


def grid_alpha_delta(X, y, alphas, deltas, etalons=m.ETALONS, metric=m.METRIC):
    """Точность, доля отказов и доля неуверенных ответов на сетке значений α и δ."""
    rows = []
    for a in alphas:
        for d in deltas:
            p, _, unc = predict(X, etalons, a, d, metric)
            rows.append({"α": a, "δ": d, "Accuracy": np.mean(p == y),
                         "Отказы": np.mean(p == ATYPICAL), "«Нельзя исключить»": np.mean(unc)})
    return pd.DataFrame(rows)


# ---------- загрузка реальных случаев ----------

def load_cases(df: pd.DataFrame):
    """CSV: столбцы x1..x9 и diagnosis.

    x1 можно задать температурой в °C (значение > 30 переводится по формуле 15);
    x2..x9 можно задать числом из [0, 1] или термом шкалы («умеренно выражен» и т. п.).
    """
    X = np.zeros((len(df), 9))
    for j, code in enumerate(m.FEATURE_CODES):
        col = df[code]
        if j == 0:
            X[:, 0] = [m.temperature_membership(v) if v > 30 else float(v) for v in col.astype(float)]
        else:
            X[:, j] = [m.TERMS[v.strip()] if isinstance(v, str) and v.strip() in m.TERMS else float(v) for v in col]
    y = np.array([m.DISEASES.index(d.strip()) for d in df["diagnosis"]])
    return X, y
