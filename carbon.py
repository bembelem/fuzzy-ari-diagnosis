"""Проверка модели на реальных данных Carbon Health (п. 3.5).

Источник: Coronavirus Disease 2019 (COVID-19) Clinical Data Repository,
https://github.com/mdcollab/covidclinicaldata (амбулаторные пациенты, США, апрель–октябрь 2020).

В данных нет признаков x8 (чихание) и x9 (начало болезни), поэтому при сравнении
с эталонами им присваивается нулевой вес.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

import model as m

DATA_DIR = Path(__file__).parent / "data" / "carbonhealth"
WEIGHTS = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=float)   # x8, x9 отсутствуют
COVID, TONSIL = 2, 3
COUGH_SEVERITY = {"Mild": 0.25, "Moderate": 0.5, "Severe": 0.75}
SYMPTOMS = ["cough", "fever", "runny_nose", "sore_throat", "muscle_sore", "fatigue",
            "loss_of_smell", "loss_of_taste", "headache", "sob", "diarrhea"]


def load(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in sorted(glob.glob(str(data_dir / "*.csv")))],
                   ignore_index=True)
    for c in SYMPTOMS:
        df[c] = df[c].astype(str).eq("True")
    df["symptomatic"] = df[SYMPTOMS].any(axis=1)
    return df


def to_fuzzy(df: pd.DataFrame, temperature: str = "fuzzy") -> np.ndarray:
    """Перевод записей в нечёткие множества пациентов.

    temperature:
      "fuzzy":  измеренная температура через формулу (13);
      "crisp":  та же температура, огрублённая до «лихорадка есть/нет» (порог 38 °C);
      "report": только жалоба на лихорадку (да/нет), без измерения.
    """
    X = np.zeros((len(df), 9))
    t = df["temperature"].to_numpy(dtype=float)
    if temperature == "fuzzy":
        X[:, 0] = [m.temperature_membership(v) for v in t]
    elif temperature == "crisp":
        X[:, 0] = (t >= 38.0).astype(float)
    else:
        X[:, 0] = df["fever"].astype(float)
    sev = df["cough_severity"].map(COUGH_SEVERITY)
    X[:, 1] = np.where(df["cough"], sev.fillna(0.5), 0.0)
    X[:, 2] = df["runny_nose"].astype(float)
    X[:, 3] = df["sore_throat"].astype(float)
    X[:, 4] = df["muscle_sore"].astype(float)
    X[:, 5] = df["fatigue"].astype(float)
    X[:, 6] = (df["loss_of_smell"] | df["loss_of_taste"]).astype(float)
    return X


def similarities(X: np.ndarray, etalons: np.ndarray = m.ETALONS, metric: str = m.METRIC,
                 weights: np.ndarray = WEIGHTS, rules: bool = True) -> np.ndarray:
    S = np.array([[m.METRICS[metric](a, e, weights) for e in etalons] for a in X])
    return np.array([m.apply_rules(a, s) for a, s in zip(X, S)]) if rules else S


def disease_score(S: np.ndarray, k: int) -> np.ndarray:
    """Насколько заболевание k выделяется среди остальных: S_k − max(S_j, j ≠ k)."""
    others = np.delete(S, k, axis=1)
    return S[:, k] - others.max(axis=1)
