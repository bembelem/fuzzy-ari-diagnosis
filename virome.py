"""Данные проекта Virome of Manhattan (Galanti et al., PLOS Biology, 2024): еженедельные мазки с ПЦР-панелью
на 18 респираторных вирусов и ежедневные оценки 9 симптомов по шкале 0–3 (S1 Data, S24 Data статьи).

Эпизод: образец с найденным вирусом ОРВИ (риновирус, сезонный коронавирус, RSV) и максимальные оценки
симптомов за 7 дней вокруг даты взятия мазка (±3 дня), переведённые в [0, 1] делением на 3.
Используются только эпизоды хотя бы с одним симптомом.
Когорты D1, D2, SC (детские сады и школа) служат для оценки κ и эталона ОРВИ, когорты MC, ER (взрослые) для проверки.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path(__file__).parent / "data" / "virome"
SYMPTOMS = ["Fever", "Chills", "MusclePain", "WateryEye", "RunnyNose", "Sneezing", "SoreThroat", "Cough", "ChestPain"]
BUILD, TEST = ["D1", "D2", "SC"], ["MC", "ER"]


def _is_orvi(t: str) -> bool:
    return t.startswith("Rhinovirus") or t.startswith("Coronavirus") or t.startswith("Respiratory syncytial")


def episodes(cohorts) -> np.ndarray:
    """Матрица эпизодов ОРВИ в признаках модели: x1 лихорадка, x2 кашель, x3 насморк, x4 боль в горле,
    x5 ломота, x8 чихание или слезотечение; x6, x7, x9 в данных нет (NaN)."""
    meta = pd.read_csv(DIR / "samples_metadata.csv", keep_default_na=False)
    meta["date"] = pd.to_datetime(meta.Date_collected, format="%d-%b-%y")
    sym = {}
    for c in cohorts:
        for k, v in pd.read_excel(DIR / "symptoms" / f"sympt_{c}.xlsx", sheet_name=None).items():
            v = v.rename(columns={v.columns[0]: "date"})
            v["date"] = pd.to_datetime(v["date"], errors="coerce")
            sym[str(k).strip()] = v.dropna(subset=["date"])
    rows = []
    for _, r in meta[meta.Cohort.isin(cohorts)].iterrows():
        if r.ID not in sym or not _is_orvi(r.NEW_Infection_type):
            continue
        v = sym[r.ID]
        win = v[(v.date >= r.date - pd.Timedelta(days=3)) & (v.date <= r.date + pd.Timedelta(days=3))]
        if len(win) < 4:
            continue
        mx = win[SYMPTOMS].astype(float).max()
        if mx.sum() == 0:
            continue
        rows.append([mx.Fever, mx.Cough, mx.RunnyNose, mx.SoreThroat, mx.MusclePain, np.nan, np.nan,
                     max(mx.Sneezing, mx.WateryEye), np.nan])
    return np.array(rows) / 3


def kappa_and_mean(X: np.ndarray):
    """p_i (доля эпизодов с симптомом), κ_i (средняя выраженность при наличии), m_i (средняя выраженность)."""
    p, k, mm = [], [], []
    for i in range(X.shape[1]):
        col = X[:, i]
        if np.all(np.isnan(col)):
            p.append(np.nan); k.append(np.nan); mm.append(np.nan); continue
        p.append(np.mean(col > 0)); k.append(col[col > 0].mean()); mm.append(col.mean())
    return np.array(p), np.array(k), np.array(mm)
