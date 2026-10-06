"""Сравнение способов использовать частотные эталоны (обсуждение п. 2.3).

A. Евклидова мера (модель работы): эталон трактуется как средняя выраженность.
B. Max-min композиция (Санчес, CADIAG-2): S_k = max_i min(a_i, p_ki).
C. Средний min: S_k = Σ min(a_i, p_ki) / Σ max(a_i, p_ki) (нечёткое сходство Жаккара).
D. «Нечёткий» наивный Байес: частота трактуется как вероятность наличия симптома,
   значение пациента как степень наличия: log L_k = Σ a_i ln p_ki + (1 − a_i) ln(1 − p_ki).
Во всех вариантах неизвестные признаки пропускаются, правила (18), (19) применяются одинаково.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

import carbon as cb
import model as m
import textbook_cases as tc

EPS = 0.02


def scores(method, a, w, E):
    k = w > 0
    a, E = a[k], E[:, k]
    if method == "A":
        s = 1 - np.sqrt(((a - E) ** 2).mean(axis=1))
    elif method == "B":
        s = np.minimum(a, E).max(axis=1)
    elif method == "C":
        s = np.minimum(a, E).sum(axis=1) / np.maximum(np.maximum(a, E).sum(axis=1), 1e-9)
    else:
        p = np.clip(E, EPS, 1 - EPS)
        ll = (a * np.log(p) + (1 - a) * np.log(1 - p)).sum(axis=1)
        s = np.exp(ll - ll.max())
        s = s / s.sum()
    return s


def decide(method, a, w, exam=None, rules=True):
    E = m.ETALONS
    rule_a = a
    if exam is not None:
        known = ~np.isnan(exam)
        a = np.concatenate([a, np.where(known, exam, 0.0)])
        w = np.concatenate([w, known.astype(float)])
        E = np.hstack([E, m.EXAM_ETALONS])
    s = scores(method, a, w, E)
    return m.apply_rules(rule_a, s) if rules else s


NAMES = {"A": "евклидова мера (сейчас)", "B": "max-min (Санчес, CADIAG-2)",
         "C": "средний min (Жаккар)", "D": "нечёткий наивный Байес"}
cases = [c for c in tc.load_cases() if c["эталон"] in m.DISEASES and c["группа"] == "проверка"]
X, y = m.synthetic_cases(300, seed=1)
df = cb.load()
sym = df[df.symptomatic & df.temperature.notna()].reset_index(drop=True)
yc = (sym.covid19_test_results == "Positive").to_numpy().astype(int)
Xc = cb.to_fuzzy(sym, "fuzzy")

for meth, name in NAMES.items():
    res = []
    for exam in (False, True):
        ok = 0
        for c in cases:
            a, w, ex = tc.case_vector(c)
            s = decide(meth, a, w, ex if exam else None)
            ok += m.DISEASES[int(np.argmax(s))] == c["эталон"]
        res.append(ok)
    syn = np.mean([np.argmax(decide(meth, a, np.ones(9))) == t for a, t in zip(X, y)])
    S = np.array([decide(meth, a, cb.WEIGHTS) for a in Xc])
    auc = roc_auc_score(yc, cb.disease_score(S, cb.COVID))
    print(f"{name:30s} задачи: {res[0]}/17 без осмотра, {res[1]}/17 с осмотром; синтетика {syn:.3f}; Carbon AUC {auc:.3f}")
