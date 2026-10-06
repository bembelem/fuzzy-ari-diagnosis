"""Проверка эталонов вида «частота × типичная выраженность» (μ = p · κ).

κ[k, i] есть средняя выраженность признака i у больных заболеванием k, у которых признак есть.
Оценивается по ситуационным задачам методом «исключи одну»: для проверяемой задачи κ
считается по всем остальным задачам, поэтому сама задача в оценку не входит.
Значения, заданные прямым методом, не масштабируются: они уже выражают типичную степень.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

import carbon as cb
import model as m
import textbook_cases as tc

MIN_OBS = 2          # при меньшем числе наблюдений κ = 1 (эталон не меняется)
CASES = [c for c in tc.load_cases() if c["эталон"] in m.DISEASES]
TRAPS = [c for c in tc.load_cases() if c["эталон"] == "нет среди пяти"]


def present_value(c, i):
    """Выраженность признака i в задаче, если он есть (для x1 степень лихорадки при t ≥ 38 °C)."""
    v = c["x"][m.FEATURE_CODES[i]]
    if v is None:
        return None
    if i == 0:
        return m.temperature_membership(v) if v >= 38.0 else None
    return v if v > 0 else None


def kappa(cases):
    K = np.ones_like(m.ETALONS)
    n = np.zeros_like(m.ETALONS)
    for k, d in enumerate(m.DISEASES):
        for i in range(9):
            vals = [v for c in cases if c["эталон"] == d and (v := present_value(c, i)) is not None]
            n[k, i] = len(vals)
            if len(vals) >= MIN_OBS:
                K[k, i] = np.mean(vals)
    return K, n


def scaled(K):
    E = m.ETALONS * K
    for r, col in m.EXPERT_CELLS:
        E[r, col] = m.ETALONS[r, col]
    return E


def evaluate(case, E, exam=True):
    a, w, ex = tc.case_vector(case)
    d = m.diagnose(a, etalons=E, alpha=0, weights=w, exam=ex if exam else None)
    s = d.similarities
    k = m.DISEASES.index(case["эталон"])
    return int(d.order[0] == k), d.order.index(k) + 1, s[k] - np.delete(s, k).max()


def alpha_for(E):
    """Как в п. 3.5: 5-й процентиль сходства с эталоном истинного заболевания на синтетике."""
    X, y = m.synthetic_cases(300, seed=0)
    s_true = [m.diagnose(a, etalons=E, alpha=0).similarities[k] for a, k in zip(X, y)]
    return float(np.percentile(s_true, 5))


K_all, n_all = kappa(CASES)
np.set_printoptions(precision=2, suppress=True)
print("κ по всем задачам (1 означает, что наблюдений меньше двух):")
for d, row, nn in zip(m.DISEASES, K_all, n_all):
    print(f"  {d:20s} {row}  n = {nn.astype(int)}")

for exam in (False, True):
    print(f"\n=== Ситуационные задачи, проверочная группа, данные осмотра: {'да' if exam else 'нет'} ===")
    rows = []
    for c in CASES:
        if c["группа"] != "проверка":
            continue
        K, _ = kappa([o for o in CASES if o["id"] != c["id"]])
        b, e = evaluate(c, m.ETALONS, exam), evaluate(c, scaled(K), exam)
        rows.append((c["id"], c["эталон"], *b, *e))
        print(f"  {c['id']:5s} {c['эталон']:20s} частоты: верно={b[0]} место={b[1]} отрыв={b[2]:+.3f} | "
              f"p·κ: верно={e[0]} место={e[1]} отрыв={e[2]:+.3f}")
    r = np.array([x[2:] for x in rows], dtype=float)
    print(f"  ИТОГО (n = {len(r)}): верно {int(r[:, 0].sum())} → {int(r[:, 3].sum())}; "
          f"среднее место {r[:, 1].mean():.2f} → {r[:, 4].mean():.2f}; "
          f"средний отрыв {r[:, 2].mean():+.3f} → {r[:, 5].mean():+.3f}; "
          f"отрыв вырос в {int((r[:, 5] > r[:, 2]).sum())} из {len(r)}")

E_all = scaled(K_all)
a0, a1 = m.ALPHA, alpha_for(E_all)
print(f"\nПорог α: частоты {a0:.3f}, p·κ {a1:.3f}")
for name, E, al in (("частоты", m.ETALONS, a0), ("p·κ", E_all, a1)):
    hits = 0
    for c in TRAPS:
        if c["группа"] != "проверка":
            continue
        a, w, ex = tc.case_vector(c)
        hits += m.diagnose(a, etalons=E, alpha=al, weights=w, exam=ex).status == "atypical"
    print(f"  ловушки ({name}): распознано как нетипичное {hits} из {sum(c['группа'] == 'проверка' for c in TRAPS)}")

df = cb.load()
sym = df[df.symptomatic & df.temperature.notna()].reset_index(drop=True)
y = (sym.covid19_test_results == "Positive").to_numpy().astype(int)
X = cb.to_fuzzy(sym, "fuzzy")
for name, E in (("частоты", m.ETALONS), ("p·κ", E_all)):
    s = cb.disease_score(cb.similarities(X, etalons=E), cb.COVID)
    print(f"Carbon Health, AUC выделения COVID-19 ({name}): {roc_auc_score(y, s):.3f}")
