"""Расчёты для п. 3.5: проверка модели на реальных пациентах Carbon Health.

Запуск: .venv\\Scripts\\python analyze_carbon.py
"""
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import carbon as cb
import model as m

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)


def auc_ci(y, s, n=1000):
    """AUC и 95 % доверительный интервал (бутстрэп)."""
    base = roc_auc_score(y, s)
    boots = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            boots.append(roc_auc_score(y[i], s[i]))
    return base, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def decisions(X, k, etalons=m.ETALONS, alpha=0.6, delta=0.05, metric="Хэмминг"):
    """top-1 == k и «k среди ответов» (основной диагноз или «нельзя исключить»)."""
    top1, in_list, refused = [], [], []
    for a in X:
        d = m.diagnose(a, etalons, alpha, delta, metric, weights=cb.WEIGHTS)
        name = m.DISEASES[k]
        top1.append(d.diagnosis == name)
        in_list.append(d.diagnosis == name or name in d.cannot_exclude)
        refused.append(d.status == "atypical")
    return np.array(top1), np.array(in_list), np.array(refused)


df = cb.load()
sym = df[df.symptomatic & df.temperature.notna()].reset_index(drop=True)
y = (sym.covid19_test_results == "Positive").to_numpy().astype(int)
print(f"Пациентов с симптомами и измеренной температурой: {len(sym)}; COVID-19+: {y.sum()}, COVID-19−: {len(y) - y.sum()}")
print(f"Средняя температура: COVID+ {sym.temperature[y == 1].mean():.2f}, COVID− {sym.temperature[y == 0].mean():.2f}; "
      f"доля ≥ 38 °C: {np.mean(sym.temperature[y == 1] >= 38):.3f} / {np.mean(sym.temperature[y == 0] >= 38):.3f}")

print("\n=== 1. Выделение COVID-19: AUC оценки S_COVID − max(S_других) ===")
for metric in m.METRICS:
    for mode in ["fuzzy", "crisp", "report"]:
        X = cb.to_fuzzy(sym, mode)
        s = cb.disease_score(cb.similarities(X, metric=metric), cb.COVID)
        a, lo, hi = auc_ci(y, s, 300)
        print(f"{metric:10s} температура={mode:6s}  AUC = {a:.3f} [{lo:.3f}; {hi:.3f}]")

print("\n=== 2. Нечёткая vs чёткая температура (Хэмминг), парный бутстрэп разности AUC ===")
sf = cb.disease_score(cb.similarities(cb.to_fuzzy(sym, "fuzzy")), cb.COVID)
sc = cb.disease_score(cb.similarities(cb.to_fuzzy(sym, "crisp")), cb.COVID)
diffs = []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y))
    diffs.append(roc_auc_score(y[i], sf[i]) - roc_auc_score(y[i], sc[i]))
print(f"ΔAUC (нечёткая − чёткая) = {roc_auc_score(y, sf) - roc_auc_score(y, sc):.4f} "
      f"[{np.percentile(diffs, 2.5):.4f}; {np.percentile(diffs, 97.5):.4f}], доля бутстрэп-выборок с ΔAUC > 0: {np.mean(np.array(diffs) > 0):.3f}")

print("\n=== 3. Решения алгоритма (Хэмминг, α = 0,6, δ = 0,05, нечёткая температура) ===")
X = cb.to_fuzzy(sym, "fuzzy")
t1, il, rf = decisions(X, cb.COVID)
print(f"COVID+: top-1 = COVID {t1[y == 1].mean():.3f}; COVID среди ответов {il[y == 1].mean():.3f}; отказы {rf[y == 1].mean():.3f}")
print(f"COVID−: top-1 = COVID {t1[y == 0].mean():.3f}; COVID среди ответов {il[y == 0].mean():.3f}; отказы {rf[y == 0].mean():.3f}")

print("\n=== 4. Эталон COVID-19 2020 года vs «омикрон» (на данных 2020 года) ===")
E_om = m.ETALONS.copy(); E_om[cb.COVID] = m.COVID_OMICRON
s_om = cb.disease_score(cb.similarities(X, E_om), cb.COVID)
print(f"AUC 2020: {roc_auc_score(y, sf):.3f}; AUC «омикрон»: {roc_auc_score(y, s_om):.3f}")

print("\n=== 5. Модель сравнения: логистическая регрессия на тех же 7 признаках (5-кратная CV) ===")
Xl = X[:, :7]
p = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced"), Xl, y,
                      cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
a, lo, hi = auc_ci(y, p, 300)
print(f"Логистическая регрессия: AUC = {a:.3f} [{lo:.3f}; {hi:.3f}]")
print("Вклад отдельных признаков (AUC одного признака):")
for j in range(7):
    print(f"  x{j + 1} {m.FEATURES[j]:28s} AUC = {roc_auc_score(y, X[:, j]):.3f}")

print("\n=== 6. Стрептококковый тонзиллит (экспресс-тест) ===")
st_ = df[df.symptomatic & df.rapid_strep_results.isin(["Positive", "Negative"])].reset_index(drop=True)
ys = (st_.rapid_strep_results == "Positive").to_numpy().astype(int)
Xs = cb.to_fuzzy(st_, "report")          # температура часто не измерена: используем жалобу на лихорадку
ss = cb.disease_score(cb.similarities(Xs), cb.TONSIL)
a, lo, hi = auc_ci(ys, ss, 1000)
t1s, ils, _ = decisions(Xs, cb.TONSIL)
print(f"Протестировано: {len(ys)}, положительных: {ys.sum()}")
print(f"AUC = {a:.3f} [{lo:.3f}; {hi:.3f}]; strep+: top-1 = тонзиллит {t1s[ys == 1].mean():.3f}, среди ответов {ils[ys == 1].mean():.3f}; "
      f"strep−: top-1 = тонзиллит {t1s[ys == 0].mean():.3f}")
