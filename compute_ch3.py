"""Все числа для пп. 3.4 и 3.5 на итоговой версии модели (с правилами (18)–(19)).

Запуск: .venv\\Scripts\\python compute_ch3.py
"""
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

import carbon as cb
import evaluation as ev
import model as m

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
ALPHA = {"Хэмминг": 0.6, "Евклид": 0.52, "Корреляция": None}


def auc_ci(y, s, n=1000):
    base = roc_auc_score(y, s); b = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            b.append(roc_auc_score(y[i], s[i]))
    return base, np.percentile(b, 2.5), np.percentile(b, 97.5)


print("=================== 3.4 CARBON HEALTH ===================")
df = cb.load()
print("всего записей:", len(df), "| с симптомами:", int(df.symptomatic.sum()), "| период:", df.batch_date.min(), "–", df.batch_date.max())
sym = df[df.symptomatic & df.temperature.notna()].reset_index(drop=True)
y = (sym.covid19_test_results == "Positive").to_numpy().astype(int)
print(f"выборка: {len(y)}, COVID+ {y.sum()}, COVID- {len(y) - y.sum()}")
X = cb.to_fuzzy(sym, "fuzzy")
print("частоты признаков COVID+:", np.round(X[y == 1][:, :7].mean(0), 2))
print("частоты признаков COVID-:", np.round(X[y == 0][:, :7].mean(0), 2))
print("доля t>=38: COVID+ %.3f COVID- %.3f; средняя t %.2f / %.2f" % (np.mean(sym.temperature[y == 1] >= 38), np.mean(sym.temperature[y == 0] >= 38), sym.temperature[y == 1].mean(), sym.temperature[y == 0].mean()))

for metric in m.METRICS:
    for mode in ("fuzzy", "crisp", "report"):
        s = cb.disease_score(cb.similarities(cb.to_fuzzy(sym, mode), metric=metric), cb.COVID)
        a, lo, hi = auc_ci(y, s, 300)
        print(f"AUC {metric:10s} {mode:6s} {a:.3f} [{lo:.3f}; {hi:.3f}]")

sf = cb.disease_score(cb.similarities(cb.to_fuzzy(sym, "fuzzy")), cb.COVID)
sc = cb.disease_score(cb.similarities(cb.to_fuzzy(sym, "crisp")), cb.COVID)
d = []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y)); d.append(roc_auc_score(y[i], sf[i]) - roc_auc_score(y[i], sc[i]))
print("ΔAUC нечёткая−чёткая %.4f [%.4f; %.4f], доля >0: %.3f" % (roc_auc_score(y, sf) - roc_auc_score(y, sc), np.percentile(d, 2.5), np.percentile(d, 97.5), np.mean(np.array(d) > 0)))

top1, inl = [], []
for a in X:
    r = m.diagnose(a, weights=cb.WEIGHTS)
    top1.append(r.diagnosis == "COVID-19"); inl.append(r.diagnosis == "COVID-19" or "COVID-19" in r.cannot_exclude)
top1, inl = np.array(top1), np.array(inl)
print("решения: COVID+ top1 %.3f, в ответе %.3f | COVID- top1 %.3f, в ответе %.3f" % (top1[y == 1].mean(), inl[y == 1].mean(), top1[y == 0].mean(), inl[y == 0].mean()))

E_om = m.ETALONS.copy(); E_om[2] = m.COVID_OMICRON
print("AUC эталон 2020 %.3f vs омикрон %.3f" % (roc_auc_score(y, sf), roc_auc_score(y, cb.disease_score(cb.similarities(X, E_om), cb.COVID))))

p = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced"), X[:, :7], y,
                      cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
print("логрегрессия AUC %.3f [%.3f; %.3f]" % auc_ci(y, p, 300))
print("AUC отдельных признаков:", [round(roc_auc_score(y, X[:, j]), 3) for j in range(7)])

res, W = [], []
for seed in range(5):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y, random_state=seed)
    w = np.zeros(9)
    for j in range(7):
        w[j] = max(roc_auc_score(y[tr], X[tr][:, j]) - 0.5, 0)
    W.append(w / w.sum())
    res.append((roc_auc_score(y[te], cb.disease_score(cb.similarities(X[te]), 2)),
                roc_auc_score(y[te], cb.disease_score(cb.similarities(X[te], weights=w), 2))))
res = np.array(res)
print("веса по данным (среднее по 5 разбиениям):", np.round(np.mean(W, 0), 2))
print("AUC на отложенной половине: равные веса %.3f±%.3f, веса по данным %.3f±%.3f" % (res[:, 0].mean(), res[:, 0].std(), res[:, 1].mean(), res[:, 1].std()))

st = df[df.symptomatic & df.rapid_strep_results.isin(["Positive", "Negative"])].reset_index(drop=True)
ys = (st.rapid_strep_results == "Positive").to_numpy().astype(int)
Xs = cb.to_fuzzy(st, "report")
ss = cb.disease_score(cb.similarities(Xs), cb.TONSIL)
print("стрептококк: n=%d, + %d, AUC %.3f [%.3f; %.3f]" % ((len(ys), ys.sum()) + auc_ci(ys, ss, 1000)))
print("доля с болью в горле среди протестированных: %.3f" % st.sore_throat.mean())
print("грипп+: %d, протестировано %d" % ((df.rapid_flu_results == "Positive").sum(), df.rapid_flu_results.isin(["Positive", "Negative"]).sum()))

print("\n=================== 3.5 SYNTHETIC ===================")
Xs, ys = m.synthetic_cases(300)
print("синтетика:", len(ys))
for metric in m.METRICS:
    S = np.array([m.apply_rules(a, np.array([m.METRICS[metric](a, e) for e in m.ETALONS])) for a in Xs])
    st_ = S[np.arange(len(ys)), ys]
    print(f"{metric}: процентили S_true 5/50: {np.percentile(st_, 5):.3f} {np.percentile(st_, 50):.3f}")
    al = ALPHA[metric] if ALPHA[metric] is not None else round(float(np.percentile(st_, 5)), 2)
    pt, ci, cm, pc, _ = ev.evaluate(Xs, ys, alpha=al, metric=metric, n_boot=300)
    print(f"   α={al}: " + "; ".join(f"{k} {v:.3f} [{ci[k][0]:.3f}; {ci[k][1]:.3f}]" if k in ci else f"{k} {v:.3f}" for k, v in pt.items()))
    if metric == "Хэмминг":
        print(pc.round(3).to_string(index=False)); print(cm.to_string())
base = ev.baseline_logreg(Xs, ys)
print("логрегрессия:", {k: round(v, 3) for k, v in base.items()})
g = ev.grid_alpha_delta(Xs, ys, [0.5, 0.55, 0.6, 0.65, 0.7], [0.0, 0.03, 0.05, 0.08])
print(g.round(3).to_string(index=False))
for eps in (0.1, 0.2):
    print("устойчивость ±%.1f:" % eps, ev.robustness_expert(Xs, eps=eps, n_runs=100))
