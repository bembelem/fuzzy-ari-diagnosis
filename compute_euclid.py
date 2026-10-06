"""Числа для перевода работы на евклидову меру (пп. 2.3, 2.4, 3.3–3.5)."""
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import carbon as cb
import evaluation as ev
import model as m
import textbook_cases as tc

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
E, A = "Евклид", 0.52

print("== 2.3 таблица 6 (Евклид)")
print(np.round(m.etalon_similarity_matrix(metric=E), 2))

print("\n== 2.4 пример")
a = np.array([0.8, 0.5, 0.25, 0.25, 0.75, 0.75, 0, 0, 1])
for k, e in enumerate(m.ETALONS):
    sq = np.sum((a - e) ** 2)
    print(f"{m.DISEASES[k]:22s} Σsq={sq:.3f} d={np.sqrt(sq / 9):.3f} S={1 - np.sqrt(sq / 9):.3f}")
d = m.diagnose(a, alpha=A, metric=E)
print(d.status, d.diagnosis, round(d.gap, 3), d.cannot_exclude, [(x[0], round(x[3], 2)) for x in d.explanation])

print("\n== чувствительность S к изменению одного признака на 0,25 (синтетика)")
X, y = m.synthetic_cases(300)
ch = []
for x, t in zip(X[:600], y[:600]):
    s0 = m.euclid_similarity(x, m.ETALONS[t])
    for i in range(9):
        x2 = x.copy(); x2[i] = min(1, x2[i] + 0.25) if x2[i] <= 0.75 else x2[i] - 0.25
        ch.append(abs(m.euclid_similarity(x2, m.ETALONS[t]) - s0))
print("среднее |ΔS| %.3f, медиана %.3f, 90-й процентиль %.3f" % (np.mean(ch), np.median(ch), np.percentile(ch, 90)))

print("\n== 3.3 задачи (Евклид, α=0.52)")
cases = {c["id"]: c for c in tc.load_cases()}
for use_exam in (False, True):
    r = tc.run(E, group="проверка", use_exam=use_exam)
    typ = [x for x in r if not (cases[x["id"]]["эталон"] == "нет среди пяти" or "атипич" in cases[x["id"]]["тип"])]
    print("осмотр", use_exam, "| типичные верно %d/%d, в списке %d" % (sum(x["итог"] == "верно" for x in typ), len(typ), sum(x["итог"] in ("верно", "в списке") for x in typ)))
    by = {}
    for x in typ:
        k = cases[x["id"]]["эталон"]; by.setdefault(k, [0, 0, 0]); by[k][0] += 1; by[k][1] += x["итог"] == "верно"; by[k][2] += x["итог"] in ("верно", "в списке")
    print("   ", by)
    for x in r:
        print("   ", x["id"], "|", x["ответ"], "|", x["итог"], "|", x["сходство"])
for g in ("разработка",):
    for rules in (False, True):
        out = []
        for c in tc.load_cases():
            if c["группа"] != g:
                continue
            av, w, ex = tc.case_vector(c)
            out.append((c["id"], tc.verdict(c, m.diagnose(av, alpha=A, metric=E, weights=w, rules=rules))))
        print("разработка rules", rules, out)
for rules in (False, True):
    res = [m.diagnose(x, alpha=A, metric=E, rules=rules) for x in X]
    t1 = np.mean([r.diagnosis == m.DISEASES[t] for r, t in zip(res, y)])
    il = np.mean([(r.diagnosis == m.DISEASES[t]) or (m.DISEASES[t] in r.cannot_exclude) for r, t in zip(res, y)])
    print("синтетика rules", rules, "top1 %.3f в ответе %.3f" % (t1, il))

print("\n== 3.4 Carbon (Евклид)")
df = cb.load(); sym = df[df.symptomatic & df.temperature.notna()].reset_index(drop=True)
yc = (sym.covid19_test_results == "Positive").to_numpy().astype(int)
Xc = cb.to_fuzzy(sym, "fuzzy")
for rules in (False, True):
    print("AUC rules", rules, round(roc_auc_score(yc, cb.disease_score(cb.similarities(Xc, metric=E, rules=rules), 2)), 3))
sf = cb.disease_score(cb.similarities(Xc, metric=E), 2)
sc = cb.disease_score(cb.similarities(cb.to_fuzzy(sym, "crisp"), metric=E), 2)
dd = []
for _ in range(1000):
    i = rng.integers(0, len(yc), len(yc)); dd.append(roc_auc_score(yc[i], sf[i]) - roc_auc_score(yc[i], sc[i]))
print("ΔAUC fuzzy-crisp %.4f [%.4f; %.4f] P>0 %.3f" % (roc_auc_score(yc, sf) - roc_auc_score(yc, sc), np.percentile(dd, 2.5), np.percentile(dd, 97.5), np.mean(np.array(dd) > 0)))
t1, il = [], []
for x in Xc:
    r = m.diagnose(x, alpha=A, metric=E, weights=cb.WEIGHTS)
    t1.append(r.diagnosis == "COVID-19"); il.append(r.diagnosis == "COVID-19" or "COVID-19" in r.cannot_exclude)
t1, il = np.array(t1), np.array(il)
print("решения COVID+ top1 %.3f в ответе %.3f | COVID- top1 %.3f" % (t1[yc == 1].mean(), il[yc == 1].mean(), t1[yc == 0].mean()))
Eo = m.ETALONS.copy(); Eo[2] = m.COVID_OMICRON
print("омикрон AUC %.3f" % roc_auc_score(yc, cb.disease_score(cb.similarities(Xc, Eo, metric=E), 2)))
res, W = [], []
for seed in range(5):
    tr, te = train_test_split(np.arange(len(yc)), test_size=0.5, stratify=yc, random_state=seed)
    w = np.zeros(9)
    for j in range(7):
        w[j] = max(roc_auc_score(yc[tr], Xc[tr][:, j]) - 0.5, 0)
    res.append((roc_auc_score(yc[te], cb.disease_score(cb.similarities(Xc[te], metric=E), 2)),
                roc_auc_score(yc[te], cb.disease_score(cb.similarities(Xc[te], metric=E, weights=w), 2))))
res = np.array(res)
print("веса: равные %.3f±%.3f, по данным %.3f±%.3f" % (res[:, 0].mean(), res[:, 0].std(), res[:, 1].mean(), res[:, 1].std()))
st = df[df.symptomatic & df.rapid_strep_results.isin(["Positive", "Negative"])].reset_index(drop=True)
ys = (st.rapid_strep_results == "Positive").to_numpy().astype(int)
ss = cb.disease_score(cb.similarities(cb.to_fuzzy(st, "report"), metric=E), 3)
b = []
for _ in range(1000):
    i = rng.integers(0, len(ys), len(ys))
    if ys[i].min() != ys[i].max():
        b.append(roc_auc_score(ys[i], ss[i]))
print("strep AUC %.3f [%.3f; %.3f]" % (roc_auc_score(ys, ss), np.percentile(b, 2.5), np.percentile(b, 97.5)))

print("\n== 3.5 синтетика (Евклид)")
g = ev.grid_alpha_delta(X, y, [0.45, 0.5, 0.52, 0.55, 0.6], [0.0, 0.03, 0.05, 0.08], metric=E)
print(g.round(3).to_string(index=False))
pt, ci, cm, pc, _ = ev.evaluate(X, y, alpha=A, metric=E, n_boot=300)
print({k: round(v, 3) for k, v in pt.items()})
print(pc.round(3).to_string(index=False)); print(cm.to_string())
for eps in (0.1, 0.2):
    print("устойчивость ±%.1f" % eps, ev.robustness_expert(X, eps=eps, n_runs=100, alpha=A, metric=E))
