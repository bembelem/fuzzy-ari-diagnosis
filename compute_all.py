"""Все числа для глав 2 и 3 итоговой (гибридной) версии модели.

Запуск: python compute_all.py   (результаты печатаются и сохраняются в results.json)
"""
import json
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

import carbon as cb
import model as m
import textbook_cases as tc
import virome as vm

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
R = {}
ALPHAS = {}


def auc_ci(y, s, n=1000):
    b = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            b.append(roc_auc_score(y[i], s[i]))
    return [round(roc_auc_score(y, s), 3), round(np.percentile(b, 2.5), 3), round(np.percentile(b, 97.5), 3)]


def paired_auc(y, s1, s0, n=1000):
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s0[i]))
    return [round(roc_auc_score(y, s1) - roc_auc_score(y, s0), 4), round(np.percentile(d, 2.5), 4), round(np.percentile(d, 97.5), 4)]


# =============== 2.3 эталоны ===============
Xb = vm.episodes(vm.BUILD)
p, k, mm = vm.kappa_and_mean(Xb)
R["virome_build"] = {"n": len(Xb), "p": np.round(p, 2).tolist(), "kappa": np.round(k, 2).tolist(), "m": np.round(mm, 2).tolist()}
R["kappa"] = m.KAPPA.tolist()
R["etalons"] = m.ETALONS.tolist()
R["frequencies"] = m.FREQUENCIES.tolist()
R["etalon_similarity"] = np.round(m.etalon_similarity_matrix(), 2).tolist()

# =============== 2.4 пример и обоснование δ ===============
a = np.array([0.8, 0.5, 0.25, 0.25, 0.75, 0.75, 0, 0, 1])
ex = []
for kk, e in enumerate(m.ETALONS):
    sq = float(np.sum((a - e) ** 2))
    ex.append([m.DISEASES[kk], round(sq, 3), round(np.sqrt(sq / 9), 3), round(1 - np.sqrt(sq / 9), 3)])
d = m.diagnose(a)
R["example"] = {"rows": ex, "status": d.status, "diagnosis": d.diagnosis, "gap": round(d.gap, 3),
                "cannot_exclude": d.cannot_exclude, "rules": {kk: round(v, 2) for kk, v in m.rule_degrees(a).items()},
                "contrast": [(f, pa, pe, round(df, 2)) for f, pa, pe, df in d.contrast],
                "explanation": [(f, pa, pe, round(df, 2)) for f, pa, pe, df in d.explanation]}
for smell in (0.25, 0.5):
    a2 = a.copy(); a2[6] = smell
    d2 = m.diagnose(a2)
    R["example"][f"smell_{smell}"] = {"gap": round(d2.gap, 3), "status": d2.status, "diagnosis": d2.diagnosis,
                                      "cannot_exclude": d2.cannot_exclude,
                                      "S": np.round(d2.similarities, 3).tolist()}

Xs, ys = m.synthetic_cases(300, seed=0)
ch = []
for x, t in zip(Xs[:600], ys[:600]):
    s0 = m.euclid_similarity(x, m.ETALONS[t])
    for i in range(9):
        x2 = x.copy(); x2[i] = x2[i] + 0.25 if x2[i] <= 0.75 else x2[i] - 0.25
        ch.append(abs(m.euclid_similarity(x2, m.ETALONS[t]) - s0))
R["delta_sens"] = [round(np.mean(ch), 3), round(np.percentile(ch, 90), 3)]

# =============== 3.5 синтетика (здесь, т. к. α нужен дальше) ===============
for met in m.METRICS:
    st = [m.METRICS[met](x, m.ETALONS[t]) for x, t in zip(Xs, ys)]
    ALPHAS[met] = round(float(np.percentile(st, 5)), 2)
R["alphas"] = ALPHAS


def syn_metrics(met, alpha, X=Xs, y=ys):
    dec = [m.diagnose(x, metric=met, alpha=alpha) for x in X]
    pred = np.array([m.DISEASES.index(dd.diagnosis) if dd.diagnosis in m.DISEASES else -1 for dd in dec])
    top2 = np.array([dd.order[:2] for dd in dec])
    out = {"acc": float(np.mean(pred == y)), "f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro")),
           "kappa": float(cohen_kappa_score(y, pred)), "top2": float(np.mean([y[i] in top2[i] for i in range(len(y))])),
           "refuse": float(np.mean(pred == -1)), "uncertain": float(np.mean([dd.status == "uncertain" for dd in dec])),
           "in_answer": float(np.mean([m.DISEASES[t] in dd.answer() for dd, t in zip(dec, y)])),
           "flag_ton": float(np.mean([any(f == "Острый тонзиллит" for f, _ in dd.flags) for dd in dec])),
           "flag_ar": float(np.mean([any(f == "Аллергический ринит" for f, _ in dd.flags) for dd in dec]))}
    b = []
    for _ in range(500):
        i = rng.integers(0, len(y), len(y)); b.append(np.mean(pred[i] == y[i]))
    out["acc_ci"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    return {kk: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 3)) for kk, v in out.items()}, pred


R["synthetic"] = {}
preds = {}
for met in m.METRICS:
    R["synthetic"][met], preds[met] = syn_metrics(met, ALPHAS[met])
proba = cross_val_predict(LogisticRegression(max_iter=2000), Xs, ys, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")
pl = proba.argmax(1)
R["synthetic"]["LR"] = {"acc": round(float(np.mean(pl == ys)), 3), "f1": round(float(f1_score(ys, pl, average="macro")), 3),
                        "kappa": round(float(cohen_kappa_score(ys, pl)), 3),
                        "top2": round(float(np.mean([ys[i] in np.argsort(-proba[i])[:2] for i in range(len(ys))])), 3)}
# без порога: евклид против хэмминга (парный бутстрэп)
pe = np.array([np.argmax([m.euclid_similarity(x, e) for e in m.ETALONS]) for x in Xs])
ph = np.array([np.argmax([m.hamming_similarity(x, e) for e in m.ETALONS]) for x in Xs])
dd_ = []
for _ in range(1000):
    i = rng.integers(0, len(ys), len(ys)); dd_.append(np.mean(pe[i] == ys[i]) - np.mean(ph[i] == ys[i]))
R["synthetic"]["noalpha_euclid_minus_hamming"] = [round(float(np.mean(pe == ys)), 3), round(float(np.mean(ph == ys)), 3),
                                                  round(float(np.mean(pe == ys) - np.mean(ph == ys)), 3),
                                                  round(float(np.percentile(dd_, 2.5)), 3), round(float(np.percentile(dd_, 97.5)), 3)]
cmx = np.zeros((3, 4), int)
for pr, t in zip(preds["Евклид"], ys):
    cmx[t, pr if pr >= 0 else 3] += 1
R["synthetic"]["confusion"] = cmx.tolist()
R["synthetic"]["delta_curve"] = {}
for dl in (0.0, 0.03, 0.05, 0.08):
    dec = [m.diagnose(x, delta=dl) for x in Xs]
    R["synthetic"]["delta_curve"][dl] = [round(float(np.mean([dd.status == "uncertain" for dd in dec])), 3),
                                         round(float(np.mean([m.DISEASES[t] in dd.answer() for dd, t in zip(dec, ys)])), 3)]
R["synthetic"]["alpha_refuse"] = {a_: round(float(np.mean([m.euclid_similarity(x, m.ETALONS[t]) < a_ for x, t in zip(Xs, ys)]) if False else
                                                  np.mean([max(m.euclid_similarity(x, e) for e in m.ETALONS) < a_ for x in Xs])), 3)
                                  for a_ in (0.5, 0.57, 0.6, 0.65)}
base = preds["Евклид"]
rob = {}
for rel in (0.1, 0.2):
    same = []
    for r_ in range(100):
        E2 = m.perturb(m.ETALONS, rel, seed=r_)
        p2 = np.array([m.DISEASES.index(dd.diagnosis) if dd.diagnosis in m.DISEASES else -1
                       for dd in (m.diagnose(x, etalons=E2) for x in Xs)])
        same.append(np.mean(p2 == base))
    rob[rel] = [round(float(np.mean(same)), 3), round(float(np.min(same)), 3)]
R["synthetic"]["robustness"] = rob
k_same = []
for r_ in range(50):
    kap = np.clip(m.KAPPA + rng.uniform(-0.1, 0.1, 9) * (m.KAPPA < 1), 0.2, 1)
    E2 = np.vstack([m.ETALONS[0], np.round(m.FREQUENCIES * kap, 2)])
    p2 = np.array([m.DISEASES.index(dd.diagnosis) if dd.diagnosis in m.DISEASES else -1 for dd in (m.diagnose(x, etalons=E2) for x in Xs)])
    k_same.append(np.mean(p2 == base))
R["synthetic"]["robust_kappa"] = [round(float(np.mean(k_same)), 3), round(float(np.min(k_same)), 3)]

# =============== 3.3 ситуационные задачи ===============
R["tasks"] = {}
for use_exam in (False, True):
    rows = tc.run(group="проверка", use_exam=use_exam)
    groups = {}
    for r in rows:
        g = r["эталон"] if r["типичная"] else ("ловушки" if r["эталон"] == "нет среди пяти" else "нетипичные известных")
        groups.setdefault(g, [0, 0, 0])
        groups[g][0] += 1; groups[g][1] += r["итог"] == "верно"; groups[g][2] += r["итог"] in ("верно", "в ответе")
    R["tasks"]["exam" if use_exam else "noexam"] = {"groups": groups, "rows": [[r["id"], r["эталон"], r["ответ"], r["итог"]] for r in rows]}
dev = tc.run(group="разработка", use_exam=False)
R["tasks"]["dev"] = [[r["id"], r["эталон"], r["ответ"], r["итог"]] for r in dev]

# =============== 3.4 Carbon Health ===============
df = cb.load()
sym = df[df.symptomatic & df.temperature.notna()].reset_index(drop=True)
y = (sym.covid19_test_results == "Positive").to_numpy().astype(int)
R["carbon"] = {"records": len(df), "sample": len(y), "pos": int(y.sum()), "neg": int(len(y) - y.sum()),
               "t38_pos": round(float(np.mean(sym.temperature[y == 1] >= 38)), 3),
               "fever_report_pos": round(float(sym.fever[y == 1].mean()), 3), "fever_report_neg": round(float(sym.fever[y == 0].mean()), 3),
               "flu_pos": int((df.rapid_flu_results == "Positive").sum()), "flu_tested": int(df.rapid_flu_results.isin(["Positive", "Negative"]).sum())}
scores = {}
for met in m.METRICS:
    for mode in ("fuzzy", "crisp", "report"):
        s = cb.disease_score(cb.similarities(cb.to_fuzzy(sym, mode), metric=met), cb.COVID)
        scores[(met, mode)] = s
        R["carbon"][f"auc_{met}_{mode}"] = auc_ci(y, s, 500)
R["carbon"]["fuzzy_minus_crisp"] = paired_auc(y, scores[("Евклид", "fuzzy")], scores[("Евклид", "crisp")])
R["carbon"]["euclid_minus_hamming"] = paired_auc(y, scores[("Евклид", "fuzzy")], scores[("Хэмминг", "fuzzy")])
R["carbon"]["report_minus_fuzzy"] = paired_auc(y, scores[("Евклид", "report")], scores[("Евклид", "fuzzy")])
X = cb.to_fuzzy(sym, "fuzzy")
dec = [m.diagnose(x, weights=cb.WEIGHTS) for x in X]
first = np.array([dd.diagnosis == "COVID-19" for dd in dec]); inans = np.array(["COVID-19" in dd.answer() for dd in dec])
R["carbon"]["decisions"] = {"pos_first": round(float(first[y == 1].mean()), 3), "pos_in": round(float(inans[y == 1].mean()), 3),
                            "neg_first": round(float(first[y == 0].mean()), 3), "neg_in": round(float(inans[y == 0].mean()), 3),
                            "flag_ton": round(float(np.mean([any(f == "Острый тонзиллит" for f, _ in dd.flags) for dd in dec])), 3),
                            "flag_ar": round(float(np.mean([any(f == "Аллергический ринит" for f, _ in dd.flags) for dd in dec])), 3),
                            "atypical": round(float(np.mean([dd.status == "atypical" for dd in dec])), 3)}
Eo = m.ETALONS.copy(); Eo[2] = m.COVID_OMICRON
R["carbon"]["omicron"] = round(roc_auc_score(y, cb.disease_score(cb.similarities(X, Eo), cb.COVID)), 3)
pr = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced"), X[:, :7], y,
                       cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
R["carbon"]["logreg"] = auc_ci(y, pr, 500)
# max-min композиция по частотам (частота по прямому смыслу) против евклидовой меры по эталонам
Pfreq = np.vstack([m.ETALONS[0] / m.KAPPA, m.FREQUENCIES]).clip(0, 1)
Xraw = cb.to_fuzzy(sym, "fuzzy"); Xraw[:, 2:7] = Xraw[:, 2:7] / m.KAPPA[2:7]
kk = cb.WEIGHTS > 0
Smm = np.array([[np.minimum(x[kk], e[kk]).max() for e in Pfreq] for x in Xraw])
R["carbon"]["maxmin"] = round(roc_auc_score(y, cb.disease_score(Smm, cb.COVID)), 3)
res = []
for seed in range(5):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y, random_state=seed)
    w = np.zeros(9)
    for j in range(7):
        w[j] = max(roc_auc_score(y[tr], X[tr][:, j]) - 0.5, 0)
    res.append((roc_auc_score(y[te], cb.disease_score(cb.similarities(X[te]), 2)),
                roc_auc_score(y[te], cb.disease_score(cb.similarities(X[te], weights=w), 2))))
res = np.array(res)
R["carbon"]["weights_split"] = [round(res[:, 0].mean(), 3), round(res[:, 1].mean(), 3)]
st_ = df[df.symptomatic & df.rapid_strep_results.isin(["Positive", "Negative"])].reset_index(drop=True)
yst = (st_.rapid_strep_results == "Positive").to_numpy().astype(int)
ton = np.array([m.rule_degrees(x)["Острый тонзиллит"] for x in cb.to_fuzzy(st_, "report")])
R["carbon"]["strep"] = {"n": len(yst), "pos": int(yst.sum()), "auc": auc_ci(yst, ton, 1000),
                        "sens": round(float(np.mean(ton[yst == 1] >= m.THETA)), 3), "spec": round(float(np.mean(ton[yst == 0] < m.THETA)), 3)}

# =============== 3.4 Virome of Manhattan, взрослые ===============
Xt = vm.episodes(vm.TEST)
wt = (~np.isnan(Xt[0])).astype(float)
Xt0 = np.nan_to_num(Xt)
R["virome_test"] = {"n": len(Xt)}
for met in ("Евклид", "Хэмминг"):
    dec = [m.diagnose(x, weights=wt, metric=met, alpha=ALPHAS[met]) for x in Xt0]
    R["virome_test"][met] = {"first": round(float(np.mean([dd.diagnosis == "ОРВИ" for dd in dec])), 3),
                             "in": round(float(np.mean(["ОРВИ" in dd.answer() for dd in dec])), 3),
                             "flu_first": round(float(np.mean([dd.diagnosis == "Грипп" for dd in dec])), 3),
                             "covid_first": round(float(np.mean([dd.diagnosis == "COVID-19" for dd in dec])), 3),
                             "atyp": round(float(np.mean([dd.status == "atypical" for dd in dec])), 3),
                             "flag_ton": round(float(np.mean([any(f == "Острый тонзиллит" for f, _ in dd.flags) for dd in dec])), 3),
                             "flag_ar": round(float(np.mean([any(f == "Аллергический ринит" for f, _ in dd.flags) for dd in dec])), 3)}
pt, kt, mt = vm.kappa_and_mean(Xt)
R["virome_test"]["kappa_adults"] = np.round(kt, 2).tolist()
R["virome_test"]["m_adults"] = np.round(mt, 2).tolist()

# =============== поверхность отклика ===============
prof = np.array([0, 0.5, 0.25, 0.25, 0.5, 0.75, 0, 0, 1.0])
temps = np.round(np.arange(37.0, 40.01, 0.1), 1); smells = np.round(np.arange(0, 1.001, 0.05), 2)
Z = np.zeros((len(smells), len(temps)))
for i, sm in enumerate(smells):
    for j, t in enumerate(temps):
        a3 = prof.copy(); a3[0] = m.temperature_membership(t); a3[6] = sm
        Z[i, j] = m.euclid_similarity(a3, m.ETALONS[2]) - m.euclid_similarity(a3, m.ETALONS[1])
np.save("surface.npy", Z)
bnd = {}
for t in (37.0, 37.5, 38.0, 38.5, 39.0, 40.0):
    j = int(np.argmin(abs(temps - t)))
    pos = np.where(Z[:, j] > 0)[0]
    bnd[t] = float(smells[pos[0]]) if len(pos) else None
R["surface_threshold_smell"] = bnd

json.dump(R, open("results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
