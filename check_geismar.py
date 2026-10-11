"""Проверка отклонённого варианта эталонов по Geismar et al. (2023): частоты из двух разных когорт (п. 2.3).
Запуск: python check_geismar.py"""
import sys, warnings; warnings.filterwarnings("ignore")
import os; REPO = os.path.dirname(os.path.abspath(__file__)); GAL = os.path.join(REPO, "data", "virome_raw")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import carbon as cb, textbook_cases as tc, model as m0
rng = np.random.default_rng(0)

FEAT = ["лихорадка", "кашель", "насморк", "боль в горле", "слабость", "обоняние", "чихание/зуд"]
IDX9 = [0, 1, 2, 3, 5, 6, 7]                       # из старых x1..x9 остаются x1,x2,x3,x4,x6,x7,x8
DIS = ["ОРВИ", "Грипп", "COVID-19"]
# p: Geismar et al. 2023, табл. 1 (ПЦР, общественная когорта); ОРВИ = риновирус+сез. CoV+RSV (взвеш. по n);
# COVID-19: исходный вариант (2020); обоняние: COVID-19 40 % (там же), ОРВИ и грипп: 22 % у SARS-CoV-2-отриц. (Menni 2020)
w = np.array([283, 191, 84.]); w /= w.sum()
orvi = w @ np.array([[.47, .87, .93, .81, .28, .22, .90], [.51, .79, .95, .72, .10, .22, .90], [.38, .83, .88, .58, .60, .22, .77]])
P = np.array([orvi, [.74, .91, .89, .72, .54, .22, .76], [.23, .65, .35, .40, .71, .40, .36]])
KAPPA = np.array([.46, .56, .50, .53, .53, .53, .53])   # Virome of Manhattan, когорты D1, D2, SC
E = np.round(P * KAPPA, 2)
THETA = 0.5

def sim(a, e, wt):
    return 1 - np.sqrt(np.sum(wt * (a - e) ** 2) / np.sum(wt))

def rules(a, exam=None):
    """Степени истинности правил (операции min и дополнения, п. 1.2)."""
    ton = min(a[3], 1 - a[1])                      # сильная боль в горле И нет кашля
    if exam is not None and not np.all(np.isnan(exam)):
        ton = min(ton, np.nanmax(exam))            # И налёт или лимфоузлы при осмотре
    ar = min(a[6], 1 - a[0])                       # чихание/зуд И нет лихорадки
    return {"Острый тонзиллит": ton, "Аллергический ринит": ar}

def diagnose(a, wt, alpha, delta=0.05, exam=None):
    s = np.array([sim(a, e, wt) for e in E]); o = np.argsort(-s)
    flags = [k for k, v in rules(a, exam).items() if v >= THETA]
    if s[o[0]] < alpha:
        return (flags[0] if flags else "нетипично"), [], flags, s
    others = [DIS[k] for k in o[1:] if s[k] >= alpha and s[o[0]] - s[k] < delta]
    return DIS[o[0]], others, flags, s

# ---------- синтетика (3 вирусных) → α ----------
terms = np.array([.25, .5, .75, 1.])
def tprobs(k):
    lo, hi = -50., 50.
    for _ in range(100):
        l = (lo + hi) / 2; p = np.exp(l * terms); p /= p.sum(); lo, hi = (l, hi) if p @ terms < k else (lo, l)
    return p
TP = [tprobs(k) for k in KAPPA]
Xs, ys = [], []
for k in range(3):
    for _ in range(500):
        pres = rng.random(7) < P[k]
        Xs.append([rng.choice(terms, p=TP[i]) if pres[i] else 0 for i in range(7)]); ys.append(k)
Xs, ys = np.array(Xs), np.array(ys); ones = np.ones(7)
s_true = np.array([sim(a, E[k], ones) for a, k in zip(Xs, ys)])
ALPHA = round(float(np.percentile(s_true, 5)), 2)
print("Эталоны (p·κ):"); [print(f"  {d:9s}", E[i]) for i, d in enumerate(DIS)]
print("Попарное сходство:", np.round([[sim(E[i], E[j], ones) for j in range(3)] for i in range(3)], 2).tolist())
ch = []
for a, k in zip(Xs[:600], ys[:600]):
    s0 = sim(a, E[k], ones)
    for i in range(7):
        b = a.copy(); b[i] = b[i] + .25 if b[i] <= .75 else b[i] - .25; ch.append(abs(sim(b, E[k], ones) - s0))
print(f"α = {ALPHA} (5-й процентиль); |ΔS| при сдвиге на градацию: среднее {np.mean(ch):.3f}, 90-й процентиль {np.percentile(ch, 90):.3f}")
res = [diagnose(a, ones, ALPHA) for a in Xs]
print("Синтетика: верно %.1f%%, в ответе %.1f%%, отказ %.1f%%, ложные флаги ТОНЗ %.1f%% АР %.1f%%" % (
    100 * np.mean([r[0] == DIS[k] for r, k in zip(res, ys)]), 100 * np.mean([r[0] == DIS[k] or DIS[k] in r[1] for r, k in zip(res, ys)]),
    100 * np.mean([r[0] == "нетипично" for r in res]), 100 * np.mean(["Острый тонзиллит" in r[2] for r in res]), 100 * np.mean(["Аллергический ринит" in r[2] for r in res])))
cm = np.zeros((3, 4), int)
for r, k in zip(res, ys): cm[k, DIS.index(r[0]) if r[0] in DIS else 3] += 1
print("  матрица ошибок (ОРВИ, грипп, COVID, нетипично):", cm.tolist())

# ---------- Carbon Health: COVID-19 против остальных ОРЗ (грипп в 2020 почти не циркулировал) ----------
df = cb.load()
def carbon_vec(d):
    X = np.zeros((len(d), 7))
    X[:, 0] = d["fever"].astype(float) * KAPPA[0]               # жалоба на лихорадку, как в Geismar
    X[:, 1] = np.where(d["cough"], d["cough_severity"].map(cb.COUGH_SEVERITY).fillna(.5), 0)
    X[:, 2] = d["runny_nose"] * KAPPA[2]; X[:, 3] = d["sore_throat"] * KAPPA[3]
    X[:, 4] = d["fatigue"] * KAPPA[4]; X[:, 5] = (d["loss_of_smell"] | d["loss_of_taste"]) * KAPPA[5]
    return X.astype(float)
wc = np.array([1, 1, 1, 1, 1, 1, 0.])                              # чихания в данных нет
sym = df[df.symptomatic].reset_index(drop=True)
y = (sym.covid19_test_results == "Positive").to_numpy().astype(int)
Xc = carbon_vec(sym)
Sc = np.array([[sim(a, e, wc) for e in E] for a in Xc])
sc = Sc[:, 2] - Sc[:, :2].max(1)
def ci(y, s, n=500):
    b = [roc_auc_score(y[i], s[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return roc_auc_score(y, s), np.percentile(b, 2.5), np.percentile(b, 97.5)
print("\nCarbon Health (n=%d, COVID+ %d): AUC COVID-19 против ОРВИ/гриппа %.3f [%.3f; %.3f]" % ((len(y), y.sum()) + ci(y, sc)))
d = [diagnose(a, wc, ALPHA) for a in Xc]
print("  решения: COVID+ → COVID первым %.1f%%, в ответе %.1f%% | COVID− → COVID первым %.1f%%" % (
    100 * np.mean([r[0] == "COVID-19" for r, t in zip(d, y) if t]), 100 * np.mean([r[0] == "COVID-19" or "COVID-19" in r[1] for r, t in zip(d, y) if t]),
    100 * np.mean([r[0] == "COVID-19" for r, t in zip(d, y) if not t])))
st = df[df.symptomatic & df.rapid_strep_results.isin(["Positive", "Negative"])].reset_index(drop=True)
ys_ = (st.rapid_strep_results == "Positive").to_numpy().astype(int)
ton = np.array([rules(a)["Острый тонзиллит"] for a in carbon_vec(st)])
print("  стрептококк (n=%d, +%d): AUC правила тонзиллита %.3f [%.3f; %.3f]" % ((len(ys_), ys_.sum()) + ci(ys_, ton)))

# ---------- Virome of Manhattan, взрослые ----------
meta = pd.read_csv(os.path.join(REPO, "data", "virome", "samples_metadata.csv"), keep_default_na=False); meta["date"] = pd.to_datetime(meta.Date_collected, format="%d-%b-%y")
SS = {}
for c in ["MC", "ER"]:
    for k, v in pd.read_excel(os.path.join(REPO, "data", "virome", "symptoms", f"sympt_{c}.xlsx"), sheet_name=None).items():
        v = v.rename(columns={v.columns[0]: "date"}); v["date"] = pd.to_datetime(v["date"], errors="coerce"); SS[str(k).strip()] = v.dropna(subset=["date"])
Xg = []
for _, r in meta.iterrows():
    t = r.NEW_Infection_type
    if r.ID not in SS or not (t.startswith("Rhino") or t.startswith("Corona") or t.startswith("Respiratory")): continue
    v = SS[r.ID]; win = v[(v.date >= r.date - pd.Timedelta(days=3)) & (v.date <= r.date + pd.Timedelta(days=3))]
    if len(win) < 4: continue
    mx = win.iloc[:, 1:].astype(float).max()
    if mx.sum() == 0: continue
    Xg.append(np.array([mx.Fever, mx.Cough, mx.RunnyNose, mx.SoreThroat, 0, 0, max(mx.Sneezing, mx.WateryEye)]) / 3)
wg = np.array([1, 1, 1, 1, 0, 0, 1.])
dg = [diagnose(a, wg, ALPHA) for a in Xg]
print("\nVirome of Manhattan, взрослые с ОРВИ (n=%d): ОРВИ первой %.0f%%, в ответе %.0f%%, нетипично %.0f%%; флаг АР %.0f%%, флаг тонзиллита %.0f%%" % (
    len(Xg), 100 * np.mean([r[0] == "ОРВИ" for r in dg]), 100 * np.mean([r[0] == "ОРВИ" or "ОРВИ" in r[1] for r in dg]),
    100 * np.mean([r[0] in ("нетипично",) for r in dg]), 100 * np.mean(["Аллергический ринит" in r[2] for r in dg]), 100 * np.mean(["Острый тонзиллит" in r[2] for r in dg])))

# ---------- ситуационные задачи ----------
cases = {c["id"]: c for c in tc.load_cases()}
for exam_on in (False, True):
    print(f"\nСитуационные задачи (проверочная группа), осмотр: {'да' if exam_on else 'нет'}")
    tot = ok = inl = 0
    for c in cases.values():
        if c["группа"] != "проверка": continue
        a9, w9, ex = tc.case_vector(c)
        a, wt = a9[IDX9], w9[IDX9]
        r = diagnose(a, wt, ALPHA, exam=ex if exam_on else None)
        truth = c["эталон"]; typ = "атипич" in c["тип"] or truth == "нет среди пяти"
        if truth in DIS: good, inlist = r[0] == truth, r[0] == truth or truth in r[1]
        elif truth == "нет среди пяти": good = inlist = r[0] == "нетипично"
        else: good = r[0] == truth; inlist = good or truth in r[2]
        mark = "верно" if good else ("в списке" if inlist else "ошибка")
        if not typ: tot += 1; ok += good; inl += inlist
        print(f"  {c['id']:4s} {'нетип.' if typ else 'тип.  '} {truth[:22]:22s} → {r[0]:18s} | можно не искл.: {r[1]} | флаги: {r[2]} | {mark}")
    print(f"  ТИПИЧНЫЕ: верно {ok}/{tot}, верно или в ответе {inl}/{tot}")
