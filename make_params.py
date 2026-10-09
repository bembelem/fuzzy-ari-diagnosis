"""Рисунки для п. 3.5: выбор параметров α и δ и матрица ошибок на синтетической выборке."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

import model as m

OUT = Path(__file__).parent / "docs"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12, "axes.unicode_minus": True})
COMMA = FuncFormatter(lambda v, p: ("%g" % round(v, 3)).replace(".", ","))
PCT = FuncFormatter(lambda v, p: f"{v:.0f} %")
BLUE, DARK, GRAY = "#2a6fb0", "#1b3a5c", "#8a8a8a"

X, y = m.synthetic_cases(300, seed=0)
dec = [m.diagnose(a, alpha=0) for a in X]
s_true = np.array([d.similarities[k] for d, k in zip(dec, y)])
alpha = float(np.percentile(s_true, 5))
print(f"5-й процентиль: {alpha:.3f}")

# ---------- рисунок: выбор α и δ ----------
plt.rcParams["font.size"] = 14
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4))
a1.hist(s_true, bins=np.arange(0.35, 0.96, 0.02), color=BLUE, edgecolor="white", linewidth=0.8)
a1.axvline(m.ALPHA, color=DARK, linewidth=2)
a1.text(m.ALPHA - 0.008, a1.get_ylim()[1] * 0.93, "α = 0,52\n5 % пациентов\nлевее линии",
        ha="right", va="top", fontsize=13)
a1.set_xlim(0.26, 0.92)
a1.set_xlabel("Степень сходства с эталоном своего заболевания")
a1.set_ylabel("Число пациентов")
a1.set_title("а) выбор порога α", fontsize=14)
a1.xaxis.set_major_formatter(COMMA)

deltas = np.round(np.arange(0, 0.121, 0.005), 3)
unc, inans = [], []
for dl in deltas:
    res = [m.diagnose(a, delta=dl) for a in X]                 # те же правила, что в модели
    unc.append(100 * np.mean([d.status == "uncertain" for d in res]))
    inans.append(100 * np.mean([d.diagnosis == m.DISEASES[k] or m.DISEASES[k] in d.cannot_exclude
                                for d, k in zip(res, y)]))
a2.plot(deltas, inans, color=DARK, linewidth=2, marker="o", markersize=4)
a2.plot(deltas, unc, color=BLUE, linewidth=2, linestyle="--", marker="s", markersize=4)
a2.axvline(0.05, color=GRAY, linewidth=1.2, linestyle=":")
i5 = int(np.argmin(abs(deltas - 0.05)))
a2.text(0.004, inans[1] - 5, "верный диагноз\nвошёл в ответ", ha="left", va="top", fontsize=13, color=DARK)
a2.text(0.066, unc[13] - 14, "доля ответов\n«нельзя исключить»", ha="left", va="top", fontsize=13, color=BLUE)
a2.text(0.052, 4, f"δ = 0,05: {inans[i5]:.0f} % и {unc[i5]:.0f} %", ha="left", va="bottom", fontsize=13)
a2.set_xlabel("Минимальный разрыв δ")
a2.set_ylabel("Доля пациентов")
a2.set_ylim(0, 100)
a2.set_title("б) выбор разрыва δ", fontsize=14)
a2.xaxis.set_major_formatter(COMMA)
a2.yaxis.set_major_formatter(PCT)
for ax in (a1, a2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e3e3e3", linewidth=0.8)
    ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "fig_params.png", dpi=220)
plt.close(fig)
print("δ:", {float(d): (round(u, 1), round(h, 1)) for d, u, h in zip(deltas, unc, inans) if d in (0.03, 0.05, 0.08)})

# ---------- рисунок: матрица ошибок ----------
names = ["ОРВИ", "Грипп", "COVID-19", "Тонзиллит", "Аллерг.\nринит"]
cm = np.zeros((5, 6), dtype=int)
for a, k in zip(X, y):
    d = m.diagnose(a)
    cm[k, 5 if d.status == "atypical" else d.order[0]] += 1
fig, ax = plt.subplots(figsize=(7.6, 4.8))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=300)
for i in range(5):
    for j in range(6):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
                color="white" if cm[i, j] > 150 else "black")
ax.set_xticks(range(6), ["ОРВИ", "Грипп", "COVID-19", "Тонзил-\nлит", "Аллерг.\nринит", "Нети-\nпично"])
ax.set_yticks(range(5), names)
ax.set_xlabel("Ответ модели")
ax.set_ylabel("Истинное заболевание")
ax.set_xticks(np.arange(-0.5, 6), minor=True)
ax.set_yticks(np.arange(-0.5, 5), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="minor", length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
cb.set_label("Число пациентов")
fig.tight_layout()
fig.savefig(OUT / "fig_confusion.png", dpi=220)
plt.close(fig)
print(cm)
print("accuracy", round(np.trace(cm[:, :5]) / cm.sum() * 100, 1))
