"""Рисунки для глав 2 и 3: выбор α и δ, матрица ошибок, карта предпочтения COVID-19/грипп."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

import model as m

OUT = Path(__file__).parent / "docs"
OUT.mkdir(exist_ok=True)
FONT = "Times New Roman" if any("Times New Roman" in f.name for f in matplotlib.font_manager.fontManager.ttflist) else "Liberation Serif"
plt.rcParams.update({"font.family": FONT, "font.size": 14, "axes.unicode_minus": True})
COMMA = FuncFormatter(lambda v, p: ("%g" % round(v, 3)).replace(".", ","))
PCT = FuncFormatter(lambda v, p: f"{v:.0f} %")
BLUE, DARK, GRAY = "#2a6fb0", "#1b3a5c", "#8a8a8a"

X, y = m.synthetic_cases(300, seed=0)
s_true = np.array([m.euclid_similarity(a, m.ETALONS[k]) for a, k in zip(X, y)])

# ---------- выбор α и δ ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4))
a1.hist(s_true, bins=np.arange(0.35, 0.96, 0.02), color=BLUE, edgecolor="white", linewidth=0.8)
a1.axvline(m.ALPHA, color=DARK, linewidth=2)
a1.text(m.ALPHA - 0.01, a1.get_ylim()[1] * 0.95, "α = 0,57\n5 % больных\nлевее линии", ha="right", va="top", fontsize=13)
a1.set_xlabel("Степень сходства с эталоном своего заболевания")
a1.set_ylabel("Число больных")
a1.set_title("а) выбор порога α", fontsize=14)
a1.xaxis.set_major_formatter(COMMA)

deltas = np.round(np.arange(0, 0.121, 0.005), 3)
unc, inans = [], []
for dl in deltas:
    res = [m.diagnose(a, delta=dl) for a in X]
    unc.append(100 * np.mean([d.status == "uncertain" for d in res]))
    inans.append(100 * np.mean([m.DISEASES[k] in d.answer() for d, k in zip(res, y)]))
a2.plot(deltas, inans, color=DARK, linewidth=2, marker="o", markersize=4)
a2.plot(deltas, unc, color=BLUE, linewidth=2, linestyle="--", marker="s", markersize=4)
a2.axvline(0.05, color=GRAY, linewidth=1.2, linestyle=":")
i5 = int(np.argmin(abs(deltas - 0.05)))
a2.text(0.004, 60, "верный диагноз\nвошёл в ответ", ha="left", va="top", fontsize=13, color=DARK)
a2.text(0.07, unc[14] - 22, "доля ответов\n«нельзя исключить»", ha="left", va="top", fontsize=13, color=BLUE)
a2.text(0.052, 3, f"δ = 0,05: {inans[i5]:.0f} % и {unc[i5]:.0f} %", ha="left", va="bottom", fontsize=13)
a2.set_xlabel("Минимальный разрыв δ")
a2.set_ylabel("Доля больных")
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

# ---------- матрица ошибок ----------
cm = np.zeros((3, 4), dtype=int)
for a, k in zip(X, y):
    d = m.diagnose(a)
    cm[k, m.DISEASES.index(d.diagnosis) if d.diagnosis in m.DISEASES else 3] += 1
fig, ax = plt.subplots(figsize=(8.2, 4.0))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=300)
for i in range(3):
    for j in range(4):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14, color="white" if cm[i, j] > 150 else "black")
ax.set_xticks(range(4), ["ОРВИ", "Грипп", "COVID-19", "Нетипично\nили правило"], fontsize=12)
ax.set_yticks(range(3), m.DISEASES)
ax.set_xlabel("Основной диагноз модели")
ax.set_ylabel("Истинное заболевание")
ax.set_xticks(np.arange(-0.5, 4), minor=True)
ax.set_yticks(np.arange(-0.5, 3), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="minor", length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
cb.set_label("Число больных", labelpad=14)
fig.tight_layout()
fig.savefig(OUT / "fig_confusion.png", dpi=220)
plt.close(fig)

# ---------- карта: S(COVID-19) − S(грипп) ----------
prof = np.array([0, 0.5, 0.25, 0.25, 0.5, 0.75, 0, 0, 1.0])
temps = np.linspace(37.0, 40.0, 121)
smells = np.linspace(0, 1, 101)
Z = np.zeros((len(smells), len(temps)))
for i, sm in enumerate(smells):
    for j, t in enumerate(temps):
        a3 = prof.copy(); a3[0] = m.temperature_membership(t); a3[6] = sm
        Z[i, j] = m.euclid_similarity(a3, m.ETALONS[2]) - m.euclid_similarity(a3, m.ETALONS[1])
fig, ax = plt.subplots(figsize=(7.6, 4.6))
lim = float(np.abs(Z).max())
pc = ax.pcolormesh(temps, smells, Z, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), shading="auto")
cs = ax.contour(temps, smells, Z, levels=[0], colors="black", linewidths=2.2)
ax.contour(temps, smells, Z, levels=[-0.05, 0.05], colors="black", linewidths=1, linestyles="--")
ax.text(37.15, 0.93, "ближе COVID-19", fontsize=13, va="top")
ax.text(38.9, 0.12, "ближе грипп", fontsize=13)
ax.set_xlabel("Температура тела, °C")
ax.set_ylabel("Снижение обоняния μ(x₇)")
ax.xaxis.set_major_formatter(COMMA)
ax.yaxis.set_major_formatter(COMMA)
cb = fig.colorbar(pc, ax=ax, pad=0.03)
cb.set_label("S(COVID-19) − S(грипп)")
cb.ax.yaxis.set_major_formatter(COMMA)
fig.tight_layout()
fig.savefig(OUT / "fig_surface.png", dpi=220)
plt.close(fig)
print("готово:", FONT)
