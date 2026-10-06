"""Поверхности отклика для п. 3.5 (рисунки 6 и 7)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import model as m

OUT = Path(__file__).parent / "docs"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12, "axes.unicode_minus": True})
from matplotlib.ticker import FuncFormatter
COMMA = FuncFormatter(lambda v, p: ("%g" % round(v, 3)).replace(".", ",").replace("-", "−"))


def commas(ax):
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_formatter(COMMA)


def sim(a, exam=None):
    return m.diagnose(np.array(a, float), exam=exam).similarities


# ---------- рис. 6: грипп против COVID-19 в зависимости от температуры и обоняния ----------
T = np.linspace(36.0, 40.0, 41)
X7 = np.linspace(0, 1, 41)
Z = np.zeros((len(X7), len(T)))
for i, x7 in enumerate(X7):
    for j, t in enumerate(T):
        s = sim([m.temperature_membership(t), 0.5, 0.25, 0.25, 0.5, 0.75, x7, 0.0, 0.75])
        Z[i, j] = s[2] - s[1]          # S(COVID-19) − S(грипп)
TT, XX = np.meshgrid(T, X7)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(TT, XX, Z, cmap="coolwarm", edgecolor="k", linewidth=0.15, alpha=0.9)
ax.contour(TT, XX, Z, levels=[0], zdir="z", offset=Z.min() - 0.02, colors="k", linewidths=2)
ax.set_xlabel("Температура тела, °C", labelpad=8)
ax.set_ylabel("Снижение обоняния μ(x7)", labelpad=8)
ax.set_zlabel("S(COVID-19) − S(грипп)", labelpad=8)
ax.set_zlim(Z.min() - 0.02, Z.max())
ax.view_init(elev=24, azim=-128)
commas(ax)
fig.tight_layout()
fig.savefig(OUT / "fig6_surface_covid_flu.png", dpi=220)
plt.close(fig)

# граница: при каком обонянии COVID-19 становится основным диагнозом
for t in (37.0, 38.0, 39.0, 40.0):
    j = np.argmin(abs(T - t))
    col = Z[:, j]
    k = np.argmax(col > 0) if (col > 0).any() else None
    print(f"t={t}: разность при x7=0 {col[0]:+.3f}, при x7=1 {col[-1]:+.3f}, COVID-19 впереди при x7 ≥ {X7[k]:.2f}" if k is not None else f"t={t}: COVID не впереди")

# ---------- рис. 7: тонзиллит против ОРВИ в зависимости от боли в горле и налёта ----------
X4 = np.linspace(0.25, 1, 31)
X10 = np.linspace(0, 1, 41)
Z2 = np.zeros((len(X10), len(X4)))
for i, x10 in enumerate(X10):
    for j, x4 in enumerate(X4):
        s = sim([0.4, 0.25, 0.25, x4, 0.25, 0.5, 0.0, 0.0, 0.5], exam=[x10, 0.25])
        Z2[i, j] = s[3] - s[0]         # S(тонзиллит) − S(ОРВИ)
A4, A10 = np.meshgrid(X4, X10)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(A4, A10, Z2, cmap="coolwarm", edgecolor="k", linewidth=0.15, alpha=0.9)
ax.contour(A4, A10, Z2, levels=[0], zdir="z", offset=Z2.min() - 0.02, colors="k", linewidths=2)
ax.set_xlabel("Боль в горле μ(x4)", labelpad=8)
ax.set_ylabel("Налёт на миндалинах μ(x10)", labelpad=8)
ax.set_zlabel("S(тонзиллит) − S(ОРВИ)", labelpad=8)
ax.set_zlim(Z2.min() - 0.02, Z2.max())
ax.view_init(elev=24, azim=-128)
commas(ax)
fig.tight_layout()
fig.savefig(OUT / "fig7_surface_tonsil.png", dpi=220)
plt.close(fig)
for x10 in (0.0, 0.5, 1.0):
    i = np.argmin(abs(X10 - x10)); row = Z2[i]
    k = np.argmax(row > 0) if (row > 0).any() else None
    print(f"налёт {x10}: x4=0.25 {row[0]:+.3f}, x4=0.5 {row[10]:+.3f}, x4=1 {row[-1]:+.3f}; тонзиллит впереди при x4 ≥ {X4[k]:.3f}" if k is not None else f"налёт {x10}: не впереди ни при каком x4 (макс {row.max():+.3f})")
print("ok")
