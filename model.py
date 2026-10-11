"""Гибридная нечёткая модель дифференциальной диагностики острых респираторных заболеваний.

Реализует главу 2 курсовой работы:
  - универсальное множество признаков X = {x1, ..., x9} (п. 2.2);
  - перевод температуры и словесных оценок в степени принадлежности (формула 13, шкала термов);
  - эталонные нечёткие множества ОРВИ, гриппа и COVID-19 как средняя выраженность признаков (п. 2.3);
  - правила для острого тонзиллита и аллергического ринита (п. 2.3);
  - меры близости (формулы 8–11) и гибридный алгоритм принятия решения (п. 2.4).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------- признаки и шкала (п. 2.2) ----------

FEATURES = [
    "Температура тела",
    "Кашель",
    "Насморк",
    "Боль в горле",
    "Ломота в мышцах и суставах",
    "Слабость",
    "Снижение обоняния и вкуса",
    "Чихание, зуд в носу и глазах",
    "Внезапное начало болезни",
]
FEATURE_CODES = [f"x{i}" for i in range(1, 10)]

# лингвистическая шкала термов для x2..x9
TERMS = {
    "отсутствует": 0.0,
    "слабо выражен": 0.25,
    "умеренно выражен": 0.5,
    "сильно выражен": 0.75,
    "очень сильно выражен": 1.0,
}


def temperature_membership(t: float, a: float = 37.0, b: float = 39.0) -> float:
    """Формула (13): линейная функция принадлежности между полярными значениями a и b."""
    return float(min(1.0, max(0.0, (t - a) / (b - a))))


# ---------- эталоны (п. 2.3) ----------
# Эталон есть средняя выраженность признака у больных: μ_E(x_i) = p_i · κ_i, формула (14).

DISEASES = ["ОРВИ", "Грипп", "COVID-19"]                         # заболевания, заданные эталонами
RULE_DISEASES = ["Острый тонзиллит", "Аллергический ринит"]       # заболевания, заданные правилами
ALL_DISEASES = DISEASES + RULE_DISEASES

# κ_i: средняя выраженность признака у тех, у кого он есть. Virome of Manhattan (Galanti et al., 2024),
# когорты D1, D2, SC, 75 эпизодов ОРВИ с подтверждённым вирусом (virome.py). Для x6, x7 данных нет: среднее по остальным;
# x9 (начало болезни) не является выраженностью: κ = 1.
KAPPA = np.array([0.44, 0.58, 0.51, 0.55, 0.59, 0.53, 0.53, 0.52, 1.00])

# частоты признаков p_i у больных гриппом и COVID-19 (Zayet et al., 2020: одна выборка, ПЦР) и острого начала
# при гриппе (Ebell et al., 2025); для COVID-19 частота острого начала задана прямым методом (0,30)
FREQUENCIES = np.array([
    [0.93, 0.81, 0.56, 0.44, 0.70, 0.87, 0.17, 0.46, 0.52],  # Грипп
    [0.76, 0.80, 0.49, 0.20, 0.59, 0.93, 0.53, 0.19, 0.30],  # COVID-19
])

# ОРВИ: x1–x5, x8: средняя выраженность в эпизодах ОРВИ (Virome of Manhattan, когорты D1, D2, SC, n = 75);
# x6, x7, x9 в когорте не оценивались: p · κ по прежним источникам (Menni 2020; Ebell 2025)
ETALON_ORVI = np.array([0.05, 0.29, 0.44, 0.29, 0.07, round(0.40 * 0.53, 2), round(0.22 * 0.53, 2), 0.30, 0.40])

ETALONS = np.vstack([ETALON_ORVI, np.round(FREQUENCIES * KAPPA, 2)])
EXPERT_CELLS = {(2, 8)}            # единственное значение прямого метода: острое начало при COVID-19

# эталон COVID-19 для варианта «омикрон» (Menni et al., 2022): обоняние 16,7 %, боль в горле 70,5 %
COVID_OMICRON = ETALONS[2].copy()
COVID_OMICRON[6] = round(0.17 * KAPPA[6], 2)
COVID_OMICRON[3] = round(0.71 * KAPPA[3], 2)

# ---------- правила для заболеваний без эталонов (п. 2.3) ----------
# данные осмотра (заполняет врач) участвуют только в правиле острого тонзиллита
EXAM_FEATURES = ["Налёт на миндалинах", "Увеличение и болезненность шейных лимфоузлов"]
EXAM_CODES = ["x10", "x11"]
EXAM_TERMS = {"нет": 0.0, "незначительно": 0.25, "умеренно": 0.5, "выражено": 0.75, "резко выражено": 1.0}
THETA = 0.5   # правило считается выполненным, если степень истинности не меньше 0,5


def rule_degrees(a: np.ndarray, exam: np.ndarray | None = None) -> dict[str, float]:
    """Степени истинности правил (15) и (16), вычисленные операциями min и дополнения (п. 1.2).

    (15) ЕСЛИ боль в горле И НЕ кашель [И (налёт ИЛИ лимфоузлы)], ТО острый тонзиллит;
    (16) ЕСЛИ чихание и зуд И НЕ лихорадка, ТО аллергический ринит.
    Неупомянутый симптом считается отсутствующим (μ = 0).
    """
    ton = min(a[3], 1 - a[1])
    if exam is not None and not np.all(np.isnan(exam)):
        ton = min(ton, float(np.nanmax(exam)))
    ar = min(a[7], 1 - a[0])
    return {"Острый тонзиллит": float(ton), "Аллергический ринит": float(ar)}


# ---------- меры близости (п. 1.3) ----------

def hamming_similarity(a: np.ndarray, e: np.ndarray, w: np.ndarray | None = None) -> float:
    """Формулы (8) и (11): S = 1 − d_H; при заданных весах взвешенный вариант."""
    diff = np.abs(a - e)
    if w is None:
        return float(1 - diff.mean())
    return float(1 - np.sum(w * diff) / np.sum(w))


def euclid_similarity(a: np.ndarray, e: np.ndarray, w: np.ndarray | None = None) -> float:
    """Формулы (9) и (11): S = 1 − d_E; при заданных весах формула (18)."""
    sq = (a - e) ** 2
    if w is None:
        return float(1 - np.sqrt(sq.mean()))
    return float(1 - np.sqrt(np.sum(w * sq) / np.sum(w)))


def correlation_similarity(a: np.ndarray, e: np.ndarray, w: np.ndarray | None = None) -> float:
    """Формула (10), приведённая к [0, 1]: S_r = (1 + r) / 2. Неизвестные признаки исключаются."""
    if w is not None:
        k = np.asarray(w) > 0
        a, e = a[k], e[k]
    if np.std(a) == 0 or np.std(e) == 0:
        return 0.5
    return (1 + float(np.corrcoef(a, e)[0, 1])) / 2


METRICS = {"Хэмминг": hamming_similarity, "Евклид": euclid_similarity, "Корреляция": correlation_similarity}
METRIC = "Евклид"
ALPHA = 0.57   # 5-й процентиль степени сходства с эталоном истинного заболевания на синтетике (п. 3.5)
DELTA = 0.05


# ---------- алгоритм принятия решения (п. 2.4) ----------

@dataclass
class Decision:
    similarities: np.ndarray            # S(A, E_k) для ОРВИ, гриппа, COVID-19
    order: list[int]                    # индексы по убыванию сходства
    status: str                         # "confident" | "uncertain" | "rule" | "atypical"
    diagnosis: str | None               # предварительный диагноз
    confidence: float                   # S(1)
    gap: float                          # Δ = S(1) − S(2)
    cannot_exclude: list[str] = field(default_factory=list)
    flags: list[tuple[str, float]] = field(default_factory=list)   # выполненные правила и их степени
    explanation: list[tuple[str, float, float, float]] = field(default_factory=list)  # отличие от эталона k(1)
    contrast: list[tuple[str, float, float, float]] = field(default_factory=list)     # отличие от эталона k(2)

    def answer(self) -> set[str]:
        """Все заболевания, названные в ответе."""
        out = set(self.cannot_exclude) | {f for f, _ in self.flags}
        if self.diagnosis:
            out.add(self.diagnosis)
        return out


def _explain(a, e, w, feats, n):
    diff = np.abs(a - e) * (w > 0)
    idx = np.argsort(-diff)[:n]
    return [(feats[i], float(a[i]), float(e[i]), float(diff[i])) for i in idx if diff[i] > 0]


def diagnose(a: np.ndarray, etalons: np.ndarray = ETALONS, alpha: float = ALPHA, delta: float = DELTA,
             metric: str = METRIC, weights: np.ndarray | None = None, n_explain: int = 3,
             exam: np.ndarray | None = None, theta: float = THETA) -> Decision:
    """Шаги 2–7 алгоритма из п. 2.4.

    weights: 1, если признак известен, 0, если неизвестен (формула 18); exam: [x10, x11], NaN, если не оценивался.
    """
    a = np.asarray(a, dtype=float)
    w = np.ones(len(a)) if weights is None else np.asarray(weights, dtype=float)
    sim = np.array([METRICS[metric](a, e, w) for e in etalons])            # шаг 2
    order = list(np.argsort(-sim, kind="stable"))                            # шаг 3
    s1, gap = float(sim[order[0]]), float(sim[order[0]] - sim[order[1]])
    rules = rule_degrees(a, exam)                                            # шаг 4
    flags = sorted([(k, v) for k, v in rules.items() if v >= theta], key=lambda t: -t[1])

    if s1 < alpha:                                                           # шаг 5
        if flags:
            return Decision(sim, order, "rule", flags[0][0], s1, gap, [], flags)
        return Decision(sim, order, "atypical", None, s1, gap, [], flags)

    k1, k2 = order[0], order[1]
    expl = _explain(a, etalons[k1], w, FEATURES, n_explain)                  # шаг 7
    contrast = []
    diff = (np.abs(a - etalons[k2]) - np.abs(a - etalons[k1])) * (w > 0)     # чем пациент ближе к k1, чем к k2
    for i in np.argsort(-diff)[:n_explain]:
        if diff[i] > 0:
            contrast.append((FEATURES[i], float(a[i]), float(etalons[k2][i]), float(diff[i])))
    if gap >= delta:                                                         # шаг 6
        return Decision(sim, order, "confident", DISEASES[k1], s1, gap, [], flags, expl, contrast)
    others = [DISEASES[k] for k in order[1:] if sim[k] >= alpha and s1 - sim[k] < delta]
    return Decision(sim, order, "uncertain", DISEASES[k1], s1, gap, others, flags, expl, contrast)


def patient_vector(temperature: float, terms: list[str]) -> np.ndarray:
    """Шаг 1: нечёткое множество пациента из температуры и восьми словесных оценок."""
    return np.array([temperature_membership(temperature)] + [TERMS[t] for t in terms])


def etalon_similarity_matrix(etalons: np.ndarray = ETALONS, metric: str = METRIC) -> np.ndarray:
    """Таблица 5: попарное сходство эталонов."""
    n = len(etalons)
    return np.array([[METRICS[metric](etalons[i], etalons[j]) for j in range(n)] for i in range(n)])


# ---------- синтетические данные (п. 3.5) ----------

def _term_probs(kappa: float, terms=np.array([0.25, 0.5, 0.75, 1.0])) -> np.ndarray:
    """Распределение по термам шкалы со средним kappa."""
    if kappa >= 0.999:
        return np.array([0, 0, 0, 1.0])
    lo, hi = -50.0, 50.0
    for _ in range(100):
        lam = (lo + hi) / 2
        p = np.exp(lam * terms); p /= p.sum()
        lo, hi = (lam, hi) if p @ terms < kappa else (lo, lam)
    return p


def synthetic_cases(n_per_disease: int = 300, etalons: np.ndarray | None = None, seed: int = 0):
    """Синтетические больные ОРВИ, гриппом и COVID-19 в соответствии с определением эталона μ = p · κ.

    Признак i присутствует с вероятностью p_i = μ_E(x_i) / κ_i; его выраженность выбирается из термов
    «слабо» … «очень сильно» со средним κ_i. Тогда средняя выраженность признака равна значению эталона.
    """
    etalons = ETALONS if etalons is None else etalons
    rng = np.random.default_rng(seed)
    terms = np.array([0.25, 0.5, 0.75, 1.0])
    probs = [_term_probs(k) for k in KAPPA]
    X, y = [], []
    for k, e in enumerate(etalons):
        p = np.clip(e / KAPPA, 0, 1)
        for _ in range(n_per_disease):
            present = rng.random(9) < p
            X.append([rng.choice(terms, p=probs[i]) if present[i] else 0.0 for i in range(9)])
            y.append(k)
    return np.array(X), np.array(y)


def perturb(etalons: np.ndarray = ETALONS, rel: float = 0.1, seed: int = 0) -> np.ndarray:
    """Случайное относительное изменение всех значений эталонов на ±rel (анализ устойчивости, п. 3.5)."""
    rng = np.random.default_rng(seed)
    return np.clip(etalons * (1 + rng.uniform(-rel, rel, etalons.shape)), 0, 1)
