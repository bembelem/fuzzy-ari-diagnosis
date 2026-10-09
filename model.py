"""Нечёткая модель дифференциальной диагностики острых респираторных заболеваний.

Реализует главу 2 курсовой работы:
  - универсальное множество признаков X = {x1, ..., x9} (п. 2.2);
  - перевод температуры и словесных оценок в степени принадлежности (формула 13, шкала термов);
  - эталонные нечёткие множества заболеваний (п. 2.3, таблица 4);
  - меры близости (формулы 8–11) и алгоритм принятия решения (п. 2.4).
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


# ---------- эталоны (п. 2.3, таблица 4) ----------

DISEASES = ["ОРВИ", "Грипп", "COVID-19", "Острый тонзиллит", "Аллергический ринит"]

ETALONS = np.array([
    [0.10, 0.40, 0.60, 0.60, 0.48, 0.40, 0.22, 0.60, 0.40],  # E1 ОРВИ
    [0.93, 0.81, 0.56, 0.44, 0.70, 0.87, 0.17, 0.46, 0.52],  # E2 Грипп
    [0.76, 0.80, 0.49, 0.20, 0.59, 0.93, 0.53, 0.19, 0.30],  # E3 COVID-19
    [0.50, 0.26, 0.05, 1.00, 0.30, 0.60, 0.05, 0.05, 0.60],  # E4 Острый тонзиллит
    [0.00, 0.30, 0.80, 0.30, 0.05, 0.30, 0.43, 0.83, 0.60],  # E5 Аллергический ринит
])

# ---------- дополнительный блок: данные осмотра (заполняет врач, п. 3.3) ----------
# Добавлен после проверки на ситуационных задачах: без данных осмотра острый тонзиллит
# не отличается от ОРВИ. Если осмотр не проводился, блок не участвует в сравнении.
EXAM_FEATURES = ["Налёт на миндалинах", "Увеличение и болезненность шейных лимфоузлов"]
EXAM_CODES = ["x10", "x11"]
EXAM_ETALONS = np.array([
    [0.16, 0.25],   # ОРВИ: частота у пациентов без стрептококка с болью в горле × доля ОРВИ с болью в горле
    [0.05, 0.30],   # Грипп: прямой метод
    [0.05, 0.30],   # COVID-19: прямой метод
    [0.57, 0.67],   # Острый тонзиллит: частота у больных стрептококковым тонзиллитом
    [0.05, 0.05],   # Аллергический ринит: прямой метод
])
EXAM_TERMS = {
    "нет": 0.0,
    "незначительно": 0.25,
    "умеренно": 0.5,
    "выражено": 0.75,
    "резко выражено": 1.0,
}

# значения, заданные прямым методом (строка, столбец); остальные получены частотным методом
EXPERT_CELLS = {(2, 8), (3, 2), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
                (4, 1), (4, 3), (4, 4), (4, 5), (4, 8)}

# эталон COVID-19 для варианта «омикрон» (Menni et al., 2022): обоняние 16,7 %, боль в горле 70,5 %
COVID_OMICRON = ETALONS[2].copy()
COVID_OMICRON[6] = 0.17
COVID_OMICRON[3] = 0.71


# ---------- меры близости (п. 1.3) ----------

def hamming_similarity(a: np.ndarray, e: np.ndarray, w: np.ndarray | None = None) -> float:
    """Формулы (8) и (11): S = 1 − d_H; при заданных весах взвешенный вариант."""
    diff = np.abs(a - e)
    if w is None:
        return float(1 - diff.mean())
    return float(1 - np.sum(w * diff) / np.sum(w))


def euclid_similarity(a: np.ndarray, e: np.ndarray, w: np.ndarray | None = None) -> float:
    """Формулы (9) и (11): S = 1 − d_E; при заданных весах формула (16)."""
    sq = (a - e) ** 2
    if w is None:
        return float(1 - np.sqrt(sq.mean()))
    return float(1 - np.sqrt(np.sum(w * sq) / np.sum(w)))


def correlation_similarity(a: np.ndarray, e: np.ndarray, w: np.ndarray | None = None) -> float:
    """Формула (10), приведённая к [0, 1]: S_r = (1 + r) / 2. Для постоянного профиля r = 0."""
    if np.std(a) == 0 or np.std(e) == 0:
        return 0.5
    r = float(np.corrcoef(a, e)[0, 1])
    return (1 + r) / 2


# В модели используется евклидова мера (выбор обоснован в п. 3.5); остальные меры оставлены
# только для сравнительного анализа в скриптах compute_ch3.py и compute_euclid.py.
METRICS = {
    "Хэмминг": hamming_similarity,
    "Евклид": euclid_similarity,
    "Корреляция": correlation_similarity,
}
METRIC = "Евклид"
ALPHA = 0.52   # 5-й процентиль степени сходства с эталоном истинного заболевания на синтетике (п. 3.5)


# ---------- алгоритм принятия решения (п. 2.4) ----------

@dataclass
class Decision:
    similarities: np.ndarray            # S(A, E_k) в порядке DISEASES
    order: list[int]                    # индексы заболеваний по убыванию сходства
    status: str                         # "atypical" | "confident" | "uncertain"
    diagnosis: str | None               # k(1) или None, если картина нетипична
    confidence: float                   # S(1)
    gap: float                          # Δ = S(1) − S(2), формула (15)
    cannot_exclude: list[str] = field(default_factory=list)
    explanation: list[tuple[str, float, float, float]] = field(default_factory=list)

    @property
    def top2(self) -> list[str]:
        return [DISEASES[i] for i in self.order[:2]]


# Правила (17) и (18): обязательный и исключающий признаки (по клиническим рекомендациям, п. 3.3).
# Каждое правило ограничивает сверху степень сходства с эталоном: S_k := min(S_k, μ),
# где μ есть степень истинности условия, вычисленная операциями нечёткой логики (п. 1.2).
KEY_RULES = {
    3: ("боль в горле обязательна", lambda a: min(1.0, 4 * a[3])),  # тонзиллит: S ≤ min(1, 4·μ_A(x4)), т. е. симптом должен присутствовать хотя бы слабо
    4: ("лихорадка исключает", lambda a: 1 - a[0]),           # аллергический ринит: S ≤ 1 − μ_A(x1)
}


def apply_rules(a: np.ndarray, sim: np.ndarray) -> np.ndarray:
    """Применяет правила.

    Отсутствие жалобы на обязательный симптом трактуется как его отсутствие (μ = 0):
    при остром тонзиллите пациент всегда жалуется на боль в горле.
    """
    sim = sim.copy()
    for k, (_, cond) in KEY_RULES.items():
        sim[k] = min(sim[k], float(cond(a)))
    return sim


def diagnose(a: np.ndarray, etalons: np.ndarray = ETALONS, alpha: float = ALPHA, delta: float = 0.05,
             metric: str = METRIC, weights: np.ndarray | None = None, n_explain: int = 3,
             rules: bool = True, exam: np.ndarray | None = None) -> Decision:
    """Шаги 2–6 алгоритма из п. 2.4 (с учётом правил обязательных и исключающих признаков).

    exam: необязательные данные осмотра [x10, x11]; NaN означает, что признак не оценивался.
    """
    a = np.asarray(a, dtype=float)
    feats = FEATURES
    if exam is not None:                                                   # блок осмотра
        ex = np.asarray(exam, dtype=float)
        known = ~np.isnan(ex)
        w9 = np.ones(len(a)) if weights is None else np.asarray(weights, dtype=float)
        weights = np.concatenate([w9, known.astype(float)])
        etalons = np.hstack([etalons, EXAM_ETALONS])
        rule_a = a
        a = np.concatenate([a, np.where(known, ex, 0.0)])
        feats = FEATURES + EXAM_FEATURES
    else:
        rule_a = a
    sim = np.array([METRICS[metric](a, e, weights) for e in etalons])     # шаг 2
    if rules:
        sim = apply_rules(rule_a, sim)
    order = list(np.argsort(-sim, kind="stable"))                          # шаг 3
    s1 = float(sim[order[0]])
    gap = float(s1 - sim[order[1]])

    if s1 < alpha:                                                         # шаг 4
        return Decision(sim, order, "atypical", None, s1, gap)

    k1 = order[0]
    diff = np.abs(a - etalons[k1])                                         # шаг 6
    if weights is not None:
        diff = diff * (np.asarray(weights) > 0)                            # неизвестные признаки не поясняем
    idx = np.argsort(-diff)[:n_explain]
    explanation = [(feats[i], float(a[i]), float(etalons[k1][i]), float(diff[i])) for i in idx]

    if gap >= delta:                                                       # шаг 5
        return Decision(sim, order, "confident", DISEASES[k1], s1, gap, [], explanation)
    others = [DISEASES[k] for k in order[1:] if sim[k] >= alpha and s1 - sim[k] < delta]
    return Decision(sim, order, "uncertain", DISEASES[k1], s1, gap, others, explanation)


def patient_vector(temperature: float, terms: list[str]) -> np.ndarray:
    """Шаг 1: нечёткое множество пациента из температуры и восьми словесных оценок."""
    return np.array([temperature_membership(temperature)] + [TERMS[t] for t in terms])


def etalon_similarity_matrix(etalons: np.ndarray = ETALONS, metric: str = METRIC) -> np.ndarray:
    """Таблица 5: попарное сходство эталонов."""
    n = len(etalons)
    return np.array([[METRICS[metric](etalons[i], etalons[j]) for j in range(n)] for i in range(n)])


# ---------- синтетические данные для проверки самосогласованности ----------

def synthetic_cases(n_per_disease: int = 200, etalons: np.ndarray = ETALONS, seed: int = 0):
    """Генерирует пациентов в соответствии с частотной трактовкой эталонов.

    Признак i у больного k присутствует с вероятностью μ_Ek(xi); если присутствует,
    его выраженность выбирается из термов «слабо» … «очень сильно»,
    для температуры из диапазона 38–40 °C (иначе 36,4–37,6 °C).
    Такая выборка проверяет самосогласованность модели, но не заменяет реальные данные.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k, e in enumerate(etalons):
        for _ in range(n_per_disease):
            present = rng.random(9) < e
            v = np.where(present, rng.choice([0.25, 0.5, 0.75, 1.0], size=9), 0.0)
            t = rng.uniform(38.0, 40.0) if present[0] else rng.uniform(36.4, 37.6)
            v[0] = temperature_membership(t)
            X.append(v)
            y.append(k)
    return np.array(X), np.array(y)


def perturb_expert(etalons: np.ndarray = ETALONS, eps: float = 0.1, seed: int = 0) -> np.ndarray:
    """Случайный сдвиг экспертных значений на ±eps с обрезкой в [0, 1] (анализ устойчивости, п. 3.5)."""
    rng = np.random.default_rng(seed)
    out = etalons.copy()
    for r, c in EXPERT_CELLS:
        out[r, c] = float(np.clip(out[r, c] + rng.uniform(-eps, eps), 0, 1))
    return out
