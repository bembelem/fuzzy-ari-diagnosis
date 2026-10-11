"""Ситуационные задачи из учебных сборников: сопоставление ответа модели с эталоном (п. 3.3).

Задачи хранятся в data/textbook_cases.json. Признак со значением null неизвестен
и не участвует в сравнении (вес 0); для правил обязательных признаков он считается равным 0.
Данные осмотра (x10, x11) указываются, только если они есть в тексте задачи.
"""
import json
from pathlib import Path

import numpy as np

import model as m

CASES_FILE = Path(__file__).parent / "data" / "textbook_cases.json"


def load_cases():
    return json.loads(CASES_FILE.read_text(encoding="utf-8"))["cases"]


def case_vector(case):
    raw = [case["x"][c] for c in m.FEATURE_CODES]
    a = np.array([0.0 if v is None else float(v) for v in raw])
    if raw[0] is not None:
        a[0] = m.temperature_membership(raw[0])
    w = np.array([0.0 if v is None else 1.0 for v in raw])
    ex = [case["x"].get(c) for c in m.EXAM_CODES]
    exam = None if all(v is None for v in ex) else np.array([np.nan if v is None else float(v) for v in ex])
    return a, w, exam


def verdict(case, d):
    """«верно» (основной диагноз совпал), «в ответе» (заболевание названо среди «нельзя исключить»
    или правилом) или «ошибка». Для ловушек верным считается ответ «нетипично»."""
    truth = case["эталон"]
    if truth == "нет среди пяти":
        return "верно" if d.status == "atypical" else "ошибка"
    if d.diagnosis == truth:
        return "верно"
    if truth in d.answer():
        return "в ответе"
    return "ошибка"


def is_typical(case):
    return case["эталон"] != "нет среди пяти" and "атипич" not in case["тип"]


def run(metric=m.METRIC, alpha=None, delta=m.DELTA, group=None, use_exam=True):
    alpha = m.ALPHA if alpha is None else alpha
    rows = []
    for c in load_cases():
        if group and c["группа"] != group:
            continue
        a, w, exam = case_vector(c)
        d = m.diagnose(a, alpha=alpha, delta=delta, metric=metric, weights=w, exam=exam if use_exam else None)
        answer = "нетипично" if d.status == "atypical" else d.diagnosis
        extra = d.cannot_exclude + [f for f, _ in d.flags if f != d.diagnosis]
        if extra:
            answer += " (+ " + ", ".join(extra) + ")"
        top = ", ".join(f"{m.DISEASES[i]} {d.similarities[i]:.2f}" for i in d.order)
        rows.append({"id": c["id"], "группа": c["группа"], "эталон": c["эталон"], "типичная": is_typical(c),
                     "ответ": answer, "итог": verdict(c, d), "сходство": top, "осмотр": exam is not None})
    return rows


if __name__ == "__main__":
    for use_exam in (False, True):
        rows = run(use_exam=use_exam, group="проверка")
        print(f"\n### данные осмотра: {'учитываются' if use_exam else 'не учитываются'}")
        for r in rows:
            print(f"  {r['id']:3s}{'*' if r['осмотр'] else ' '} {r['эталон'][:20]:20s} → {r['итог']:8s} | {r['ответ']}  [{r['сходство']}]")
