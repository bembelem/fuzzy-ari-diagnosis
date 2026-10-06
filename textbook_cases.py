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
    """«верно», «в списке» (верный диагноз среди «нельзя исключить») или «ошибка»."""
    truth = case["эталон"]
    if truth == "нет среди пяти":
        return "верно" if d.status == "atypical" else "ошибка"
    if d.diagnosis == truth:
        return "верно"
    if truth in d.cannot_exclude:
        return "в списке"
    return "ошибка"


def run(metric=m.METRIC, alpha=None, delta=0.05, group=None, use_exam=True):
    alpha = alpha if alpha is not None else (0.52 if metric == "Евклид" else 0.6)  # калибровка: 5-й процентиль S_true на синтетике
    rows = []
    for c in load_cases():
        if group and c["группа"] != group:
            continue
        a, w, exam = case_vector(c)
        d = m.diagnose(a, alpha=alpha, delta=delta, metric=metric, weights=w, exam=exam if use_exam else None)
        answer = "нетипично" if d.status == "atypical" else d.diagnosis
        if d.cannot_exclude:
            answer += " (нельзя исключить: " + ", ".join(d.cannot_exclude) + ")"
        top = ", ".join(f"{m.DISEASES[i]} {d.similarities[i]:.2f}" for i in d.order[:3])
        rows.append({"id": c["id"], "группа": c["группа"], "эталон": c["эталон"], "ответ": answer,
                     "итог": verdict(c, d), "сходство": top, "осмотр": exam is not None})
    return rows


if __name__ == "__main__":
    for use_exam in (False, True):
        rows = run(use_exam=use_exam)
        print(f"\n### {m.METRIC}, данные осмотра: {'учитываются' if use_exam else 'не учитываются'}")
        for r in rows:
            mark = "*" if r["осмотр"] else " "
            print(f"  {r['id']:3s}{mark} [{r['группа']:10s}] эталон: {r['эталон']:18s} → {r['итог']:9s} | {r['ответ']}  [{r['сходство']}]")
        for g in ("разработка", "проверка"):
            sub = [r for r in rows if r["группа"] == g]
            ok = sum(r["итог"] == "верно" for r in sub)
            inl = sum(r["итог"] in ("верно", "в списке") for r in sub)
            print(f"  {g}: верно {ok}/{len(sub)}, верно или в списке {inl}/{len(sub)}")
