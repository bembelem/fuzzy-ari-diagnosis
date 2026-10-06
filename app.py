"""Streamlit-приложение: система поддержки предварительной диагностики ОРЗ на основе нечёткой логики.

Запуск:  .venv\\Scripts\\streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import evaluation as ev
import model as m

st.set_page_config(page_title="Нечёткая диагностика ОРЗ", page_icon="🩺", layout="wide")

# ---------- параметры модели (боковая панель) ----------
st.sidebar.header("Параметры модели")
metric = m.METRIC
st.sidebar.markdown("**Мера близости:** евклидова, формулы (11) и (12)")
alpha = st.sidebar.number_input("Порог сходства α", 0.40, 0.90, m.ALPHA, 0.01, format="%.2f")
st.sidebar.caption("Если сходство пациента ни с одним заболеванием не достигает α, "
                   "картина считается нетипичной и модель не ставит диагноз.")
delta = st.sidebar.number_input("Минимальный разрыв δ", 0.00, 0.15, 0.05, 0.01, format="%.2f")
st.sidebar.caption("Если два наиболее похожих заболевания различаются по сходству меньше чем на δ, "
                   "модель указывает оба: второе попадает в список «нельзя исключить».")

if "etalons" not in st.session_state:
    st.session_state.etalons = m.ETALONS.copy()
etalons = st.session_state.etalons.copy()

st.title("Нечёткая модель дифференциальной диагностики ОРЗ")
PAGES = ["Диагностика", "Эталоны", "Оценка качества", "Анализ чувствительности"]
_qp = st.query_params.get("page")  # позволяет открыть вкладку по ссылке, например ?page=Эталоны
if "page" not in st.session_state and _qp in PAGES:
    st.session_state.page = _qp
page = st.radio("Раздел", PAGES, horizontal=True, label_visibility="collapsed", key="page")
st.divider()


@st.cache_data(show_spinner="Расчёт показателей…")
def cached_evaluate(X, y, E, alpha, delta, metric, n_boot=500):
    return ev.evaluate(X, y, E, alpha, delta, metric, n_boot=n_boot)


@st.cache_data(show_spinner="Обучение модели сравнения…")
def cached_baseline(X, y):
    return ev.baseline_logreg(X, y)


@st.cache_data(show_spinner="Анализ устойчивости…")
def cached_robust(X, E, eps, alpha, delta, metric):
    return ev.robustness_expert(X, E, eps, 50, alpha=alpha, delta=delta, metric=metric)


@st.cache_data(show_spinner="Перебор параметров…")
def cached_grid(X, y, E, metric):
    return ev.grid_alpha_delta(X, y, np.round(np.arange(0.4, 0.71, 0.05), 2), [0.0, 0.03, 0.05, 0.08, 0.1], E, metric)


# короткие подписи для шкалы термов (п. 2.2) и шкалы осмотра
SCALE = {"нет": "отсутствует", "слабо": "слабо выражен", "умеренно": "умеренно выражен",
         "сильно": "сильно выражен", "очень сильно": "очень сильно выражен"}


# род признака для согласования: м: мужской, ж: женский, с: средний, мн: множественное число
GENDER = {
    "Кашель": "м", "Насморк": "м", "Боль в горле": "ж", "Ломота в мышцах и суставах": "ж",
    "Слабость": "ж", "Снижение обоняния и вкуса": "с", "Чихание, зуд в носу и глазах": "мн",
    "Внезапное начало болезни": "с", "Налёт на миндалинах": "м",
    "Увеличение и болезненность шейных лимфоузлов": "мн",
}
EXPRESSED = {"м": "выражен", "ж": "выражена", "с": "выражено", "мн": "выражены"}
ABSENT = {"м": "отсутствует", "ж": "отсутствует", "с": "отсутствует", "мн": "отсутствуют"}
MEETS = {"м": "встречается", "ж": "встречается", "с": "встречается", "мн": "встречаются"}


def explain_text(feature, patient, etalon):
    """Пояснение отличия пациента от эталона обычными словами."""
    if feature == "Температура тела":
        return "**Температура тела** " + ("выше" if patient > etalon else "ниже") + ", чем обычно бывает при этом заболевании"
    g = GENDER.get(feature, "м")
    if patient > etalon:
        return f"**{feature}** {EXPRESSED[g]} сильнее, чем обычно бывает при этом заболевании"
    if patient == 0:
        return f"**{feature}** {ABSENT[g]}, хотя при этом заболевании {MEETS[g]} часто"
    return f"**{feature}** {EXPRESSED[g]} слабее, чем обычно бывает при этом заболевании"


# ---------- вкладка 1: диагностика ----------
if page == "Диагностика":
    st.subheader("Данные пациента")
    t = st.number_input("Температура тела, °C", 35.0, 42.0, 38.6, 0.1)
    defaults = ["умеренно", "слабо", "слабо", "сильно", "сильно", "нет", "нет", "очень сильно"]
    cols = st.columns(2)
    terms = []
    for i, name in enumerate(m.FEATURES[1:]):
        with cols[i % 2]:
            v = st.segmented_control(name, list(SCALE), default=defaults[i], key=f"t{i}")
            terms.append(SCALE[v or "нет"])
    exam = None
    with st.expander("Данные осмотра (заполняет врач)"):
        if st.checkbox("Осмотр ротоглотки проведён", key="exam_done"):
            ecols = st.columns(2)
            exam = []
            for i, name in enumerate(m.EXAM_FEATURES):
                with ecols[i]:
                    v = st.segmented_control(name, list(m.EXAM_TERMS), default="нет", key=f"ex{i}")
                    exam.append(m.EXAM_TERMS[v or "нет"])
        else:
            st.caption("Без осмотра эти признаки не участвуют в сравнении.")

    a = m.patient_vector(t, terms)
    d = m.diagnose(a, etalons, alpha, delta, metric, exam=exam)

    st.divider()
    st.subheader("Результат")
    if d.status == "atypical":
        st.error("**Картина нетипична** ни для одного из пяти заболеваний. Требуется очное обследование.")
    elif d.status == "confident":
        st.success(f"Предварительный диагноз: **{d.diagnosis}**")
    else:
        st.warning(f"Предварительный диагноз: **{d.diagnosis}**  \n"
                   f"Нельзя исключить: **{', '.join(d.cannot_exclude)}**. Рекомендуется дифференциальная проверка.")

    if d.explanation:
        with st.expander(f"Почему такой ответ: чем пациент отличается от типичной картины «{d.diagnosis}»"):
            for feature, pa, pe, _ in d.explanation:
                st.markdown("- " + explain_text(feature, pa, pe))

    df_s = pd.DataFrame({"Заболевание": m.DISEASES, "Степень сходства": d.similarities}) \
        .sort_values("Степень сходства")
    fig = px.bar(df_s, x="Степень сходства", y="Заболевание", orientation="h", text_auto=".3f",
                 range_x=[0, 1], color_discrete_sequence=["#4C78A8"])
    fig.add_vline(x=alpha, line_dash="dash", line_color="crimson", annotation_text=f"α = {alpha:.2f}".replace(".", ","), annotation_position="top")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), separators=", ")
    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})
    st.caption(f"Степень сходства с основным диагнозом {d.confidence:.3f}; порог α = {alpha:.2f}; "
               f"разрыв со следующим заболеванием Δ = {d.gap:.3f} (δ = {delta:.2f}); "
               f"температура {t:.1f} °C соответствует степени принадлежности {m.temperature_membership(t):.2f}.")

# ---------- вкладка 2: эталоны ----------
elif page == "Эталоны":
    st.subheader("Эталонные нечёткие множества (таблица 5)")
    st.caption("Значения можно редактировать; экспертные значения (прямой метод) помечены в таблице курсовой звёздочкой.")
    df_e = pd.DataFrame(st.session_state.etalons, index=m.DISEASES, columns=m.FEATURE_CODES)
    edited = st.data_editor(df_e, use_container_width=True,
                            column_config={c: st.column_config.NumberColumn(c, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
                                           for c in m.FEATURE_CODES})
    c1, c2 = st.columns(2)
    if c1.button("Применить изменения"):
        st.session_state.etalons = edited.to_numpy(dtype=float)
        st.rerun()
    if c2.button("Вернуть значения из курсовой"):
        st.session_state.etalons = m.ETALONS.copy()
        st.rerun()

    st.subheader("Сходство эталонов между собой (таблица 6)")
    sm = m.etalon_similarity_matrix(etalons, metric)
    fig = px.imshow(sm, x=m.DISEASES, y=m.DISEASES, text_auto=".2f", zmin=0.4, zmax=1,
                    color_continuous_scale="Blues")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ---------- вкладка 3: оценка качества ----------
elif page == "Оценка качества":
    st.subheader("Оценка качества на наборе случаев")
    src = st.radio("Источник данных", ["Загрузить CSV с реальными случаями", "Синтетическая выборка (проверка самосогласованности)"])
    X = y = None
    if src.startswith("Загрузить"):
        st.caption("Формат: столбцы x1…x9 и diagnosis. x1 можно указать температурой в °C, x2…x9 можно указать числом или термом шкалы. "
                   "Шаблон: data/cases_template.csv")
        up = st.file_uploader("CSV-файл", type="csv")
        if up is not None:
            X, y = ev.load_cases(pd.read_csv(up))
    else:
        n = st.slider("Случаев на каждое заболевание", 20, 500, 200, 20)
        seed = st.number_input("Зерно генератора", 0, 9999, 0)
        st.info("Пациенты генерируются из самих эталонов, поэтому результат характеризует согласованность модели, "
                "а не её точность на реальных пациентах.")
        X, y = m.synthetic_cases(n, etalons, seed=int(seed))

    if X is not None:
        point, ci, cm_df, pc, p = cached_evaluate(X, y, etalons, alpha, delta, metric)
        st.markdown(f"**Случаев:** {len(y)}")
        st.dataframe(pd.DataFrame([{"Показатель": k, "Значение": v,
                                    "95 % ДИ": f"{ci[k][0]:.3f} – {ci[k][1]:.3f}" if k in ci else ""}
                                   for k, v in point.items()]).style.format({"Значение": "{:.3f}"}),
                     hide_index=True, use_container_width=True)
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown("**Матрица ошибок** (строки: истинный диагноз, столбцы: ответ модели)")
            fig = px.imshow(cm_df.values, x=list(cm_df.columns), y=list(cm_df.index), text_auto=True,
                            color_continuous_scale="Blues")
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("**Чувствительность и специфичность**")
            st.dataframe(pc.style.format({"Чувствительность": "{:.3f}", "Специфичность": "{:.3f}"}),
                         hide_index=True, use_container_width=True)
            base = cached_baseline(X, y)
            if base:
                st.markdown("**Сравнение с логистической регрессией** (5-кратная перекрёстная проверка)")
                keys = ["Доля верных ответов (accuracy)", "Сбалансированная точность", "Macro-F1", "Каппа Коэна", "Top-2 accuracy"]
                st.dataframe(pd.DataFrame({"Показатель": keys, "Нечёткая модель": [point[k] for k in keys],
                                           "Логистическая регрессия": [base[k] for k in keys]})
                             .style.format({"Нечёткая модель": "{:.3f}", "Логистическая регрессия": "{:.3f}"}),
                             hide_index=True, use_container_width=True)
        st.session_state["last_X"], st.session_state["last_y"] = X, y

# ---------- вкладка 4: анализ чувствительности ----------
elif page == "Анализ чувствительности":
    st.subheader("Анализ чувствительности")
    if "last_X" not in st.session_state:
        st.info("Сначала выберите набор случаев на вкладке «Оценка качества».")
    else:
        X, y = st.session_state["last_X"], st.session_state["last_y"]
        st.markdown("**1. Устойчивость к экспертным значениям.** Все 12 значений, заданных прямым методом, "
                    "случайно сдвигаются на ±ε; считается доля случаев, у которых ответ модели не изменился.")
        eps = st.slider("ε", 0.05, 0.3, 0.1, 0.05)
        mean_same, min_same = cached_robust(X, etalons, eps, alpha, delta, metric)
        st.metric("Ответ не изменился (в среднем / в худшем прогоне)", f"{mean_same:.1%} / {min_same:.1%}")

        st.markdown("**2. Влияние порога α и разрыва δ.**")
        grid = cached_grid(X, y, etalons, metric)
        fig = px.line(grid, x="α", y="Accuracy", color="δ", markers=True)
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(grid.style.format({"Accuracy": "{:.3f}", "Отказы": "{:.3f}", "«Нельзя исключить»": "{:.3f}"}),
                     hide_index=True, use_container_width=True)

        st.markdown("**3. Сравнение мер близости.**")
        rows = []
        for name in m.METRICS:
            pt, _, _, _, _ = cached_evaluate(X, y, etalons, alpha, delta, name, n_boot=1)
            rows.append({"Мера": name, **{k: pt[k] for k in ["Доля верных ответов (accuracy)", "Macro-F1", "Top-2 accuracy", "Доля отказов («нетипично»)"]}})
        st.dataframe(pd.DataFrame(rows).style.format({c: "{:.3f}" for c in rows[0] if c != "Мера"}),
                     hide_index=True, use_container_width=True)
