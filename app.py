"""Streamlit-приложение: система поддержки предварительной диагностики ОРЗ на основе нечёткой логики.

Запуск:  .venv\\Scripts\\streamlit run app.py
"""
import pandas as pd
import plotly.express as px
import streamlit as st

import model as m

st.set_page_config(page_title="Нечёткая диагностика ОРЗ", page_icon="🩺", layout="wide")

# ---------- параметры модели (боковая панель) ----------
st.sidebar.header("Параметры модели")
metric = m.METRIC
st.sidebar.markdown("**Мера близости:** евклидова, формулы (9) и (11)")
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
PAGES = ["Диагностика", "Эталоны"]
_qp = st.query_params.get("page")  # позволяет открыть вкладку по ссылке, например ?page=Эталоны
if "page" not in st.session_state and _qp in PAGES:
    st.session_state.page = _qp
page = st.radio("Раздел", PAGES, horizontal=True, label_visibility="collapsed", key="page")
st.divider()


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
    st.subheader("Эталонные нечёткие множества (таблица 4)")
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

    st.subheader("Сходство эталонов между собой (таблица 5)")
    sm = m.etalon_similarity_matrix(etalons, metric)
    fig = px.imshow(sm, x=m.DISEASES, y=m.DISEASES, text_auto=".2f", zmin=0.4, zmax=1,
                    color_continuous_scale="Blues")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
