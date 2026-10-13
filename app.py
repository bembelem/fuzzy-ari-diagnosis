"""Streamlit-приложение: система поддержки предварительной диагностики ОРЗ на основе нечёткой логики.

Запуск:  .venv\\Scripts\\streamlit run app.py
"""
import numpy as np
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


def explain_text(feature, patient, etalon, other):
    """Чем пациент отличается от типичной картины второго по сходству заболевания."""
    if feature == "Температура тела":
        return "**Температура тела** " + ("выше" if patient > etalon else "ниже") + f", чем обычно бывает при заболевании «{other}»"
    g = GENDER.get(feature, "м")
    if patient > etalon:
        return f"**{feature}** {EXPRESSED[g]} сильнее, чем обычно бывает при заболевании «{other}»"
    if patient == 0:
        return f"**{feature}** {ABSENT[g]}, хотя при заболевании «{other}» {MEETS[g]} часто"
    return f"**{feature}** {EXPRESSED[g]} слабее, чем обычно бывает при заболевании «{other}»"


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
            st.caption("Без осмотра правило острого тонзиллита проверяется только по жалобам.")

    a = m.patient_vector(t, terms)
    d = m.diagnose(a, etalons, alpha, delta, metric, exam=None if exam is None else np.array(exam, dtype=float))

    st.divider()
    st.subheader("Результат")
    if not np.any(a > 0) and (exam is None or not any(exam)):   # нет ни симптомов, ни лихорадки, ни находок осмотра
        st.info("Симптомы ОРЗ не указаны: температура нормальная, все признаки отсутствуют. Модель предназначена "
                "для пациентов с признаками острого респираторного заболевания, поэтому диагноз не ставится.")
        st.stop()
    flags = [f for f, _ in d.flags if f != d.diagnosis]
    if d.status == "atypical":
        st.error("**Картина нетипична** для ОРВИ, гриппа и COVID-19, правила других заболеваний не выполнены. "
                 "Требуется очное обследование.")
    elif d.status == "rule":
        st.error(f"Картина нетипична для ОРВИ, гриппа и COVID-19. Подозрение: **{d.diagnosis}**, требуется осмотр.")
    elif d.status == "confident" or not d.cannot_exclude:   # остальные заболевания ниже порога α: исключать нечего
        st.success(f"Предварительный диагноз: **{d.diagnosis}**")
    else:
        st.warning(f"Предварительный диагноз: **{d.diagnosis}**  \n"
                   f"Нельзя исключить: **{', '.join(d.cannot_exclude)}**. Рекомендуется дифференциальная проверка.")
    if flags:
        st.info("Выполнено правило: **" + ", ".join(flags) + "**. Рекомендуется осмотр или тест для подтверждения.")

    if d.contrast:
        with st.expander(f"Почему «{d.diagnosis}», а не «{m.DISEASES[d.order[1]]}»"):
            for feature, pa, pe, _ in d.contrast:
                st.markdown("- " + explain_text(feature, pa, pe, m.DISEASES[d.order[1]]))

    df_s = pd.DataFrame({"Заболевание": m.DISEASES, "Степень сходства": d.similarities}).sort_values("Степень сходства")
    fig = px.bar(df_s, x="Степень сходства", y="Заболевание", orientation="h", text_auto=".3f",
                 range_x=[0, 1], color_discrete_sequence=["#4C78A8"])
    fig.add_vline(x=alpha, line_dash="dash", line_color="crimson", annotation_text=f"α = {alpha:.2f}".replace(".", ","), annotation_position="top")
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10), separators=", ")
    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})
    rd = m.rule_degrees(a, None if exam is None else np.array(exam, dtype=float))
    st.caption(f"Степень сходства с основным диагнозом {d.confidence:.3f}; порог α = {alpha:.2f}; "
               f"разрыв Δ = {d.gap:.3f} (δ = {delta:.2f}). Степени истинности правил: "
               + "; ".join(f"{k.lower()} {v:.2f}" for k, v in rd.items()) + f" (порог θ = {m.THETA:.1f}).")

# ---------- вкладка 2: эталоны ----------
elif page == "Эталоны":
    st.subheader("Эталонные нечёткие множества (таблица 4)")
    st.caption("Значение эталона есть средняя выраженность признака у больных (p · κ). Значения можно редактировать.")
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
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Правила для заболеваний без эталонов")
    st.markdown("- **(15) Острый тонзиллит:** ЕСЛИ боль в горле И НЕ кашель [И (налёт на миндалинах ИЛИ увеличение лимфоузлов): "
                "если проведён осмотр], ТО острый тонзиллит;\n"
                "- **(16) Аллергический ринит:** ЕСЛИ чихание и зуд И НЕ лихорадка, ТО аллергический ринит.\n\n"
                f"Правило считается выполненным при степени истинности не менее θ = {m.THETA:.1f}.")
