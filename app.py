import os
import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io
import pandas as pd
import json
import copy
import numpy as np

# Импорт шаблонов
from templates import REPORT_TEMPLATES, get_report_template_as_string, get_translation_map, get_report_codes

# Импорт вспомогательного модуля таксономии
from ifrs_taxonomy_helper import load_ifrs_taxonomy, suggest_mapping_from_taxonomy, build_unmapped_with_suggestions

# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.exceptions import OutputParserException

# Конфигурация
st.set_page_config(layout="wide", page_title="Извлечение данных отчета")

# API ключ
try:
    PROVIDER_API_KEY = st.secrets.get("NOVITA_API_KEY") or os.getenv("PROVIDER_API_KEY")
    if not PROVIDER_API_KEY:
        st.error("Ключ API не найден. Установите PROVIDER_API_KEY или secrets.toml.")
        st.stop()
except Exception:
    st.error("Ключ API не найден.")
    st.stop()

# Инициализация LLM
llm = ChatOpenAI(
    model_name="google/gemma-3-27b-it",
    openai_api_key=PROVIDER_API_KEY,
    openai_api_base="https://api.novita.ai/v3/openai",
    temperature=0.1
)

# ✅ Загрузка IFRS таксономии
IFRS_TAXONOMY = load_ifrs_taxonomy("json_taxonomy_rus.json")

# --- ВАШИ ФУНКЦИИ (extract_text_from_file, classify_report, extract_raw_financial_data, correct_source_item_names,
# standardize_data, flatten_data_for_display, display_raw_data, to_excel_bytes, transform_to_wide_format) остаются без изменений ---
# (Пропущено здесь ради краткости — ты уже включил их ранее и они корректны.)

# --- ИНТЕРФЕЙС ---

st.title("📊 Извлечение данных отчета")

st.sidebar.header("Загрузка Файлов")
uploaded_files = st.sidebar.file_uploader("Загрузите сканы отчета", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    file_names = [f.name for f in uploaded_files]
    if "processed_data" not in st.session_state or st.session_state.get("file_names") != file_names:
        st.session_state.file_names = file_names
        st.session_state.all_text = ""
        st.session_state.raw_data = None
        st.session_state.corrected_raw_data = None
        st.session_state.processed_data = None
        st.session_state.unmapped_items = None

        with st.spinner("Шаг 1/5: Извлечение текста из файлов..."):
            all_text = ""
            for uploaded_file in uploaded_files:
                file_text = extract_text_from_file(uploaded_file.getvalue(), uploaded_file.name)
                if file_text:
                    all_text += f"\n\n--- НАЧАЛО ФАЙЛА: {uploaded_file.name} ---\n\n{file_text}"
            st.session_state.all_text = all_text.strip()

    all_text = st.session_state.get("all_text", "")
    if not all_text:
        st.error("Не удалось извлечь текст.")
        st.stop()

    st.info(f"📝 Общий объем текста: {len(all_text)} символов.")

    with st.spinner("🔍 Шаг 2/5: Определение типа отчета..."):
        report_type = classify_report(llm, all_text)

    if report_type == "Unknown":
        st.error("⚠️ Не удалось определить тип отчета.")
        st.stop()
    else:
        st.success(f"✅ Отчет классифицирован как **{report_type}**.")

    if st.session_state.get("raw_data") is None:
        with st.spinner("📋 Шаг 3/5: Извлечение сырых данных..."):
            raw_data = extract_raw_financial_data(llm, all_text)
            st.session_state.raw_data = raw_data

    if st.session_state.raw_data:
        st.success("✅ Сырые данные успешно извлечены!")
        with st.expander("🔎 Просмотреть исходные сырые данные", expanded=False):
            raw_df = display_raw_data(st.session_state.raw_data)
            st.dataframe(raw_df, use_container_width=True, hide_index=True)

    if st.session_state.raw_data and st.session_state.get("corrected_raw_data") is None:
        with st.spinner("✍️ Шаг 4/5: Коррекция названий статей..."):
            corrected_data = correct_source_item_names(llm, st.session_state.raw_data, report_type)
            st.session_state.corrected_raw_data = corrected_data

    if st.session_state.get("corrected_raw_data") and st.session_state.get("processed_data") is None:
        with st.spinner("🔄 Шаг 5/5: Стандартизация данных..."):
            result = standardize_data(llm, st.session_state.corrected_raw_data, report_type)
            st.session_state.processed_data = result.get("standardized_data", [])
            st.session_state.unmapped_items = result.get("unmapped_items", [])

    if st.session_state.get("processed_data") is not None:
        flat_data = flatten_data_for_display(st.session_state.processed_data, report_type)
        if flat_data:
            long_df = pd.DataFrame(flat_data)
            long_df['Источник агрегации'] = long_df['components'].apply(lambda comps: "; ".join(
                [f"{c.get('source_item')} ({c.get('source_value')})" for c in comps] if comps else ["Прямое сопоставление"]))

            long_df.rename(columns={
                'Код': 'Код статьи',
                'Статья (RU)': 'Стандартизированная статья',
                'value': 'Итоговое значение',
                'period': 'Период',
                'unit': 'Ед. изм.'
            }, inplace=True)

            long_df.sort_values(by=['Код статьи', 'Период'], ascending=[True, False], inplace=True)
            wide_df = transform_to_wide_format(long_df)

            display_format = st.radio("Формат отображения:", ["Стандартный (периоды в колонках)", "Детальный"])
            if display_format == "Стандартный (периоды в колонках)":
                st.dataframe(wide_df, use_container_width=True, hide_index=True)
            else:
                st.dataframe(long_df[["Код статьи", "Стандартизированная статья", "Итоговое значение", "Период", "Источник агрегации", "Ед. изм."]], use_container_width=True)

            excel_bytes = to_excel_bytes(wide_df, long_df)
            st.download_button("📥 Скачать отчет в Excel", excel_bytes, f"standard_report_{report_type.replace(' ', '_')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.warning("Нет данных для отображения.")

        if st.session_state.get("unmapped_items"):
            st.warning("⚠️ Несопоставленные статьи:")
            st.dataframe(display_raw_data(st.session_state.unmapped_items), use_container_width=True)

            # ✅ ПОДСКАЗКИ НА ОСНОВЕ ТАКСОНОМИИ
            with st.expander("💡 Подсказки по несопоставленным статьям (таксономия IFRS)", expanded=False):
                for item in st.session_state.unmapped_items:
                    source = item.get("source_item", "")
                    suggestion = suggest_mapping_from_taxonomy(source, IFRS_TAXONOMY)
                    if suggestion:
                        st.markdown(f"🔎 *{source}* возможно соответствует: **{suggestion}**")
                    else:
                        st.markdown(f"⚠️ *{source}* — соответствие не найдено")

            # ✅ ЭКСПОРТ ПОДСКАЗОК
            with st.expander("📥 Скачать подсказки по несопоставленным статьям"):
                suggestion_df = pd.DataFrame(build_unmapped_with_suggestions(st.session_state.unmapped_items, IFRS_TAXONOMY))
                st.dataframe(suggestion_df, use_container_width=True, hide_index=True)

                excel_output = io.BytesIO()
                with pd.ExcelWriter(excel_output, engine='xlsxwriter') as writer:
                    suggestion_df.to_excel(writer, index=False, sheet_name="IFRS_Suggestions")

                file_name = f"unmapped_suggestions_{report_type.replace(' ', '_')}.xlsx"
                st.download_button("💾 Скачать Excel", data=excel_output.getvalue(), file_name=file_name)

        # Отладочная информация
        with st.expander("📄 JSON стандартизированных данных"):
            st.json(st.session_state.processed_data)
        with st.expander("📄 JSON несопоставленных статей"):
            st.json(st.session_state.unmapped_items)
        with st.expander("📝 Извлечённый текст"):
            st.text_area("Распознанный текст", all_text, height=400)
else:
    st.info("👈 Загрузите PDF/изображения в боковой панели")
