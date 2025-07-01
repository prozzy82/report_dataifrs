# app.py
import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io
import pandas as pd
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

## ИЗМЕНЕНИЕ: импортируем новую функцию get_translation_map
from templates import REPORT_TEMPLATES, get_report_template_as_string, get_translation_map

# --- КОНФИГУРАЦИЯ ---
st.set_page_config(layout="wide", page_title="Анализатор Финансовых Отчетов")

# --- ИНИЦИАЛИЗАЦИЯ LLM ---
try:
    PROVIDER_API_KEY = st.secrets["NOVITA_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("Не удалось найти NOVITA_API_KEY. Создайте файл .streamlit/secrets.toml и добавьте ключ.")
    st.stop()

llm = ChatOpenAI(
    model_name="deepseek/deepseek-coder",
    openai_api_key=PROVIDER_API_KEY,
    openai_api_base="https://api.novita.ai/v3/openai",
    temperature=0.1,
    max_tokens=4096
)

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
# ... (функции extract_text_from_file, classify_report, extract_data_with_template остаются БЕЗ ИЗМЕНЕНИЙ) ...
@st.cache_data
def extract_text_from_file(file_bytes, filename):
    # ... код без изменений
    pass

@st.cache_data
def classify_report(_llm, text: str) -> str:
    # ... код без изменений
    pass

@st.cache_data
def extract_data_with_template(_llm, text: str, report_type: str) -> list:
    # ... код без изменений
    pass

# Оставим тут заглушки, чтобы показать, что они не меняются
extract_text_from_file = st.session_state.get('extract_text_from_file_func', extract_text_from_file)
classify_report = st.session_state.get('classify_report_func', classify_report)
extract_data_with_template = st.session_state.get('extract_data_with_template_func', extract_data_with_template)
# Полный код этих функций есть в предыдущем ответе.

def to_excel_bytes(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Report')
    return output.getvalue()

## ИЗМЕНЕНИЕ: Новая функция для обогащения данных русскими названиями
def enrich_data_with_russian_names(data: list, report_type: str) -> list:
    """Добавляет русские названия статей в извлеченные данные."""
    translation_map = get_translation_map(report_type)
    enriched_data = []
    
    for item in data:
        english_name = item.get("line_item")
        if not english_name:
            continue
            
        russian_name = translation_map.get(english_name, "Статья не найдена в шаблоне")
        
        enriched_data.append({
            "Статья (RU)": russian_name,
            "Line Item (EN)": english_name,
            "value": item.get("value"),
            "year": item.get("year"),
            "unit": item.get("unit")
        })
    return enriched_data

# --- ИНТЕРФЕЙС ПРИЛОЖЕНИЯ ---
st.title("🤖 Автоматизация разбора финансовых отчетов")

st.sidebar.header("Загрузка файла")
uploaded_file = st.sidebar.file_uploader(
    "Загрузите скан отчета",
    type=["pdf", "png", "jpg", "jpeg"]
)

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    
    with st.spinner("1/3 Извлечение текста из файла..."):
        # Используем полный код функций из предыдущего ответа
        # Здесь для краткости опущен
        extracted_text = extract_text_from_file(file_bytes, uploaded_file.name)
    
    if not extracted_text:
        st.error("Не удалось извлечь текст из файла.")
    else:
        st.success("✅ Шаг 1: Текст успешно извлечен.")
        
        with st.spinner("2/3 Классификация типа отчета..."):
            report_type = classify_report(llm, extracted_text)
        
        if report_type not in REPORT_TEMPLATES:
            st.error(f"Не удалось определить тип отчета. LLM вернул: '{report_type}'")
        else:
            st.success(f"✅ Шаг 2: Отчет классифицирован как **{report_type}**.")
            
            with st.spinner(f"3/3 Извлечение данных по шаблону..."):
                structured_data = extract_data_with_template(llm, extracted_text, report_type)
            
            if not structured_data:
                st.error("Не удалось извлечь структурированные данные.")
            else:
                st.success("✅ Шаг 3: Данные успешно извлечены!")

                ## ИЗМЕНЕНИЕ: Добавляем шаг обогащения данных
                final_data = enrich_data_with_russian_names(structured_data, report_type)
                
                # Фильтруем строки, где не было найдено значение
                df = pd.DataFrame([item for item in final_data if item.get('value') is not None])
                
                st.header("Извлеченные и стандартизированные данные")
                # Устанавливаем порядок колонок для красивого вывода
                if not df.empty:
                    df = df[["Статья (RU)", "Line Item (EN)", "value", "year", "unit"]]
                st.dataframe(df, use_container_width=True)
                
                # Кнопка для скачивания
                excel_bytes = to_excel_bytes(df)
                st.download_button(
                    label="📥 Скачать отчет в Excel",
                    data=excel_bytes,
                    file_name=f"standard_report_{uploaded_file.name.split('.')[0]}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                with st.expander("Показать полный JSON от LLM (до обогащения)"):
                    st.json(structured_data)
else:
    st.info("Пожалуйста, загрузите файл в боковой панели, чтобы начать.")

# Сохраняем функции в session_state, чтобы избежать переопределения при перезапуске
st.session_state['extract_text_from_file_func'] = extract_text_from_file
st.session_state['classify_report_func'] = classify_report
st.session_state['extract_data_with_template_func'] = extract_data_with_template
