import os
import sys
import streamlit as st

# Остальные импорты
import fitz
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io
import pandas as pd
import json
import tempfile

# Настройка путей для Tesseract в Streamlit Cloud
if os.path.exists('/app'):
    os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/tessdata'
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

## ИЗМЕНЕНИЕ: импортируем новую функцию get_translation_map
from templates import REPORT_TEMPLATES, get_report_template_as_string, get_translation_map

# --- КОНФИГУРАЦИЯ ---
st.set_page_config(layout="wide", page_title="Унификация отчета")

# --- ИНИЦИАЛИЗАЦИЯ LLM ---
try:
    PROVIDER_API_KEY = os.getenv("PROVIDER_API_KEY")
    if not PROVIDER_API_KEY:
        st.error("API ключ не найден в переменных окружения.")
        st.stop()
except Exception as e:
    st.error(f"Ошибка при получении API ключа: {e}")
    st.stop()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

llm = ChatOpenAI(
    model_name="deepseek/deepseek-r1-0528",
    openai_api_key=PROVIDER_API_KEY,
    openai_api_base="https://api.novita.ai/v3/openai",
    temperature=0.1
)

# --- РЕАЛИЗАЦИЯ ФУНКЦИЙ ---

@st.cache_data
def extract_text_from_file(file_bytes, filename):
    """Извлекает текст из PDF или изображения с поддержкой OCR."""
    try:
        ext = filename.split(".")[-1].lower()
        text = ""
        
        if ext == "pdf":
            # Обработка PDF
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()
            
            # Если текст не извлекся, пробуем OCR
            if not text.strip():
                st.warning("PDF не содержит текста. Использую OCR...")
                images = convert_from_bytes(file_bytes, dpi=300)
                for image in images:
                    text += pytesseract.image_to_string(image, lang='rus+eng')
                    
        elif ext in ["png", "jpg", "jpeg"]:
            # Обработка изображений
            image = Image.open(io.BytesIO(file_bytes))
            # Улучшаем качество изображения для OCR
            if image.mode != 'RGB':
                image = image.convert('RGB')
            text = pytesseract.image_to_string(image, lang='rus+eng')
        else:
            st.error(f"Неподдерживаемый формат файла: {ext}")
            return None
            
        return text.strip()
    
    except Exception as e:
        st.error(f"Ошибка при извлечении текста: {e}")
        return None

@st.cache_data
def classify_report(_llm, text: str) -> str:
    """Определяет тип отчета (баланс, ОПУ и т.д.) с помощью LLM."""
    # Запрос к LLM
    prompt = ChatPromptTemplate.from_template(
        "Текст финансового отчета: {text}\n\n"
        "Определи тип финансового отчета. Выбери один из: {report_types}.\n"
        "Ответ выдай только как строку, без пояснений."
    )
    chain = prompt | _llm | StrOutputParser()
    report_types_str = ", ".join(REPORT_TEMPLATES.keys())
    return chain.invoke({"text": text, "report_types": report_types_str})

@st.cache_data
def extract_data_with_template(_llm, text: str, report_type: str) -> list:
    """Извлекает данные из текста отчета по заданному шаблону."""
    # Получаем шаблон для данного типа отчета
    template = get_report_template_as_string(report_type)
    
    # Создаем парсер
    parser = JsonOutputParser()
    
    # Создаем промпт
    prompt = ChatPromptTemplate.from_template(
        template + "\n\nИзвлеки данные из следующего текста финансового отчета:\n{text}"
    )
    
    chain = prompt | _llm | parser
    return chain.invoke({"text": text})

def to_excel_bytes(df):
    """Конвертирует DataFrame в байты Excel файла."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Report')
    return output.getvalue()

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
st.title("🤖 Унификация отчета")

st.sidebar.header("Загрузка файлов")
uploaded_files = st.sidebar.file_uploader(
    "Загрузите сканы отчета (один или несколько файлов)",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    all_text = ""
    
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        
        with st.spinner(f"Извлечение текста из {uploaded_file.name}..."):
            extracted_text = extract_text_from_file(file_bytes, uploaded_file.name)
        
        if extracted_text:
            st.success(f"✅ Текст из {uploaded_file.name} успешно извлечен!")
            all_text += f"\n\n--- ФАЙЛ: {uploaded_file.name} ---\n\n{extracted_text}"
        else:
            st.error(f"Не удалось извлечь текст из {uploaded_file.name}.")
    
    if not all_text.strip():
        st.error("Не удалось извлечь текст ни из одного файла.")
        st.stop()
    
    # Покажем общий объем извлеченного текста
    st.info(f"Общий объем извлеченного текста: {len(all_text)} символов")
    
    with st.expander("Показать извлеченный текст"):
        st.text(all_text[:5000] + ("..." if len(all_text) > 5000 else ""))
    
    with st.spinner("Классификация типа отчета..."):
        report_type = classify_report(llm, all_text)
    
    if report_type not in REPORT_TEMPLATES:
        st.error(f"Не удалось определить тип отчета. LLM вернул: '{report_type}'")
    else:
        st.success(f"✅ Отчет классифицирован как **{report_type}**.")
        
        with st.spinner(f"Извлечение данных по шаблону..."):
            structured_data = extract_data_with_template(llm, all_text, report_type)
        
        if not structured_data:
            st.error("Не удалось извлечь структурированные данные.")
        else:
            st.success("✅ Данные успешно извлечены!")

            final_data = enrich_data_with_russian_names(structured_data, report_type)
            
            # Фильтруем строки с значениями
            df = pd.DataFrame([item for item in final_data if item.get('value') is not None])
            
            st.header("Извлеченные и стандартизированные данные")
            if not df.empty:
                df = df[["Статья (RU)", "Line Item (EN)", "value", "year", "unit"]]
                st.dataframe(df, use_container_width=True)
                
                # Кнопка для скачивания
                excel_bytes = to_excel_bytes(df)
                st.download_button(
                    label="📥 Скачать отчет в Excel",
                    data=excel_bytes,
                    file_name="standard_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("Нет данных для отображения.")
            
            with st.expander("Показать полный JSON от LLM (до обогащения)"):
                st.json(structured_data)
else:
    st.info("Пожалуйста, загрузите файлы в боковой панели, чтобы начать.")

# Сохраняем функции в session_state
st.session_state['extract_text_from_file_func'] = extract_text_from_file
st.session_state['classify_report_func'] = classify_report
st.session_state['extract_data_with_template_func'] = extract_data_with_template
