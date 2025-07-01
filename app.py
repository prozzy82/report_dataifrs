import os
import sys
import streamlit as st

# Настройка путей для Tesseract в Streamlit Cloud
if os.path.exists('/app'):
    os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/tessdata'
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Остальные импорты
import fitz
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io
import pandas as pd
import json

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

llm = ChatOpenAI(
    model_name="deepseek/deepseek-r1-0528",
    openai_api_key=PROVIDER_API_KEY,
    openai_api_base="https://api.novita.ai/v3/openai",
    temperature=0.1
)

# --- РЕАЛИЗАЦИЯ ФУНКЦИИ EXTRACT_TEXT_FROM_FILE ---
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
                images = convert_from_bytes(file_bytes)
                for image in images:
                    text += pytesseract.image_to_string(image, lang='rus')
                    
        elif ext in ["png", "jpg", "jpeg"]:
            # Обработка изображений
            image = Image.open(io.BytesIO(file_bytes))
            text = pytesseract.image_to_string(image, lang='rus')
        else:
            st.error(f"Неподдерживаемый формат файла: {ext}")
            return None
            
        return text.strip()
    
    except Exception as e:
        st.error(f"Ошибка при извлечении текста: {e}")
        return None

# ... (остальные функции остаются без изменений)

# --- ИНТЕРФЕЙС ПРИЛОЖЕНИЯ С ПОДДЕРЖКОЙ МНОЖЕСТВЕННОЙ ЗАГРУЗКИ ---
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
