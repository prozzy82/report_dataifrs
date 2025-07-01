import os
import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io
import pandas as pd
import json

# Импорт из вашего файла шаблонов
from templates import REPORT_TEMPLATES, get_report_template_as_string, get_translation_map

# Импорты LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.exceptions import OutputParserException

# --- КОНФИГУРАЦИЯ ---
st.set_page_config(layout="wide", page_title="Унификация Отчета")

# --- ИНИЦИАЛИЗАЦИЯ LLM ---
try:
    PROVIDER_API_KEY = st.secrets.get("NOVITA_API_KEY") or os.getenv("PROVIDER_API_KEY")
    if not PROVIDER_API_KEY:
        raise KeyError
except (FileNotFoundError, KeyError):
    st.error("Ключ API не найден. Установите переменную окружения PROVIDER_API_KEY или создайте .streamlit/secrets.toml.")
    st.stop()

llm = ChatOpenAI(
    model_name="google/gemma-3-27b-it",
    openai_api_key=PROVIDER_API_KEY,
    openai_api_base="https://api.novita.ai/v3/openai",
    temperature=0.1
)

# --- РЕАЛИЗАЦИЯ ФУНКЦИЙ ---

@st.cache_data
def extract_text_from_file(file_bytes, filename):
    # Эта функция остается без изменений
    try:
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".pdf":
            with fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf") as doc:
                text = "".join(page.get_text() for page in doc)
            if len(text.strip()) < 150:
                st.warning(f"Текстовый слой в '{filename}' пуст. Используется OCR...")
                images = convert_from_bytes(file_bytes, dpi=300)
                text = "\n".join([pytesseract.image_to_string(img, lang='rus+eng', config='--psm 6') for img in images])
        elif ext in [".png", ".jpg", ".jpeg"]:
            image = Image.open(io.BytesIO(file_bytes))
            text = pytesseract.image_to_string(image, lang='rus+eng', config='--psm 6')
        else: return None
        return text.strip()
    except Exception as e:
        st.error(f"Ошибка при извлечении текста из '{filename}': {e}")
        return None

@st.cache_data
def classify_report(_llm, text: str) -> str:
    # Эта функция остается без изменений
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_template(
        "Определи тип финансового отчета. Ответь ТОЛЬКО одним из вариантов: {report_types}.\n\nТекст:\n---\n{text_snippet}\n---"
    )
    chain = prompt | _llm | parser
    report_types_str = ", ".join(REPORT_TEMPLATES.keys())
    response = chain.invoke({"text_snippet": text[:4000], "report_types": report_types_str})
    for report_type in REPORT_TEMPLATES.keys():
        if report_type in response:
            return report_type
    return "Unknown"

@st.cache_data
def extract_raw_financial_data(_llm, text: str) -> list:
    # Эта функция остается без изменений
    parser = JsonOutputParser()
    prompt_text = (
        "Ты — финансовый аналитик. Извлеки ВСЕ финансовые показатели из текста отчета. "
        "Ответь ТОЛЬКО JSON-массивом объектов. Каждый объект должен содержать:\n"
        "- source_item: название статьи на языке оригинала\n"
        "- unit: единица измерения (если есть)\n"
        "- values: массив объектов с полями period (год) и value (число)\n\n"
        "ПРАВИЛА:\n"
        "1. Включай ВСЕ статьи с числовыми значениями\n"
        "2. Для статей с несколькими периодами создай один объект с массивом values\n"
        "3. Период указывай в формате 'YYYY'\n"
        "4. Если период не указан, используй 'N/A'\n"
        "5. Если значение не числовое, пропускай его\n"
        "6. Не пытайся классифицировать статьи!\n"
        "7. Сохраняй оригинальные названия статей\n\n"
        "Пример вывода для отчета с двумя периодами:\n"
        "["
        "  {{\"source_item\": \"Выручка\", \"unit\": \"тыс. руб.\", \"values\": [ {{\"period\": \"2024\", \"value\": 150000}}, {{\"period\": \"2023\", \"value\": 120000}} ]}},"
        "  {{\"source_item\": \"Себестоимость продаж\", \"unit\": \"тыс. руб.\", \"values\": [ {{\"period\": \"2024\", \"value\": 90000}}, {{\"period\": \"2023\", \"value\": 75000}} ]}}"
        "]"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        ("user", "Текст отчета:\n---\n{text}\n---")
    ])
    chain = prompt | _llm | parser
    try:
        result = chain.invoke({"text": text[:100000]})
        return result
    except Exception as e:
        st.error(f"Ошибка извлечения сырых данных: {e}")
        return []

# ИЗМЕНЕНО: Функция стандартизации теперь возвращает словарь с двумя списками
@st.cache_data
def standardize_data(_llm, raw_data: list, report_type: str) -> dict:
    """Сопоставляет сырые данные со стандартным шаблоном и возвращает сопоставленные и несопоставленные элементы."""
    template_items = get_report_template_as_string(report_type)
    if not template_items: return {"standardized_data": [], "unmapped_items": raw_data}

    # ИЗМЕНЕНО: Промпт теперь требует JSON-объект с двумя ключами
    # ВНИМАНИЕ: Все фигурные скобки в примере JSON ниже были экранированы (продублированы)
    # Вместо { стало {{ и вместо } стало }}
    prompt_text = (
        "Ты — эксперт по финансовой отчетности. Сопоставь сырые финансовые данные со стандартным шаблоном. "
        "Ответь ТОЛЬКО JSON-объектом с ДВУМЯ ключами: `standardized_data` и `unmapped_items`.\n\n"
        "1. Ключ `standardized_data` должен содержать МАССИВ объектов, сопоставленных с шаблоном, в формате:\n"
        # ---- НАЧАЛО ИЗМЕНЕНИЯ ----
        "   `{{\"line_item\": \"название_из_шаблона\", \"unit\": \"ед_изм\", \"values_by_period\": [{{\"period\": \"2024\", \"value\": число, \"components\": [{{\"source_item\": \"исходная_статья\", \"source_value\": число}}]}}]}}`\n\n"
        # ---- КОНЕЦ ИЗМЕНЕНИЯ ----
        "2. Ключ `unmapped_items` должен содержать МАССИВ объектов из сырых данных, которые НЕ удалось сопоставить ни с одной статьей шаблона. Сохрани их исходный формат.\n\n"
        "ПРАВИЛА СОПОСТАВЛЕНИЯ:\n"
        "- Используй ТОЛЬКО статьи из этого шаблона: {template_items}\n" # Это настоящая переменная, ее не трогаем
        "- Для каждой стандартной статьи найди соответствующие сырые статьи и агрегируй (суммируй) их значения.\n"
        "- Если стандартная статья состоит из нескольких сырых, укажи все компоненты.\n"
        "- Если сырая статья не соответствует шаблону, помести ее в `unmapped_items`.\n"
        "- Если для стандартной статьи нет данных, не включай ее в `standardized_data`.\n\n"
        "Сырые данные:\n{raw_data}" # Это тоже настоящая переменная, ее не трогаем
    )

    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        ("user", "Выполни сопоставление строго по правилам и верни JSON с ключами 'standardized_data' и 'unmapped_items'.")
    ])

    chain = prompt | _llm | parser
    try:
        raw_data_str = json.dumps(raw_data, ensure_ascii=False, indent=2)[:100000]
        # При вызове chain.invoke() передаем только настоящие переменные
        result = chain.invoke({
            "template_items": template_items,
            "raw_data": raw_data_str
        })
        # Убедимся, что оба ключа присутствуют
        if "standardized_data" not in result: result["standardized_data"] = []
        if "unmapped_items" not in result: result["unmapped_items"] = []
        return result
    except Exception as e:
        st.error(f"Ошибка стандартизации: {e}")
        # Возвращаем пустую структуру при ошибке
        return {"standardized_data": [], "unmapped_items": raw_data}

# Функции flatten_data_for_display, display_raw_data, to_excel_bytes остаются без изменений
def flatten_data_for_display(data: list, report_type: str) -> list:
    flat_list = []
    translation_map = get_translation_map(report_type)
    for item in data:
        english_name = item.get("line_item")
        russian_name = translation_map.get(english_name, english_name)
        unit = item.get("unit")
        values_by_period = item.get("values_by_period", [])
        if not values_by_period:
            continue
        for period_data in values_by_period:
            value = period_data.get("value")
            if value is None:
                continue
            flat_list.append({
                "Статья (RU)": russian_name,
                "Line Item (EN)": english_name,
                "unit": unit,
                "period": period_data.get("period"),
                "value": value,
                "components": period_data.get("components", [])
            })
    return flat_list

def display_raw_data(raw_data):
    if not raw_data:
        return pd.DataFrame()
    rows = []
    for item in raw_data:
        for val in item.get("values", []):
            rows.append({
                "Исходная статья": item.get('source_item', 'N/A'),
                "Период": val.get("period", "N/A"),
                "Значение": val.get("value", "N/A"),
                "Ед. изм.": item.get("unit", "")
            })
    return pd.DataFrame(rows)

def to_excel_bytes(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Report')
    return output.getvalue()

# --- ОБНОВЛЕННЫЙ ИНТЕРФЕЙС ПРИЛОЖЕНИЯ ---
st.title("📊 Унификация Финансового Отчета")

st.sidebar.header("Загрузка Файлов")
uploaded_files = st.sidebar.file_uploader(
    "Загрузите сканы отчета", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True
)

if uploaded_files:
    file_names = [f.name for f in uploaded_files]
    # Инициализация сессионных переменных
    if "processed_data" not in st.session_state or st.session_state.get("file_names") != file_names:
        st.session_state.file_names = file_names
        st.session_state.all_text = ""
        st.session_state.raw_data = None
        st.session_state.processed_data = None
        st.session_state.unmapped_items = None # НОВЫЙ КЛЮЧ В СЕССИИ

        with st.spinner("Извлечение текста из файлов..."):
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

    # Шаг 1: Классификация отчета
    with st.spinner("🔍 Шаг 1/4: Определение типа отчета..."):
        report_type = classify_report(llm, all_text)

    if report_type == "Unknown":
        st.error("⚠️ Не удалось определить тип отчета.")
        st.stop()
    else:
        st.success(f"✅ Отчет классифицирован как **{report_type}**.")

    # Шаг 2: Извлечение сырых данных
    if st.session_state.get("raw_data") is None:
        with st.spinner("📋 Шаг 2/4: Извлечение сырых данных..."):
            raw_data = extract_raw_financial_data(llm, all_text)
            st.session_state.raw_data = raw_data

    if st.session_state.raw_data:
        st.success("✅ Сырые данные успешно извлечены!")
        with st.expander("🔎 Просмотреть сырые данные", expanded=False):
            raw_df = display_raw_data(st.session_state.raw_data)
            if not raw_df.empty:
                st.dataframe(raw_df, use_container_width=True)
            else:
                st.warning("Сырые данные не содержат записей")

    # Шаг 3: Стандартизация данных
    # ИЗМЕНЕНО: Обработка словаря, возвращаемого из standardize_data
    if st.session_state.raw_data and st.session_state.get("processed_data") is None:
        with st.spinner("🔄 Шаг 3/4: Стандартизация данных..."):
            response_dict = standardize_data(llm, st.session_state.raw_data, report_type)
            st.session_state.processed_data = response_dict.get("standardized_data", [])
            st.session_state.unmapped_items = response_dict.get("unmapped_items", [])

    # Шаг 4: Отображение результатов
    if st.session_state.get("processed_data") is not None:
        st.success("✅ Данные успешно стандартизированы!")

        flat_data = flatten_data_for_display(st.session_state.processed_data, report_type)
        if flat_data:
            df = pd.DataFrame(flat_data)

            def format_components(components_list):
                if not components_list or not isinstance(components_list, list):
                    return "Прямое сопоставление"
                return "; ".join([
                    f"{c.get('source_item', 'N/A')} ({c.get('source_value', 'N/A')})"
                    for c in components_list
                ])

            df['Источник агрегации'] = df['components'].apply(format_components)
            df.sort_values(by=['Статья (RU)', 'period'], ascending=[True, False], inplace=True)
            df = df[["Статья (RU)", "value", "period", "Источник агрегации", "unit"]]
            df.rename(columns={
                'Статья (RU)': 'Стандартизированная статья',
                'value': 'Итоговое значение',
                'period': 'Период',
                'unit': 'Ед. изм.'
            }, inplace=True)

            st.dataframe(df, use_container_width=True, hide_index=True)

            excel_bytes = to_excel_bytes(df)
            st.download_button(
                "📥 Скачать отчет в Excel",
                excel_bytes,
                f"standard_report_{report_type.replace(' ', '_')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("После стандартизации не осталось данных для отображения.")

        # НОВЫЙ БЛОК: Отображение непринятых статей для отладки
        if st.session_state.get("unmapped_items"):
            st.warning("⚠️ Следующие статьи из исходного отчета не были сопоставлены с шаблоном:")
            unmapped_df = display_raw_data(st.session_state.unmapped_items)
            if not unmapped_df.empty:
                st.dataframe(unmapped_df, use_container_width=True, hide_index=True)

        # Отладочная информация
        with st.expander("📄 Показать полный JSON от LLM (стандартизированные данные)"):
            st.json(st.session_state.processed_data)
        
        with st.expander("📄 Показать JSON непринятых статей"):
            st.json(st.session_state.unmapped_items)

        with st.expander("📝 Показать весь извлеченный текст"):
            st.text_area("Распознанный текст", all_text, height=400)
else:
    st.info("👈 Пожалуйста, загрузите файлы в боковой панели, чтобы начать анализ.")
