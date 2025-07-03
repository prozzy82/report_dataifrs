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

# --- НОВЫЕ ИМПОРТЫ ---
# Импорт из вашего нового файла таксономии
from taxonomy import IFRS_TAXONOMY, IFRS_FLAT_LIST
# Импорт из вашего старого файла для шаблонов отображения
from templates import get_translation_map, get_report_codes, REPORT_TEMPLATES

# Импорты LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# --- КОНФИГУРАЦИЯ ---
st.set_page_config(layout="wide", page_title="Извлечение и стандартизация отчета")

# --- ИНИЦИАЛИЗАЦИЯ LLM ---
try:
    PROVIDER_API_KEY = st.secrets.get("NOVITA_API_KEY") or os.getenv("PROVIDER_API_KEY")
    if not PROVIDER_API_KEY:
        st.error("Ключ API не найден.")
        st.stop()
except Exception:
    st.error("Ключ API не найден.")
    st.stop()

llm = ChatOpenAI(
    model_name="google/gemma-3-27b-it",
    openai_api_key=PROVIDER_API_KEY,
    openai_api_base="https://api.novita.ai/v3/openai",
    temperature=0.0
)

# --- РЕАЛИЗАЦИЯ ФУНКЦИЙ ---

# Функции extract_text_from_file, classify_report, extract_raw_financial_data, correct_source_item_names
# остаются такими же, как в предыдущей версии. Я их оставлю здесь для полноты файла.

@st.cache_data
def extract_text_from_file(file_bytes, filename):
    try:
        ext = os.path.splitext(filename)[1].lower()
        text = ""
        if ext == ".pdf":
            with fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf") as doc:
                text = "".join(page.get_text() for page in doc)
            if len(text.strip()) < 150:
                st.warning(f"Текстовый слой в '{filename}' пуст. Используется OCR...")
                images = convert_from_bytes(file_bytes, dpi=300)
                text = "\n".join([pytesseract.image_to_string(img, lang='rus+eng', config='--psm 6') for img in images])
        elif ext in [".png", ".jpg", ".jpeg"]:
            image = Image.open(io.BytesIO(file_bytes))
            if image.mode != 'RGB': image = image.convert('RGB')
            text = pytesseract.image_to_string(image, lang='rus+eng', config='--psm 6')
        else:
            st.error(f"Неподдерживаемый формат файла: {ext}"); return None
        return text.strip()
    except Exception as e:
        st.error(f"Ошибка при извлечении текста из '{filename}': {e}"); return None

@st.cache_data
def classify_report(_llm, text: str) -> str:
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_template("Определи тип финансового отчета. Ответь ТОЛЬКО одним из вариантов: {report_types}.\n\nТекст:\n---\n{text_snippet}\n---")
    chain = prompt | _llm | parser
    report_types_str = ", ".join(REPORT_TEMPLATES.keys())
    response = chain.invoke({"text_snippet": text[:4000], "report_types": report_types_str})
    for report_type in REPORT_TEMPLATES.keys():
        if report_type in response: return report_type
    return "Unknown"

@st.cache_data
def extract_raw_financial_data(_llm, text: str) -> list:
    parser = JsonOutputParser()
    prompt_text = (
        "Ты — финансовый аналитик. Извлеки ВСЕ финансовые показатели из текста отчета. "
        "Ответь ТОЛЬКО JSON-массивом объектов. Каждый объект должен содержать:\n"
        "- source_item: название статьи на языке оригинала\n"
        "- unit: единица измерения (если есть)\n"
        "- values: массив объектов с полями period (год) и value (число)\n\n"
        "ПРАВИЛА:\n"
        "1. Включай ВСЕ статьи с числовыми значениями.\n2. Период указывай в формате 'YYYY'.\n"
        "3. Сохраняй оригинальные названия статей.\n\n"
        "Пример: [ {{\"source_item\": \"Выручка\", \"unit\": \"тыс. руб.\", \"values\": [ {{\"period\": \"2024\", \"value\": 150000}}, {{\"period\": \"2023\", \"value\": 120000}} ]}} ]"
    )
    prompt = ChatPromptTemplate.from_messages([("system", prompt_text), ("user", "Текст отчета:\n---\n{text}\n---")])
    chain = prompt | _llm | parser
    try:
        return chain.invoke({"text": text[:100000]})
    except Exception as e:
        st.error(f"Ошибка извлечения сырых данных: {e}"); return []

@st.cache_data
def correct_source_item_names(_llm, raw_data: list) -> list:
    if not raw_data: return []
    source_items_to_correct = list(set([item['source_item'] for item in raw_data]))
    parser = JsonOutputParser()
    prompt_text = (
        "Ты — корректор, исправляешь ошибки OCR в названиях финансовых статей. Ответь ТОЛЬКО JSON-объектом, где ключ — исходная строка, а значение — исправленная.\n\n"
        "ПРАВИЛА:\n"
        "1. Исправляй только явные ошибки OCR и орфографии. 'Bupyuka' -> 'Выручка'.\n"
        "2. Если название корректно, верни его без изменений.\n"
        "3. В качестве ориентира используй этот список стандартных финансовых терминов: {template_items}\n\n"
        "Список для исправления:\n{items_to_correct}"
    )
    prompt = ChatPromptTemplate.from_messages([("system", prompt_text), ("user", "Исправь этот список и верни JSON-объект.")])
    chain = prompt | _llm | parser
    try:
        items_json_string = json.dumps(source_items_to_correct, ensure_ascii=False)
        correction_map = chain.invoke({"template_items": "\n".join(IFRS_FLAT_LIST), "items_to_correct": items_json_string})
        corrected_raw_data = copy.deepcopy(raw_data)
        for item in corrected_raw_data:
            original_item = item['source_item']
            if original_item in correction_map: item['source_item'] = correction_map[original_item]
        return corrected_raw_data
    except Exception as e:
        st.error(f"Ошибка при коррекции названий статей: {e}"); return raw_data

# --- ОБНОВЛЕННАЯ ФУНКЦИЯ СТАНДАРТИЗАЦИИ (БЕЗ РАСЧЕТА) ---
@st.cache_data
def standardize_data_with_aggregation(_llm, corrected_data: list, report_type: str) -> dict:
    """
    Сопоставляет сырые данные со стандартным шаблоном и агрегирует их,
    используя полную таксономию как контекст.
    """
    # Получаем целевой плоский шаблон для вывода
    template = REPORT_TEMPLATES.get(report_type, {})
    template_items_str = "\n".join([f"- {en_name} ({details['ru']})" for en_name, details in template.items()])
    
    # Полная таксономия для контекста
    full_taxonomy_str = json.dumps(IFRS_TAXONOMY, ensure_ascii=False, indent=2)[:5000]

    parser = JsonOutputParser()
    prompt_text = (
        "Ты — эксперт по МСФО. Твоя задача — сопоставить сырые финансовые данные с ЦЕЛЕВЫМ шаблоном и агрегировать значения. "
        "Ответь ТОЛЬКО JSON-объектом с ДВУМЯ ключами: `standardized_data` и `unmapped_items`.\n\n"
        "ЦЕЛЕВОЙ ШАБЛОН (используй ТОЛЬКО эти статьи в `standardized_data`):\n{target_template}\n\n"
        "ПОЛНАЯ ТАКСОНОМИЯ МСФО ДЛЯ КОНТЕКСТА (помогает понять, что куда относится):\n{full_taxonomy}\n\n"
        "ПРАВИЛА:\n"
        "1. Для КАЖДОЙ статьи из ЦЕЛЕВОГО ШАБЛОНА найди в сырых данных все соответствующие ей компоненты.\n"
        "2. АГРЕГИРУЙ (суммируй) значения найденных компонентов для каждого периода.\n"
        "   Пример: если в целевом шаблоне есть 'Assets', а в сырых данных есть 'NoncurrentAssets' и 'CurrentAssets', "
        "   то значением для 'Assets' будет сумма 'NoncurrentAssets' + 'CurrentAssets'.\n"
        "3. В `standardized_data` включай ТОЛЬКО статьи из ЦЕЛЕВОГО ШАБЛОНА.\n"
        "4. В `unmapped_items` помести сырые статьи, которые не удалось использовать для агрегации ни в одну целевую статью.\n"
        "5. Результат для `standardized_data` должен быть в формате:\n"
        "   [ {{\"line_item\": \"название_из_ЦЕЛЕВОГО_шаблона\", \"unit\": \"ед_изм\", \"values\": [ {{\"period\": \"2024\", \"value\": АГРЕГИРОВАННОЕ_число}}] }} ]\n\n"
        "Сырые данные:\n{raw_data}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        ("user", "Выполни сопоставление и агрегацию строго по правилам и верни JSON.")
    ])
    chain = prompt | _llm | parser
    
    try:
        raw_data_str = json.dumps(corrected_data, ensure_ascii=False, indent=2)[:100000]
        result = chain.invoke({
            "target_template": template_items_str,
            "full_taxonomy": full_taxonomy_str,
            "raw_data": raw_data_str
        })
        
        if "standardized_data" not in result: result["standardized_data"] = []
        if "unmapped_items" not in result: result["unmapped_items"] = []
            
        return result
    except Exception as e:
        st.error(f"Ошибка стандартизации: {e}")
        return {"standardized_data": [], "unmapped_items": corrected_data}


# --- УПРОЩЕННЫЕ ФУНКЦИИ ОТОБРАЖЕНИЯ ---

def flatten_data_for_display(data: list, report_type: str) -> list:
    """Преобразует стандартизированные данные в плоский формат для DataFrame"""
    flat_list = []
    translation_dict = get_translation_map(report_type)
    
    for item in data:
        english_name = item.get("line_item")
        item_info = translation_dict.get(english_name, {})
        russian_name = item_info.get("ru", english_name)
        code = item_info.get("code", "N/A")
        
        for period_data in item.get("values", []):
            flat_list.append({
                "Код статьи": code,
                "Стандартизированная статья": russian_name,
                "Ед. изм.": item.get("unit", "N/A"),
                "Период": period_data.get("period"),
                "Итоговое значение": period_data.get("value")
            })
    return flat_list

def display_raw_data(raw_data):
    """
    Создает DataFrame для отображения сырых или несопоставленных данных.
    УСТОЙЧИВАЯ ВЕРСИЯ: Проверяет, что каждый элемент является словарем.
    """
    if not raw_data: 
        return pd.DataFrame()
        
    rows = []
    for item in raw_data:
        # --- ДОБАВЛЕНА ПРОВЕРКА ---
        # Если item не является словарем, пропускаем его и переходим к следующему
        if not isinstance(item, dict):
            continue
        # --------------------------
            
        for val in item.get("values", []):
            # Дополнительная проверка, что 'val' тоже является словарем
            if not isinstance(val, dict):
                continue

            rows.append({
                "Исходная статья": item.get('source_item', 'N/A'),
                "Период": val.get("period", "N/A"),
                "Значение": val.get("value", "N/A"),
                "Ед. изм.": item.get("unit", "")
            })
    return pd.DataFrame(rows)

def to_excel_bytes(df):
    """Конвертирует один DataFrame в байты Excel файла"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Отчет')
    return output.getvalue()

def transform_to_wide_format(long_df):
    """Преобразует данные из длинного формата в широкий"""
    if long_df.empty: return pd.DataFrame()
    wide_df = long_df.pivot_table(
        index=['Код статьи', 'Стандартизированная статья', 'Ед. изм.'],
        columns='Период',
        values='Итоговое значение',
        aggfunc='first'
    ).reset_index()
    wide_df.columns.name = None
    period_cols = sorted([col for col in wide_df.columns if col not in ['Код статьи', 'Стандартизированная статья', 'Ед. изм.']], reverse=True)
    return wide_df[['Код статьи', 'Стандартизированная статья', 'Ед. изм.'] + period_cols]


# --- ОБНОВЛЕННЫЙ ИНТЕРФЕЙС ПРИЛОЖЕНИЯ ---
st.title("📊 Извлечение и стандартизация финансовой отчетности")

st.sidebar.header("Загрузка Файлов")
uploaded_files = st.sidebar.file_uploader(
    "Загрузите сканы отчета", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True
)

if uploaded_files:
    file_names = [f.name for f in uploaded_files]
    # Сброс состояния при изменении файлов
    if "processed_data" not in st.session_state or st.session_state.get("file_names") != file_names:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.file_names = file_names

    # --- ПОШАГОВОЕ ВЫПОЛНЕНИЕ ---

    if "all_text" not in st.session_state:
        with st.spinner("Шаг 1/5: Извлечение текста..."):
            all_text = ""
            for uploaded_file in uploaded_files:
                file_text = extract_text_from_file(uploaded_file.getvalue(), uploaded_file.name)
                if file_text: all_text += f"\n\n--- НАЧАЛО ФАЙЛА: {uploaded_file.name} ---\n\n{file_text}"
            st.session_state.all_text = all_text.strip()
    
    if "report_type" not in st.session_state:
        with st.spinner("Шаг 2/5: Определение типа отчета..."):
            st.session_state.report_type = classify_report(llm, st.session_state.all_text)
    report_type = st.session_state.report_type
    if report_type == "Unknown": st.error("⚠️ Не удалось определить тип отчета."); st.stop()
    st.success(f"✅ Отчет классифицирован как **{report_type}**.")

    if "raw_data" not in st.session_state:
        with st.spinner("Шаг 3/5: Извлечение сырых данных..."):
            st.session_state.raw_data = extract_raw_financial_data(llm, st.session_state.all_text)

    if "corrected_data" not in st.session_state:
        with st.spinner("Шаг 4/5: Коррекция названий статей..."):
            st.session_state.corrected_data = correct_source_item_names(llm, st.session_state.raw_data)

    if "processed_data" not in st.session_state:
        with st.spinner("Шаг 5/5: Стандартизация и агрегация данных..."):
            response_dict = standardize_data_with_aggregation(llm, st.session_state.corrected_data, report_type)
            st.session_state.processed_data = response_dict.get("standardized_data", [])
            st.session_state.unmapped_items = response_dict.get("unmapped_items", [])

    # --- ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ ---
    st.success("✅ Обработка завершена!")
    st.header("Итоговый стандартизированный отчет")

    processed_data = st.session_state.processed_data
    if processed_data:
        flat_data = flatten_data_for_display(processed_data, report_type)
        long_df = pd.DataFrame(flat_data)
        wide_df = transform_to_wide_format(long_df)
        
        st.dataframe(wide_df, use_container_width=True, hide_index=True)
        
        excel_bytes = to_excel_bytes(wide_df)
        st.download_button(
            "📥 Скачать отчет в Excel", 
            excel_bytes, 
            f"standard_report_{report_type.replace(' ', '_')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("Не удалось стандартизировать данные для отображения.")

    unmapped_items = st.session_state.unmapped_items
    if unmapped_items:
        with st.expander("⚠️ Несопоставленные статьи"):
            unmapped_df = display_raw_data(unmapped_items)
            st.dataframe(unmapped_df, use_container_width=True, hide_index=True)

    with st.expander("🔍 Детализация процесса (для отладки)"):
        st.subheader("Скорректированные сырые данные (вход для стандартизации)")
        st.json(st.session_state.corrected_data)
        st.subheader("Стандартизированные данные (выход)")
        st.json(st.session_state.processed_data)

else:
    st.info("👈 Пожалуйста, загрузите файлы в боковой панели, чтобы начать.")
