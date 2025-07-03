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
# Импорт из вашего старого файла для совместимости отображения
from templates import get_translation_map, get_report_codes, REPORT_TEMPLATES

# Импорты LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.exceptions import OutputParserException

# --- КОНФИГУРАЦИЯ ---
st.set_page_config(layout="wide", page_title="Извлечение и анализ отчета")

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

# Функция extract_text_from_file остается без изменений

@st.cache_data
def extract_text_from_file(file_bytes, filename):
    """Извлекает текст из файла (PDF или изображения) с использованием OCR при необходимости"""
    try:
        ext = os.path.splitext(filename)[1].lower()
        text = ""
        
        if ext == ".pdf":
            # Попытка извлечь текст напрямую из PDF
            with fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf") as doc:
                text = "".join(page.get_text() for page in doc)
            
            # Если текста недостаточно, используем OCR
            if len(text.strip()) < 150:
                st.warning(f"Текстовый слой в '{filename}' пуст. Используется OCR...")
                images = convert_from_bytes(file_bytes, dpi=300)
                text = "\n".join([pytesseract.image_to_string(img, lang='rus+eng', config='--psm 6') for img in images])
        
        elif ext in [".png", ".jpg", ".jpeg"]:
            # Обработка изображений с помощью OCR
            image = Image.open(io.BytesIO(file_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            text = pytesseract.image_to_string(image, lang='rus+eng', config='--psm 6')
        else:
            st.error(f"Неподдерживаемый формат файла: {ext}")
            return None
            
        return text.strip()
    
    except Exception as e:
        st.error(f"Ошибка при извлечении текста из '{filename}': {e}")
        return None

# Функция classify_report остается без изменений

@st.cache_data
def classify_report(_llm, text: str) -> str:
    """Определяет тип финансового отчета с помощью LLM"""
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_template(
        "Определи тип финансового отчета. Ответь ТОЛЬКО одним из вариантов: {report_types}.\n\nТекст:\n---\n{text_snippet}\n---"
    )
    chain = prompt | _llm | parser
    report_types_str = ", ".join(REPORT_TEMPLATES.keys())
    response = chain.invoke({"text_snippet": text[:4000], "report_types": report_types_str})
    
    # Поиск действительного типа отчета в ответе LLM
    for report_type in REPORT_TEMPLATES.keys():
        if report_type in response:
            return report_type
    return "Unknown"

# Функция extract_raw_financial_data остается без изменений
@st.cache_data
def extract_raw_financial_data(_llm, text: str) -> list:
    """Извлекает финансовые показатели в сыром виде из текста отчета"""
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

# Функция correct_source_item_names остается без изменений, но использует новый список статей
@st.cache_data
def correct_source_item_names(_llm, raw_data: list) -> list:
    """Исправляет названия статей, используя полную таксономию как справочник"""
    if not raw_data: return []

    source_items_to_correct = list(set([item['source_item'] for item in raw_data]))

    parser = JsonOutputParser()
    prompt_text = (
        "Ты — корректор, исправляешь ошибки OCR в названиях финансовых статей. "
        "Ответь ТОЛЬКО JSON-объектом, где ключ — исходная строка, а значение — исправленная.\n\n"
        "ПРАВИЛА:\n"
        "1. Исправляй только явные ошибки OCR и орфографии. 'Bupyuka' -> 'Выручка'.\n"
        "2. Сохраняй смысл. 'neGuropoxaa anomxenocr' -> 'Дебиторская задолженность'.\n"
        "3. Если название корректно, верни его без изменений.\n"
        "4. В качестве ориентира используй этот список стандартных финансовых терминов: {template_items}\n\n"
        "Список для исправления:\n{items_to_correct}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        ("user", "Исправь этот список и верни JSON-объект. Включи в ответ ВСЕ ключи из исходного списка.")
    ])
    chain = prompt | _llm | parser

    try:
        items_json_string = json.dumps(source_items_to_correct, ensure_ascii=False)
        correction_map = chain.invoke({
            "template_items": "\n".join(IFRS_FLAT_LIST), # Используем полную таксономию
            "items_to_correct": items_json_string
        })

        corrected_raw_data = copy.deepcopy(raw_data)
        for item in corrected_raw_data:
            original_item = item['source_item']
            if original_item in correction_map:
                item['source_item'] = correction_map[original_item]
        return corrected_raw_data
    except Exception as e:
        st.error(f"Ошибка при коррекции названий статей: {e}")
        return raw_data

# --- ОБНОВЛЕННАЯ ФУНКЦИЯ СТАНДАРТИЗАЦИИ ---
@st.cache_data
def standardize_data_with_taxonomy(_llm, corrected_data: list) -> dict:
    """Сопоставляет сырые данные с полной таксономией МСФО"""
    parser = JsonOutputParser()
    prompt_text = (
        "Ты — эксперт по МСФО. Сопоставь каждую статью из сырых данных с ОДНОЙ, наиболее подходящей и наиболее ДЕТАЛИЗИРОВАННОЙ статьей из полной таксономии МСФО. "
        "Ответь ТОЛЬКО JSON-объектом с двумя ключами: `mapped_data` и `unmapped_items`.\n\n"
        "1. `mapped_data`: МАССИВ объектов, сопоставленных с таксономией, в формате:\n"
        "   {{\"ifrs_item\": \"Название_статьи_из_таксономии_МСФО\", \"source_item\": \"исходное_название\", \"unit\": \"ед_изм\", \"values\": [...]}}\n\n"
        "2. `unmapped_items`: МАССИВ объектов из сырых данных, которые НЕ удалось сопоставить.\n\n"
        "ПРАВИЛА:\n"
        "- Используй ТОЛЬКО статьи из этого полного списка таксономии: {ifrs_taxonomy_list}\n"
        "- Для каждой сырой статьи выбери одну, самую точную и детальную статью МСФО. 'Выручка от продажи товаров' -> 'RevenueFromSaleOfGoods', а не просто 'Revenue'.\n"
        "- Если сырая статья является итогом (например, 'Итого активы'), сопоставь ее с родительской статьей ('Assets').\n"
        "- Не агрегируй данные сам, просто сопоставь 1 к 1.\n\n"
        "Сырые данные:\n{raw_data}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        ("user", "Выполни сопоставление 1 к 1 и верни JSON.")
    ])
    chain = prompt | _llm | parser
    
    try:
        raw_data_str = json.dumps(corrected_data, ensure_ascii=False, indent=2)[:100000]
        result = chain.invoke({
            "ifrs_taxonomy_list": "\n".join(IFRS_FLAT_LIST),
            "raw_data": raw_data_str
        })
        
        if "mapped_data" not in result: result["mapped_data"] = []
        if "unmapped_items" not in result: result["unmapped_items"] = corrected_data
            
        return result
    except Exception as e:
        st.error(f"Ошибка стандартизации: {e}")
        return {"mapped_data": [], "unmapped_items": corrected_data}


# --- НОВАЯ ФУНКЦИЯ РАСЧЕТА И ВАЛИДАЦИИ ---
@st.cache_data
def calculate_and_validate(mapped_data: list, taxonomy: dict, periods: list) -> tuple:
    """Рассчитывает итоговые показатели и выполняет валидацию"""
    # Структура для хранения всех значений по периодам: {period: {ifrs_item: value}}
    report_by_period = {p: {} for p in periods}
    warnings = []
    
    # 1. Заполняем исходными сопоставленными данными
    for item in mapped_data:
        ifrs_item = item.get("ifrs_item")
        for val in item.get("values", []):
            period = val.get("period")
            value = val.get("value")
            if period in report_by_period and ifrs_item and isinstance(value, (int, float)):
                # Если статья уже есть, агрегируем (на случай, если LLM сопоставил несколько сырых статей с одной)
                report_by_period[period][ifrs_item] = report_by_period[period].get(ifrs_item, 0) + value

    # 2. Итеративно рассчитываем родительские статьи для каждого периода
    for period in periods:
        for i in range(5): # Несколько проходов для расчета зависимых итогов
            for parent, formulas in taxonomy.items():
                if parent in report_by_period[period]:
                    continue # Уже рассчитано или было в исходных
                
                # Пытаемся рассчитать по самой детальной формуле
                for formula in formulas:
                    try:
                        calculated_value = 0
                        all_components_found = True
                        for component in formula:
                            child_name = component['child']
                            sign = component['sign']
                            if child_name not in report_by_period[period]:
                                all_components_found = False
                                break # Не хватает компонента в этой формуле
                            calculated_value += sign * report_by_period[period][child_name]
                        
                        if all_components_found:
                            report_by_period[period][parent] = calculated_value
                            break # Успешно рассчитали по одной из формул, переходим к следующему родителю
                    except Exception:
                        continue # Ошибка в расчете, пробуем следующую формулу

    # 3. Валидация
    for period in periods:
        # Проверка баланса
        try:
            assets = report_by_period[period].get('Assets')
            liabilities = report_by_period[period].get('Liabilities')
            equity = report_by_period[period].get('Equity')
            
            if all(v is not None for v in [assets, liabilities, equity]):
                if abs(assets - (liabilities + equity)) > 10: # Допуск на ошибки округления
                     warnings.append(f"Период {period}: Баланс не сходится! Активы={assets}, Пассивы={liabilities + equity}")
            else:
                 warnings.append(f"Период {period}: Не удалось проверить баланс, отсутствуют ключевые компоненты (Активы/Обязательства/Капитал).")
        except Exception as e:
            warnings.append(f"Период {period}: Ошибка при проверке баланса - {e}")

    return report_by_period, warnings

# Функции для отображения display_raw_data, to_excel_bytes, transform_to_wide_format остаются без изменений
def display_raw_data(raw_data):
    """Создает DataFrame для отображения сырых данных"""
    if not raw_data:
        return pd.DataFrame()
    rows = []
    for item in raw_data:
        for val in item.get("values", []):
            rows.append({
                "Исходная статья": item.get('source_item', item.get('ifrs_item', 'N/A')),
                "Период": val.get("period", "N/A"),
                "Значение": val.get("value", "N/A"),
                "Ед. изм.": item.get("unit", "")
            })
    return pd.DataFrame(rows)

def to_excel_bytes(wide_df, long_df):
    """Конвертирует DataFrame в байты Excel файла с двумя листами"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        wide_df.to_excel(writer, index=False, sheet_name='Отчет')
        long_df.to_excel(writer, index=False, sheet_name='Детализация')
    return output.getvalue()

def transform_to_wide_format(long_df):
    """Преобразует данные из длинного формата в широкий с колонками для каждого периода"""
    # Создаем сводную таблицу
    wide_df = long_df.pivot_table(
        index=['Код статьи', 'Стандартизированная статья', 'Ед. изм.'],
        columns='Период',
        values='Итоговое значение',
        aggfunc='first'  # Берем первое значение (должно быть только одно)
    ).reset_index()
    
    # Переименуем колонки с периодами
    wide_df.columns.name = None  # Убираем название индекса колонок
    
    # Переставляем колонки: последний период первым
    period_cols = [col for col in wide_df.columns if col not in ['Код статьи', 'Стандартизированная статья', 'Ед. изм.']]
    period_cols.sort(reverse=True)  # Сортируем периоды по убыванию (новые сначала)
    wide_df = wide_df[['Код статьи', 'Стандартизированная статья', 'Ед. изм.'] + period_cols]
    
    return wide_df

# --- ОБНОВЛЕННЫЙ ИНТЕРФЕЙС ПРИЛОЖЕНИЯ ---
st.title("📊 Извлечение и анализ финансовой отчетности")

st.sidebar.header("Загрузка Файлов")
uploaded_files = st.sidebar.file_uploader(
    "Загрузите сканы отчета", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True
)

if uploaded_files:
    # --- ШАГ 1: ИЗВЛЕЧЕНИЕ ТЕКСТА ---
    if "all_text" not in st.session_state:
        with st.spinner("Шаг 1/6: Извлечение текста..."):
            all_text = ""
            for uploaded_file in uploaded_files:
                file_text = extract_text_from_file(uploaded_file.getvalue(), uploaded_file.name)
                if file_text:
                    all_text += f"\n\n--- НАЧАЛО ФАЙЛА: {uploaded_file.name} ---\n\n{file_text}"
            st.session_state.all_text = all_text.strip()

    all_text = st.session_state.all_text
    if not all_text:
        st.error("Не удалось извлечь текст.")
        st.stop()
    st.info(f"📝 Общий объем текста: {len(all_text)} символов.")

    # --- ШАГ 2: КЛАССИФИКАЦИЯ ---
    if "report_type" not in st.session_state:
        with st.spinner("Шаг 2/6: Определение типа отчета..."):
            st.session_state.report_type = classify_report(llm, all_text)
    report_type = st.session_state.report_type
    if report_type == "Unknown":
        st.error("⚠️ Не удалось определить тип отчета.")
        st.stop()
    st.success(f"✅ Отчет классифицирован как **{report_type}**.")

    # --- ШАГ 3: ИЗВЛЕЧЕНИЕ СЫРЫХ ДАННЫХ ---
    if "raw_data" not in st.session_state:
        with st.spinner("Шаг 3/6: Извлечение сырых данных..."):
            st.session_state.raw_data = extract_raw_financial_data(llm, all_text)
            
    # --- ШАГ 4: КОРРЕКЦИЯ НАЗВАНИЙ ---
    if "corrected_data" not in st.session_state:
        with st.spinner("Шаг 4/6: Коррекция названий..."):
            st.session_state.corrected_data = correct_source_item_names(llm, st.session_state.raw_data)
    
    # --- ШАГ 5: СТАНДАРТИЗАЦИЯ ---
    if "mapped_data_dict" not in st.session_state:
        with st.spinner("Шаг 5/6: Сопоставление с таксономией МСФО..."):
            st.session_state.mapped_data_dict = standardize_data_with_taxonomy(llm, st.session_state.corrected_data)
    
    mapped_data = st.session_state.mapped_data_dict["mapped_data"]
    unmapped_items = st.session_state.mapped_data_dict["unmapped_items"]

    # --- ШАГ 6: РАСЧЕТ И ВАЛИДАЦИЯ ---
    if "final_report" not in st.session_state:
        with st.spinner("Шаг 6/6: Расчет итогов и валидация..."):
            # Определяем все уникальные периоды в данных
            all_periods = set()
            for item in st.session_state.raw_data:
                for v in item.get('values', []):
                    all_periods.add(v.get('period'))
            sorted_periods = sorted(list(filter(None, all_periods)), reverse=True)

            final_report, warnings = calculate_and_validate(mapped_data, IFRS_TAXONOMY, sorted_periods)
            st.session_state.final_report = final_report
            st.session_state.validation_warnings = warnings

    # --- ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ ---
    st.success("✅ Обработка завершена!")

    if st.session_state.validation_warnings:
        st.warning("⚠️ Обнаружены проблемы при расчете и валидации:")
        for warning in st.session_state.validation_warnings:
            st.markdown(f"- `{warning}`")

    # Формируем плоскую таблицу для отображения
    final_report_data = st.session_state.final_report
    flat_list = []
    translation_dict = get_translation_map(report_type)
    codes_dict = get_report_codes(report_type) # {EN: code}
    # Инвертируем словарь для поиска по коду
    codes_to_en = {details["code"]: en_name for en_name, details in REPORT_TEMPLATES.get(report_type, {}).items()}

    # Отображаем только статьи из исходного "плоского" шаблона
    template_keys = list(REPORT_TEMPLATES.get(report_type, {}).keys())
    
    for period, values in final_report_data.items():
        for item_en, value in values.items():
            if item_en in template_keys: # Отображаем только то, что есть в шаблоне
                item_info = translation_dict.get(item_en, {})
                russian_name = item_info.get("ru", item_en)
                code = item_info.get("code", "N/A")
                
                flat_list.append({
                    "Код статьи": code,
                    "Стандартизированная статья": russian_name,
                    "Ед. изм.": "N/A", # Единицы измерения можно извлекать дополнительно
                    "Период": period,
                    "Итоговое значение": value,
                })

    if flat_list:
        long_df = pd.DataFrame(flat_list)
        long_df.sort_values(by=['Код статьи', 'Период'], ascending=[True, False], inplace=True)
        
        wide_df = transform_to_wide_format(long_df)
        
        st.header("Итоговый стандартизированный отчет")
        st.dataframe(wide_df, use_container_width=True, hide_index=True)
        
        excel_bytes = to_excel_bytes(wide_df, long_df)
        st.download_button(
            "📥 Скачать отчет в Excel", 
            excel_bytes, 
            f"standard_report_{report_type.replace(' ', '_')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("Не удалось сформировать итоговый отчет.")
        
    # Отображение несопоставленных статей
    if unmapped_items:
        with st.expander("⚠️ Несопоставленные статьи"):
            unmapped_df = display_raw_data(unmapped_items)
            st.dataframe(unmapped_df, use_container_width=True, hide_index=True)

    # Отладочная информация
    with st.expander("🔍 Детализация процесса (для отладки)"):
        st.subheader("Шаг 3: Сырые данные")
        st.json(st.session_state.raw_data)
        st.subheader("Шаг 5: Сопоставленные данные (до расчета)")
        st.json(mapped_data)
        st.subheader("Шаг 6: Полный рассчитанный отчет (все статьи)")
        st.json(st.session_state.final_report)

else:
    st.info("👈 Пожалуйста, загрузите файлы в боковой панели, чтобы начать.")
