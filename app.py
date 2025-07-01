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
    model_name="mistralai/mistral-7b-instruct",
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

# --- ИЗМЕНЕНИЕ №1: ОБНОВЛЕННЫЙ ПРОМПТ ДЛЯ НЕСКОЛЬКИХ ПЕРИОДОВ ---
@st.cache_data
def extract_data_with_template(_llm, text: str, report_type: str) -> list | None:
    """Извлекает данные по всем периодам из текста отчета."""
    template_items = get_report_template_as_string(report_type)
    if not template_items: return []

    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Ты — высокоточный финансовый аналитик. Твоя задача — анализировать текст отчета и возвращать результат в строго заданном формате JSON. "
         "Тебе ЗАПРЕЩЕНО добавлять любые пояснения или текст, кроме чистого JSON."
         "\n\nПРАВИЛА:"
         "\n1. Проанализируй текст отчета, который может содержать данные за НЕСКОЛЬКО ПЕРИОДОВ (годов)."
         "\n2. Для каждой СТАНДАРТНОЙ статьи из предоставленного списка в соответствии с стандартами IFRS определи СООТВЕТСТВУЮЩУЮ статью в исходном тексте из загруженного отчета, в случае необходимости произведи агрегацию значений статей."
         "\n3. Для КАЖДОГО периода, найденного в тексте (например, для каждого года-колонки), извлеки числовое значение."
         "\n4. Создай объект для каждой СТАНДАРТНОЙ статьи. Внутри него, в массиве `values_by_period`, создай отдельный объект для КАЖДОГО периода."
         "\n5. В поле 'components' для каждого периода укажи, из каких исходных статей было получено значение."
         "\n6. Если для статьи из загруженного отчета не найдено ни одной стандартной статьи из шаблона ни за один период, укажи наименование соответствующей статьи и ее значение в исходном виде."
         "\n7. Числовые значения должны быть в формате `float` или `int`. Десятичный разделитель - точка."
         "\n8. Твой ответ должен быть ТОЛЬКО JSON-массивом объектов."
         "\n\nСПИСОК СТАНДАРТНЫХ СТАТЕЙ:\n{template_items}"
         "\n\nФОРМАТ ВЫВОДА:\n```json\n"
         "[\n"
         "  {{\n"
         "    \"line_item\": \"Английское название стандартной статьи\",\n"
         "    \"unit\": \"единица измерения\",\n"
         "    \"values_by_period\": [\n"
         "      {{\n"
         "        \"period\": \"2024\",\n"
         "        \"value\": <число или null>,\n"
         "        \"components\": [ {{ \"source_item\": \"Исходная статья 1\", \"source_value\": <число> }} ]\n"
         "      }},\n"
         "      {{\n"
         "        \"period\": \"2023\",\n"
         "        \"value\": <число или null>,\n"
         "        \"components\": [ {{ \"source_item\": \"Исходная статья 1\", \"source_value\": <число> }} ]\n"
         "      }}\n"
         "    ]\n"
         "  }}\n"
         "]\n"
         "```"
         ),
        ("user", "Вот текст для анализа. Извлеки данные по ВСЕМ ПЕРИОДАМ строго по правилам и верни ТОЛЬКО JSON.\n\nТЕКСТ:\n---\n{text}\n---")
    ])

    chain = prompt | _llm | parser
    try:
        return chain.invoke({"text": text, "template_items": template_items})
    except Exception as e:
        # Упрощенная обработка ошибок для краткости
        st.error(f"Ошибка при вызове или парсинге ответа LLM: {e}")
        # Попробуем "починить" JSON, если он обернут в текст
        if hasattr(e, 'llm_output'):
            raw_output = e.llm_output
            st.warning("Пытаюсь извлечь JSON из ответа...")
            try:
                json_part = raw_output[raw_output.find('['):raw_output.rfind(']')+1]
                return json.loads(json_part)
            except Exception:
                st.error("Не удалось исправить JSON.")
                st.code(raw_output)
        return None

# --- ИЗМЕНЕНИЕ №2: НОВАЯ ФУНКЦИЯ ДЛЯ "РАЗВОРАЧИВАНИЯ" ДАННЫХ ---
def flatten_data_for_display(data: list, report_type: str) -> list:
    """Преобразует вложенную структуру данных в плоский список для DataFrame."""
    flat_list = []
    translation_map = get_translation_map(report_type)
    
    for item in data:
        english_name = item.get("line_item")
        russian_name = translation_map.get(english_name, "N/A")
        unit = item.get("unit")
        
        if not item.get("values_by_period"):
            continue

        for period_data in item["values_by_period"]:
            if period_data.get("value") is not None:
                flat_list.append({
                    "Статья (RU)": russian_name,
                    "Line Item (EN)": english_name,
                    "unit": unit,
                    "period": period_data.get("period"),
                    "value": period_data.get("value"),
                    "components": period_data.get("components", [])
                })
    return flat_list

def to_excel_bytes(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Report')
    return output.getvalue()

# --- ИНТЕРФЕЙС ПРИЛОЖЕНИЯ ---
st.title("Унификация Отчета")

st.sidebar.header("Загрузка Файлов")
uploaded_files = st.sidebar.file_uploader(
    "Загрузите сканы отчета", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True
)

if uploaded_files:
    if "processed_data" not in st.session_state or st.session_state.get("file_names") != [f.name for f in uploaded_files]:
        st.session_state.file_names = [f.name for f in uploaded_files]
        all_text = ""
        with st.spinner("Извлечение текста из файлов..."):
            for uploaded_file in uploaded_files:
                all_text += f"\n\n--- НАЧАЛО ФАЙЛА: {uploaded_file.name} ---\n\n{extract_text_from_file(uploaded_file.getvalue(), uploaded_file.name)}"
        st.session_state.all_text = all_text.strip()
        st.session_state.processed_data = None

    all_text = st.session_state.get("all_text", "")
    if not all_text:
        st.error("Не удалось извлечь текст.")
        st.stop()

    st.info(f"Общий объем текста: {len(all_text)} символов.")
    
    with st.spinner("Шаг 1/2: Классификация типа отчета..."):
        report_type = classify_report(llm, all_text)

    if report_type == "Unknown":
        st.error("Не удалось определить тип отчета.")
    else:
        st.success(f"✅ Отчет классифицирован как **{report_type}**.")
        with st.spinner("Шаг 2/2: Извлечение данных по всем периодам..."):
            structured_data = extract_data_with_template(llm, all_text, report_type)
        if structured_data is None:
            st.error("Не удалось извлечь данные.")
        else:
            st.success("✅ Данные успешно извлечены!")
            st.session_state.processed_data = structured_data

    # --- ИЗМЕНЕНИЕ №3: ОБНОВЛЕННАЯ ЛОГИКА ОТОБРАЖЕНИЯ ---
    if st.session_state.get("processed_data"):
        # Используем новую функцию для подготовки данных к отображению
        flat_data = flatten_data_for_display(st.session_state.processed_data, report_type)
        df = pd.DataFrame(flat_data)
        
        st.header("Извлеченные и стандартизированные данные")
        if not df.empty:
            def format_components(components_list):
                if not isinstance(components_list, list) or not components_list: return "Прямое сопоставление"
                def format_val(v):
                    try: return f"{float(v):,.0f}".replace(",", " ")
                    except (ValueError, TypeError): return str(v)
                return "; ".join([f"{c.get('source_item', 'N/A')} ({format_val(c.get('source_value'))})" for c in components_list])

            df['Источник агрегации'] = df['components'].apply(format_components)
            
            # Сортируем для наглядности: сначала по статье, потом по периоду (в обратном порядке)
            df.sort_values(by=['Статья (RU)', 'period'], ascending=[True, False], inplace=True)
            
            # Финальный порядок колонок
            df = df[["Статья (RU)", "value", "period", "Источник агрегации", "unit"]]
            df.rename(columns={
                'Статья (RU)': 'Стандартизированная статья',
                'value': 'Итоговое значение',
                'period': 'Период',
                'unit': 'Ед. изм.'
            }, inplace=True)

            st.dataframe(df, use_container_width=True, hide_index=True)
            
            excel_bytes = to_excel_bytes(df)
            st.download_button("📥 Скачать отчет в Excel", excel_bytes, f"standard_report_{report_type.replace(' ', '_')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.warning("В тексте не найдено ни одной статьи из шаблона с числовым значением.")
        
        with st.expander("Показать полный JSON от LLM (вложенная структура)"):
            st.json(st.session_state.processed_data)
        with st.expander("Показать весь извлеченный текст"):
            st.text_area("Распознанный текст", all_text, height=400)
else:
    st.info("Пожалуйста, загрузите файлы в боковой панели, чтобы начать анализ.")

# Сохраняем функции в session_state
st.session_state['extract_text_from_file_func'] = extract_text_from_file
st.session_state['classify_report_func'] = classify_report
st.session_state['extract_data_with_template_func'] = extract_data_with_template
