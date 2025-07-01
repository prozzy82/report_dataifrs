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
st.set_page_config(layout="wide", page_title="Унификация Финансовых Отчетов")

# --- ИНИЦИАЛИЗАЦИЯ LLM ---
# Используем st.secrets для ключа, что является лучшей практикой для Streamlit
try:
    # Для локального запуска можно использовать переменные окружения
    # Для Streamlit Cloud/Community используйте st.secrets["NOVITA_API_KEY"]
    PROVIDER_API_KEY = st.secrets.get("NOVITA_API_KEY") or os.getenv("PROVIDER_API_KEY")
    if not PROVIDER_API_KEY:
        raise KeyError
except (FileNotFoundError, KeyError):
    st.error("Ключ API не найден. Пожалуйста, установите переменную окружения PROVIDER_API_KEY или создайте файл .streamlit/secrets.toml с ключом NOVITA_API_KEY.")
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
    """Извлекает текст из PDF или изображения, предпочитая текстовый слой, а затем OCR."""
    try:
        ext = os.path.splitext(filename)[1].lower()
        text = ""

        if ext == ".pdf":
            with fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf") as doc:
                text = "".join(page.get_text() for page in doc)
            if len(text.strip()) < 150:
                st.warning(f"Текстовый слой в '{filename}' пуст. Используется OCR...")
                images = convert_from_bytes(file_bytes, dpi=300)
                ocr_texts = [pytesseract.image_to_string(img, lang='rus+eng', config='--psm 6') for img in images]
                text = "\n".join(ocr_texts)
        elif ext in [".png", ".jpg", ".jpeg"]:
            image = Image.open(io.BytesIO(file_bytes))
            text = pytesseract.image_to_string(image, lang='rus+eng', config='--psm 6')
        else:
            st.error(f"Неподдерживаемый формат файла: {ext}")
            return None
        return text.strip()
    except Exception as e:
        st.error(f"Ошибка при извлечении текста из '{filename}': {e}")
        return None

@st.cache_data
def classify_report(_llm, text: str) -> str:
    """Определяет тип отчета с помощью LLM."""
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_template(
        "Определи тип финансового отчета из текста. Ответь ТОЛЬКО одним из вариантов: {report_types}. Никаких других слов.\n\n"
        "Текст для анализа (первые 4000 символов):\n---\n{text_snippet}\n---"
    )
    chain = prompt | _llm | parser
    report_types_str = ", ".join(REPORT_TEMPLATES.keys())
    response = chain.invoke({"text_snippet": text[:4000], "report_types": report_types_str})
    for report_type in REPORT_TEMPLATES.keys():
        if report_type in response:
            return report_type
    return "Unknown"

@st.cache_data
def extract_data_with_template(_llm, text: str, report_type: str) -> list | None:
    """Извлекает данные из текста отчета, включая детализацию агрегации."""
    template_items = get_report_template_as_string(report_type)
    if not template_items:
        return []

    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Ты — высокоточный финансовый аналитик. Твоя задача — анализировать текст отчета и возвращать результат в строго заданном формате JSON. "
         "Тебе ЗАПРЕЩЕНО добавлять любые пояснения, комментарии или любой текст, кроме чистого JSON."
         "\n\nПРАВИЛА:"
         "\n1. Проанализируй текст финансового отчета."
         "\n2. Для каждой СТАНДАРТНОЙ статьи из предоставленного списка найди одну или несколько СООТВЕТСТВУЮЩИХ статей в исходном тексте."
         "\n3. Рассчитай итоговое значение 'value', просуммировав значения найденных исходных статей."
         "\n4. В поле 'components' предоставь детализацию: список всех исходных статей и их значений, которые были использованы для расчета 'value'."
         "\n5. Если статья НЕ найдена, 'value' должно быть `null`, а 'components' — пустым массивом `[]`."
         "\n6. Числовые значения должны быть в формате `float` или `int`, без пробелов-разделителей. Десятичный разделитель - точка."
         "\n7. Твой ответ должен быть ТОЛЬКО JSON-массивом объектов. Начинай с `[` и заканчивай `]`."
         "\n\nСПИСОК СТАНДАРТНЫХ СТАТЕЙ ДЛЯ ИЗВЛЕЧЕНИЯ:\n{template_items}"
         "\n\nФОРМАТ ВЫВОДА ДЛЯ КАЖДОЙ СТАТЬИ:\n```json\n"
         "  {{\n"
         "    \"line_item\": \"Английское название стандартной статьи\",\n"
         "    \"value\": <итоговое число или null>,\n"
         "    \"year\": <год или период как строка>,\n"
         "    \"unit\": \"единица измерения как строка\",\n"
         "    \"components\": [\n"
         "      {{ \"source_item\": \"Название статьи из исходного текста\", \"source_value\": <число> }},\n"
         "      {{ \"source_item\": \"Другая статья из исходного текста\", \"source_value\": <число> }}\n"
         "    ]\n"
         "  }}\n"
         "```"
         ),
        ("user",
         "Вот текст для анализа. Извлеки данные строго по правилам и верни ТОЛЬКО JSON.\n\n"
         "ТЕКСТ ОТЧЕТА:\n---\n{text}\n---"
         )
    ])

    chain = prompt | _llm | parser
    
    try:
        return chain.invoke({"text": text, "template_items": template_items})
    except OutputParserException as e:
        st.error(f"Ошибка парсинга ответа от LLM.")
        raw_output = e.llm_output
        st.warning("Ответ LLM содержал лишний текст. Пытаюсь извлечь JSON...")
        try:
            json_start = raw_output.find('[')
            json_end = raw_output.rfind(']') + 1
            if json_start != -1 and json_end != 0:
                json_part = raw_output[json_start:json_end]
                parsed_json = json.loads(json_part)
                st.success("JSON успешно извлечен из некорректного ответа!")
                return parsed_json
            else: raise ValueError("JSON-массив не найден в ответе")
        except (ValueError, json.JSONDecodeError):
            st.error("Не удалось исправить JSON. Ответ от LLM был полностью некорректным.")
            st.code(raw_output, language='text')
            return None
    except Exception as e:
        st.error(f"Произошла непредвиденная ошибка при вызове LLM: {e}")
        return None

def enrich_data_with_russian_names(data: list, report_type: str) -> list:
    """Добавляет русские названия статей в извлеченные данные, сохраняя остальные поля."""
    translation_map = get_translation_map(report_type)
    enriched_data = []
    for item in data:
        english_name = item.get("line_item")
        russian_name = translation_map.get(english_name, "Статья не найдена в шаблоне")
        
        # Копируем все исходные поля и добавляем русское название
        new_item = item.copy()
        new_item["Статья (RU)"] = russian_name
        new_item["Line Item (EN)"] = english_name
        enriched_data.append(new_item)
    return enriched_data

def to_excel_bytes(df):
    """Конвертирует DataFrame в байты Excel файла."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Report')
    return output.getvalue()

# --- ИНТЕРФЕЙС ПРИЛОЖЕНИЯ ---
st.title("Унификация и Анализ Финансовых Отчетов")

st.sidebar.header("Загрузка Файлов")
uploaded_files = st.sidebar.file_uploader(
    "Загрузите сканы отчета (можно несколько файлов)",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    # Используем session_state для хранения обработанных данных
    if "processed_data" not in st.session_state or st.session_state.get("file_names") != [f.name for f in uploaded_files]:
        st.session_state.file_names = [f.name for f in uploaded_files]
        all_text = ""
        with st.spinner("Извлечение текста из файлов..."):
            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.getvalue()
                extracted_text = extract_text_from_file(file_bytes, uploaded_file.name)
                if extracted_text:
                    all_text += f"\n\n--- НАЧАЛО ФАЙЛА: {uploaded_file.name} ---\n\n{extracted_text}"
        st.session_state.all_text = all_text.strip()
        st.session_state.processed_data = None # Сбрасываем старые данные
    
    if not st.session_state.get("all_text"):
        st.error("Не удалось извлечь текст ни из одного файла.")
        st.stop()
    
    all_text = st.session_state.get("all_text")
    st.info(f"Общий объем извлеченного текста: {len(all_text)} символов.")
    
    with st.spinner("Шаг 1/2: Классификация типа отчета..."):
        report_type = classify_report(llm, all_text)

    if report_type not in REPORT_TEMPLATES:
        st.error(f"Не удалось определить тип отчета. LLM вернул: '{report_type}'")
    else:
        st.success(f"✅ Отчет классифицирован как **{report_type}**.")
        with st.spinner("Шаг 2/2: Извлечение данных с детализацией (может занять несколько минут)..."):
            structured_data = extract_data_with_template(llm, all_text, report_type)
        if structured_data is None:
            st.error("Не удалось извлечь структурированные данные. Проверьте логи ошибок выше.")
        else:
            st.success("✅ Данные успешно извлечены и структурированы!")
            st.session_state.processed_data = structured_data

    # Отображение результата
    if st.session_state.get("processed_data"):
        final_data = enrich_data_with_russian_names(st.session_state.processed_data, report_type)
        df = pd.DataFrame([item for item in final_data if item.get('value') is not None])
        
        st.header("Извлеченные и стандартизированные данные")
        if not df.empty:
            def format_components(components_list):
                if not isinstance(components_list, list) or not components_list: return "Прямое сопоставление"
                def format_val(v):
                    try:
                        num = float(v)
                        return f"{num:,.0f}".replace(",", " ") if abs(num) > 1000 else str(num)
                    except (ValueError, TypeError): return str(v)
                return "; ".join([f"{comp.get('source_item', 'N/A')} ({format_val(comp.get('source_value'))})" for comp in components_list])

            df['Источник агрегации'] = df['components'].apply(format_components)
            
            df = df[["Статья (RU)", "value", "Источник агрегации", "year", "unit", "Line Item (EN)"]]
            df.rename(columns={
                'Статья (RU)': 'Стандартизированная статья',
                'value': 'Итоговое значение',
                'year': 'Год',
                'unit': 'Ед. изм.'
            }, inplace=True)

            st.dataframe(df, use_container_width=True, hide_index=True)
            
            excel_bytes = to_excel_bytes(df)
            st.download_button("📥 Скачать отчет в Excel", excel_bytes, f"standard_report_{report_type.replace(' ', '_')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.warning("В тексте не найдено ни одной статьи из шаблона с числовым значением.")
        
        with st.expander("Показать полный JSON от LLM (с деталями агрегации)"):
            st.json(st.session_state.processed_data)
        with st.expander("Показать весь извлеченный текст"):
            st.text_area("Распознанный текст", all_text, height=400)
else:
    st.info("Пожалуйста, загрузите файлы в боковой панели, чтобы начать анализ.")

# Сохраняем функции в session_state
st.session_state['extract_text_from_file_func'] = extract_text_from_file
st.session_state['classify_report_func'] = classify_report
st.session_state['extract_data_with_template_func'] = extract_data_with_template
