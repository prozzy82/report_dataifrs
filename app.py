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
st.set_page_config(layout="wide", page_title="Унификация отчета")

# --- ИНИЦИАЛИЗАЦИЯ LLM ---
# Используем st.secrets для ключа, что является лучшей практикой для Streamlit
try:
    PROVIDER_API_KEY = os.getenv("PROVIDER_API_KEY")
except (FileNotFoundError, KeyError):
    st.error("Ключ 'NOVITA_API_KEY' не найден. Пожалуйста, создайте файл .streamlit/secrets.toml и добавьте его.")
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
            # Попытка №1: Извлечь текстовый слой (быстро и точно)
            with fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf") as doc:
                text = "".join(page.get_text() for page in doc)

            # Попытка №2: Если текста мало (вероятно, скан), используем OCR
            if len(text.strip()) < 150:
                st.warning(f"В '{filename}' текстовый слой пуст или мал. Используется OCR (это может занять время)...")
                images = convert_from_bytes(file_bytes, dpi=300)
                ocr_texts = []
                for i, image in enumerate(images):
                    st.sidebar.text(f"  - Обработка OCR стр. {i+1}/{len(images)}...")
                    ocr_texts.append(pytesseract.image_to_string(image, lang='rus+eng', config='--psm 6'))
                text = "\n".join(ocr_texts)

        elif ext in [".png", ".jpg", ".jpeg"]:
            # Обработка изображений через OCR
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

    # Постобработка для надежности
    for report_type in REPORT_TEMPLATES.keys():
        if report_type in response:
            return report_type
    return "Unknown"

@st.cache_data
def extract_data_with_template(_llm, text: str, report_type: str) -> list | None:
    """Извлекает данные из текста отчета по заданному шаблону с улучшенным промптом."""
    template_items = get_report_template_as_string(report_type)
    if not template_items:
        return []

    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Ты — высокоточный робот для извлечения данных. Твоя единственная задача — анализировать текст и возвращать результат в формате JSON. "
         "Тебе ЗАПРЕЩЕНО добавлять любые пояснения, комментарии, размышления или любой другой текст, кроме чистого JSON."
         "\n\nПРАВИЛА:"
         "\n1. Проанализируй текст финансового отчета."
         "\n2. Найди значения для каждой статьи из предоставленного списка."
         "\n3. Если статья из списка найдена в тексте, извлеки ее числовое значение ('value'), год ('year') и единицу измерения ('unit'). Числовое значение должно быть в формате `float` или `int`, без пробелов-разделителей. Десятичный разделитель - точка."
         "\n4. Если статья из списка НЕ найдена в тексте, значение 'value' должно быть `null`."
         "\n5. Твой ответ должен быть ТОЛЬКО JSON-массивом объектов. Начинай ответ с `[` и заканчивай `]`."
         "\n6. Не учитывай колонку с кодами статей или примечаниями к статьям - Прим. или Код, обрабатывай только колонки со значениями статей по периодам. "
         "\n\nСПИСОК СТАТЕЙ ДЛЯ ИЗВЛЕЧЕНИЯ:\n{template_items}"
         "\n\nФОРМАТ ВЫВОДА:\n```json\n"
         "[\n"
         "  {{\n"
         "    \"line_item\": \"Английское название статьи из списка\",\n"
         "    \"value\": <число или null>,\n"
         "    \"year\": <год или период как строка>,\n"
         "    \"unit\": \"единица измерения как строка\"\n"
         "  }}\n"
         "]\n"
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
            # Ищем начало и конец JSON-массива в "грязном" ответе
            json_start = raw_output.find('[')
            json_end = raw_output.rfind(']') + 1
            if json_start != -1 and json_end != 0:
                json_part = raw_output[json_start:json_end]
                parsed_json = json.loads(json_part)
                st.success("JSON успешно извлечен из некорректного ответа!")
                return parsed_json
            else:
                raise ValueError("JSON-массив не найден в ответе")
        except (ValueError, json.JSONDecodeError):
            st.error("Не удалось исправить JSON. Ответ от LLM был полностью некорректным.")
            st.code(raw_output, language='text') # Показываем "сырой" ответ для отладки
            return None
    except Exception as e:
        st.error(f"Произошла непредвиденная ошибка при вызове LLM: {e}")
        return None

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

def to_excel_bytes(df):
    """Конвертирует DataFrame в байты Excel файла."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Report')
    return output.getvalue()

# --- ИНТЕРФЕЙС ПРИЛОЖЕНИЯ ---
st.title("🤖 Унификация финансовой отчетности")

st.sidebar.header("Загрузка файлов")
uploaded_files = st.sidebar.file_uploader(
    "Загрузите сканы отчета (один или несколько файлов)",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    # Используем session_state для хранения обработанных данных
    if "processed_data" not in st.session_state or st.session_state.get("file_names") != [f.name for f in uploaded_files]:
        st.session_state.file_names = [f.name for f in uploaded_files]
        all_text = ""
        st.sidebar.subheader("Прогресс обработки:")
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()
            st.sidebar.text(f"-> Файл: {uploaded_file.name}")
            extracted_text = extract_text_from_file(file_bytes, uploaded_file.name)
            if extracted_text:
                all_text += f"\n\n--- НАЧАЛО ФАЙЛА: {uploaded_file.name} ---\n\n{extracted_text}\n\n--- КОНЕЦ ФАЙЛА: {uploaded_file.name} ---"
        st.session_state.all_text = all_text.strip()
        st.session_state.processed_data = None # Сбрасываем старые данные
    
    if not st.session_state.get("all_text"):
        st.error("Не удалось извлечь текст ни из одного файла.")
        st.stop()
    
    all_text = st.session_state.get("all_text")
    st.info(f"Общий объем извлеченного текста: {len(all_text)} символов.")
    
    # Классификация и извлечение данных
    with st.spinner("Шаг 1/2: Классификация типа отчета..."):
        report_type = classify_report(llm, all_text)

    if report_type not in REPORT_TEMPLATES:
        st.error(f"Не удалось определить тип отчета. LLM вернул: '{report_type}'")
    else:
        st.success(f"✅ Отчет классифицирован как **{report_type}**.")
        
        with st.spinner("Шаг 2/2: Извлечение данных по шаблону (может занять несколько минут)..."):
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
            df = df[["Статья (RU)", "Line Item (EN)", "value", "year", "unit"]]
            st.dataframe(df, use_container_width=True)
            
            excel_bytes = to_excel_bytes(df)
            st.download_button(
                label="📥 Скачать отчет в Excel",
                data=excel_bytes,
                file_name=f"standard_report_{report_type.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("В тексте не найдено ни одной статьи из шаблона с числовым значением.")
        
        with st.expander("Показать полный JSON от LLM (до обогащения)"):
            st.json(st.session_state.processed_data)
        with st.expander("Показать весь извлеченный текст"):
            st.text_area("Распознанный текст", all_text, height=400)

else:
    st.info("Пожалуйста, загрузите файлы в боковой панели, чтобы начать анализ.")

# Сохраняем функции в session_state
st.session_state['extract_text_from_file_func'] = extract_text_from_file
st.session_state['classify_report_func'] = classify_report
st.session_state['extract_data_with_template_func'] = extract_data_with_template
