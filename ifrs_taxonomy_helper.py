import json
from difflib import get_close_matches

# Загрузка таксономии IFRS из JSON-файла
def load_ifrs_taxonomy(path="json_taxonomy_rus.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Поиск возможного соответствия по смыслу
def suggest_mapping_from_taxonomy(source_item: str, ifrs_dict: dict, max_suggestions=1, cutoff=0.6):
    candidates = list(ifrs_dict.values())  # русские названия из таксономии
    matches = get_close_matches(source_item, candidates, n=max_suggestions, cutoff=cutoff)
    return matches[0] if matches else None

# Построение таблицы подсказок
def build_unmapped_with_suggestions(unmapped_items: list, ifrs_dict: dict) -> list:
    rows = []
    for item in unmapped_items:
        source = item.get("source_item", "")
        suggestion = suggest_mapping_from_taxonomy(source, ifrs_dict)
        for val in item.get("values", []):
            rows.append({
                "Исходная статья": source,
                "Период": val.get("period", "N/A"),
                "Значение": val.get("value", "N/A"),
                "Ед. изм.": item.get("unit", ""),
                "Возможное соответствие IFRS": suggestion or "—"
            })
    return rows