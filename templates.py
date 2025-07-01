# templates.py

# Структура: { "Тип отчета": { "Английское название (ключ для LLM)": "Русское название (для вывода)" } }
REPORT_TEMPLATES = {
    "Balance Sheet": {
        "Property, plant and equipment": "Основные средства",
        "Investment property": "Инвестиционная собственность",
        "Goodwill": "Гудвилл",
        "Intangible assets other than goodwill": "Нематериальные активы, кроме гудвилла",
        "Investment accounted for using equity method": "Инвестиции, учитываемые долевым методом",
        "Investments in subsidiaries, joint ventures and associates": "Инвестиции в дочерние, совместные и ассоциированные компании",
        "Non-current biological assets": "Внеоборотные биологические активы",
        "Trade and other non-current receivables": "Долгосрочная дебиторская задолженность",
        "Non-current inventories": "Внеоборотные запасы",
        "Deferred tax assets": "Отложенные налоговые активы",
        "Current tax assets, non-current": "Долгосрочная дебиторская задолженность по текущему налогу",
        "Other non-current financial assets": "Прочие внеоборотные финансовые активы",
        "Other non-current non-financial assets": "Прочие внеоборотные нефинансовые активы",
        "Non-current non-cash assets pledged as collateral for which transferee has right by contract or custom to sell or repledge collateral": "Внеоборотные неденежные активы, находящиеся в залоге",
        "Non-current assets": "Итого внеоборотных активов",
        "Current inventories": "Запасы",
        "Trade and other current receivables": "Краткосрочная дебиторская задолженность",
        "Current tax assets, current": "Краткосрочная дебиторская задолженность по текущему налогу",
        "Current biological assets": "Оборотные биологические активы",
        "Other current financial assets": "Прочие оборотные финансовые активы",
        "Other current non-financial assets": "Прочие оборотные нефинансовые активы",
        "Cash and cash equivalents": "Денежные средства и эквиваленты денежных средств",
        "Current non-cash assets pledged as collateral for which transferee has right by contract or custom to sell or repledge collateral": "Оборотные неденежные активы, находящиеся в залоге",
        "Non-current assets or disposal groups classified as held for sale or as held for distribution to owners": "Внеоборотные активы и группы выбытия для продажи",
        "Current assets": "Итого оборотных активов",
        "ASSETS": "АКТИВЫ (БАЛАНС)",
        "Issued (share) capital": "Акционерный (уставный) капитал",
        "Share premium": "Эмиссионный доход",
        "Treasury shares": "Собственные акции, выкупленные у акционеров",
        "Other equity interest": "Прочий капитал организации",
        "Other reserves": "Прочие фонды",
        "Retained earnings": "Нераспределенная прибыль",
        "Non-controlling interests": "Неконтролируемые доли",
        "Equity": "Итого капитал",
        "Non-current provisions for employee benefits": "Долгосрочные резервы на вознаграждения работников",
        "Other non-current provisions": "Прочие долгосрочные резервы",
        "Trade and other non-current payables": "Долгосрочная кредиторская задолженность",
        "Deferred tax liabilities": "Отложенные налоговые обязательства",
        "Current tax liabilities, non-current": "Долгосрочная задолженность по текущему налогу",
        "Other non-current financial liabilities": "Прочие долгосрочные финансовые обязательства",
        "Other non-current non-financial liabilities": "Прочие долгосрочные нефинансовые обязательства",
        "Non-current liabilities": "Итого долгосрочных обязательств",
        "Current provisions for employee benefits": "Краткосрочные резервы на вознаграждения работников",
        "Other current provisions": "Прочие краткосрочные резервы",
        "Trade and other current payables": "Краткосрочная кредиторская задолженность",
        "Current tax liabilities, current": "Краткосрочные обязательства по текущему налогу",
        "Other current financial liabilities": "Прочие краткосрочные финансовые обязательства",
        "Other current non-financial liabilities": "Прочие краткосрочные нефинансовые обязательства",
        "Liabilities included in disposal groups classified as held for sale": "Краткосрочные обязательства для продажи",
        "Current liabilities": "Итого краткосрочных обязательств",
        "Liabilities": "Итого обязательств",
        "EQUITY AND LIABILITIES": "КАПИТАЛ И ОБЯЗАТЕЛЬСТВА (БАЛАНС)"
    },
    "Income Statement": {
        "Revenue": "Выручка",
        "Cost of sales": "Себестоимость продаж",
        "Gross profit": "Валовая прибыль",
        "Other income": "Прочие доходы",
        "Distribution costs": "Коммерческие расходы",
        "Administrative expense": "Управленческие расходы",
        "Other expense": "Прочие расходы",
        "Other gains (losses)": "Прочие прибыли (убытки)",
        "Profit (loss) from operating activities": "Прибыль (убыток) от операционной деятельности",
        "Finance income": "Финансовые доходы",
        "Finance costs": "Расходы на финансирование",
        "Profit (loss) before tax": "Прибыль (убыток) до налогообложения",
        "Income tax expense (from continuing operations)": "Расходы по налогу на прибыль",
        "Profit (loss) from continuing operations": "Прибыль (убыток) от продолжающейся деятельности",
        "Profit (loss) from discontinued operations": "Прибыль (убыток) от прекращаемой деятельности",
        "Profit (loss)": "Чистая прибыль (убыток)",
        "Other comprehensive income": "Прочий совокупный доход",
        "COMPREHENSIVE INCOME": "СОВОКУПНЫЙ ДОХОД"
    }
}

def get_report_template_as_string(report_type: str) -> str:
    """Возвращает список АНГЛИЙСКИХ названий в виде строки для промпта."""
    if report_type not in REPORT_TEMPLATES:
        return ""
    # Мы передаем в LLM только ключи (английские названия)
    english_keys = REPORT_TEMPLATES[report_type].keys()
    return "\n".join([f"- {item}" for item in english_keys])

def get_translation_map(report_type: str) -> dict:
    """Возвращает словарь для перевода { 'EN': 'RU' }."""
    return REPORT_TEMPLATES.get(report_type, {})
