# Структура: { "Тип отчета": { "Английское название (ключ для LLM)": "Русское название (для вывода)" } }
REPORT_TEMPLATES = {
    "Balance Sheet": {
        "Property, plant and equipment": {"ru": "Основные средства", "code": "BS_PPE"},
    "Investment property": {"ru": "Инвестиционная собственность", "code": "BS_INVPROP"},
    "Goodwill": {"ru": "Гудвилл", "code": "BS_GDW"},
    "Intangible assets other than goodwill": {"ru": "Нематериальные активы, кроме гудвилла", "code": "BS_INTANG"},
    "Investment accounted for using equity method": {"ru": "Инвестиции, учитываемые долевым методом", "code": "BS_EQUITYINV"},
    "Investments in subsidiaries, joint ventures and associates": {"ru": "Инвестиции в дочерние, совместные и ассоциированные компании", "code": "BS_SUBSINV"},
    "Non-current biological assets": {"ru": "Внеоборотные биологические активы", "code": "BS_NC_BIO"},
    "Trade and other non-current receivables": {"ru": "Долгосрочная дебиторская задолженность", "code": "BS_NC_RECV"},
    "Non-current inventories": {"ru": "Внеоборотные запасы", "code": "BS_NC_INV"},
    "Deferred tax assets": {"ru": "Отложенные налоговые активы", "code": "BS_DTA"},
    "Current tax assets, non-current": {"ru": "Долгосрочная дебиторская задолженность по текущему налогу", "code": "BS_NC_TAXRECV"},
    "Other non-current financial assets": {"ru": "Прочие внеоборотные финансовые активы", "code": "BS_NC_FINAST"},
    "Other non-current non-financial assets": {"ru": "Прочие внеоборотные нефинансовые активы", "code": "BS_NC_NONFIN"},
    "Non-current non-cash assets pledged as collateral for which transferee has right by contract or custom to sell or repledge collateral": {"ru": "Внеоборотные неденежные активы, находящиеся в залоге", "code": "BS_NC_PLEDGE"},
    "Current inventories": {"ru": "Запасы", "code": "BS_C_INV"},
    "Trade and other current receivables": {"ru": "Краткосрочная дебиторская задолженность", "code": "BS_C_RECV"},
    "Current tax assets, current": {"ru": "Краткосрочная дебиторская задолженность по текущему налогу", "code": "BS_C_TAXRECV"},
    "Current biological assets": {"ru": "Оборотные биологические активы", "code": "BS_C_BIO"},
    "Other current financial assets": {"ru": "Прочие оборотные финансовые активы", "code": "BS_C_FINAST"},
    "Other current non-financial assets": {"ru": "Прочие оборотные нефинансовые активы", "code": "BS_C_NONFIN"},
    "Cash and cash equivalents": {"ru": "Денежные средства и эквиваленты денежных средств", "code": "BS_CASH"},
    "Current non-cash assets pledged as collateral for which transferee has right by contract or custom to sell or repledge collateral": {"ru": "Оборотные неденежные активы, находящиеся в залоге", "code": "BS_C_PLEDGE"},
    "Non-current assets or disposal groups classified as held for sale or as held for distribution to owners": {"ru": "Внеоборотные активы и группы выбытия для продажи", "code": "BS_HELDFORSALE"},
    "Issued (share) capital": {"ru": "Акционерный (уставный) капитал", "code": "BS_SHARECAP"},
    "Share premium": {"ru": "Эмиссионный доход", "code": "BS_SHAREPREM"},
    "Treasury shares": {"ru": "Собственные акции, выкупленные у акционеров", "code": "BS_TREASURY"},
    "Other equity interest": {"ru": "Прочий капитал организации", "code": "BS_OTHEQUITY"},
    "Other reserves": {"ru": "Прочие фонды", "code": "BS_RESERVES"},
    "Retained earnings": {"ru": "Нераспределенная прибыль", "code": "BS_RETAINED"},
    "Non-controlling interests": {"ru": "Неконтролируемые доли", "code": "BS_NCI"},
    "Non-current provisions for employee benefits": {"ru": "Долгосрочные резервы на вознаграждения работников", "code": "BS_NC_EMP"},
    "Other non-current provisions": {"ru": "Прочие долгосрочные резервы", "code": "BS_NC_PROV"},
    "Trade and other non-current payables": {"ru": "Долгосрочная кредиторская задолженность", "code": "BS_NC_PAY"},
    "Deferred tax liabilities": {"ru": "Отложенные налоговые обязательства", "code": "BS_DTL"},
    "Current tax liabilities, non-current": {"ru": "Долгосрочная задолженность по текущему налогу", "code": "BS_NC_TAXPAY"},
    "Other non-current financial liabilities": {"ru": "Прочие долгосрочные финансовые обязательства", "code": "BS_NC_FINLIA"},
    "Other non-current non-financial liabilities": {"ru": "Прочие долгосрочные нефинансовые обязательства", "code": "BS_NC_NONFINLIA"},
    "Current provisions for employee benefits": {"ru": "Краткосрочные резервы на вознаграждения работников", "code": "BS_C_EMP"},
    "Other current provisions": {"ru": "Прочие краткосрочные резервы", "code": "BS_C_PROV"},
    "Trade and other current payables": {"ru": "Краткосрочная кредиторская задолженность", "code": "BS_C_PAY"},
    "Current tax liabilities, current": {"ru": "Краткосрочные обязательства по текущему налогу", "code": "BS_C_TAXPAY"},
    "Other current financial liabilities": {"ru": "Прочие краткосрочные финансовые обязательства", "code": "BS_C_FINLIA"},
    "Other current non-financial liabilities": {"ru": "Прочие краткосрочные нефинансовые обязательства", "code": "BS_C_NONFINLIA"},
    "Liabilities included in disposal groups classified as held for sale": {"ru": "Краткосрочные обязательства для продажи", "code": "BS_SALELIA"}
    },
    "Income Statement": {
        "Revenue": {"ru": "Выручка", "code": "IS_REV"},
    "Cost of sales": {"ru": "Себестоимость продаж", "code": "IS_COS"},
    "Other income": {"ru": "Прочие доходы", "code": "IS_OTHINC"},
    "Distribution costs": {"ru": "Коммерческие расходы", "code": "IS_DIST"},
    "Administrative expense": {"ru": "Управленческие расходы", "code": "IS_ADMIN"},
    "Other expense": {"ru": "Прочие расходы", "code": "IS_OTHEXP"},
    "Other gains (losses)": {"ru": "Прочие прибыли (убытки)", "code": "IS_GAINLOSS"},
    "Finance income": {"ru": "Финансовые доходы", "code": "IS_FININC"},
    "Finance costs": {"ru": "Расходы на финансирование", "code": "IS_FINEXP"},
    "Income tax expense (from continuing operations)": {"ru": "Расходы по налогу на прибыль", "code": "IS_TAX"},
    "Profit (loss) from continuing operations": {"ru": "Прибыль (убыток) от продолжающейся деятельности", "code": "IS_CONT_OPS"},
    "Profit (loss) from discontinued operations": {"ru": "Прибыль (убыток) от прекращаемой деятельности", "code": "IS_DISC_OPS"},
    "Other comprehensive income": {"ru": "Прочий совокупный доход", "code": "IS_OCI"}
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

def get_report_codes(report_type: str) -> dict:
    """Возвращает словарь кодов для статей отчета"""
    template = REPORT_TEMPLATES.get(report_type, {})
    return {item: details["code"] for item, details in template.items()}
