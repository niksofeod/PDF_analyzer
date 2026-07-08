from core.document_classifier import classify_by_rules


def test_rule_based_invoice() -> None:
    result = classify_by_rules("Счет на оплату. Поставщик и покупатель указаны ниже.")
    assert result["label"] == "invoice"


def test_rule_based_contract() -> None:
    result = classify_by_rules("Договор оказания услуг. Стороны определили предмет договора.")
    assert result["label"] == "contract"


def test_rule_based_act() -> None:
    result = classify_by_rules("Акт выполненных работ. Работы сдали и приняли.")
    assert result["label"] == "act"


def test_rule_based_waybill() -> None:
    result = classify_by_rules("Товарная накладная ТОРГ-12. Грузоотправитель указан.")
    assert result["label"] == "waybill"


def test_rule_based_statement() -> None:
    result = classify_by_rules("Банковская выписка: движение средств и остаток на счете.")
    assert result["label"] == "statement"


def test_rule_based_unknown() -> None:
    result = classify_by_rules("Произвольный текст без деловых ключевых слов.")
    assert result["label"] == "unknown"
    assert result["confidence"] == 0.0

