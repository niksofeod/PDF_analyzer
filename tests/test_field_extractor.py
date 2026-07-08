from core.field_extractor import extract_fields


def test_extract_inn() -> None:
    fields = extract_fields("Поставщик: ООО Ромашка\nИНН 7701234567")
    assert fields["inn"]["value"] == "7701234567"
    assert fields["inn"]["confidence"] > 0.8


def test_extract_kpp() -> None:
    fields = extract_fields("КПП: 770101001")
    assert fields["kpp"]["value"] == "770101001"


def test_extract_date() -> None:
    fields = extract_fields("Счет № A-10 от 15.03.2026")
    assert fields["document_date"]["value"] == "15.03.2026"


def test_extract_long_form_date() -> None:
    fields = extract_fields("Акт составлен от 15 марта 2026")
    assert fields["document_date"]["value"] == "15 марта 2026"


def test_extract_quoted_long_form_date() -> None:
    fields = extract_fields("г. Москва от «05» сентября 2023 г.")
    assert fields["document_date"]["value"] == "«05» сентября 2023 г"


def test_extract_organization_near_role() -> None:
    fields = extract_fields("Исполнитель: Общество с ограниченной ответственностью Ромашка\nИНН 7701234567")
    assert fields["organization_name"]["value"] == "Общество с ограниченной ответственностью Ромашка"


def test_extract_organization_after_label_line_break() -> None:
    fields = extract_fields("Поставщик:\nООО «Альфа Сервис»\nКПП 770101001")
    assert fields["organization_name"]["value"] == "ООО «Альфа Сервис»"


def test_extract_address_by_index() -> None:
    fields = extract_fields("Юридический адрес: 123456, Россия, г. Москва, ул. Ленина, д. 10, офис 5")
    assert "Москва" in fields["address"]["value"]


def test_extract_address_by_location_label() -> None:
    fields = extract_fields("Место нахождения: 190000, г. Санкт-Петербург, Невский проспект, дом 1")
    assert "Санкт-Петербург" in fields["address"]["value"]


def test_extract_person_name_from_signature_decoding() -> None:
    fields = extract_fields("Расшифровка подписи: Иванов Иван Иванович")
    assert fields["person_name"]["value"] == "Иванов Иван Иванович"


def test_extract_person_name_with_initials() -> None:
    fields = extract_fields("Генеральный директор /Иванов И.И./")
    assert fields["person_name"]["value"] == "Иванов И.И"


def test_extract_fields_from_invoice_sample_text() -> None:
    text = """
    ЗАО ''Милана''
    129366, Москва, Ракетный бульвар, 17
    ИНН 7717027908
    КПП 671010011
    Получатель
    ЗАО ''Милана''
    Сч.№
    40702810500005042124
    СЧЕТ № 000289 от 12 июня 2007 г.
    Плательщик: ООО ''Чемпион'', ИНН 6731202344/671001001, г. Смоленск, ул. 25 Сентября, 12
    №
    Наименование товара
    Руководитель предприятия ______________________________ (Иванов В.П.)
    Главный бухгалтер _____________________________________ (Петрова Г.С.)
    """
    fields = extract_fields(text)
    assert fields["document_number"]["value"] == "000289"
    assert fields["document_date"]["value"] == "12 июня 2007 г"
    assert fields["organization_name"]["value"] == 'ЗАО "Милана"'
    assert fields["address"]["value"] == "129366, Москва, Ракетный бульвар, 17"
    assert fields["person_name"]["value"] == "Иванов В.П"


def test_extract_invoice_number_with_payment_phrase() -> None:
    fields = extract_fields("Счет на оплату № Е-00035401 от 27.01.2020")
    assert fields["document_number"]["value"] == "Е-00035401"


def test_organization_name_is_not_person_name() -> None:
    fields = extract_fields('Покупатель: ФОНД ПОДДЕРЖКИ СОЦИАЛЬНЫХ ПРОГРАММ И ИНИЦИАТИВ "ЛАВКА РАДОСТЕЙ"')
    assert fields["person_name"]["value"] is None


def test_extract_person_name_near_signature_labels() -> None:
    text = """
    Исполнитель
    Генеральный директор ООО «А-Сервис»
    должность
    Швецов
    Швецов А.С.
    подпись
    расшифровка подписи
    """
    fields = extract_fields(text)
    assert fields["person_name"]["value"] == "Швецов А.С"


def test_extract_document_number() -> None:
    fields = extract_fields("Договор № AB-77/5 от 01.02.2026")
    assert fields["document_number"]["value"] == "AB-77/5"


def test_document_number_is_empty_without_explicit_number_marker() -> None:
    fields = extract_fields("Договор оказания услуг от 01.02.2026. Стороны согласовали условия.")
    assert fields["document_number"]["value"] is None


def test_empty_text_returns_empty_fields() -> None:
    fields = extract_fields("")
    assert all(payload["value"] is None for payload in fields.values())
    assert all(payload["confidence"] == 0.0 for payload in fields.values())
    assert "total_amount" not in fields
