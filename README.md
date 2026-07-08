# Система автоматического распознавания PDF-документов

Учебный MVP-проект для программы «Прикладные задачи машинного обучения и анализа больших данных». Система принимает PDF, определяет тип PDF, извлекает текст, классифицирует документ, достает ключевые поля, ищет подписи и печати как визуальные объекты и сохраняет структурированный JSON.

## Цель

Показать реалистичную локальную Document AI-архитектуру, которую можно развивать: от простых эвристик и baseline ML до OCR, layout-aware моделей и детекторов объектов.

## Архитектура

```text
app/                 FastAPI, CLI, Pydantic-схемы
core/                Основные модули обработки PDF
data/raw/            Загружаемые PDF
data/processed/      Промежуточные данные
data/sample_texts/   Небольшой корпус для baseline-классификатора
models/              Обученные модели, например document_classifier.pkl
outputs/json/        Итоговые JSON
outputs/text/        Извлеченный текст
outputs/tables/      CSV-таблицы
outputs/visualizations/ PNG с bounding boxes
streamlit_app/       Веб-интерфейс Streamlit
tests/               Pytest-тесты
```

## Pipeline

1. Проверка входного PDF.
2. Генерация `document_id`.
3. Определение типа PDF через PyMuPDF: `text`, `scanned`, `mixed`, `unknown`.
4. Извлечение текстового слоя через PyMuPDF.
5. OCR через Tesseract/pytesseract для сканов или смешанных PDF без текста.
6. Классификация типа документа: ML-модель из `models/document_classifier.pkl` или rule-based fallback.
7. Извлечение полей регулярными выражениями и поиском значений рядом с ключевыми словами.
8. Базовое извлечение таблиц через pdfplumber или текстовый fallback.
9. Поиск подписей и печатей через OpenCV-контуры и Hough Circle Transform.
10. Постобработка ИНН, КПП и дат.
11. Сохранение `.txt`, `.json`, `.csv` и PNG-визуализаций.

## Стек

Python 3.11+, PyMuPDF, pytesseract, OpenCV, Pillow, numpy, pandas, scikit-learn, FastAPI, Uvicorn, Streamlit, Pydantic, pytest, python-multipart, pdfplumber.

Система работает локально и не использует платные API или облачные сервисы.

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Если PowerShell пишет, что выполнение сценариев отключено, активацию можно не использовать. Запускайте инструменты напрямую через Python из виртуального окружения:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
.\.venv\Scripts\python.exe -m streamlit run streamlit_app/app.py
.\.venv\Scripts\python.exe -m app.cli process path\to\file.pdf
```

Альтернативно можно разрешить выполнение скриптов только для текущего окна PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Для OCR нужен установленный системный Tesseract. Если Tesseract не найден, pipeline не падает: он добавляет warning и возвращает пустой OCR-текст.

## Запуск FastAPI

```bash
uvicorn app.main:app --reload
```

Основные роуты:

- `GET /health`
- `POST /documents/upload`
- `GET /documents/{document_id}`
- `GET /documents/{document_id}/text`
- `GET /documents/{document_id}/visualization`

## Запуск Streamlit

```bash
streamlit run streamlit_app/app.py
```

Интерфейс позволяет загрузить PDF и посмотреть тип PDF, тип документа, поля, подписи, печати, визуализации, текст и сырой JSON.

## Запуск CLI

```bash
python -m app.cli process data/raw/example.pdf
python -m app.cli process data/raw/example.pdf --json
python -m app.cli train-classifier
```

Команда `train-classifier` обучает baseline-модель классификации документов на текстах из `data/sample_texts/` и сохраняет результат в `models/document_classifier.pkl`.

## Запуск тестов

```bash
pytest
```

Тесты проверяют rule-based классификацию, извлечение полей, устойчивость PDF type detector, структуру visual mark detector и базовый контракт pipeline.

## ML-часть

`core/document_classifier.py` поддерживает два режима:

- baseline-модель TF-IDF + LogisticRegression из `models/document_classifier.pkl`;
- rule-based fallback по ключевым словам, если модель отсутствует или не загрузилась.

Для обучения используются тексты из `data/sample_texts/<class>`. В проекте уже есть небольшой учебный корпус по классам `invoice`, `contract`, `act`, `waybill`, `statement`. Чтобы улучшить качество, добавляйте реальные распознанные тексты документов в соответствующие папки и запускайте:

```powershell
.\.venv\Scripts\python.exe -m app.cli train-classifier
```

После переобучения pipeline автоматически начнет использовать обновленный `models/document_classifier.pkl`.

### Реальные внешние данные

В `data/external/` скачаны публичные датасеты:

- `CUAD` из репозитория The Atticus Project: реальные тексты договоров для класса `contract`.
- `FATURA` с Zenodo: изображения invoice-документов и layout-аннотации для OCR/layout-экспериментов.

Из CUAD импортированы реальные contract-тексты в `data/sample_texts/contract/` с помощью:

```powershell
.\.venv\Scripts\python.exe scripts\import_cuad_contracts.py --limit 20
.\.venv\Scripts\python.exe -m app.cli train-classifier
```

FATURA не добавляется напрямую в TF-IDF-классификатор, потому что содержит изображения и разметку областей, а не готовый plain text. Его можно использовать дальше для OCR, layout analysis и проверки invoice-пайплайна.

## OCR-часть

OCR реализован через `pytesseract`. Перед распознаванием применяется цепочка:

- grayscale;
- повышение контраста;
- denoising;
- thresholding.

При отсутствии Tesseract возвращается понятное предупреждение, а остальные этапы pipeline продолжают работать.

## Извлечение полей

Модуль `core/field_extractor.py` извлекает:

- `document_number`;
- `document_date`;
- `organization_name`;
- `inn`;
- `kpp`;
- `address`;
- `person_name`.

Используются регулярные выражения, словари ключевых слов и простая логика поиска значения рядом с ключом.

## Подписи и печати

`core/signature_stamp_detector.py` рендерит страницы PDF через PyMuPDF и анализирует изображения OpenCV:

- подписи ищутся как вытянутые контуры;
- печати ищутся через Hough Circle Transform;
- дополнительно учитывается близость к словам «подпись», «печать», «М.П.»;
- confidence эвристический и зависит от формы, размера и положения объекта.

Это не полноценная object detection модель, а демонстрационный CV-подход для MVP.

## Пример JSON

```json
{
  "document_id": "invoice_abc123",
  "file_name": "document.pdf",
  "pdf_type": "text",
  "document_type": {
    "label": "invoice",
    "confidence": 0.8
  },
  "fields": {
    "document_number": {"value": "INV-001", "confidence": 0.78},
    "document_date": {"value": "2026-03-01", "confidence": 0.78},
    "organization_name": {"value": "ООО Ромашка", "confidence": 0.72},
    "inn": {"value": "7701234567", "confidence": 0.95},
    "kpp": {"value": "770101001", "confidence": 0.95},
    "address": {"value": null, "confidence": 0.0},
    "person_name": {"value": null, "confidence": 0.0}
  },
  "tables": [],
  "visual_marks": {
    "signatures": [],
    "stamps": []
  },
  "text_path": "outputs/text/invoice_abc123.txt",
  "json_path": "outputs/json/invoice_abc123.json",
  "visualization_paths": [],
  "warnings": []
}
```

## Метрики качества

- OCR: CER, WER.
- Классификация: Accuracy, Precision, Recall, F1-score.
- Извлечение полей: Exact Match, Precision, Recall, F1-score.
- Подписи и печати: Precision, Recall, IoU для bounding boxes.

## Ограничения MVP

- OCR зависит от установленного Tesseract и качества скана.
- Rule-based классификация уступает обученной модели на реальных документах.
- Извлечение полей ориентировано на типовые русскоязычные документы.
- Таблицы извлекаются базово и требуют доработки для сложных layout.
- Поиск подписей и печатей эвристический, без нейросетевого object detector.

## Возможные улучшения

- PaddleOCR для более сильного OCR.
- LayoutLMv3 или Donut для layout-aware понимания документов.
- YOLO/Detectron2 для подписей и печатей.
- Обучение NER-модели для реквизитов.
- Полнотекстовый поиск по обработанным документам.
- PostgreSQL для хранения результатов и истории обработки.
- Очередь фоновых задач для больших PDF.
