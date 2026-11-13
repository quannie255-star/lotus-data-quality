# parse

职责：文档解析的独立能力层，当前提供 PDF 文本层提取。

- 文件：`pdf_extractor.py`
- 接口：`extract_pdf(path) -> ExtractResult(text, pages, warnings)`（PyMuPDF）
- 约定：扫描页没有文本层时返回 warning，不假装有内容
- 使用方：`ingest/pdf.py`
- 测试：`pytest tests/test_ingest_pdf_news.py -q`
