import fitz
from tqdm.auto import tqdm
from mini_rag_v1.models.text_page import TextPage, TextBlock
import pandas as pd


def text_formatter(text: str) -> str:
    """perform minor farmatting on text"""
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text


def open_and_read_pdf(pdf_path: str) -> list[TextPage]:
    """open and read pdf file, return list of TextPage"""
    try:
        doc = fitz.open(pdf_path)
        page_and_texts = []
        for page_number, page in tqdm(enumerate(doc)):
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)[
                "blocks"
            ]
            text_parts = []
            for block in blocks:
                if block["type"] == 0:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text_parts.append(span["text"])
            text = "".join(text_parts)
            text = text_formatter(text)
            page_and_texts.append(
                TextPage(
                    page_number=page_number + 1,
                    text=text,
                    page_char_count=len(text),
                    page_word_count=len(text.split(" ")),
                    page_sentence_count=len(text.split(".")),
                )
            )
        return page_and_texts
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []


page_and_texts = open_and_read_pdf(pdf_path="documents/tutuongHoChiMinh2.pdf")
page_and_texts[:10]
