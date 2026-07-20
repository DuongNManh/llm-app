from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel


class BlockType(Enum):
    UNKNOWN = "UNKNOWN"

    BODY = "BODY"

    HEADING = "HEADING"

    FOOTER = "FOOTER"

    WATERMARK = "WATERMARK"

    TOC = "TOC"


class TextBlock(BaseModel):
    page: int
    text: str


class TextPage(BaseModel):
    page_number: int
    text: str
    page_char_count: int
    page_word_count: int
    page_sentence_count: int
