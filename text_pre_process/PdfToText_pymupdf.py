import imp
import fitz

from .text_preprocess import cleanText


def pdfToTextWithPyMuPdf(filename, clean=True):
    text = ""
    with fitz.open(filename) as doc:
        for page in doc:
            text += page.get_text()

    if clean:
        text = cleanText(text)
    return text
