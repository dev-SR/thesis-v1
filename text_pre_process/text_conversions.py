from time import sleep
from utils.rich_console import console
from .PdfToTextPYPDF_textract import pdfToTextWithPyPDF2_n_Textract
from .PdfToText_pymupdf import pdfToTextWithPyMuPdf
import pandas as pd
from rich.console import Console
console = Console()


def convertPdfToTextManually(ID="54e025107a6b2e429357638b867d0452e1978779", OUTPUT_PATH="data/papers/text_demo/", PDF_PATH="data/papers", withRaw=False):
    FULL_PDF_PATH = f"{PDF_PATH}/{ID}.pdf"
    FULL_OUTPUT_PATH_CLEAN_V1 = f"{OUTPUT_PATH}/{ID}-pypdf2-textract-clean.txt"
    FULL_OUTPUT_PATH_CLEAN_V2 = f"{OUTPUT_PATH}/{ID}-pymupdf-clean.txt"
    FULL_OUTPUT_PATH_RAW_V1 = f"{OUTPUT_PATH}/{ID}-pypdf2-textract-raw.txt"
    FULL_OUTPUT_PATH_RAW_V2 = f"{OUTPUT_PATH}/{ID}-pymupdf-raw.txt"

    with open(FULL_OUTPUT_PATH_CLEAN_V1, "w", encoding="utf-8") as f:
        print("Writing to file: " + FULL_OUTPUT_PATH_CLEAN_V1)
        f.write(pdfToTextWithPyPDF2_n_Textract(FULL_PDF_PATH))

    with open(FULL_OUTPUT_PATH_CLEAN_V2, "w", encoding="utf-8") as f:
        print("Writing to file: " + FULL_OUTPUT_PATH_CLEAN_V2)
        f.write(pdfToTextWithPyMuPdf(
            FULL_PDF_PATH))

    if withRaw:
        with open(FULL_OUTPUT_PATH_RAW_V1, "w", encoding="utf-8") as f:
            f.write(pdfToTextWithPyPDF2_n_Textract(
                FULL_PDF_PATH, clean=False))

        with open(FULL_OUTPUT_PATH_RAW_V2, "w", encoding="utf-8") as f:
            f.write(pdfToTextWithPyMuPdf(
                FULL_PDF_PATH, clean=False))


def convertPdfToText(ID, PDF_PATH, OUTPUT_PATH, ERROR_PATH, METHOD="pymupdf"):
    try:
        FULL_PDF_PATH = f"{PDF_PATH}/{ID}.pdf"
        FULL_OUTPUT_PATH = f"{OUTPUT_PATH}/{ID}.txt"
        if METHOD == "pymupdf":
            kps = pdfToTextWithPyMuPdf(
                FULL_PDF_PATH)
            if kps:
                with open(FULL_OUTPUT_PATH, "w", encoding="utf-8") as f:
                    f.write(kps)
                    return True
        else:
            kps = pdfToTextWithPyPDF2_n_Textract(FULL_PDF_PATH)
            if kps:
                with open(FULL_OUTPUT_PATH, "w", encoding="utf-8") as f:
                    f.write(kps)
                return True
    except Exception as e:
        error = {
            "id": ID,
            "error": str(e),
            "method": METHOD
        }
        df = pd.DataFrame([error])
        df.to_csv(f"{ERROR_PATH}/conversion_failed.csv",
                  index=False, header=False, mode="a")
        return False
