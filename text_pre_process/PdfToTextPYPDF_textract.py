#!/usr/bin/env python3

import PyPDF2
import textract

from .text_preprocess import cleanText


def pdfToTextWithPyPDF2_n_Textract(filename, clean=True):
    text = ""
    try:
        pdfFileObj = open(filename, 'rb')
        pdfReader = PyPDF2.PdfReader(filename, strict=False)
        num_pages = pdfReader.numPages
        for i in range(0, num_pages):
            text += pdfReader.getPage(i).extractText()
    except Exception as e:
        # print(e)
        pass
    # This if statement exists to check if the above library returned words. It's done because PyPDF2 cannot read scanned files.
    if text != "":
        text = text
    # If the above returns as False, we run the OCR library textract to #convert scanned/image based PDF files into text.
    else:
        text = textract.process(filename, method='pdfminer', language='eng')
    if clean:
        return cleanText(text)
    else:
        return text
