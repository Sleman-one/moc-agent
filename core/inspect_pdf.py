import pymupdf
import os

PDF_PATH = "data/raw/مشروع نظام حماية المستهلك.pdf"


def inspect_pdf(path: str):

    doc = pymupdf.open(path)

    print(f"Total Pages: {len(doc)}")
    print("=" * 60)

    warning_num = 0
    for i, page in enumerate(doc):
        text = page.get_text()

        print(f"\n--- Page {i+1} ---")
        print(f"Characters extracted: {len(text)}")

        if len(text.strip()) == 0:
            warning_num += 1
            print(
                f"WARNING {warning_num}: No text found on this page - may be scanned image"
            )

        else:
            print(text[:400])
            print("...")

        if i >= 10:
            print("\n Stoped at page 10")
            break


if __name__ == "__main__":
    inspect_pdf(PDF_PATH)
