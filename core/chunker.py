import pymupdf
import re
import json

# CRITICAL: Arabic PDFs use ligature "املادة" not "المادة"
ARTICLE_PATTERN_PDF = (
    r"(املادة\s+[\u0600-\u06FF\s]+?:\s*[\u0600-\u06FF\s]{1,60}?)(?=\n)"
)
ARTICLE_PATTERN_TXT = (
    r"(المادة\s+[\u0600-\u06FF\s]+?:\s*[\u0600-\u06FF\s]{1,60}?)(?=\n)"
)


def extract_text_from_pdf(pdf_path: str) -> str:
    doc = pymupdf.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text


def extract_text_from_txt(txt_path: str) -> str:
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()


def preprocess_text(text: str) -> str:
    """
    Step 1: Remove entire lines that are chapter/section headers.

    Why whole lines: removing just the words الباب or الفصل leaves
    the rest of the line ": أحكام عامة" as floating garbage that
    contaminates the body of the next chunk.

    We also collapse split article headers. The PDF sometimes breaks
    "المادة الرابعة" across two lines as "المادة\nالرابعة". We join
    those before the colon pattern runs, otherwise the pattern misses them.
    """
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Remove entire line if it starts with a chapter or section marker
        if stripped.startswith("الباب") or stripped.startswith("الفصل"):
            continue
        # Also remove page numbers and the draft watermark
        if stripped == "مـــسـودة" or stripped.isdigit():
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)

    # Collapse split article headers: "املادة\nالرابعة:" → "املادة الرابعة:"
    # The PDF breaks long ordinals across lines. We join them if the next
    # line continues with Arabic words (no colon yet on current line).
    text = re.sub(r"(املادة)\n([\u0600-\u06FF]+)", r"\1 \2", text)
    text = re.sub(r"(المادة)\n([\u0600-\u06FF]+)", r"\1 \2", text)

    return text


def chunk_by_article(text: str, pattern: str, source_law: str) -> list[dict]:
    """
    Split text on article markers that contain a colon.

    Why require a colon: every real article header follows the pattern
    "المادة [ordinal]: [title]". False positives from mid-sentence
    fragments never have this colon structure, so requiring it eliminates
    them without needing any length thresholds to tune.
    """
    parts = re.split(pattern, text)
    chunks = []

    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        header = " ".join(header.split())
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""

        # Skip genuinely empty bodies
        if len(body) < 50:
            continue

        chunks.append(
            {
                "id": f"{source_law}_{header}",
                "article_header": header,
                "content": f"{header}\n{body}",
                "source_law": source_law,
                "char_count": len(body),
            }
        )

    return chunks


if __name__ == "__main__":
    # Consumer Protection Law
    cpl_raw = extract_text_from_pdf("data/raw/مشروع نظام حماية المستهلك.pdf")
    cpl_clean = preprocess_text(cpl_raw)
    cpl_chunks = chunk_by_article(
        cpl_clean, ARTICLE_PATTERN_PDF, "مشروع_نظام_حماية_المستهلك"
    )

    # E-Commerce Law
    ecl_raw = extract_text_from_txt("data/raw/Ecommerce_law_clean.txt")
    ecl_clean = preprocess_text(ecl_raw)
    ecl_chunks = chunk_by_article(
        ecl_clean, ARTICLE_PATTERN_TXT, "نظام_التجارة_الإلكترونية"
    )

    all_chunks = cpl_chunks + ecl_chunks

    print(f"CPL chunks: {len(cpl_chunks)}")
    print(f"ECL chunks: {len(ecl_chunks)}")
    print(f"Total: {len(all_chunks)}")

    # Coverage check — the only count that matters
    GOLDEN_ARTICLES_CPL = [
        "احلادية واألربعون",  # Article 41 - return period online
        "السادسة والثالثون",  # Article 36 - return period off-premises
        "احلادية واخلمسون",  # Article 51 - warranty duration
        "السابعة عشرة",  # Article 17 - unfair practices
        "الثامنة والسبعون",  # Article 78 - admin penalties
        "التاسعة والسبعون",  # Article 79 - criminal penalties
        "التاسعة واألربعون",  # Article 49 - statutory warranty
        "الثالثة واخلمسون",  # Article 53 - warranty breach rights
    ]
    GOLDEN_ARTICLES_ECL = [
        "الثالثة عشرة",  # Article 13 - cancellation right
        "الرابعة عشرة",  # Article 14 - late delivery
        "السابعة عشرة",  # Article 17 - Ministry powers
        "الثامنة عشرة",  # Article 18 - admin penalties
    ]

    print("\n--- Golden set coverage check ---")
    cpl_headers = [c["article_header"] for c in cpl_chunks]
    for article in GOLDEN_ARTICLES_CPL:
        found = any(article in h for h in cpl_headers)
        print(f"  CPL {article}: {'✓' if found else '✗ MISSING'}")

    ecl_headers = [c["article_header"] for c in ecl_chunks]
    for article in GOLDEN_ARTICLES_ECL:
        found = any(article in h for h in ecl_headers)
        print(f"  ECL {article}: {'✓' if found else '✗ MISSING'}")

    with open("data/processed/chunks_preview.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print("\nChunks written to data/processed/chunks_preview.json")
