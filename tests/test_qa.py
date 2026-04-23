"""
test_qa.py — Validate qa_pipeline.py works end-to-end.

Tests three cases:
- Normal question: expects a cited Arabic answer
- Out-of-scope question: expects a polite refusal
- Cross-article question: expects answer drawing from multiple articles

Run:
  cd ~/projects/moc-agent && uv run python tests/test_qa.py
"""

from __future__ import annotations

import asyncio

from core.qa_pipeline import ask

TESTS = [
    {
        "id": 1,
        "type": "normal",
        "question": "كيف أرجع منتجاً اشتريته من متجر إلكتروني؟",
    },
    {
        "id": 2,
        "type": "out_of_scope",
        "question": "كيف أسجل شركة في المملكة العربية السعودية؟",
    },
    {
        "id": 3,
        "type": "cross_article",
        "question": "اشتريت منتجاً معيباً ماذا أفعل وما حقوقي؟",
    },
]


async def main() -> None:
    for test in TESTS:
        print("=" * 72)
        print(f"[{test['type']}] {test['question']}")
        print("-" * 72)
        answer = await ask(test["question"])
        print(answer)
        print()


if __name__ == "__main__":
    asyncio.run(main())
