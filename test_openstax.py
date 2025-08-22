from deepeval import assert_test, login
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    # AnswerAccuracyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric
)
import fitz  # PyMuPDF

# 🔐 Authenticate
CONFIDENT_API_KEY = "+OYuJj5xqOwXR3/CQ/M/UBFexAHXWe0d6FK3xkCY+po="
login(CONFIDENT_API_KEY)

# 📥 Load context from a PDF
def extract_text_from_pdf(path: str) -> str:
    text = ""
    with fitz.open(path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

pdf_context = extract_text_from_pdf(r"D:\Downloads\Workplace Software and Skills.pdf" )  # <-- make sure this file exists

def test_openstax():
    print("🚀 Running OpenStax test...")

    test_case = LLMTestCase(
        input="What is OpenStax?",
        actual_output="OpenStax is part of Rice University, which is a nonprofit charitable corporation.",
        expected_output="OpenStax is a nonprofit educational initiative that publishes free textbooks and learning resources.",
        context=[pdf_context],
        retrieval_context=[pdf_context]
    )

    metrics = [
        AnswerRelevancyMetric(threshold=0.8),
        # AnswerAccuracyMetric(threshold=0.8),
        FaithfulnessMetric(threshold=0.8),
        ContextualRecallMetric(threshold=0.8),
        ContextualPrecisionMetric(threshold=0.8),
    ]

    assert_test(test_case, metrics)

if __name__ == "__main__":
    test_openstax()


#Run  command : deepeval test run test_openstax.py
