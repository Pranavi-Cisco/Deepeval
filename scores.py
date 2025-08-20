from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric
)
from openai import OpenAI
import os, json
from dotenv import load_dotenv
import PyPDF2

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

PDF_PATH = r"D:\Downloads\Workplace Software and Skills.pdf"

def extract_text_from_pdf(pdf_path, max_chars=3000):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            if len(text) >= max_chars:
                break
            text += page.extract_text()
        return text[:max_chars]

qa_pairs = [
    {
        "user_prompt": "What is OpenStax?",
        "expected_response": "OpenStax is part of Rice University, which is a nonprofit charitable corporation."
    },
    {
        "user_prompt": "What are 'data crunchers'?",
        "expected_response": "Early computers were developed to be 'data crunchers' to manage large amounts of numbers."
    },
    {
        "user_prompt": "How does today's technology impact the power of computers?",
        "expected_response": "Computing innovations changed workplaces with tech like direct deposit, smartcards, paperless docs, etc."
    },
    {
        "user_prompt": "What is the advantage of e-sports in colleges and universities?",
        "expected_response": "E-sports impacts academics via game dev programs, gaming student groups, and collegiate sports. Scholarships exist."
    },
    {
        "user_prompt": "What are wearables?",
        "expected_response": "Wearables are devices like smartwatches that use computing to send/receive data via the internet."
    }
]

metrics = [
    AnswerRelevancyMetric(model="gpt-4o-mini"),
    FaithfulnessMetric(model="gpt-4o-mini"),
    ContextualPrecisionMetric(model="gpt-4o-mini"),
    ContextualRecallMetric(model="gpt-4o-mini")
]

def run_deepeval_rag(pdf_path, qa_pairs):
    full_context = extract_text_from_pdf(pdf_path)
    results = []

    for i, pair in enumerate(qa_pairs, 1):
        print(f"\n🧪 Evaluating Q{i}: {pair['user_prompt']}")

        test_case = LLMTestCase(
            input=pair["user_prompt"],
            actual_output=None,  # Will be filled after generation
            expected_output=pair["expected_response"],
            retrieval_context=[full_context]
        )

        # === Call OpenAI model to generate answer ===
        import openai
        openai.api_key = os.environ["OPENAI_API_KEY"]
        gpt_response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"{pair['user_prompt']}\n\nUse this context:\n{full_context[:4000]}"}],
            temperature=0.2
        )
        generated_answer = gpt_response.choices[0].message.content.strip()
        print(f"🤖 GPT Answer: {generated_answer}")
        test_case.actual_output = generated_answer

        # === Score with DeepEval ===
        scores = {}
        for metric in metrics:
            result = metric.measure(test_case)
            if hasattr(result, "score"):
                scores[metric.__class__.__name__] = {
                    "score": result.score,
                    "reason": result.reason
                }
                print(f"✅ {metric.__class__.__name__}: {result.score:.2f} — {result.reason}")
            else:
                scores[metric.__class__.__name__] = {
                    "score": float(result),
                    # "reason": "No reason provided"
                }

        results.append({
            "question": pair["user_prompt"],
            "generated_answer": generated_answer,
            "expected_answer": pair["expected_response"],
            "scores": scores
        })

    with open("deepeval_rag_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n📦 Results saved to deepeval_rag_results.json")

if __name__ == "__main__":
    run_deepeval_rag(PDF_PATH, qa_pairs)
