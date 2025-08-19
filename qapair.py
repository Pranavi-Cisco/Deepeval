import os
import openai
import PyPDF2
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(pdf_path, max_pages=10):
    """Extracts text from the first few pages of a PDF."""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for i, page in enumerate(reader.pages):
            if i >= max_pages:
                break
            text += page.extract_text()
        return text

def generate_deepeval_questions(text, num_questions=30):
    """Generates DeepEval-style open-ended QA pairs using GPT."""
    system_prompt = (
        "You are a QA generation model. Given a document excerpt, generate high-quality, open-ended "
        "DeepEval-style factual questions and answers. The questions should evaluate a reader's understanding, "
        "not just recall. Make sure answers are short, clear, and correct.\n"
    )

    user_prompt = f"""Document excerpt:
{text[:6000]}

Now, generate {num_questions} QA pairs in the format:
1. Q: ...
   A: ...
2. Q: ...
   A: ...
...
"""

    response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.3
)

    qa_text = response.choices[0].message.content.strip()

    return qa_text

if __name__ == "__main__":
    # === Edit the path below ===
    pdf_path = r"D:\Downloads\Workplace Software and Skills.pdf"

    print("📄 Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)

    print("🤖 Generating DeepEval QA pairs...")
    qa_pairs = generate_deepeval_questions(text)

    output_file = "deepeval_qa_output.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(qa_pairs)

    print(f"✅ Saved {output_file}")


