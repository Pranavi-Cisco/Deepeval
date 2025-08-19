import json
import re
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from tqdm import tqdm

# === Load PDF Context ===
def load_pdf_text(path):
    loader = PyPDFLoader(path)
    pages = loader.load()
    splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = splitter.split_documents(pages)
    return " ".join([doc.page_content for doc in docs])

# === Build Instruction Prompt ===
def build_instruction(user_prompt, expected_response, pdf_context):
    return f"""
You must respond ONLY with a valid JSON object, no explanations, no markdown. Format:
{{
  "new_prompts": {{
    "prompt_1": "string",
    "prompt_2": "string"
  }}
}}

"task": "Generate two new user prompts that are semantically similar to a given user prompt and should elicit the same expected response from an HR-focused LLM bot. The bot’s knowledge base is augmented by a provided PDF document.",
"role": "QA Engineer",
"system": "HR LLM Bot",
"functionality": "Discounts and Services",
"data": "Golden set of user prompts and expected responses (provided in ‘input’ field) AND a PDF document (provided in ‘pdf_context’ field)",
"goal": "Create additional, diverse questions to effectively test the bot’s understanding and response accuracy, leveraging the information in the PDF",
"pdf_context_description": "This PDF document contains detailed information about the employee discount program, including a list of participating vendors, specific discount percentages, eligibility requirements, and instructions on how to access the discounts. The bot should be able to use this information to answer questions more comprehensively.",
"constraints": [
  "The new prompts should not be simple rewordings of the original prompt.",
  "The new prompts should explore different phrasing, context, or perspectives while maintaining the same underlying intent.",
  "The new prompts should be designed to elicit the exact same expected response as the original prompt.",
  "The new prompts should be relevant to the ‘Discounts and Services’ functionality of the HR bot.",
  "The new prompts should be designed to potentially leverage information contained within the ‘pdf_context’ document. They can ask for specific details that would be found in the PDF, even if the original prompt didn’t."
],
"input": {{
  "user_prompt": "{user_prompt}",
  "expected_response": "{expected_response}"
}},
"pdf_context": """ + f'"""{pdf_context}"""' + """,
"output_format": {{
  "type": "JSON",
  "schema": {{
    "new_prompts": {{
      "prompt_1": "string",
      "prompt_2": "string"
    }}
  }}
}}"""

# === LLM Call and Response Parser ===
def generate_prompt_variants(llm, user_prompt, expected_response, pdf_context):
    full_prompt = build_instruction(user_prompt, expected_response, pdf_context)
    response = llm.invoke(full_prompt)
    raw_output = response.content.strip()

    # Extract JSON block using regex
    try:
        json_match = re.search(r'{[\s\S]*}', raw_output)
        if not json_match:
            raise ValueError("No JSON object found in LLM output.")
        json_str = json_match.group(0)
        parsed = json.loads(json_str)
        return parsed["new_prompts"]
    except Exception as e:
        raise ValueError(f"Invalid LLM output:\n{raw_output}\n\nError: {str(e)}")

# === Main Processor ===
def process_all_pairs(input_json_path, pdf_path, output_path):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    pdf_context = load_pdf_text(pdf_path)

    with open(input_json_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    results = []
    for item in tqdm(qa_data, desc="Generating prompt variants"):
        user_prompt = item["user_prompt"]
        expected_response = item["expected_response"]
        try:
            generated = generate_prompt_variants(llm, user_prompt, expected_response, pdf_context)
            result = {
                "user_prompt": user_prompt,
                "expected_response": expected_response,
                "generated_prompts": generated
            }
        except Exception as e:
            result = {
                "user_prompt": user_prompt,
                "expected_response": expected_response,
                "error": str(e)
            }
        results.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(results)} results to {output_path}")

# === Entry point ===
if __name__ == "__main__":
    input_json_path = "deepeval_input.json"                     # Your QA pairs
    pdf_path = r"D:\Downloads\Workplace Software and Skills.pdf"  # Your PDF file
    output_path = "generated_prompt_variants.json"              # Final output
    process_all_pairs(input_json_path, pdf_path, output_path)
