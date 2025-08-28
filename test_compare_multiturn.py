# test_compare_multiturn.py
from openai import OpenAI
from deepeval import assert_test, evaluate
from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.metrics import (
    # ConversationRelevancyMetric,
    KnowledgeRetentionMetric,
    ConversationCompletenessMetric,
    RoleAdherenceMetric,
)

client = OpenAI()

from deepeval import login

CONFIDENT_API_KEY = "+OYuJj5xqOwXR3/CQ/M/UBFexAHXWe0d6FK3xkCY+po="  # Replace with env var in production
login(CONFIDENT_API_KEY)
# Multi-turn wearable questions (user side)
questions = [
    "What are wearables?",
    "Give some examples of wearable technology.",
    "How are wearables used in practice?",
    "How big is the wearables market?",
]

# Generate assistant responses using gpt-4o-mini
turns = []
for q in questions:
    turns.append(Turn(role="user", content=q))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": q}],
        temperature=1,
    )
    answer = resp.choices[0].message.content.strip()
    turns.append(Turn(role="assistant", content=answer))

# Define the test case
wearables_convo = ConversationalTestCase(
    chatbot_role=(
        "You are a concise, helpful tech explainer. "
        "Stay factual and on-topic. Cover definitions, examples, uses, and scale of wearables."
    ),
    turns=turns,
)

# Multi-turn conversational metrics
metrics = [
    # ConversationRelevancyMetric(window_size=3, threshold=0.7, verbose_mode=True),
    KnowledgeRetentionMetric(threshold=0.7, verbose_mode=True),
    ConversationCompletenessMetric(threshold=0.7, verbose_mode=True),
    RoleAdherenceMetric(threshold=0.7, verbose_mode=True),
]

# Programmatic run
if __name__ == "__main__":
    evaluate([wearables_convo], metrics)

# Pytest entrypoint
def test_wearables_multi_turn():
    assert_test(wearables_convo, metrics)
