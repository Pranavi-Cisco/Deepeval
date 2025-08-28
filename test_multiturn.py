# test_multiturn.py
# ------------------------------------------------------------
# Multi-turn QA evaluation with DeepEval (conversational metrics)
# DeepEval >= 3.x expects ConversationalTestCase(turns=[Turn(role, content), ...])
#
# Run:
#   pip install -U deepeval
#   export OPENAI_API_KEY=...   # or set $env:OPENAI_API_KEY="..." on Windows
#   deepeval test run test_multiturn.py
# ------------------------------------------------------------

from deepeval import assert_test, evaluate
from deepeval.test_case import ConversationalTestCase, Turn
# Depending on your install, these are usually available from deepeval.metrics
# If your IDE can't find them, try: from deepeval.metrics.conversational import ...
from deepeval.metrics import (
    KnowledgeRetentionMetric,
    ConversationCompletenessMetric,
    RoleAdherenceMetric,
    # ConversationRelevancyMetric,   # if missing in your env, comment out and run the rest
)
from deepeval import login

CONFIDENT_API_KEY = "+OYuJj5xqOwXR3/CQ/M/UBFexAHXWe0d6FK3xkCY+po="  # Replace with env var in production
login(CONFIDENT_API_KEY)

# --- Multi-turn conversation (role/content), based on your PDF’s wearables section ---
turns = [
    Turn(role="user",      content="What are wearables?"),
    Turn(role="assistant", content=(
        "A wearable is a device that uses computing technology to collect and receive data via the internet. "
        "They can be worn on the body, incorporated into clothing, or even embedded under the skin."
    )),

    Turn(role="user",      content="Give some examples of wearable technology."),
    Turn(role="assistant", content=(
        "Common examples include smartwatches, smart rings that track sleep and fitness, "
        "heart-rate monitors, and medical alert devices."
    )),

    Turn(role="user",      content="How are wearables used in practice?"),
    Turn(role="assistant", content=(
        "Athletes use them to track performance and concussion risks; parents use them for children’s location tracking; "
        "healthcare wearables can detect early signs of illnesses; and events use wearables for ticketing."
    )),

    Turn(role="user",      content="How big is the wearables market?"),
    Turn(role="assistant", content=(
        "There are roughly a billion active devices globally; over half are smartwatches; "
        "about a quarter of users wear them during sleep; U.S. revenues are nearly $10B."
    )),
]

wearables_convo = ConversationalTestCase(
    chatbot_role=(
        "You are a concise, helpful tech explainer. Stay on-topic and avoid speculation. "
        "When asked about wearables, cover definition, examples, uses, and scale."
    ),
    turns=turns,
)

metrics = [
    # All of these work on multi-turn chat and read `turns` with role/content:
    KnowledgeRetentionMetric(threshold=0.7, verbose_mode=True),
    ConversationCompletenessMetric(threshold=0.7, verbose_mode=True),
    RoleAdherenceMetric(threshold=0.7, verbose_mode=True),
    # ConversationRelevancyMetric(window_size=3, threshold=0.7, verbose_mode=True),  # optional if available
]

# Programmatic run
if __name__ == "__main__":
    evaluate([wearables_convo], metrics)

# Pytest-style entry point
def test_wearables_multi_turn():
    assert_test(wearables_convo, metrics)
