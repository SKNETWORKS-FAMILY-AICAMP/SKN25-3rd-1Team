from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import GPTModel
from dotenv import load_dotenv
load_dotenv()


model = GPTModel(model="gpt-4o-mini")


def evaluate_sample(sample: dict) -> dict:
    test_case = LLMTestCase(
        input=sample["question"],
        actual_output=sample["answer"],
        retrieval_context=sample["contexts"]
    )

    answer_relevancy = AnswerRelevancyMetric(model=model)
    faithfulness = FaithfulnessMetric(model=model)

    answer_relevancy.measure(test_case)
    faithfulness.measure(test_case)

    return {
        "answer_relevancy_score": answer_relevancy.score,
        "answer_relevancy_reason": answer_relevancy.reason,
        "faithfulness_score": faithfulness.score,
        "faithfulness_reason": faithfulness.reason,
    }