from src.parrotly.generation.prompts import ANSWER_PROMPT


def build_qa_chain(llm):
    return ANSWER_PROMPT | llm