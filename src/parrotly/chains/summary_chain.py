from langchain_core.output_parsers import StrOutputParser

from src.parrotly.generation.prompts import SUMMARY_PROMPT


def build_summary_chain(llm):
    return SUMMARY_PROMPT | llm