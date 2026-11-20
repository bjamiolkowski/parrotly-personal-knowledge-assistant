"""
Prompt templates for the generation module.

This module defines LangChain prompt templates used to control LLM behavior in the RAG pipeline.
"""

from langchain_core.prompts import ChatPromptTemplate


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You answer questions using retrieved document chunks.

            Guidelines:
            - Treat retrieved chunks as the only source of information.
            - Some retrieved chunks may be unrelated. Ignore content that does not help answer the question.
            - Do not add facts that are not supported by the provided context.
            - Never answer using your general knowledge.
            - Never explain topics that are not present in the retrieved context.
            - If the context does not contain the answer, respond exactly: "The answer was not found in the provided documents."
            - Keep the answer concise and focused.
            - Do not include document names, source metadata, chunk IDs, or citations in the final answer.
            """.strip(),
        ),
        (
            "human",
            """
            Conversation History:
            {history}

            Question:
            {query}

            Documents:
            {context}

            Answer:
            """.strip(),
        ),
    ]
)


SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You create summaries from retrieved document chunks.

            Guidelines:
            - Use retrieved context as the only source of information.
            - Preserve the meaning of the original documents.
            - Ignore unrelated chunks if they appear in the context.
            - Do not include assumptions or external knowledge.
            - If the context does not contain enough information, say that there is not enough information to summarize.

            Structure the response as:

            Definition:
            - Brief explanation of the topic.

            Key points:
            - Most important facts from the documents.

            Practical importance:
            - Why this topic matters in the document context.
            """.strip(),
        ),
        (
            "human",
            """
            Topic:
            {topic}

            Documents:
            {context}

            Summary:
            """.strip(),
        ),
    ]
)