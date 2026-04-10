"""
llm_chain.py — LLM answer generation using Qwen via HuggingFace.
"""
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from src.config import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    HF_TOKEN, SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",
     "Context from sensor documentation:\n"
     "---\n{context}\n---\n\n"
     "Question: {question}\n\n"
     "Answer based ONLY on the context above:"),
])


def build_llm():
    logger.info(f"Connecting to LLM: {LLM_MODEL}")
    return HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_new_tokens=LLM_MAX_TOKENS,
        huggingfacehub_api_token=HF_TOKEN or None,
    )


def generate_answer(llm, context: str, question: str) -> str:
    logger.info(f"Generating answer for: '{question[:60]}...'")
    prompt = PROMPT_TEMPLATE.format_messages(context=context, question=question)
    full_prompt = "\n".join(m.content for m in prompt)
    answer = llm.invoke(full_prompt).strip()
    logger.debug(f"Answer: {answer[:120]}...")
    return answer