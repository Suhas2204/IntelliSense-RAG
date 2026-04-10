"""
llm_chain.py — LLM answer generation using Qwen via HuggingFace.

Pipeline stage: retrieved context + question  →  grounded answer
"""
import logging
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceEndpoint
from src.config import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    HF_TOKEN, SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# ── Prompt template: injects retrieved context + user question ──
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",
     "Context from sensor documentation:\n"
     "---\n{context}\n---\n\n"
     "Question: {question}\n\n"
     "Answer based ONLY on the context above:"),
])


def build_llm():
    """Initialise the Qwen LLM via HuggingFace Inference API."""
    logger.info(f"Connecting to LLM: {LLM_MODEL}")

    llm = HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_new_tokens=LLM_MAX_TOKENS,
        huggingfacehub_api_token=HF_TOKEN or None,
    )
    return llm


def build_chain(llm):
    """Create a LangChain chain with prompt + LLM."""
    chain = LLMChain(llm=llm, prompt=PROMPT_TEMPLATE)
    logger.info("LangChain chain ready")
    return chain


def generate_answer(chain, context: str, question: str) -> str:
    """Run the chain and return the answer string."""
    logger.info(f"Generating answer for: '{question[:60]}...'")
    response = chain.invoke({"context": context, "question": question})
    answer = response.get("text", "").strip()
    logger.debug(f"Answer: {answer[:120]}...")
    return answer
