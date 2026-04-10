"""
llm_chain.py — LLM answer generation using Qwen via HuggingFace.
"""
import logging
from huggingface_hub import InferenceClient
from langchain_core.prompts import ChatPromptTemplate
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
    return InferenceClient(
        provider="novita",
        api_key=HF_TOKEN or None,
    )


def generate_answer(llm, context: str, question: str) -> str:
    logger.info(f"Generating answer for: '{question[:60]}...'")
    prompt = PROMPT_TEMPLATE.format_messages(context=context, question=question)
    
    messages = [{"role": m.type if m.type != "human" else "user", "content": m.content} 
                for m in prompt]
    
    response = llm.chat_completion(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
    )
    answer = response.choices[0].message.content.strip()
    logger.debug(f"Answer: {answer[:120]}...")
    return answer