"""
Treuno 125M —  RAG Prompt Builder
Assembles the final enriched prompt by injecting retrieved document chunks
as system context before the user's coding question.

This is the bridge between the retriever/indexer and the model:
    query → retrieve → build_rag_prompt → TreunoModel

Format:
    [SYSTEM CONTEXT — retrieved from live web]
    Source: {url}
    ---
    {chunk_text}
    ...
    [END CONTEXT]

    {base_prompt}

Usage:
    context = RAGContext(query="python requests POST", docs=[...])
    prompt  = build_rag_prompt(context, base_prompt="Write a POST request")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RAGContext:
    """Container for a retrieval context attached to a single inference call."""
    query: str
    docs: List[dict] = field(default_factory=list)   # list of {text, url, title}
    max_context_chars: int = 3000                     # total char budget for context


def build_rag_prompt(
    query: str,
    retrieved_docs: List[dict],
    base_prompt: str,
    max_context_chars: int = 3000,
    max_docs: int = 3,
) -> str:
    """
    Build a RAG-enriched prompt by prepending retrieved document chunks.

    Args:
        query:             The user's original query (used for header)
        retrieved_docs:    List of dicts with keys: text, url, title (optional)
        base_prompt:       The user's actual instruction / question
        max_context_chars: Hard character budget for all context combined
        max_docs:          Maximum number of documents to include

    Returns:
        Complete prompt string ready for tokenization and model input
    """
    if not retrieved_docs:
        return base_prompt

    context_parts = []
    chars_used = 0

    for i, doc in enumerate(retrieved_docs[:max_docs]):
        text = doc.get("text", "").strip()
        url  = doc.get("url", "")
        title = doc.get("title", "")

        if not text:
            continue

        # Trim this doc's contribution to stay within budget
        remaining = max_context_chars - chars_used
        if remaining <= 0:
            break
        text = text[:remaining]
        chars_used += len(text)

        header = f"Source: {title or url}" if (title or url) else f"Source [{i+1}]"
        context_parts.append(f"{header}\n{'-'*40}\n{text}")

    if not context_parts:
        return base_prompt

    context_block = "\n\n".join(context_parts)

    prompt = (
        f"[ANTIGRAVITY LIVE CONTEXT — Retrieved for: \"{query}\"]\n"
        f"{context_block}\n"
        f"[END CONTEXT]\n\n"
        f"{base_prompt}"
    )
    return prompt


def build_rag_system_message(retrieved_docs: List[dict], max_context_chars: int = 3000) -> str:
    """
    Build a system message string containing only the retrieved context.
    Use this when calling a chat-template model where system and user turns are separate.
    """
    if not retrieved_docs:
        return ""
    parts = []
    chars_used = 0
    for doc in retrieved_docs[:3]:
        text = doc.get("text", "").strip()[:max_context_chars - chars_used]
        url = doc.get("url", "")
        if not text:
            continue
        parts.append(f"[Source: {url}]\n{text}")
        chars_used += len(text)
        if chars_used >= max_context_chars:
            break
    return "\n\n".join(parts)
