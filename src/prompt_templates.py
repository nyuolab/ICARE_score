"""
Load generation / with-document QA prompts from an internal prompt pack.

Packs (not meant for end users to edit as part of normal usage):
  prompts/radiology_specific/   DOCUMENT_TYPE=radiology (default)
  prompts/generic/              DOCUMENT_TYPE=generic

Each pack has:
  generation.txt          placeholders: {n}, {document}, {extra_instruction}
  qa_with_document.txt    placeholders: {document}, {question}, {option_a}..{option_d}

Anti-repeat prefix and without-document QA stay in code (not domain-specific).
"""

from pathlib import Path
from typing import Dict, List, Optional

from config import Config

_REPO_ROOT = Path(__file__).resolve().parent.parent

_DOCUMENT_TYPE_TO_DIR = {
    "radiology": "prompts/radiology_specific",
    "generic": "prompts/generic",
}


def _prompt_dir() -> Path:
    doc_type = (Config.DOCUMENT_TYPE or "radiology").strip().lower()
    if doc_type not in _DOCUMENT_TYPE_TO_DIR:
        raise ValueError(
            f"Unsupported DOCUMENT_TYPE={doc_type!r}. "
            f"Choose one of: {', '.join(sorted(_DOCUMENT_TYPE_TO_DIR))}."
        )
    rel = _DOCUMENT_TYPE_TO_DIR[doc_type]
    # Optional maintainer override (not documented for deploy users)
    override = (Config.PROMPT_DIR or "").strip()
    if override:
        rel = override
    p = Path(rel)
    return p if p.is_absolute() else _REPO_ROOT / p


def _render(filename: str, **kwargs) -> str:
    path = _prompt_dir() / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing prompt file {path}. Expected generation.txt and "
            f"qa_with_document.txt under the pack for DOCUMENT_TYPE="
            f"{Config.DOCUMENT_TYPE!r}."
        )
    return path.read_text(encoding="utf-8").format(**kwargs)


def build_generation_prompt(
    document: str, n: int, previous_questions: Optional[List[str]] = None
) -> str:
    previous_questions = previous_questions or []
    previous_prefix = ""
    extra_instruction = ""
    if previous_questions:
        listed = "\n".join(f"- {q}" for q in previous_questions)
        previous_prefix = (
            f"Questions already generated for this document:\n{listed}\n\n"
        )
        extra_instruction = (
            "Do not repeat the questions listed above; "
            "cover parts of the document not addressed by them."
        )
    body = _render(
        "generation.txt",
        n=n,
        document=document,
        extra_instruction=extra_instruction,
    )
    return previous_prefix + body


def build_qa_prompt(
    document: str,
    question: str,
    options: Dict[str, str],
    with_document: bool,
) -> str:
    if with_document:
        return _render(
            "qa_with_document.txt",
            document=document,
            question=question,
            option_a=options["A"],
            option_b=options["B"],
            option_c=options["C"],
            option_d=options["D"],
        )
    # Without-document prompt has no domain wording; keep inline.
    return f"""Answer the following question:
        {question}
    
        Options:
        A) {options['A']}
        B) {options['B']}
        C) {options['C']}
        D) {options['D']}
    
        Your life depends on providing ONLY a single letter (A, B, C, or D) as your answer. 
        Do not include any other text, punctuation, or explanation.
        Format: Just the letter.
        Example correct format: A
        Example incorrect formats: A., The answer is A, Option A"""
