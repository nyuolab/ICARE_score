"""
Load generation / with-document QA prompts from ICARE_PROMPT_DIR.

Each pack is a folder with two files:
  generation.txt          placeholders: {n}, {document}, {extra_instruction}
  qa_with_document.txt    placeholders: {document}, {question}, {option_a}..{option_d}

Set via .env or sbatch:
  ICARE_PROMPT_DIR=prompts/radiology_specific   # default (current radiology wording)
  ICARE_PROMPT_DIR=prompts/generic              # "document" instead of radiology/report

Anti-repeat prefix and without-document QA stay in code (not domain-specific).
"""

from pathlib import Path
from typing import Dict, List, Optional

from config import Config

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _prompt_dir() -> Path:
    raw = (Config.PROMPT_DIR or "prompts/radiology_specific").strip()
    p = Path(raw)
    return p if p.is_absolute() else _REPO_ROOT / p


def _render(filename: str, **kwargs) -> str:
    path = _prompt_dir() / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing prompt file {path}. Set ICARE_PROMPT_DIR to a pack "
            f"with generation.txt and qa_with_document.txt"
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
    # Without-document prompt has no radiology wording; keep inline.
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
