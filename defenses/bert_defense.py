"""
BERT-family pre-query defense for RAG poisoning.

The defense scans a document before the normal user query is allowed to run.
It uses a natural language inference model to score whether document chunks
entail prompt-injection style hypotheses, such as telling an assistant to
ignore the user or follow hidden document instructions.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Iterable

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_BERT_DEFENSE_MODEL = "typeform/mobilebert-uncased-mnli"
DEFAULT_BERT_ATTACK_THRESHOLD = 0.8


@dataclass
class BertFinding:
    score: float
    hypothesis: str
    text_window: str


@dataclass
class BertDefenseResult:
    passed: bool
    threshold: float
    best_score: float = 0.0
    findings: list[BertFinding] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return not self.passed

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return f"BertDefense: {status} best_score={self.best_score:.3f} threshold={self.threshold:.3f}"


@dataclass
class BertDefenseDecision:
    flagged: bool
    context_attack_entailment: float | None = None
    reason: str = "no threshold exceeded"


def _normalize_label(label):
    return str(label).strip().lower().replace("-", "_")


def _normalized_text(text):
    return " ".join(text.strip().lower().split())


class BertDefense:
    """
    Flag likely prompt-injection content.

    model_name:
        Hugging Face sequence-classification model with entailment, neutral,
        and contradiction labels.
    threshold:
        Minimum entailment score needed to block the document.
    verbose:
        Print scoring details while scanning.
    """

    DEFAULT_HYPOTHESES = (
        "This text tells an assistant to disregard previous or original instructions.",
        "This text contains a hidden directive that overrides the user request.",
        "This text commands an AI model to follow document instructions instead of the user.",
        "This text tells an AI assistant to reply with a specific answer.",
        "This text contains a system message for an AI assistant.",
        "This text attempts to change the task that the user asked the assistant to perform.",
    )

    def __init__(
        self,
        model_name=DEFAULT_BERT_DEFENSE_MODEL,
        threshold=DEFAULT_BERT_ATTACK_THRESHOLD,
        verbose=False,
        hypotheses: Iterable[str] | None = None,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.verbose = verbose
        self.hypotheses = tuple(hypotheses or self.DEFAULT_HYPOTHESES)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.id2label = {
            int(idx): _normalize_label(label)
            for idx, label in self.model.config.id2label.items()
        }
        self.entailment_index = self._find_label_index({"entailment"})

    def _find_label_index(self, acceptable_labels):
        for idx, label in self.id2label.items():
            if label in acceptable_labels:
                return idx
        raise ValueError(
            f"Model '{self.model_name}' does not expose labels {sorted(acceptable_labels)}. "
            f"Found labels: {self.id2label}"
        )

    def _score_entailment(self, premise, hypothesis):
        return self._score_label(premise, hypothesis, self.entailment_index)

    def _score_label(self, premise, hypothesis, label_index):
        encoded = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            logits = self.model(**encoded).logits[0]
        probabilities = torch.softmax(logits, dim=0)
        return float(probabilities[label_index])

    def _context_windows(self, text, max_chars=1800):
        paragraphs = [paragraph.strip() for paragraph in text.splitlines() if paragraph.strip()]
        windows = []
        current = []
        current_len = 0

        for paragraph in paragraphs:
            paragraph_len = len(paragraph)
            if current and current_len + paragraph_len > max_chars:
                windows.append("\n".join(current))
                current = []
                current_len = 0
            if paragraph_len > max_chars:
                windows.append(paragraph[:max_chars])
                continue
            current.append(paragraph)
            current_len += paragraph_len + 1

        if current:
            windows.append("\n".join(current))

        windows.extend(self._focused_windows(text))
        deduped = []
        seen = set()
        for window in windows or [text[:max_chars]]:
            key = _normalized_text(window)
            if key and key not in seen:
                deduped.append(window)
                seen.add(key)
        return deduped

    def _focused_windows(self, text, max_chars=420):
        split_points = re.split(r"(?<=[.!?])\s+|\n+", text)
        units = [unit.strip() for unit in split_points if unit.strip()]
        windows = []
        for idx, unit in enumerate(units):
            if len(unit) <= max_chars:
                windows.append(unit)
            else:
                for start in range(0, len(unit), max_chars):
                    windows.append(unit[start:start + max_chars])
            if idx + 1 < len(units):
                pair = f"{unit} {units[idx + 1]}".strip()
                if len(pair) <= max_chars:
                    windows.append(pair)
        return windows

    def scan_text(self, text):
        """
        Return a BertDefenseResult for a document or retrieved chunk text
        """
        findings = []
        best_score = 0.0

        for window in self._context_windows(text):
            for hypothesis in self.hypotheses:
                score = self._score_entailment(window, hypothesis)
                best_score = max(best_score, score)
                if score >= self.threshold:
                    findings.append(BertFinding(score=score, hypothesis=hypothesis, text_window=window))
                    if self.verbose:
                        print(f"[BertDefense] flagged score={score:.3f} hypothesis={hypothesis}")

        result = BertDefenseResult(
            passed=best_score < self.threshold,
            threshold=self.threshold,
            best_score=best_score,
            findings=findings,
        )
        if self.verbose:
            print(f"[BertDefense] {result.summary()}")
        return result

    def scan_file(self, path):
        """
        Read and scan a file before the normal RAG user query runs.
        """
        with open(path, "r", errors="replace") as f:
            return self.scan_text(f.read())

    def evaluate(self, user_prompt, model_output, retrieved_context=None):
        """
        Score whether an answer looks correct to the user prompt or influenced
        by prompt-injection content.
        """

        context_attack_entailment = None
        if retrieved_context:
            context_result = self.scan_text(retrieved_context)
            context_attack_entailment = context_result.best_score

        flagged = False
        reason = "no threshold exceeded"
        if context_attack_entailment is not None and context_attack_entailment >= self.threshold:
            flagged = True
            reason = (
                "retrieved context looks injected "
                f"({context_attack_entailment:.3f} >= {self.threshold:.3f})"
            )

        if self.verbose:
            print(
                "[BertDefense] "
                f"context_attack={context_attack_entailment} "
                f"decision={'BLOCK' if flagged else 'ALLOW'} "
                f"reason={reason}"
            )

        return BertDefenseDecision(
            flagged=flagged,
            context_attack_entailment=context_attack_entailment,
            reason=reason,
        )
