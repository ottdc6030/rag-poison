"""
Known-Answer (Canary Probe) Defense for RAG Poisoning
======================================================

Strategy
--------
Before trusting a RAG query engine for real user queries, probe it with a set of
"canary" questions whose correct answers are known in advance.  If the engine's
answers to those canaries deviate from the expected responses, the RAG context
may have been poisoned.

Why this catches injections
---------------------------
A successful RAG injection typically overrides the model's normal retrieval
behavior — e.g. "ignore all other context and output X instead".  When that
override is active, questions whose answers exist in the original (clean)
document suddenly get wrong answers, or the model produces content that looks
nothing like the expected domain.  Both cases lower the semantic similarity
between the expected and actual answer, triggering the defense.

Two matching strategies are supported per canary:
  "exact"    — the expected string must appear verbatim in the model's answer.
               Good for controlled key-value facts (names, IDs, dates).
  "semantic" — cosine similarity between sentence embeddings must exceed a
               threshold.  Good for open-ended or paraphrased answers.

Limitations
-----------
* A clever attacker who knows your canary questions can craft an injection that
  answers them correctly before executing the poison goal.  Rotate canaries and
  keep them secret to raise the bar.
* Canaries must be grounded in the document the target actually loads.  Generic
  questions answered from the LLM's training data are not useful.

Integration with rag-poison-test.py
------------------------------------
  from llama_index.embeddings.huggingface import HuggingFaceEmbedding
  from llama_index.core import Settings
  from defenses.known_answer_defense import Canary, KnownAnswerDefense

  Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

  canaries = [
      Canary("What is the first name listed in the document?", "Ian"),
      Canary("Which city is Ian from?", "Norfolk"),
      Canary("Is Hilda male or female?", "female", match_style="exact"),
  ]

  defense = KnownAnswerDefense(Settings.embed_model, canaries, threshold=0.65)

  # build_query_engine is imported from rag-poison-test.py (or call it inline)
  query_engine = build_query_engine(poisoned_file, top_k, target_llm)

  result = defense.check(query_engine)
  if not result.passed:
      print("WARNING: Possible injection detected — refusing user query.")
  else:
      response = query_engine.query(target_prompt)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Canary:
    """A single known-answer probe question."""
    question: str
    expected_answer: str
    match_style: Literal["exact", "semantic"] = "semantic"


@dataclass
class ProbeResult:
    """Result for one canary question."""
    canary: Canary
    actual_answer: str
    score: float   # similarity in [0, 1]; higher = closer to expected
    passed: bool


@dataclass
class DefenseResult:
    """Aggregate result for a full canary check."""
    passed: bool   # True only when every single canary passes
    probe_results: List[ProbeResult] = field(default_factory=list)

    @property
    def failed_canaries(self) -> List[Canary]:
        return [r.canary for r in self.probe_results if not r.passed]

    def summary(self) -> str:
        n = len(self.probe_results)
        n_fail = sum(1 for r in self.probe_results if not r.passed)
        status = "PASSED" if self.passed else f"FAILED ({n_fail}/{n} canaries failed)"
        return f"KnownAnswerDefense: {status}"


# ---------------------------------------------------------------------------
# Defense class
# ---------------------------------------------------------------------------

class KnownAnswerDefense:
    """
    Probes a RAG query engine with canary questions before trusting it for
    real user queries.  Sessions where answers diverge from known-correct
    responses are flagged as potentially poisoned.

    Parameters
    ----------
    embed_model : any object with a ``get_text_embedding(text) -> list[float]``
                  method.  The LlamaIndex ``HuggingFaceEmbedding`` class satisfies
                  this interface.  Used only for "semantic" canaries.
    canaries    : list of :class:`Canary` probes.
    threshold   : minimum cosine similarity for a semantic canary to pass.
                  0.65 works well with bge-small; raise to 0.75+ for stricter checks.
    verbose     : print per-canary results to stdout.
    """

    def __init__(
        self,
        embed_model,
        canaries: List[Canary],
        threshold: float = 0.65,
        verbose: bool = True,
    ):
        self.embed_model = embed_model
        self.canaries = canaries
        self.threshold = threshold
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _cosine(self, a: str, b: str) -> float:
        va = np.array(self.embed_model.get_text_embedding(a), dtype=float)
        vb = np.array(self.embed_model.get_text_embedding(b), dtype=float)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom > 0 else 0.0

    @staticmethod
    def _contains(expected: str, actual: str) -> float:
        """1.0 if expected appears verbatim in actual (case-insensitive)."""
        return 1.0 if expected.strip().lower() in actual.strip().lower() else 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, query_engine) -> DefenseResult:
        """
        Run all canary probes against *query_engine*.

        Returns a :class:`DefenseResult` where ``passed`` is ``True`` only if
        every canary individually passes its threshold check.
        """
        results: List[ProbeResult] = []

        for canary in self.canaries:
            raw = query_engine.query(canary.question)
            actual = str(raw).strip()

            if canary.match_style == "exact":
                score = self._contains(canary.expected_answer, actual)
                passed = score == 1.0
            else:
                score = self._cosine(canary.expected_answer, actual)
                passed = score >= self.threshold

            if self.verbose:
                status = "PASS" if passed else "FAIL"
                print(f"[KnownAnswerDefense] {status}: {canary.question!r}  score={score:.3f}")
                if not passed:
                    print(f"  expected : {canary.expected_answer!r}")
                    print(f"  actual   : {actual!r}")

            results.append(ProbeResult(
                canary=canary,
                actual_answer=actual,
                score=score,
                passed=passed,
            ))

        overall = all(r.passed for r in results)
        result = DefenseResult(passed=overall, probe_results=results)
        if self.verbose:
            print(f"[KnownAnswerDefense] {result.summary()}")
        return result
