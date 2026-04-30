from argparse import ArgumentParser
from defenses.bert_defense import DEFAULT_BERT_DEFENSE_MODEL, BertDefense

def _fmt_optional_score(value):
    return "None" if value is None else f"{value:.3f}"


def main():
    parser = ArgumentParser(
        description=(
            "Evaluate the BERT defense on a baseline case made of an un-poisoned "
            "context file, a benign prompt file, and a clean expected response file."
        )
    )
    parser.add_argument("context_file", help="Un-poisoned file to scan as retrieved context.")
    parser.add_argument(
        "--model",
        default=DEFAULT_BERT_DEFENSE_MODEL,
        help=f"Hugging Face NLI model to use.",
    )

    args = parser.parse_args()
    with open(args.context_file, "r", errors="replace") as f:
        context = f.read()

    print(f"Loading BERT defense model: {args.model}")
    defense = BertDefense(model_name=args.model)

    print(f"Evaluating context file: {args.context_file}")
    decision = defense.evaluate(
        retrieved_context=context,
    )

    print("\n--- BERT DEFENSE RESULT ---")
    print(f"blocked: {decision.flagged}")
    print(f"context_attack_entailment: {decision.context_attack_entailment:.3f}")

    if decision.flagged:
        print("\nVerdict: BERT would block this file/context.")
    else:
        print("\nVerdict: BERT would allow this file/context.")


if __name__ == "__main__":
    main()
