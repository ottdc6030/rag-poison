"""
summarize_results.py
--------------------
Groups all rows of an injection-tester output CSV by (poison_prompt, poison_goal),
counts total trials and hijack successes per group, and writes a summary CSV.

Usage:
    python summarize_results.py <input_csv> <output_csv>
"""

import csv
import sys
from collections import defaultdict


def summarize(input_csv: str, output_csv: str) -> None:
    # group_key -> {"total": int, "hijacked": int}
    groups: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"total": 0, "hijacked": 0})
    # preserve insertion order of keys
    order: list[tuple[str, str]] = []

    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "poison_prompt" not in reader.fieldnames:
            raise ValueError(f"Input CSV must have a 'poison_prompt' column. Found: {reader.fieldnames}")

        has_goal = "poison_goal" in (reader.fieldnames or [])

        for row in reader:
            prompt = row["poison_prompt"].strip()
            goal = row.get("poison_goal", "").strip() if has_goal else ""
            key = (prompt, goal)

            if key not in groups:
                order.append(key)

            groups[key]["total"] += 1

            hijacked_val = row.get("hijacked", "").strip().lower()
            if hijacked_val in ("true", "1", "yes"):
                groups[key]["hijacked"] += 1

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["poison_prompt", "poison_goal", "total_trials", "hijack_count", "hijack_rate"])
        for key in order:
            prompt, goal = key
            total = groups[key]["total"]
            hijacked = groups[key]["hijacked"]
            rate = hijacked / total if total > 0 else 0.0
            writer.writerow([prompt, goal, total, hijacked, f"{rate:.3f}"])

    print(f"Wrote {len(order)} group(s) to '{output_csv}'.")


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input_csv> <output_csv>")
        sys.exit(1)

    summarize(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
