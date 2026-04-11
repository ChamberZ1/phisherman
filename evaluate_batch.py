"""Batch evaluation script — runs a directory of .eml files through the cascade.

Usage:
    python evaluate_batch.py --eml-dir path/to/emails --output results.csv

If --label is provided (phish or benign), accuracy metrics are printed at the end.
This is useful when evaluating a known-phishing corpus like phishing_pot to measure
the false negative rate, or a known-benign corpus to measure the false positive rate.

Example (phishing_pot repo):
    git clone https://github.com/rf-peixoto/phishing_pot
    python evaluate_batch.py --eml-dir phishing_pot/email --label phish --output results.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch .eml evaluation against the phishing cascade.")
    parser.add_argument("--eml-dir", type=Path, required=True, help="Directory containing .eml files (searched recursively)")
    parser.add_argument("--output", type=Path, default=Path("results.csv"), help="Output CSV path (default: results.csv)")
    parser.add_argument("--label", choices=["phish", "benign"], default=None, help="Ground-truth label for accuracy metrics")
    parser.add_argument("--limit", type=int, default=1000, help="Max number of files to process (default: 1000, 0 = no limit)")
    return parser.parse_args()


def _truncate(text: str, max_chars: int = 500) -> str:
    text = text.strip()
    return text[:max_chars] + "…" if len(text) > max_chars else text


def main() -> None:
    args = parse_args()

    if not args.eml_dir.exists():
        print(f"Error: directory not found: {args.eml_dir}")
        sys.exit(1)

    all_files = sorted(args.eml_dir.rglob("*.eml"))
    if not all_files:
        print(f"No .eml files found under: {args.eml_dir}")
        sys.exit(1)

    eml_files = all_files if args.limit == 0 else all_files[:args.limit]
    print(f"Found {len(all_files)} .eml files. Processing {len(eml_files)}. Loading models...")

    from src.cascade import PhishingCascade
    from src.eml_parser import parse_eml

    cascade = PhishingCascade()
    print("Models loaded. Starting evaluation...\n")

    total = len(eml_files)
    phish_count = 0
    benign_count = 0
    error_count = 0
    start = time.time()

    results_fieldnames = [
        "file", "verdict", "triggered_by",
        "rule_score", "rule_matches",
        "supervised_proba", "transformer_proba", "isolation_score",
    ]

    review_fieldnames = [
        "file", "from_address", "subject", "body_preview", "urls_found",
        "rule_score", "rule_matches",
        "supervised_proba", "transformer_proba", "isolation_score",
    ]

    review_path = args.output.with_stem(args.output.stem + "_review")
    error_path  = args.output.with_stem(args.output.stem + "_errors")

    with (
        args.output.open("w", newline="", encoding="utf-8") as f_results,
        review_path.open("w", newline="", encoding="utf-8") as f_review,
        error_path.open("w", newline="", encoding="utf-8") as f_errors,
    ):
        results_writer = csv.DictWriter(f_results, fieldnames=results_fieldnames)
        results_writer.writeheader()

        review_writer = csv.DictWriter(f_review, fieldnames=review_fieldnames)
        review_writer.writeheader()

        f_errors.write("file\terror\n")

        for i, eml_path in enumerate(eml_files, 1):
            try:
                parsed = parse_eml(eml_path.read_bytes())
                result = cascade.predict({
                    "from_address":          parsed["from_address"],
                    "subject":               parsed["subject"],
                    "body":                  parsed["body"],
                    "attachment_extensions": parsed.get("attachment_extensions", []),
                    "dkim_pass":             parsed.get("dkim_pass", False),
                    "dkim_domain":           parsed.get("dkim_domain", None),
                })

                verdict = "phish" if result["is_phish"] == 1 else "benign"
                if verdict == "phish":
                    phish_count += 1
                else:
                    benign_count += 1

                results_writer.writerow({
                    "file":              eml_path.name,
                    "verdict":           verdict,
                    "triggered_by":      result.get("triggered_by") or "",
                    "rule_score":        result.get("rule_score") or 0,
                    "rule_matches":      "|".join(result.get("rule_matches") or []),
                    "supervised_proba":  f"{result.get('supervised_proba') or 0:.4f}",
                    "transformer_proba": f"{result.get('transformer_proba') or 0:.4f}",
                    "isolation_score":   f"{result.get('isolation_score') or 0:.4f}",
                })

                # Write misclassified emails to the review file
                is_miss = (
                    (verdict == "benign" and args.label == "phish") or
                    (verdict == "phish"  and args.label == "benign")
                )
                if is_miss:
                    review_writer.writerow({
                        "file":              eml_path.name,
                        "from_address":      parsed["from_address"],
                        "subject":           parsed["subject"],
                        "body_preview":      _truncate(parsed["body"]),
                        "urls_found":        parsed["urls_found"],
                        "rule_score":        result.get("rule_score") or 0,
                        "rule_matches":      "|".join(result.get("rule_matches") or []),
                        "supervised_proba":  f"{result.get('supervised_proba') or 0:.4f}",
                        "transformer_proba": f"{result.get('transformer_proba') or 0:.4f}",
                        "isolation_score":   f"{result.get('isolation_score') or 0:.4f}",
                    })

            except Exception as e:
                error_count += 1
                error_msg = str(e).replace("\n", " ")
                f_errors.write(f"{eml_path.name}\t{error_msg}\n")
                results_writer.writerow({
                    "file": eml_path.name,
                    "verdict": "error",
                    "triggered_by": error_msg[:120],
                    "rule_score": "", "rule_matches": "",
                    "supervised_proba": "", "transformer_proba": "", "isolation_score": "",
                })

            # Progress update every 50 files
            if i % 50 == 0 or i == total:
                elapsed = time.time() - start
                rate = i / elapsed
                eta = (total - i) / rate if rate > 0 else 0
                print(f"  [{i}/{total}] {rate:.1f} emails/sec — ETA {eta:.0f}s")

    elapsed = time.time() - start
    processed = phish_count + benign_count

    print(f"\nDone in {elapsed:.1f}s.")
    print(f"  Results  → {args.output}")
    if args.label:
        label_desc = "missed phishing emails" if args.label == "phish" else "false positives"
        print(f"  Review   → {review_path}  ({label_desc})")
    print(f"  Errors   → {error_path}")
    print()
    print(f"  Processed: {processed}/{total}")
    print(f"  Phishing : {phish_count:>6} ({phish_count/total*100:.1f}%)")
    print(f"  Benign   : {benign_count:>6} ({benign_count/total*100:.1f}%)")
    if error_count:
        print(f"  Errors   : {error_count:>6} ({error_count/total*100:.1f}%)")

    if args.label and processed > 0:
        correct = phish_count if args.label == "phish" else benign_count
        missed  = benign_count if args.label == "phish" else phish_count
        print(f"\nGround truth: all {args.label} (excluding errors)")
        print(f"  Correct  : {correct}/{processed} ({correct/processed*100:.1f}%)")
        print(f"  Missed   : {missed}/{processed} ({missed/processed*100:.1f}%)")
        if args.label == "phish":
            print(f"  False negative rate: {missed/processed*100:.1f}%")
        else:
            print(f"  False positive rate: {missed/processed*100:.1f}%")


if __name__ == "__main__":
    main()
