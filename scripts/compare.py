"""Agreement of runs of ``scripts/bench.py`` / ``tools/reference.py`` on their common inputs.

    python scripts/compare.py ref-fp32.json ref-bf16.json ours-bf16.json

Per pair: identical texts and the character edit rate (Levenshtein distance over the first run's characters);
per run: its last pass's speed.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from textmetrics import compare  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+")
    args = p.parse_args()
    runs = {r.name: json.loads(r.read_text()) for r in args.runs}
    for name, r in runs.items():
        last = r["passes"][-1]
        print(f"{name}: {r['inputs']} inputs, {r['audio_seconds']:.0f} s, RTFx {last['rtfx']:.1f}")
    texts = {name: dict(zip(r["ids"], r["texts"], strict=True)) for name, r in runs.items()}
    for a, b in itertools.combinations(runs, 2):
        ids = [i for i in texts[a] if i in texts[b]]
        c = compare([texts[a][i] for i in ids], [texts[b][i] for i in ids])
        print(f"{a} vs {b}: {c['identical']}/{c['inputs']} identical, "
              f"character edit rate {100 * c['character_edit_rate']:.2f}%")


if __name__ == "__main__":
    main()
