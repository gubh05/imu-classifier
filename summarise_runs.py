"""CLI tool to compare experiment runs recorded in outputs/runs/."""
import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_run(run_dir: Path) -> dict | None:
    """Load a single run directory. Returns None if incomplete."""
    config_path = run_dir / "run_config.json"
    metrics_path = run_dir / "metrics.json"
    if not config_path.exists() or not metrics_path.exists():
        print(f"WARNING: skipping incomplete run directory: {run_dir.name}", file=sys.stderr)
        return None
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        met = json.loads(metrics_path.read_text(encoding="utf-8"))
        final = met.get("final", {})
        epoch_log = met.get("epoch_log", [])

        run_name = cfg.get("run_name", "")
        name_upper = run_name.upper()
        if name_upper in ("CNN", "LSTM", "RF"):
            model = name_upper
        else:
            model = run_name.split("_")[0].upper() if run_name else "?"

        epochs: int | str = len(epoch_log) if epoch_log else "-"

        duration_s = cfg.get("duration_seconds", 0)
        mins = int(duration_s) // 60
        secs = int(duration_s) % 60
        duration_str = f"{mins}m {secs:02d}s"

        return {
            "run_id": cfg.get("run_id", run_dir.name),
            "model": model,
            "accuracy": final.get("accuracy"),
            "f1_macro": final.get("f1_macro", final.get("f1")),
            "epochs": epochs,
            "duration": duration_str,
            "duration_s": duration_s,
        }
    except Exception as exc:
        print(f"WARNING: failed to load {run_dir.name}: {exc}", file=sys.stderr)
        return None


def load_all_runs(base_dir: Path) -> list[dict]:
    """Walk outputs/runs/ and return a list of run dicts."""
    if not base_dir.exists():
        return []
    runs = []
    for run_dir in sorted(base_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run = _load_run(run_dir)
        if run is not None:
            runs.append(run)
    return runs


# ---------------------------------------------------------------------------
# Filtering and sorting
# ---------------------------------------------------------------------------

def filter_runs(runs: list[dict], model_filter: str | None) -> list[dict]:
    if not model_filter:
        return runs
    mf = model_filter.upper()
    return [r for r in runs if mf in r["model"].upper() or mf in r["run_id"].upper()]


def sort_runs(runs: list[dict], sort_key: str) -> list[dict]:
    key_map = {"f1": "f1_macro", "accuracy": "accuracy", "acc": "accuracy"}
    field = key_map.get(sort_key.lower(), sort_key.lower())
    return sorted(runs, key=lambda r: (r.get(field) or 0.0), reverse=True)


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

def _pct(val) -> str:
    if val is None:
        return "  N/A  "
    return f"{val:.1%}"


def _str(val) -> str:
    return str(val) if val is not None else "-"


COL_WIDTHS = {
    "run_id": 33,
    "model": 6,
    "accuracy": 9,
    "f1_macro": 9,
    "epochs": 6,
    "duration": 9,
}

HEADERS = ["Run ID", "Model", "Accuracy", "F1 Macro", "Epochs", "Duration"]
KEYS = ["run_id", "model", "accuracy", "f1_macro", "epochs", "duration"]


def _row_values(run: dict) -> list[str]:
    return [
        run["run_id"][:COL_WIDTHS["run_id"]],
        run["model"],
        _pct(run["accuracy"]),
        _pct(run["f1_macro"]),
        _str(run["epochs"]),
        run["duration"],
    ]


def _box_table(runs: list[dict]) -> str:
    """Render runs as a box-drawing table."""
    widths = list(COL_WIDTHS.values())

    def _bar(left, mid, right, fill="─"):
        return left + mid.join(fill * (w + 2) for w in widths) + right

    def _row(cells: list[str]) -> str:
        parts = []
        for cell, w in zip(cells, widths):
            parts.append(f" {cell:^{w}} ")
        return "│" + "│".join(parts) + "│"

    lines = []
    lines.append(_bar("┌", "┬", "┐"))
    lines.append(_row(HEADERS))
    lines.append(_bar("├", "┼", "┤"))
    for run in runs:
        lines.append(_row(_row_values(run)))
    lines.append(_bar("└", "┴", "┘"))
    return "\n".join(lines)


def _md_table(runs: list[dict]) -> str:
    """Render runs as a Markdown pipe table."""
    lines = []
    lines.append("| " + " | ".join(HEADERS) + " |")
    lines.append("| " + " | ".join("---" for _ in HEADERS) + " |")
    for run in runs:
        lines.append("| " + " | ".join(_row_values(run)) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Ensure UTF-8 output on Windows terminals / conda run
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Summarise and compare experiment runs from outputs/runs/."
    )
    parser.add_argument("--model", default=None, help="Filter by model name (e.g. cnn)")
    parser.add_argument("--top", type=int, default=None, help="Show top N runs")
    parser.add_argument("--sort", default="run_id", help="Sort by: f1, accuracy, run_id (default)")
    parser.add_argument("--export", default=None, help="Export table to a Markdown file")
    args = parser.parse_args()

    base_dir = Path("outputs/runs")
    runs = load_all_runs(base_dir)

    if not runs:
        print("No completed runs found in", base_dir)
        return

    runs = filter_runs(runs, args.model)
    if not runs:
        print("No runs matched the filter.")
        return

    if args.sort != "run_id":
        runs = sort_runs(runs, args.sort)

    if args.top:
        runs = runs[: args.top]

    # Print box table to stdout
    print(_box_table(runs))

    # Best run by f1_macro
    best = max(runs, key=lambda r: r.get("f1_macro") or 0.0)
    f1_best = best.get("f1_macro")
    f1_str = f"{f1_best:.1%}" if f1_best is not None else "N/A"
    print(f"Best run: {best['run_id']} (F1: {f1_str})")

    # Export
    if args.export:
        export_path = Path(args.export)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(_md_table(runs), encoding="utf-8")
        print(f"Exported to: {export_path}")


if __name__ == "__main__":
    main()
