#!/usr/bin/env python3
"""
Build a local HTML dashboard (file:// compatible) that aggregates plots/configs
from a single run directory.

Output:
  - plot_dashboard.html (repo root)

This script reads existing artifacts only; it does not modify the run outputs.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as _dt
import html
import json
import os
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = REPO_ROOT / "outputs" / "cnn_results" / "251008_160341"
DEFAULT_OUT_HTML = REPO_ROOT / "plot_dashboard.html"


@dataclasses.dataclass(frozen=True)
class ImgTile:
    title: str
    rel_src: str
    rel_href: str
    note: str | None = None


def _relpath(p: Path, start: Path) -> str:
    return os.path.relpath(p, start).replace(os.sep, "/")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _read_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text())


def _fmt_float(v: Any, nd: int = 3) -> str:
    try:
        f = float(v)
    except Exception:
        return str(v)
    if abs(f) >= 1e6:
        return f"{f:.3g}"
    return f"{f:.{nd}f}"


def _safe(s: Any) -> str:
    return html.escape("" if s is None else str(s))


def _slug(s: str) -> str:
    out = []
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "_", "-", ".", "/"):
            out.append("-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "section"


def _existing_files(base: Path, names: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for name in names:
        p = base / name
        if p.exists():
            out.append(p)
    return out


def _glob_existing(base: Path, pattern: str) -> list[Path]:
    return sorted([p for p in base.glob(pattern) if p.is_file()])


def _img_tile(p: Path, root: Path, title: str, note: str | None = None) -> ImgTile:
    rel = _relpath(p, root)
    return ImgTile(title=title, rel_src=rel, rel_href=rel, note=note)


def _gallery(title: str, tiles: list[ImgTile], extra_html: str = "") -> str:
    if not tiles and not extra_html:
        return ""
    parts = [f'<h4 class="subhead">{_safe(title)}</h4>']
    if extra_html:
        parts.append(extra_html)
    if tiles:
        parts.append('<div class="grid">')
        for t in tiles:
            note = f'<div class="note">{_safe(t.note)}</div>' if t.note else ""
            parts.append(
                "\n".join(
                    [
                        '<figure class="tile">',
                        f'  <img class="thumb" loading="lazy" data-full="{_safe(t.rel_href)}" src="{_safe(t.rel_src)}" alt="{_safe(t.title)}" />',
                        '  <figcaption>',
                        f'    <div class="cap">{_safe(t.title)}</div>',
                        f'    <a class="filelink" href="{_safe(t.rel_href)}">open</a>',
                        note,
                        "  </figcaption>",
                        "</figure>",
                    ]
                )
            )
        parts.append("</div>")
    return "\n".join(parts)


def _kv_table(title: str, d: dict[str, Any] | None, keys: list[str], root: Path, source_path: Path | None) -> str:
    if not d and not source_path:
        return ""
    parts = [f'<h4 class="subhead">{_safe(title)}</h4>']
    if source_path:
        parts.append(f'<div class="minor">source: <a href="{_safe(_relpath(source_path, root))}">{_safe(_relpath(source_path, root))}</a></div>')
    if not d:
        parts.append('<div class="missing">missing</div>')
        return "\n".join(parts)

    parts.append('<table class="kv">')
    parts.append("<tbody>")
    for k in keys:
        v = d.get(k, None)
        parts.append(f"<tr><th>{_safe(k)}</th><td>{_safe(v)}</td></tr>")
    parts.append("</tbody>")
    parts.append("</table>")
    return "\n".join(parts)


def _metrics_table(title: str, summary: dict[str, Any] | None, root: Path, source_path: Path | None) -> str:
    parts = [f'<h4 class="subhead">{_safe(title)}</h4>']
    if source_path:
        parts.append(f'<div class="minor">source: <a href="{_safe(_relpath(source_path, root))}">{_safe(_relpath(source_path, root))}</a></div>')
    if not summary:
        parts.append('<div class="missing">missing</div>')
        return "\n".join(parts)

    sel = summary.get("selected_threshold", None)
    parts.append(f'<div class="minor">strategy: <code>{_safe(summary.get("strategy"))}</code>, selected threshold: <code>{_safe(sel)}</code></div>')

    def row(name: str, m: dict[str, Any] | None) -> str:
        if not m:
            return f"<tr><th>{_safe(name)}</th><td class=\"missing\">missing</td></tr>"
        cols = ["TCR", "NMR", "CMR", "F", "ACC"]
        items = " ".join([f"<code>{c}={_safe(_fmt_float(m.get(c)))}</code>" for c in cols if c in m])
        return f"<tr><th>{_safe(name)}</th><td>{items}</td></tr>"

    parts.append('<table class="kv"><tbody>')
    parts.append(row("validation_metrics", summary.get("validation_metrics")))
    parts.append(row("test_metrics_at_selected_threshold", summary.get("test_metrics_at_selected_threshold")))
    parts.append(f"<tr><th>test_best_threshold</th><td><code>{_safe(summary.get('test_best_threshold'))}</code></td></tr>")
    parts.append(row("test_best_metrics", summary.get("test_best_metrics")))
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _series_description(series_name: str) -> str:
    # Keep this intentionally short: dashboard explanation, not thesis prose.
    return {
        "latest": "Most recent comparison snapshot in this run folder.",
        "series_BS": "Baseline sanity checks / starting points (small models, default-ish settings).",
        "series_R0": "Input/loader representation checks (rect/imgsz/batch style changes).",
        "series_A": "Capacity and input resolution variations.",
        "series_B": "Augmentation strategy variations.",
        "series_C": "Pretraining and model variant experiments.",
        "series_D": "Best candidate vs alternative detector settings; includes C2 vs D2 comparisons where available.",
        "series_D_cnn_baseline": "Detector vs CNN baseline comparison at the selected operating point (Balleny fold).",
    }.get(series_name, "Series comparison plots.")


def _sorted_run_names(names: list[str]) -> list[str]:
    preferred = ["BS1", "BS2", "R0a", "R0b", "A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "D2", "F1"]
    order = {n: i for i, n in enumerate(preferred)}
    return sorted(names, key=lambda n: (order.get(n, 999), n))


def _collect_ultralytics_tiles(run_dir: Path, root: Path) -> list[ImgTile]:
    curated = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "labels.jpg",
        "train_batch0.jpg",
        "train_batch1.jpg",
        "train_batch2.jpg",
        "val_batch0_labels.jpg",
        "val_batch0_pred.jpg",
        "val_batch1_labels.jpg",
        "val_batch1_pred.jpg",
        "val_batch2_labels.jpg",
        "val_batch2_pred.jpg",
    ]
    tiles: list[ImgTile] = []
    for p in _existing_files(run_dir, curated):
        tiles.append(_img_tile(p, root, p.name))
    return tiles


def _collect_custom_eval_tiles(run_dir: Path, root: Path) -> list[ImgTile]:
    tiles: list[ImgTile] = []
    for p in _existing_files(run_dir / "plots", ["f_vs_confidence_top1.png", "tcr_vs_nmr_top1.png"]):
        tiles.append(_img_tile(p, root, f"test: {p.name}"))
    for p in _existing_files(run_dir / "validation_eval" / "plots", ["f_vs_confidence_top1.png", "tcr_vs_nmr_top1.png"]):
        tiles.append(_img_tile(p, root, f"val: {p.name}"))
    return tiles


def _file_list_links(dir_path: Path, root: Path, patterns: list[str]) -> str:
    files: list[Path] = []
    for pat in patterns:
        files.extend(_glob_existing(dir_path, pat))
    files = sorted(set(files))
    if not files:
        return '<div class="minor">No matching files.</div>'
    lis = "\n".join(
        f'<li><a href="{_safe(_relpath(p, root))}">{_safe(p.name)}</a></li>' for p in files
    )
    return f"<ul class=\"filelist\">{lis}</ul>"


def _render_run_card(run_dir: Path, root: Path, card_id: str, fold_name: str | None) -> str:
    run_name = run_dir.name
    args_path = run_dir / "args.yaml"
    cfg_path = run_dir / "training_config.json"
    sel_path = run_dir / "selected_threshold_summary.json"

    args = _read_yaml(args_path)
    cfg = _read_json(cfg_path)
    sel = _read_json(sel_path)

    # Extract a stable subset of keys (shows what we actually varied most often).
    args_keys = [
        "model",
        "epochs",
        "patience",
        "batch",
        "imgsz",
        "rect",
        "optimizer",
        "lr0",
        "close_mosaic",
        "mosaic",
        "flipud",
        "fliplr",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "mixup",
        "copy_paste",
    ]
    cfg_keys = [
        "run_name",
        "fold",
        "noise",
        "mode",
        "top1_collapse",
        "conf_sweep_step",
        "conf_sweep_min",
        "conf_sweep_max",
        "val_select_metric",
    ]

    custom_tiles = _collect_custom_eval_tiles(run_dir, root)
    ul_tiles = _collect_ultralytics_tiles(run_dir, root)

    all_imgs_details = (
        "<details class=\"files\"><summary>Files: images (run root)</summary>"
        + _file_list_links(run_dir, root, ["*.png", "*.jpg"])
        + "</details>"
    )
    all_custom_details = (
        "<details class=\"files\"><summary>Files: custom plots</summary>"
        + _file_list_links(run_dir / "plots", root, ["*.png", "*.jpg"])
        + _file_list_links(run_dir / "validation_eval" / "plots", root, ["*.png", "*.jpg"])
        + "</details>"
    )

    fold_label = f" ({fold_name})" if fold_name else ""
    attrs = [f'data-run="{_safe(run_name)}"']
    if fold_name:
        attrs.append(f'data-fold="{_safe(fold_name)}"')

    parts = [
        f'<details class="card" id="{_safe(card_id)}" {" ".join(attrs)}>',
        f'  <summary><span class="run">{_safe(run_name)}</span><span class="fold">{_safe(fold_label)}</span></summary>',
        '  <div class="cardbody">',
        '    <div class="cols">',
        '      <div class="col">',
        _kv_table("args.yaml (selected keys)", args, args_keys, root, args_path),
        _kv_table("training_config.json (selected keys)", cfg, cfg_keys, root, cfg_path),
        "      </div>",
        '      <div class="col">',
        _metrics_table("selected_threshold_summary.json", sel, root, sel_path),
        "      </div>",
        "    </div>",
        _gallery("Custom evaluation curves", custom_tiles, extra_html=all_custom_details),
        _gallery("Ultralytics diagnostics (curated)", ul_tiles, extra_html=all_imgs_details),
        "  </div>",
        "</details>",
    ]
    return "\n".join([p for p in parts if p])


def _render_comparison_block(name: str, comp_dir: Path, root: Path) -> str:
    imgs = [
        "bar_test_selected_TCR_NMR_CMR_F.png",
        "scatter_tcr_vs_nmr_test_selected.png",
        "overlay_f_vs_confidence_top1.png",
        "overlay_tcr_vs_nmr_top1.png",
    ]
    tiles: list[ImgTile] = []
    for p in _existing_files(comp_dir, imgs):
        tiles.append(_img_tile(p, root, p.name))

    csv_path = comp_dir / "comparison_runs.csv"
    csv_link = ""
    if csv_path.exists():
        rel = _relpath(csv_path, root)
        csv_link = f'<div class="minor">source: <a href="{_safe(rel)}">{_safe(rel)}</a></div>'

    desc = _series_description(name)
    return "\n".join(
        [
            f'<section class="block" id="{_safe(_slug("exploratory-"+name))}">',
            f'  <h3>{_safe(name)}</h3>',
            f'  <p class="desc">{_safe(desc)}</p>',
            csv_link,
            _gallery("Series comparison plots", tiles),
            "</section>",
        ]
    )


def _render_png_dir(title: str, pdir: Path, root: Path) -> str:
    pngs = sorted([p for p in pdir.glob("*.png") if p.is_file()])
    tiles = [_img_tile(p, root, p.name) for p in pngs]
    return _gallery(title, tiles)


def _render_cnn_section(run_root: Path, root: Path) -> str:
    eval_dir = run_root / "evaluation"
    metrics_csv = eval_dir / "all_metrics_by_fold_and_noise.csv"
    rows: list[dict[str, Any]] = []
    if metrics_csv.exists():
        with metrics_csv.open() as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
        rows.sort(key=lambda d: (d.get("noise", ""), d.get("fold", "")))

    def table_html() -> str:
        if not rows:
            return '<div class="missing">missing: evaluation/all_metrics_by_fold_and_noise.csv</div>'
        cols = ["fold", "noise", "TCR", "NMR", "CMR", "F", "ACC"]
        head = "".join(f"<th>{_safe(c)}</th>" for c in cols)
        body = []
        for r in rows:
            tds = []
            for c in cols:
                v = r.get(c, "")
                if c in ("TCR", "NMR", "CMR", "F", "ACC"):
                    v = _fmt_float(v)
                tds.append(f"<td><code>{_safe(v)}</code></td>" if c != "fold" else f"<td>{_safe(v)}</td>")
            body.append("<tr>" + "".join(tds) + "</tr>")
        rel = _relpath(metrics_csv, root)
        return "\n".join(
            [
                f'<div class="minor">source: <a href="{_safe(rel)}">{_safe(rel)}</a></div>',
                '<div class="tablewrap"><table class="data">',
                f"<thead><tr>{head}</tr></thead>",
                "<tbody>",
                "\n".join(body),
                "</tbody></table></div>",
            ]
        )

    # Plots
    tiles: list[ImgTile] = []
    box = eval_dir / "metrics_boxplots.png"
    if box.exists():
        tiles.append(_img_tile(box, root, "metrics_boxplots.png"))

    # Fold training plots
    train_pngs: list[Path] = []
    for fd in sorted(run_root.glob("fold_*")):
        if not fd.is_dir():
            continue
        train_pngs.extend(sorted(fd.glob("training_*.png")))
    train_tiles = [_img_tile(p, root, _relpath(p, run_root)) for p in train_pngs]

    return "\n".join(
        [
            '<section id="cnn-baseline">',
            "  <h2>CNN baseline</h2>",
            "  <p class=\"desc\">CNN baseline artifacts and per-fold metrics table (from the benchmark evaluation outputs).</p>",
            _gallery("CNN plots", tiles),
            _gallery("CNN training curves (all folds)", train_tiles),
            '<h3 class="subhead">Per-fold metrics (CSV)</h3>',
            table_html(),
            "</section>",
        ]
    )


def build(run_root: Path = DEFAULT_RUN_ROOT, out_html: Path = DEFAULT_OUT_HTML) -> None:
    if not run_root.exists():
        raise SystemExit(f"Run root not found: {run_root}")

    # Exploratory fold dir: prefer Balleny fold (as specified).
    balleny = run_root / "fold_BallenyIslands2015_noise_0.25"
    if not balleny.exists():
        # fall back to glob if naming differs
        cands = sorted(run_root.glob("fold_Balleny*"))
        if not cands:
            raise SystemExit("Could not locate Balleny fold directory under run root.")
        balleny = cands[0]

    comparisons_dir = balleny / "yolo" / "comparisons"
    runs_det_dir = balleny / "yolo" / "runs_det"

    series_order = ["latest", "series_BS", "series_R0", "series_A", "series_B", "series_C", "series_D", "series_D_cnn_baseline"]
    series_blocks: list[str] = []
    for s in series_order:
        p = comparisons_dir / s
        if p.exists():
            series_blocks.append(_render_comparison_block(s, p, REPO_ROOT))

    # Per-run cards (Balleny)
    run_dirs = [p for p in runs_det_dir.iterdir() if p.is_dir()]
    run_names = _sorted_run_names([p.name for p in run_dirs])
    name_to_dir = {p.name: p for p in run_dirs}
    run_cards = []
    for rn in run_names:
        rd = name_to_dir.get(rn)
        if not rd:
            continue
        run_cards.append(_render_run_card(rd, REPO_ROOT, f"run-{_slug('balleny-'+rn)}", fold_name=balleny.name))

    # Cross-fold F1
    f1_dirs = sorted(run_root.glob("fold_*/yolo/runs_det/F1"))
    fold_cards = []
    for d in f1_dirs:
        # d = run_root / fold_* / yolo / runs_det / F1
        fold_name = d.parents[2].name
        fold_cards.append(_render_run_card(d, REPO_ROOT, f"run-{_slug(fold_name+'-F1')}", fold_name=fold_name))

    manual_compare = run_root / "yolo" / "analysis" / "manual_compare"
    agg_blocks = []
    for sub in ["07_compare_F1_agg_with_cnn_curves", "01_compare_F1_by_fold", "05_aggregate_F1_all_folds"]:
        p = manual_compare / sub / "plots"
        if p.exists():
            agg_blocks.append(_render_png_dir(sub, p, REPO_ROOT))

    other_agg_blocks = []
    for sub in [
        "02_compare_F1_aggregated",
        "03_compare_F1_C2_across_folds",
        "04_compare_F1_C2_default_mode",
    ]:
        p = manual_compare / sub / "plots"
        if p.exists():
            other_agg_blocks.append(_render_png_dir(sub, p, REPO_ROOT))

    # Per-fold F1 vs CNN plots
    per_fold_comp = manual_compare / "per_fold_F1_vs_CNN"
    per_fold_blocks: list[str] = []
    if per_fold_comp.exists():
        for fold_dir in sorted([p for p in per_fold_comp.iterdir() if p.is_dir()]):
            p = fold_dir / "plots"
            if not p.exists():
                continue
            per_fold_blocks.append(
                "\n".join(
                    [
                        f'<details class="block"><summary>{_safe(fold_dir.name)}: F1 vs CNN plots</summary>',
                        _render_png_dir("plots", p, REPO_ROOT),
                        "</details>",
                    ]
                )
            )

    # Header metadata
    generated_at = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config_json = _read_json(run_root / "config.json") or {}
    dataset = config_json.get("DATA_DIR", None)

    # C2 availability (for quick interpretability)
    c2_dirs = sorted(run_root.glob("fold_*/yolo/runs_det/C2"))
    f1_folds_n = len(f1_dirs)
    c2_folds_n = len(c2_dirs)
    c2_warning = ""
    if c2_folds_n and c2_folds_n < f1_folds_n:
        c2_warning = (
            f'<div class="missing">C2 is present in {c2_folds_n}/{f1_folds_n} folds in this run root; cross-fold C2 comparisons are incomplete.</div>'
        )
    elif not c2_folds_n:
        c2_warning = (
            f'<div class="minor">No cross-fold C2 run folders found under <code>fold_*/yolo/runs_det/C2</code> in this run root.</div>'
        )

    # Build HTML
    html_out = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Plot Dashboard - 251008_160341</title>
  <style>
    :root {{
      --bg: #fbfbf8;
      --fg: #111;
      --muted: #555;
      --card: #fff;
      --line: #e6e6df;
      --accent: #0b5;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
    }}
    body {{
      margin: 0;
      font-family: var(--sans);
      color: var(--fg);
      background: var(--bg);
      line-height: 1.4;
    }}
    a {{ color: inherit; }}
    header {{
      position: sticky;
      top: 0;
      background: rgba(251,251,248,0.92);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid var(--line);
      z-index: 10;
    }}
    .wrap {{ display: grid; grid-template-columns: 280px 1fr; gap: 16px; max-width: 1520px; margin: 0 auto; padding: 16px; }}
    nav {{
      position: sticky;
      top: 76px;
      align-self: start;
      border: 1px solid var(--line);
      background: var(--card);
      border-radius: 10px;
      padding: 12px;
    }}
    nav h3 {{ margin: 0 0 8px 0; font-size: 14px; color: var(--muted); letter-spacing: .02em; }}
    nav a {{ display: block; padding: 6px 8px; border-radius: 8px; text-decoration: none; font-size: 14px; }}
    nav a:hover {{ background: #f2f2eb; }}
    main {{
      min-width: 0;
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      letter-spacing: .01em;
    }}
    .head {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      padding: 12px 16px;
      max-width: 1520px;
      margin: 0 auto;
    }}
    .meta {{ font-size: 12px; color: var(--muted); }}
    section {{ margin: 0 0 28px 0; }}
    .desc {{ color: var(--muted); font-size: 14px; margin-top: 6px; }}
    .minor {{ color: var(--muted); font-size: 12px; margin: 6px 0; }}
    code {{ font-family: var(--mono); font-size: 12px; background: #f3f3ee; padding: 1px 6px; border-radius: 8px; }}
    .subhead {{ margin: 12px 0 8px 0; font-size: 14px; color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 12px;
      margin: 8px 0 0 0;
    }}
    .tile {{
      margin: 0;
      border: 1px solid var(--line);
      background: var(--card);
      border-radius: 10px;
      overflow: hidden;
      display: grid;
      grid-template-rows: auto 1fr;
    }}
    .thumb {{
      width: 100%;
      height: 170px;
      object-fit: cover;
      cursor: zoom-in;
      background: #fff;
    }}
    figcaption {{
      padding: 10px 10px 12px 10px;
      border-top: 1px solid var(--line);
      display: grid;
      gap: 6px;
    }}
    .cap {{ font-size: 12px; font-family: var(--mono); word-break: break-word; }}
    .filelink {{ font-size: 12px; color: var(--muted); text-decoration: none; }}
    .filelink:hover {{ text-decoration: underline; }}
    .note {{ font-size: 12px; color: var(--muted); }}
    details.block, details.card {{
      border: 1px solid var(--line);
      background: var(--card);
      border-radius: 10px;
      padding: 10px 12px;
      margin: 10px 0;
    }}
    details > summary {{
      cursor: pointer;
      font-weight: 600;
    }}
    .card summary {{ display: flex; justify-content: space-between; gap: 10px; }}
    .run {{ font-family: var(--mono); }}
    .fold {{ color: var(--muted); font-weight: 400; }}
    .cardbody {{ margin-top: 10px; }}
    .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .col {{ min-width: 0; }}
    table.kv {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    table.kv th {{
      text-align: left;
      font-weight: 600;
      color: var(--muted);
      width: 42%;
      padding: 6px 8px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    table.kv td {{
      padding: 6px 8px;
      border-bottom: 1px solid var(--line);
      font-family: var(--mono);
      word-break: break-word;
    }}
    .missing {{ color: #a22; font-size: 12px; }}
    .files summary {{ color: var(--muted); font-weight: 500; }}
    ul.filelist {{
      margin: 8px 0 0 16px;
      padding: 0;
      font-size: 12px;
      color: var(--muted);
    }}
    ul.filelist li {{ margin: 3px 0; }}
    .search {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
      border: 1px solid var(--line);
      background: var(--card);
      border-radius: 10px;
      padding: 12px;
      margin: 0 0 12px 0;
    }}
    .search input {{
      width: 100%;
      padding: 10px 10px;
      border-radius: 10px;
      border: 1px solid var(--line);
      font-size: 14px;
      background: #fff;
    }}
    .tablewrap {{
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--card);
    }}
    table.data {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    table.data th, table.data td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      white-space: nowrap;
    }}
    table.data th {{ color: var(--muted); font-weight: 600; }}
    .modal {{
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.72);
      display: none;
      align-items: center;
      justify-content: center;
      padding: 18px;
      z-index: 100;
    }}
    .modal.open {{ display: flex; }}
    .modalbox {{
      max-width: min(1400px, 98vw);
      max-height: 92vh;
      background: #111;
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid rgba(255,255,255,0.15);
      display: grid;
      /* minmax(0, 1fr) prevents grid overflow clipping inside the modal */
      grid-template-rows: auto minmax(0, 1fr);
    }}
    .modalhead {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 10px 12px;
      color: #ddd;
      font-family: var(--mono);
      font-size: 12px;
      border-bottom: 1px solid rgba(255,255,255,0.12);
    }}
    .modalhead a {{ color: #ddd; text-decoration: none; }}
    .modalhead a:hover {{ text-decoration: underline; }}
    .closebtn {{
      border: 1px solid rgba(255,255,255,0.25);
      background: transparent;
      color: #ddd;
      border-radius: 10px;
      padding: 6px 10px;
      cursor: pointer;
      font-size: 12px;
      font-family: var(--mono);
    }}
    .modalbody {{
      min-height: 0;
      overflow: auto;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #111;
    }}
    .full {{
      /* Fit within the available space; if extremely tall, modalbody allows scroll */
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }}
    @media (max-width: 1000px) {{
      .wrap {{ grid-template-columns: 1fr; }}
      nav {{ position: relative; top: 0; }}
      header {{ position: relative; }}
      .cols {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="head">
      <div>
        <h1>Plot Dashboard: outputs/cnn_results/251008_160341</h1>
        <div class="meta">generated: {_safe(generated_at)} | dataset: <code>{_safe(dataset)}</code></div>
      </div>
      <div class="meta">
        <a href="{_safe(_relpath(REPO_ROOT / 'docs' / 'RUN_RESULTS.md', REPO_ROOT))}">docs/RUN_RESULTS.md</a>
        &nbsp;|&nbsp;
        <a href="{_safe(_relpath(REPO_ROOT / 'docs' / 'THESIS_PLAN.md', REPO_ROOT))}">docs/THESIS_PLAN.md</a>
      </div>
    </div>
  </header>

  <div class="wrap">
    <nav>
      <h3>Navigation</h3>
      <a href="#overview">Overview</a>
      <a href="#exploratory">Exploratory (Balleny)</a>
      <a href="#exploratory-series">Exploratory: series comparisons</a>
      <a href="#exploratory-runs">Exploratory: per-run cards</a>
      <a href="#confirmatory">Confirmatory (F1 across folds)</a>
      <a href="#confirmatory-aggregate">F1: aggregate plots</a>
      <a href="#confirmatory-other">Other cross-fold analyses</a>
      <a href="#confirmatory-folds">F1: per-fold drilldowns</a>
      <a href="#cnn-baseline">CNN baseline</a>
      <a href="#appendix">Appendix</a>
    </nav>

    <main>
      <section id="overview">
        <h2>Overview</h2>
        <p class="desc">
          Local dashboard for browsing plots and key artifacts for a single run directory:
          <code>outputs/cnn_results/251008_160341</code>. All images/values are pulled directly
          from files on disk; missing tiles mean missing artifacts.
        </p>
        <h3 class="subhead">Legend (custom plots)</h3>
        <ul class="filelist">
          <li><code>f_vs_confidence_top1.png</code>: custom benchmark F vs confidence (top-1 collapse), typically for val and test.</li>
          <li><code>tcr_vs_nmr_top1.png</code>: TCR vs NMR trade-off curve across confidence thresholds.</li>
        </ul>
      </section>

      <section id="exploratory">
        <h2>Exploratory single-fold (Balleny)</h2>
        <p class="desc">
          Chronological experiment series comparisons and per-run drilldowns for
          <code>{_safe(_relpath(balleny, run_root))}</code>.
        </p>
      </section>

      <section id="exploratory-series">
        <h2>Exploratory: series comparisons</h2>
        {''.join(series_blocks)}
      </section>

      <section id="exploratory-runs">
        <h2>Exploratory: per-run cards</h2>
        <div class="search">
          <input id="q" type="search" placeholder="Filter runs/folds (e.g. F1, C2, Balleny, kerguelen2015)..." />
          <div class="minor">Filters apply to all run cards below (exploratory + cross-fold).</div>
        </div>
        {''.join(run_cards)}
      </section>

      <section id="confirmatory">
        <h2>Confirmatory cross-fold (F1)</h2>
        <p class="desc">
          Cross-fold summary plots plus per-fold drilldowns for <code>runs_det/F1</code> under each fold.
        </p>
      </section>

      <section id="confirmatory-aggregate">
        <h2>F1: aggregate plots</h2>
        {''.join(agg_blocks)}
        {c2_warning}
        <h3 class="subhead">Per-fold F1 vs CNN comparison plots</h3>
        {''.join(per_fold_blocks) if per_fold_blocks else '<div class="minor">No per-fold comparison plots found.</div>'}
      </section>

      <section id="confirmatory-other">
        <h2>Other cross-fold analyses</h2>
        <p class="desc">Additional cross-fold plot groups present under <code>yolo/analysis/manual_compare</code>.</p>
        {''.join(other_agg_blocks) if other_agg_blocks else '<div class="minor">No additional plot groups found.</div>'}
      </section>

      <section id="confirmatory-folds">
        <h2>F1: per-fold drilldowns</h2>
        <p class="desc">Each card is a full Ultralytics run folder plus the custom evaluation artifacts for that fold.</p>
        {''.join(fold_cards)}
      </section>

      {_render_cnn_section(run_root, REPO_ROOT)}

      <section id="appendix">
        <h2>Appendix</h2>
        <div class="minor">Run root: <a href="{_safe(_relpath(run_root, REPO_ROOT))}">{_safe(_relpath(run_root, REPO_ROOT))}</a></div>
        <div class="minor">Exploratory comparisons dir: <a href="{_safe(_relpath(comparisons_dir, REPO_ROOT))}">{_safe(_relpath(comparisons_dir, REPO_ROOT))}</a></div>
        <div class="minor">Exploratory runs dir: <a href="{_safe(_relpath(runs_det_dir, REPO_ROOT))}">{_safe(_relpath(runs_det_dir, REPO_ROOT))}</a></div>
        <div class="minor">Cross-fold analysis dir: <a href="{_safe(_relpath(manual_compare, REPO_ROOT))}">{_safe(_relpath(manual_compare, REPO_ROOT))}</a></div>
      </section>
    </main>
  </div>

  <div class="modal" id="modal" role="dialog" aria-modal="true" aria-label="Image preview">
    <div class="modalbox">
      <div class="modalhead">
        <div><span id="modaltitle">image</span> | <a id="modalopen" href="#" target="_blank" rel="noopener">open</a></div>
        <button class="closebtn" id="modalclose" type="button">close</button>
      </div>
      <div class="modalbody">
        <img class="full" id="modalimg" alt="Full-size plot" />
      </div>
    </div>
  </div>

  <script>
    (function() {{
      const modal = document.getElementById('modal');
      const modalImg = document.getElementById('modalimg');
      const modalTitle = document.getElementById('modaltitle');
      const modalOpen = document.getElementById('modalopen');
      const closeBtn = document.getElementById('modalclose');

      function openModal(src, title) {{
        modal.classList.add('open');
        modalImg.src = src;
        modalTitle.textContent = title || src;
        modalOpen.href = src;
      }}
      function closeModal() {{
        modal.classList.remove('open');
        modalImg.src = '';
      }}
      closeBtn.addEventListener('click', closeModal);
      modal.addEventListener('click', (e) => {{
        if (e.target === modal) closeModal();
      }});
      document.addEventListener('keydown', (e) => {{
        if (e.key === 'Escape') closeModal();
      }});

      document.querySelectorAll('img.thumb').forEach((img) => {{
        img.addEventListener('click', () => {{
          const full = img.getAttribute('data-full') || img.getAttribute('src');
          openModal(full, img.getAttribute('alt') || full);
        }});
      }});

      const q = document.getElementById('q');
      function applyFilter() {{
        const v = (q.value || '').trim().toLowerCase();
        document.querySelectorAll('details.card').forEach((card) => {{
          const run = (card.getAttribute('data-run') || '').toLowerCase();
          const fold = (card.getAttribute('data-fold') || '').toLowerCase();
          const hay = run + ' ' + fold;
          card.style.display = (!v || hay.includes(v)) ? '' : 'none';
        }});
      }}
      if (q) {{
        q.addEventListener('input', applyFilter);
      }}
    }})();
  </script>
</body>
</html>
"""

    out_html.write_text(html_out, encoding="utf-8")


if __name__ == "__main__":
    build()
