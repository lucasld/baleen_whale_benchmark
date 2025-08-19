#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import Counter
from PIL import Image, ImageDraw
import scipy.signal as spsig
import scipy.io.wavfile as wavfile


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return json.load(f)


def load_koogu(site_dir: str, wav_base: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    npz_path = os.path.join(site_dir, f"{wav_base}.wav.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Koogu file not found: {npz_path}")
    data = np.load(npz_path)
    clips = data['clips']  # shape (num_clips, T)
    labels = data['labels']  # shape (num_clips, C)
    classes_file = os.path.join(site_dir, 'classes_list.json')
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            label_list = json.load(f)
    else:
        label_list = [str(i) for i in range(labels.shape[1])]
    return clips, labels, label_list


def load_annotations(ann_dir: str, wav_base: str) -> pd.DataFrame:
    ann_path = os.path.join(ann_dir, f"{wav_base}_annotations.txt")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")
    df = pd.read_csv(ann_path, sep='\t')
    required = ['Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'Tags']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {ann_path}")
    return df


def compute_overlaps_for_clip(clip_start: float, clip_end: float, ann_df: pd.DataFrame) -> pd.DataFrame:
    # Overlap where selection interval intersects [clip_start, clip_end]
    sel_start = ann_df['Begin Time (s)'].astype(float).values
    sel_end = ann_df['End Time (s)'].astype(float).values
    mask = (sel_end > clip_start) & (sel_start < clip_end)
    return ann_df.loc[mask]


def tags_to_class_ids(tags_str: str, tag2id: Dict[str, int]) -> List[int]:
    if pd.isna(tags_str):
        return []
    ids = []
    for t in str(tags_str).split(','):
        t = t.strip()
        if not t:
            continue
        if t in tag2id:
            ids.append(tag2id[t])
    return ids


def build_yolo_boxes(overlaps: pd.DataFrame, clip_start: float, clip_end: float, fs: float, tag2id: Dict[str, int]) -> List[Tuple[int, float, float, float, float]]:
    # Map selections to YOLO normalized boxes for a 15s window and freq range [0, fs/2]
    duration = clip_end - clip_start
    f_max = fs / 2.0  # 125 Hz for fs=250
    boxes = []
    for _, row in overlaps.iterrows():
        sel_start = float(row['Begin Time (s)'])
        sel_end = float(row['End Time (s)'])
        low_f = float(row['Low Freq (Hz)'])
        high_f = float(row['High Freq (Hz)'])
        inter_start = max(sel_start, clip_start)
        inter_end = min(sel_end, clip_end)
        if inter_end <= inter_start:
            continue
        x_center = ((inter_start + inter_end) / 2.0 - clip_start) / duration
        w_norm = (inter_end - inter_start) / duration
        low_n = max(0.0, min(1.0, low_f / f_max))
        high_n = max(0.0, min(1.0, high_f / f_max))
        if high_n <= low_n:
            continue
        y_center = 1.0 - ((low_n + high_n) / 2.0)
        h_norm = (high_n - low_n)
        # Clamp
        x_center = float(max(0.0, min(1.0, x_center)))
        y_center = float(max(0.0, min(1.0, y_center)))
        w_norm = float(max(0.0, min(1.0, w_norm)))
        h_norm = float(max(0.0, min(1.0, h_norm)))
        for cls_id in tags_to_class_ids(row.get('Tags'), tag2id):
            boxes.append((cls_id, x_center, y_center, w_norm, h_norm))
    return boxes


def spectrogram_image_from_clip(clip: np.ndarray) -> Image.Image:
    # Reproduce generate_spectrograms.py processing
    fs = 250.0
    sos = spsig.iirfilter(20, [5, 124], btype='band', ftype='butter', output='sos', fs=fs)
    filtered = spsig.sosfilt(sos, clip)
    f, t, Sxx = spsig.spectrogram(filtered, fs=fs, window='hamming', nperseg=256, noverlap=250,
                                  nfft=3570, detrend=False, return_onesided=True, scaling='density', axis=-1,
                                  mode='magnitude')
    Sxx = 1 - Sxx
    per = np.percentile(Sxx.flatten(), 98)
    I = (Sxx - Sxx.min()) / (per - Sxx.min())
    I[I > 1] = 1
    im = np.array(np.flipud(I) * 255, dtype=np.uint8)
    img = Image.fromarray(im).resize((30, 90), Image.Resampling.LANCZOS)
    # Ensure RGB for colored overlays
    return img.convert('RGB')


def main():
    ap = argparse.ArgumentParser(description='Validate Koogu clips vs annotations and YOLO box mapping for a single WAV or batch')
    ap.add_argument('--output_dir', type=str, required=True, help='Preprocessed dataset root (contains annotations/, koogu_output/)')
    ap.add_argument('--site', type=str, help='Site name (e.g., BallenyIslands2015)')
    ap.add_argument('--wav_base', type=str, help='WAV base name (e.g., 20150510_100000)')
    ap.add_argument('--config', type=str, default=os.path.join('src', 'config.json'))
    ap.add_argument('--max_clips', type=int, default=60)
    ap.add_argument('--save_overlays', type=int, default=5, help='How many overlay examples to save')
    ap.add_argument('--out', type=str, default=os.path.join('outputs', 'analysis'))
    # New: search for best time origin offset
    ap.add_argument('--search_offset', action='store_true', help='Search for a global time offset that best aligns overlaps with Koogu labels')
    ap.add_argument('--search_window', type=float, default=120.0, help='Search +/- window (seconds) around min annotation time')
    ap.add_argument('--search_step', type=float, default=0.5, help='Step (seconds) for offset search')
    # New: optional audio alignment check from raw WAV
    ap.add_argument('--prepared_dataset', type=str, default=os.path.join('datasets', 'prepared_dataset'), help='Path to prepared_dataset to read raw WAV')
    ap.add_argument('--site_config', type=str, default=os.path.join('src', 'custom_preprocessing', 'site_config.json'), help='Path to site_config.json for WAV sample rate')
    ap.add_argument('--audio_check_clips', type=int, default=0, help='If >0, extract this many clips from raw WAV at best offset and correlate with Koogu clips')
    # Batch mode
    ap.add_argument('--batch', action='store_true', help='Run on multiple WAVs automatically')
    ap.add_argument('--sites', type=str, default=None, help='Comma-separated list of sites to include (or omit to use --site)')
    ap.add_argument('--num_wavs', type=int, default=3, help='Number of WAVs per site to validate in batch mode')
    args = ap.parse_args()

    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)
    cfg = load_config(args.config)
    categories = cfg.get('CATEGORIES', [])
    print(f"Categories ({len(categories)}): {categories}")

    # Helper to run single validation on a given (site, wav_base)
    def run_single(site: str, wav_base: str):
        koogu_site_dir = os.path.join(args.output_dir, 'koogu_output', site)
        ann_site_dir = os.path.join(args.output_dir, 'annotations', site)
        print(f"Koogu dir: {koogu_site_dir}")
        print(f"Annotations dir: {ann_site_dir}")

        clips, labels, koogu_label_list = load_koogu(koogu_site_dir, wav_base)
        print(f"Loaded Koogu: clips={clips.shape}, labels={labels.shape}, classes={koogu_label_list}")

        ann_df = load_annotations(ann_site_dir, wav_base)
        print(f"Loaded annotations: rows={len(ann_df)}")

        out_dir = os.path.join(out_root, f"validate_{site}_{wav_base}")
        os.makedirs(out_dir, exist_ok=True)

        # Build tag mapping to Koogu class ids
        tag2id = {c: i for i, c in enumerate(koogu_label_list)}
        tags_all = Counter()
        for v in ann_df['Tags'].dropna().tolist():
            for t in str(v).split(','):
                t = t.strip()
                if t:
                    tags_all[t] += 1
        print(f"Tag counts (top 20): {tags_all.most_common(20)}")
        missing_in_koogu = sorted([t for t in tags_all if t not in tag2id])
        if missing_in_koogu:
            print(f"WARNING: Tags not present in Koogu label list and will be ignored: {missing_in_koogu[:20]}{' ...' if len(missing_in_koogu)>20 else ''}")

        # Per-clip overlap vs Koogu labels (assume origin at 0 initially)
        clip_len = 15.0
        advance = 2.5
        fs = 250.0
        num_clips = clips.shape[0]
        max_k = min(num_clips, args.max_clips)
        rows = []
        overlay_saved = 0
        for k in range(max_k):
            c_start = k * advance
            c_end = c_start + clip_len
            ov = compute_overlaps_for_clip(c_start, c_end, ann_df)
            # Classes from overlaps
            ov_classes = []
            for _, r in ov.iterrows():
                ov_classes += tags_to_class_ids(r.get('Tags'), tag2id)
            ov_counts = Counter(ov_classes)
            # Koogu labels
            label_vec = labels[k]
            nz = [i for i, v in enumerate(label_vec) if v > 0.5]
            top = int(np.argmax(label_vec)) if label_vec.size > 0 else -1
            rows.append({
                'clip_idx': k,
                'clip_start_s': c_start,
                'num_overlaps': int(len(ov)),
                'overlap_class_ids': ';'.join(map(str, sorted(set(ov_classes)))) if ov_classes else '',
                'overlap_counts': json.dumps({int(k2): int(v2) for k2, v2 in ov_counts.items()}),
                'koogu_nonzero_ids': ';'.join(map(str, nz)) if nz else '',
                'koogu_top_id': int(top),
                'koogu_top_label': koogu_label_list[top] if 0 <= top < len(koogu_label_list) else '',
                'koogu_vec_sum': float(np.sum(label_vec)),
            })

            # Save overlays for first few clips with overlaps
            if overlay_saved < args.save_overlays and len(ov) > 0:
                try:
                    img = spectrogram_image_from_clip(clips[k])
                    # Draw YOLO boxes
                    boxes = build_yolo_boxes(ov, c_start, c_end, fs, tag2id)
                    draw = ImageDraw.Draw(img)
                    for (cls_id, x_c, y_c, w_n, h_n) in boxes:
                        x0 = int((x_c - w_n / 2) * 30)
                        x1 = int((x_c + w_n / 2) * 30)
                        y0 = int((y_c - h_n / 2) * 90)
                        y1 = int((y_c + h_n / 2) * 90)
                        x0 = max(0, min(29, x0)); x1 = max(0, min(29, x1))
                        y0 = max(0, min(89, y0)); y1 = max(0, min(89, y1))
                        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=1)
                    out_path = os.path.join(out_dir, f"overlay_clip_{k:04d}.png")
                    img.save(out_path)
                    overlay_saved += 1
                except Exception as e:
                    print(f"WARN: failed to save overlay for clip {k}: {e}")

        df_rows = pd.DataFrame(rows)
        csv_out = os.path.join(out_dir, 'per_clip_overlap_vs_koogu.csv')
        df_rows.to_csv(csv_out, index=False)
        print(f"Wrote per-clip comparison to: {csv_out}")
        print("Examples:")
        print(df_rows.head(10).to_string(index=False))

        # Simple summary stats
        with_overlaps = (df_rows['num_overlaps'] > 0).sum()
        with_nonzero_koogu = (df_rows['koogu_vec_sum'] > 0).sum()
        print(f"Clips examined: {len(df_rows)} | with overlaps: {with_overlaps} | with nonzero Koogu labels: {with_nonzero_koogu}")

        # Save label list and tag coverage
        with open(os.path.join(out_dir, 'koogu_classes.json'), 'w') as f:
            json.dump(koogu_label_list, f, indent=2)
        with open(os.path.join(out_dir, 'tag_counts.json'), 'w') as f:
            json.dump({k: int(v) for k, v in tags_all.items()}, f, indent=2)

        print(f"Overlays saved: {overlay_saved} (in {out_dir})")

        # Optional: search for best global time origin offset to align overlaps to Koogu labels
        if args.search_offset and len(ann_df) > 0:
            min_ann = float(ann_df['Begin Time (s)'].min())
            start = min_ann - abs(args.search_window)
            end = min_ann + abs(args.search_window)
            steps = int(max(1, round((end - start) / args.search_step)))
            print(f"Searching offset in [{start:.2f}, {end:.2f}] with step {args.search_step:.2f} ({steps} steps)...")
            best_score = -1.0
            best_t0 = None
            for i in range(steps + 1):
                t0 = start + i * args.search_step
                score = 0.0
                for k in range(max_k):
                    c_start = t0 + k * advance
                    c_end = c_start + clip_len
                    ov = compute_overlaps_for_clip(c_start, c_end, ann_df)
                    ov_ids = []
                    for _, r in ov.iterrows():
                        ov_ids += tags_to_class_ids(r.get('Tags'), tag2id)
                    ov_set = set(ov_ids)
                    nz = set([i2 for i2, v in enumerate(labels[k]) if v > 0.5])
                    # score by intersection size
                    score += len(ov_set & nz)
                if score > best_score:
                    best_score = score
                    best_t0 = t0
            
            print(f"Best offset t0 = {best_t0:.3f}s with score {best_score}")

        # Recompute per-clip table at best_t0 and save
        rows2 = []
        overlay_saved2 = 0
        for k in range(max_k):
            c_start = best_t0 + k * advance
            c_end = c_start + clip_len
            ov = compute_overlaps_for_clip(c_start, c_end, ann_df)
            ov_classes = []
            for _, r in ov.iterrows():
                ov_classes += tags_to_class_ids(r.get('Tags'), tag2id)
            ov_counts = Counter(ov_classes)
            label_vec = labels[k]
            nz = [i for i, v in enumerate(label_vec) if v > 0.5]
            top = int(np.argmax(label_vec)) if label_vec.size > 0 else -1
            rows2.append({
                'clip_idx': k,
                'clip_start_s': c_start,
                'num_overlaps': int(len(ov)),
                'overlap_class_ids': ';'.join(map(str, sorted(set(ov_classes)))) if ov_classes else '',
                'overlap_counts': json.dumps({int(k2): int(v2) for k2, v2 in ov_counts.items()}),
                'koogu_nonzero_ids': ';'.join(map(str, nz)) if nz else '',
                'koogu_top_id': int(top),
                'koogu_top_label': koogu_label_list[top] if 0 <= top < len(koogu_label_list) else '',
                'koogu_vec_sum': float(np.sum(label_vec)),
            })
            if overlay_saved2 < args.save_overlays and len(ov) > 0:
                try:
                    img = spectrogram_image_from_clip(clips[k])
                    boxes = build_yolo_boxes(ov, c_start, c_end, fs, tag2id)
                    draw = ImageDraw.Draw(img)
                    for (cls_id, x_c, y_c, w_n, h_n) in boxes:
                        x0 = int((x_c - w_n / 2) * 30)
                        x1 = int((x_c + w_n / 2) * 30)
                        y0 = int((y_c - h_n / 2) * 90)
                        y1 = int((y_c + h_n / 2) * 90)
                        x0 = max(0, min(29, x0)); x1 = max(0, min(29, x1))
                        y0 = max(0, min(89, y0)); y1 = max(0, min(89, y1))
                        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=1)
                    out_path = os.path.join(out_dir, f"overlay_best_t0_clip_{k:04d}.png")
                    img.save(out_path)
                    overlay_saved2 += 1
                except Exception as e:
                    print(f"WARN: failed to save best-t0 overlay for clip {k}: {e}")
        df2 = pd.DataFrame(rows2)
        df2_path = os.path.join(out_dir, 'per_clip_overlap_vs_koogu_best_t0.csv')
        df2.to_csv(df2_path, index=False)
        print(f"Best-offset per-clip table saved to: {df2_path}")

        # Optional audio correlation check against raw WAV
        if args.audio_check_clips > 0:
            raw_wav_path = os.path.join(args.prepared_dataset, args.site, 'wav', f'{args.wav_base}.wav')
            if os.path.exists(raw_wav_path) and os.path.exists(args.site_config):
                try:
                    with open(args.site_config, 'r') as f:
                        site_cfg = json.load(f)
                    site_fs = site_cfg.get(args.site)
                    if not site_fs:
                        print(f"WARN: site sample rate not found for {args.site} in {args.site_config}")
                    else:
                        sr, data = wavfile.read(raw_wav_path)
                        if sr != site_fs:
                            print(f"WARN: WAV header rate {sr} != site_config rate {site_fs}")
                        data = data.astype(np.float32)
                        # Normalize to [-1,1] if int16
                        if data.dtype.kind in ('i', 'u'):
                            maxv = np.iinfo(data.dtype).max
                            data = data / maxv
                        checked = 0
                        for k in range(min(args.audio_check_clips, max_k)):
                            c_start = best_t0 + k * advance
                            c_end = c_start + clip_len
                            s0 = int(c_start * sr)
                            s1 = int(c_end * sr)
                            if s1 <= s0 or s1 > len(data):
                                continue
                            seg = data[s0:s1]
                            # Resample to 250 Hz
                            g = np.gcd(sr, int(fs))
                            up = int(fs) // g
                            down = sr // g
                            seg_250 = spsig.resample_poly(seg, up, down)
                            koogu_seg = clips[k]
                            # Pad/trim to same length
                            L = min(len(seg_250), len(koogu_seg))
                            if L <= 0:
                                continue
                            a = seg_250[:L]
                            b = koogu_seg[:L]
                            # Normalize
                            a = (a - np.mean(a)) / (np.std(a) + 1e-8)
                            b = (b - np.mean(b)) / (np.std(b) + 1e-8)
                            corr = float(np.correlate(a, b, mode='valid') / L)
                            print(f"Audio corr clip {k}: {corr:.3f}")
                            checked += 1
                        if checked == 0:
                            print("WARN: No clips checked for audio correlation (bounds or lengths invalid)")
                except Exception as e:
                    print(f"WARN: audio correlation step failed: {e}")

        return best_t0, best_score, with_overlaps, with_nonzero_koogu

    # Batch or single
    results = []
    if args.batch:
        # Determine sites
        if args.sites:
            sites = [s.strip() for s in args.sites.split(',') if s.strip()]
        elif args.site:
            sites = [args.site]
        else:
            # all sites under koogu_output
            koogu_root = os.path.join(args.output_dir, 'koogu_output')
            sites = sorted([d for d in os.listdir(koogu_root) if os.path.isdir(os.path.join(koogu_root, d))])
        print(f"Batch validating sites: {sites}")
        for site in sites:
            koogu_site_dir = os.path.join(args.output_dir, 'koogu_output', site)
            wav_bases = [os.path.splitext(os.path.splitext(f)[0])[0] for f in sorted(os.listdir(koogu_site_dir)) if f.endswith('.wav.npz')]
            wav_bases = wav_bases[:max(0, args.num_wavs)]
            for wb in wav_bases:
                print(f"\n=== Validating {site} / {wb} ===")
                try:
                    t0, score, ov_cnt, nz_cnt = run_single(site, wb)
                    results.append({'site': site, 'wav_base': wb, 'best_t0': t0, 'score': score, 'clips_with_overlaps_at_t0': ov_cnt, 'clips_with_koogu': nz_cnt})
                except Exception as e:
                    print(f"ERROR: failed {site}/{wb}: {e}")
                    results.append({'site': site, 'wav_base': wb, 'best_t0': None, 'score': -1, 'clips_with_overlaps_at_t0': 0, 'clips_with_koogu': 0, 'error': str(e)})
        df_res = pd.DataFrame(results)
        out_csv = os.path.join(out_root, 'validate_batch_summary.csv')
        df_res.to_csv(out_csv, index=False)
        print(f"\nBatch summary saved to: {out_csv}")
    else:
        if not args.site or not args.wav_base:
            print("ERROR: --site and --wav_base are required in single mode", file=sys.stderr)
            sys.exit(2)
        run_single(args.site, args.wav_base)
    print("Done.")


if __name__ == '__main__':
    main()


