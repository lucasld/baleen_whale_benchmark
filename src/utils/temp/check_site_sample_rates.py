#!/usr/bin/env python3
import os
import sys
import json
import time
import glob
import argparse
from collections import Counter, defaultdict
import wave


def read_wav_rate(wav_path: str, verbose: bool = False) -> int:
    try:
        with wave.open(wav_path, 'rb') as w:
            return w.getframerate()
    except Exception:
        if verbose:
            print(f"WARN: failed to read WAV header: {wav_path}")
        return -1


def scan_site_wavs(site_wav_dir: str, max_files: int = 200, verbose: bool = False, progress_every: int = 50) -> Counter:
    wavs = sorted(glob.glob(os.path.join(site_wav_dir, '*.wav')))
    if not wavs:
        return Counter()
    # Sample up to max_files for speed
    if len(wavs) > max_files:
        wavs = wavs[:max_files]
    rates = Counter()
    total = len(wavs)
    if verbose:
        print(f"  Found {total} wav files, scanning up to {max_files}")
    t0 = time.time()
    for idx, wp in enumerate(wavs, 1):
        r = read_wav_rate(wp, verbose=verbose)
        rates[r] += 1
        if verbose and (idx % progress_every == 0 or idx == total):
            elapsed = time.time() - t0
            print(f"    Scanned {idx}/{total} files... elapsed {elapsed:.1f}s | current rates: {dict(rates)}")
    return rates


def main():
    parser = argparse.ArgumentParser(description='Check site sample rates against site_config.json')
    parser.add_argument('--prepared_dataset', type=str, default=os.path.join('datasets', 'prepared_dataset'),
                        help='Path to prepared_dataset (contains <site>/wav/*.wav)')
    parser.add_argument('--site_config', type=str, default=os.path.join('src', 'custom_preprocessing', 'site_config.json'),
                        help='Path to site_config.json with expected sample rates per site')
    parser.add_argument('--max_files', type=int, default=200,
                        help='Max WAVs to read per site (for speed)')
    parser.add_argument('--out', type=str, default=os.path.join('outputs', 'analysis'),
                        help='Directory to write the JSON report')
    parser.add_argument('--site', type=str, default=None,
                        help='If set, only scan this site directory')
    parser.add_argument('--verbose', action='store_true', help='Print detailed progress logs')
    parser.add_argument('--progress_every', type=int, default=50, help='Print progress every N files when verbose')
    args = parser.parse_args()

    prepared_dataset = os.path.abspath(args.prepared_dataset)
    site_config_path = os.path.abspath(args.site_config)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(prepared_dataset):
        print(f"ERROR: prepared_dataset not found: {prepared_dataset}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isfile(site_config_path):
        print(f"ERROR: site_config.json not found: {site_config_path}", file=sys.stderr)
        sys.exit(2)

    with open(site_config_path, 'r') as f:
        site_cfg = json.load(f)

    sites_on_disk = sorted([d for d in os.listdir(prepared_dataset) if os.path.isdir(os.path.join(prepared_dataset, d))])
    if args.site:
        if args.site in sites_on_disk:
            sites_on_disk = [args.site]
        else:
            print(f"ERROR: requested site '{args.site}' not found under {prepared_dataset}", file=sys.stderr)
            sys.exit(2)
    report = {
        'prepared_dataset': prepared_dataset,
        'site_config': site_config_path,
        'timestamp': time.strftime('%y%m%d_%H%M%S'),
        'sites_on_disk': sites_on_disk,
        'sites_in_config': sorted(site_cfg.keys()),
        'per_site': {}
    }

    print(f"Starting scan of {len(sites_on_disk)} site(s) under {prepared_dataset}")
    for site in sites_on_disk:
        wav_dir = os.path.join(prepared_dataset, site, 'wav')
        print(f"- Site: {site} | wav_dir: {wav_dir}")
        rates = scan_site_wavs(wav_dir, max_files=args.max_files, verbose=args.verbose, progress_every=args.progress_every)
        expected = site_cfg.get(site)
        summary = {
            'wav_dir': wav_dir,
            'num_scanned': sum(rates.values()),
            'observed_rates_counts': dict(rates),
            'expected_rate': expected,
            'match': (expected in rates and rates.get(expected, 0) > 0) if expected is not None else None,
            'all_match_expected': (len(rates) == 1 and expected in rates) if expected is not None and rates else None,
        }
        report['per_site'][site] = summary

    # Sites in config but not on disk
    missing_sites = sorted([s for s in site_cfg.keys() if s not in sites_on_disk])
    report['sites_in_config_but_missing_on_disk'] = missing_sites

    out_path = os.path.join(out_dir, f'site_rate_check_{report["timestamp"]}.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Pretty-print a brief summary
    print(f"Scanned prepared dataset: {prepared_dataset}")
    for site in sites_on_disk:
        s = report['per_site'][site]
        print(f"  {site}: expected={s['expected_rate']} | observed={s['observed_rates_counts']} | scanned={s['num_scanned']} | all_match={s['all_match_expected']}")
    if missing_sites:
        print(f"Sites in config but not on disk: {missing_sites}")
    print(f"Report saved to: {out_path}")


if __name__ == '__main__':
    main()


