import os
import random
import wave

BASE_DIR = "/share/klab/danthes/data/ArcticWhales_AADC/"
N_FILES = 5

def get_sample_rate(wav_path):
    try:
        with wave.open(wav_path, 'rb') as wf:
            return wf.getframerate()
    except wave.Error as e:
        return f"WaveError: {e}"
    except Exception as e:
        return f"Error: {e}"

def main():
    for location in sorted(os.listdir(BASE_DIR)):
        loc_path = os.path.join(BASE_DIR, location)
        wav_dir = os.path.join(loc_path, "wav")
        if not os.path.isdir(wav_dir):
            continue
        wav_files = [f for f in os.listdir(wav_dir) if f.lower().endswith(".wav")]
        if not wav_files:
            continue
        print(f"\nLocation: {location}")
        sample_files = random.sample(wav_files, min(N_FILES, len(wav_files)))
        rates = []
        for fname in sample_files:
            wav_path = os.path.join(wav_dir, fname)
            rate = get_sample_rate(wav_path)
            print(f"  {fname}: {rate}")
            rates.append(rate)
        unique_rates = set(rates)
        if len(unique_rates) == 1:
            print(f"  All files have the same sample rate: {unique_rates.pop()}")
        else:
            print(f"  WARNING: Multiple sample rates found: {unique_rates}")

if __name__ == "__main__":
    main()