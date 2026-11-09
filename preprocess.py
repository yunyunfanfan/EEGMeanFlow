"""Prepare the official FACED Processed_data pickle files for EEGMeanFlow."""
import argparse
import json
import pickle
import re
from pathlib import Path
import lmdb
import numpy as np
from scipy import signal

def process_subject(array, input_unit):
    x = np.asarray(array, dtype=np.float64)
    if x.shape != (28, 32, 7500) or not np.isfinite(x).all():
        raise ValueError(f'Expected finite (28,32,7500) data, received {x.shape}')
    if input_unit == 'V':
        x = x * 1e6
    x = x.reshape(28, 32, 3, 2500).transpose(0, 2, 1, 3)
    x = signal.resample(x, 2000, axis=-1)
    sos = signal.butter(4, [.3, 45], btype='bandpass', fs=200, output='sos')
    return signal.sosfiltfilt(sos, x, axis=-1)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input-unit', choices=['uV', 'V'], required=True)
    args = parser.parse_args()
    files = list(args.input.glob('*.pkl'))
    def number(p):
        m = re.search(r'(\d+)$', p.stem)
        if not m:
            raise ValueError(f'No subject number in {p.name}')
        return int(m[1])
    files.sort(key=number)
    if len(files) != 123 or set(map(number, files)) not in [set(range(123)), set(range(1,124))]:
        parser.error('Expected all 123 subject files numbered 0–122 or 1–123')
    if args.output.exists():
        parser.error('Choose a new output directory')
    labels = [0]*3+[1]*3+[2]*3+[3]*3+[4]*4+[5]*3+[6]*3+[7]*3+[8]*3
    splits = dict(train=files[:80], val=files[80:100], test=files[100:])
    keys = {name: [] for name in splits}
    args.output.mkdir(parents=True)
    db = lmdb.open(str(args.output), map_size=16*1024**3)
    try:
        for name, subjects in splits.items():
            for path in subjects:
                with path.open('rb') as f:
                    eeg = process_subject(pickle.load(f), args.input_unit)
                with db.begin(write=True) as tx:
                    for video, label in enumerate(labels):
                        for segment in range(3):
                            key = f'{path.name}-{video}-{segment}'
                            tx.put(key.encode(), pickle.dumps(dict(sample=eeg[video,segment].reshape(32,10,200), label=label)))
                            keys[name].append(key)
                print(f'{name}: {path.name}', flush=True)
        with db.begin(write=True) as tx:
            tx.put(b'__keys__', pickle.dumps(keys))
    finally:
        db.close()
    metadata = dict(dataset='faced', sample_rate=200, stored_unit='uV',
                    output_shape=[32,10,200], input_unit=args.input_unit,
                    counts={k:len(v) for k,v in keys.items()},
                    subjects={k:[p.name for p in v] for k,v in splits.items()},
                    preprocessing='10s segments; FFT resampling; Butterworth N=4 0.3–45Hz sosfiltfilt',
                    model_scaling='divide by 100 in the loader')
    (args.output/'manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(metadata['counts'])

if __name__ == '__main__':
    main()
