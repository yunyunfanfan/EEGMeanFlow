"""Read EEG LMDB records; convert microvolts to model units once."""
import pickle
import lmdb
import numpy as np
import torch

CLASSES = ['Anger', 'Disgust', 'Fear', 'Sadness', 'Neutral',
           'Amusement', 'Inspiration', 'Joy', 'Tenderness']

def load_split(path, split):
    db = lmdb.open(str(path), readonly=True, lock=False)
    try:
        with db.begin() as tx:
            keys = pickle.loads(tx.get(b'__keys__'))[split]
            if not keys:
                raise ValueError(f'Empty {split} split')
            x = np.empty((len(keys), 32, 2000), dtype=np.float32)
            y = np.empty(len(keys), dtype=np.int64)
            for i, key in enumerate(keys):
                item = pickle.loads(tx.get(key.encode()))
                x[i] = np.asarray(item['sample']).reshape(32, 2000) / 100
                y[i] = int(item['label'])
        if not np.isfinite(x).all() or not np.isin(y, np.arange(9)).all():
            raise ValueError('Invalid EEG samples or labels')
        return torch.from_numpy(x), torch.from_numpy(y)
    finally:
        db.close()
