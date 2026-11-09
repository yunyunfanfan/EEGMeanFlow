"""Generate class-conditional EEG with one network evaluation."""
import argparse
from pathlib import Path
import numpy as np
import torch
from eegmeanflow.data import CLASSES
from eegmeanflow.models.eeg_band_spatial_meanflow import EEGBandSpatialMeanFlow
from eegmeanflow.meanflow import sample_flow

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('outputs/generated.npz'))
    p.add_argument('--classes',type=int,nargs='+',default=list(range(9)))
    p.add_argument('--per-class',type=int,default=128)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--seed',type=int,default=21888)
    p.add_argument('--device',default='cuda:0')
    args = p.parse_args()
    if min(args.per_class,args.batch_size)<1 or not set(args.classes)<=set(range(9)):
        p.error('Use positive counts and class IDs from 0 to 8')
    if args.output.exists():p.error('Choose a new output filename')
    torch.set_num_threads(2)
    state = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model = EEGBandSpatialMeanFlow(**state['model_config']).to(args.device).eval()
    model.load_state_dict(state['model'])
    labels = torch.tensor(args.classes).repeat_interleave(args.per_class)
    result=[]
    with torch.no_grad():
        for start in range(0,len(labels),args.batch_size):
            yy=labels[start:start+args.batch_size].to(args.device)
            z=torch.stack([torch.randn(32,2000,generator=torch.Generator().manual_seed(args.seed+i))
                           for i in range(start,start+len(yy))]).to(args.device)
            result.append((sample_flow(model,z,yy,steps=1)*100).cpu().numpy())
    x=np.concatenate(result)
    if not np.isfinite(x).all():raise RuntimeError('Generated EEG contains nonfinite values')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('wb') as f:
        np.savez_compressed(f,x=x,y=labels.numpy(),sample_rate=200,unit='uV',class_names=np.array(CLASSES))
    print(f'Saved {len(x)} EEG segments of shape (32,2000) to {args.output}')

if __name__ == '__main__':
    main()
