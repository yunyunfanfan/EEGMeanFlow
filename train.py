"""Train conditional EEGMeanFlow with spectral/lag relationship loss."""
import argparse
import copy
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from eegmeanflow.data import load_split
from eegmeanflow.models.eeg_band_spatial_meanflow import EEGBandSpatialMeanFlow
from eegmeanflow.meanflow import meanflow_loss, sample_intervals, drop_labels_once
from eegmeanflow.relationship import EEGCrossLag
from eegmeanflow.validation import Validator

MODEL = dict(in_channels=32,time_points=2000,num_classes=9,embed_dim=512,
             patch_size=5,depth=4,num_heads=16,mlp_ratio=4.,class_dropout_prob=.1,
             learn_sigma=False,branch_width=64,fs=200.)

def save(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state,temporary)
    temporary.replace(path)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('outputs/eegmeanflow'))
    p.add_argument('--epochs',type=int,default=2000)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--microbatch',type=int,default=8)
    p.add_argument('--lr',type=float,default=5e-4)
    p.add_argument('--relationship-weight',type=float,default=.1)
    p.add_argument('--seed',type=int,default=8888)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--val-every',type=int,default=1)
    p.add_argument('--val-per-class',type=int,default=128)
    p.add_argument('--resume',action='store_true')
    args = p.parse_args()
    if min(args.epochs,args.batch_size,args.microbatch,args.val_every,args.val_per_class)<1 or args.relationship_weight<0:
        p.error('Invalid training settings')
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        p.error('Output is nonempty; use --resume or a new directory')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);threadpool_limits(2)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device = torch.device(args.device)
    config = {k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='resume'}
    if args.resume:
        previous = json.loads((args.output/'config.json').read_text())
        if any(previous[k]!=v for k,v in config.items() if k not in ['device','microbatch']):
            p.error('Resume must retain the original training settings and data path')
    else:
        (args.output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('Loading training and validation data…',flush=True)
    x,y = load_split(args.data,'train')
    validation = load_split(args.data,'val')
    print('Fitting the real-training feature bank and validation transform…',flush=True)
    bank = EEGCrossLag.from_training(x,y).to(device)
    validator = Validator(x,validation,args.val_per_class)
    model = EEGBandSpatialMeanFlow(**MODEL).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.05)
    steps = math.ceil(len(x)/args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs*steps,eta_min=1e-6)
    start,step,best = 1,0,float('inf')
    if args.resume:
        state = torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        start,step,best = state['epoch']+1,state['step'],state['best']
    def select(epoch):
        nonlocal best
        scores = {}
        for name, candidate in [('model',model),('ema',ema)]:
            score = validator.score(candidate,device)
            if not math.isfinite(score):raise RuntimeError('Nonfinite validation score')
            scores[name] = score
            if score<best:
                best = score
                save(args.output/'best.pt',dict(model=candidate.state_dict(),model_config=MODEL,
                    epoch=epoch,validation_mmd=score,selected_weights=name))
        return scores
    if not args.resume:select(0)
    for epoch in range(start,args.epochs+1):
        started = time.perf_counter();total=0.
        order = torch.randperm(len(x),generator=torch.Generator().manual_seed(args.seed+31000000+104729*epoch))
        strength = args.relationship_weight*min(1.,max(0.,(epoch-50)/100))
        model.train()
        for indices in order.split(args.batch_size):
            xx,yy = x[indices].to(device),y[indices].to(device)
            torch.manual_seed(args.seed+100003*(step+1))
            optimizer.zero_grad(set_to_none=True)
            noise = torch.randn_like(xx);t,r = sample_intervals(len(xx),device)
            fixed = drop_labels_once(model,yy)
            probability = model.y_embedder.dropout_prob;model.y_embedder.dropout_prob=0.
            batch_loss = 0.
            try:
                for offset in range(0,len(xx),args.microbatch):
                    sl = slice(offset,min(offset+args.microbatch,len(xx)))
                    loss = meanflow_loss(model,xx[sl],fixed[sl],noise=noise[sl],t=t[sl],r=r[sl])['loss']
                    fraction = len(xx[sl])/len(xx)
                    (loss*fraction).backward();batch_loss+=float(loss.detach())*fraction
            finally:
                model.y_embedder.dropout_prob=probability
            if strength and (step+1)%4==0:
                rng = torch.Generator(device=device).manual_seed(args.seed+91000000+733*(step+1))
                labels = torch.arange(9,device=device).repeat_interleave(8)
                z = torch.randn(72,32,2000,device=device,generator=rng)
                model.eval()
                fake = z-model(z,torch.ones(72,device=device),labels,r=torch.zeros(72,device=device))
                refs = torch.Generator().manual_seed(args.seed+93000000+379*(step+1))
                loss = bank.loss(fake,labels,refs)['loss']
                (4*strength*loss).backward();batch_loss+=float(loss.detach())*4*strength
                model.train()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();scheduler.step();step+=1;total+=batch_loss
            with torch.no_grad():
                for target,source in zip(ema.parameters(),model.parameters()):
                    target.lerp_(source,1-.9999)
        scores = select(epoch) if epoch%args.val_every==0 or epoch==args.epochs else {}
        save(args.output/'last.pt',dict(model=model.state_dict(),ema=ema.state_dict(),
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),model_config=MODEL,
            epoch=epoch,step=step,best=best))
        row = dict(epoch=epoch,loss=total/steps,relationship_weight=strength,
                   validation=scores,best_mmd=best,seconds=time.perf_counter()-started)
        with (args.output/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)

if __name__ == '__main__':
    main()
