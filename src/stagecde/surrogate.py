"""Conditional-solver emulator. Neither CFD truth nor experimental validation."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from assembly import ROOT

OUT=ROOT/'results/research/E/surrogate'

def load_split(role):
    files=sorted((ROOT/'results/research/E/transitions').glob(role+'-*.npz'))
    if not files:raise FileNotFoundError('Generate teacher transitions for '+role)
    blocks=[np.load(p) for p in files]
    return np.concatenate([b['X'] for b in blocks]),np.concatenate([b['Y'] for b in blocks]),files

class Emulator:
    def __init__(self,directory=OUT):
        self.directory=Path(directory);self.meta=json.loads((self.directory/'model.json').read_text())
        for name,expected in self.meta.get('weights_sha256',{}).items():
            if hashlib.sha256((self.directory/name).read_bytes()).hexdigest()!=expected:raise ValueError('Frozen emulator weights changed: '+name)
        self.members=[dict(np.load(self.directory/name)) for name in self.meta['weights']]
    def predict(self,x,member=None):
        x=np.atleast_2d(x);ys=[]
        for b in self.members if member is None else [self.members[member]]:
            z=(x-b['xm'])/b['xs']
            for i in range(3):
                z=z@b[f'w{i}'].T+b[f'b{i}']
                if i<2:z=np.tanh(z)
            y=z.copy();y[:,:19]=y[:,:19]*b['ys']+b['ym'];y[:,19]=1/(1+np.exp(-np.clip(y[:,19],-40,40)))
            ys.append(y)
        return np.mean(ys,axis=0)
    def out_of_distribution(self,x):
        x=np.asarray(x);b=self.members[0]
        # Operational flag, not a calibrated probabilistic distance.
        return bool(np.max(np.abs((x-b['xm'])/b['xs']))>8 or not np.isfinite(x).all())

def physical(y,air):
    y=np.atleast_2d(y)
    return {'T':y[:,:8]+np.asarray(air)[:,None], 'water':.01*np.expm1(np.clip(y[:,8:16],0,12)),
            'pv':np.maximum(0,y[:,16]),'support':y[:,19]}

def metrics(y,p,x):
    # Air is raw input 18, following 18 state features.
    a=physical(y,x[:,18]);b=physical(p,x[:,18]);dt=b['T']-a['T'];dw=b['water']-a['water'];de=b['pv']-a['pv']
    return {'samples':len(y),'temperature_MAE_K_by_node':np.mean(abs(dt),0).tolist(),
      'temperature_RMSE_K_by_node':np.sqrt(np.mean(dt**2,0)).tolist(),'surface_max_abs_p95_K':float(np.quantile(np.max(abs(dt[:,[0,6]]),1),.95)),
      'water_MAE_L_by_rain_cond_evap_drain_loss_film':np.mean(abs(dw),0).tolist(),
      'drain_RMSE_L':float(np.sqrt(np.mean(dw[:,3]**2))),'drain_mean_reference_L':float(a['water'][:,3].mean()),
      'pv_MAE_Wh':float(np.mean(abs(de))),'pv_RMSE_Wh':float(np.sqrt(np.mean(de**2))),
      'unsupported_reference':int((y[:,19]<.5).sum()),'false_accept_at_0_9':int(((y[:,19]<.5)&(p[:,19]>=.9)).sum()),
      'supported_reference':int((y[:,19]>=.5).sum()),'true_accept_at_0_9':int(((y[:,19]>=.5)&(p[:,19]>=.9)).sum())}

def train(epochs=600,members=3):
    if (OUT/'model.json').exists():raise FileExistsError('Frozen emulator already exists; archive a separate research version before training')
    import torch
    torch.set_num_threads(1)
    from torch import nn
    x,y,files=load_split('train');vx,vy,vfiles=load_split('selection')
    xm=x.mean(0);xs=x.std(0);xs=np.where(xs<1e-6,1.,xs)
    ym=y[:,:19].mean(0);ys=y[:,:19].std(0);ys=np.where(ys<1e-5,1.,ys)
    tx=torch.tensor((x-xm)/xs,dtype=torch.float32);ty=torch.tensor((y[:,:19]-ym)/ys,dtype=torch.float32);support=torch.tensor(y[:,19:20],dtype=torch.float32)
    tvx=torch.tensor((vx-xm)/xs,dtype=torch.float32);tvy=torch.tensor((vy[:,:19]-ym)/ys,dtype=torch.float32);vs=torch.tensor(vy[:,19:20],dtype=torch.float32)
    OUT.mkdir(parents=True,exist_ok=True);hist=[];names=[];start=time.time()
    for j in range(members):
        seed=7+10*j;torch.manual_seed(seed);rng=np.random.default_rng(seed)
        net=nn.Sequential(nn.Linear(x.shape[1],128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,20))
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-4);best=float('inf');wait=0;beststate=None
        for ep in range(epochs):
            net.train()
            for ids in np.array_split(rng.permutation(len(x)),max(1,int(np.ceil(len(x)/512)))):
                z=net(tx[ids]);loss=((z[:,:19]-ty[ids])**2).mean()+.2*nn.functional.binary_cross_entropy_with_logits(z[:,19:],support[ids])
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
            net.eval()
            with torch.no_grad():
                z=net(tvx);val=float(((z[:,:19]-tvy)**2).mean()+.2*nn.functional.binary_cross_entropy_with_logits(z[:,19:],vs))
            if val<best-1e-5:best=val;wait=0;beststate={k:v.detach().clone() for k,v in net.state_dict().items()};bestep=ep+1
            else:wait+=1
            if wait>=70:break
        net.load_state_dict(beststate);b={'xm':xm,'xs':xs,'ym':ym,'ys':ys,'xlo':x.min(0),'xhi':x.max(0)}
        for k,layer in enumerate([net[0],net[2],net[4]]):b[f'w{k}']=layer.weight.detach().numpy();b[f'b{k}']=layer.bias.detach().numpy()
        name=f'member-{seed}.npz';np.savez_compressed(OUT/name,**b);names.append(name)
        hist.append({'seed':seed,'epochs_completed':ep+1,'selected_epoch':bestep,'selection_loss':best})
        print(json.dumps(hist[-1]),flush=True)
    meta={'version':'conditional-emulator-1','weights':names,'training':hist,'runtime_s':time.time()-start,
      'train_groups':[p.stem for p in files],'selection_groups':[p.stem for p in vfiles],
      'test_used_for_model_selection':False,'teacher':'cassette-v2.0 + B1 wall/ground','CFD_surrogate':False,
      'independent_experimental_validation':False,'physical_product_qualification':False,'training_seed_policy':[7+10*j for j in range(members)],
      'uncertainty':'ensemble spread is model disagreement, not full physical uncertainty'}
    (OUT/'model.json').write_text(json.dumps(meta,indent=2))
    model=Emulator();vp=model.predict(vx)
    residual=np.abs(physical(vy,vx[:,18])['water'][:,3]-physical(vp,vx[:,18])['water'][:,3])
    meta['selection_metrics']=metrics(vy,vp,vx);meta['selection_drain_abs_residual_p95_L']=float(np.quantile(residual,.95))
    meta['weights_sha256']={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in names}
    (OUT/'model.json').write_text(json.dumps(meta,indent=2))

def test():
    lock=OUT/'test-metrics.json'
    if lock.exists():raise FileExistsError('Final test already recorded; preserve it and use a new protocol/version for retuning')
    model=Emulator();x,y,files=load_split('test');p=model.predict(x)
    result={'split':'2023 + four unseen cities','model_sha256':hashlib.sha256((OUT/'model.json').read_bytes()).hexdigest(),
      'overall':metrics(y,p,x),'by_group':{},'independent_experiment':False,'qualified_for_product_claims':False}
    for f in files:
        b=np.load(f);result['by_group'][f.stem]=metrics(b['Y'],model.predict(b['X']),b['X'])
    lock.write_text(json.dumps(result,indent=2));print(json.dumps(result['overall'],indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('task',choices=['train','test']);p.add_argument('--epochs',type=int,default=600);p.add_argument('--members',type=int,default=3);a=p.parse_args()
    train(a.epochs,a.members) if a.task=='train' else test()
if __name__=='__main__':main()
