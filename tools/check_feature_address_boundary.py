"""Opt-in FP16 sparse feature addressing regression (large GPU memory required).
Run with the intended rebuilt cumm/spconv installed; prints JSON and fails on errors.
"""
import os,sys,json,argparse,time,math

a=argparse.ArgumentParser();a.add_argument('--pairs',action='store_true');a.add_argument('--ci',type=int,default=512);a.add_argument('--co',type=int,default=16);a.add_argument('--delta',type=int,default=32);a.add_argument('--seed',type=int,default=11);a.add_argument('--n',type=int);args=a.parse_args()
os.environ.update(CUMM_DISABLE_JIT='1',SPCONV_DISABLE_JIT='1')

import torch,spconv.pytorch as sp
from spconv.core import ConvAlgo
import spconv.core_cc as ext
ci,co=args.ci,args.co;N=args.n or ((2**31)//(max(ci,co)*2)+args.delta);period=250 if args.pairs else 251
torch.manual_seed(args.seed);torch.backends.cuda.matmul.allow_tf32=False
torch.cuda.reset_peak_memory_stats();start=time.time()
# Disconnected voxels or disconnected adjacent pairs; all 27 weights are nonzero.
# Reference uses only the centre tap and, in pairs mode, the two adjacent taps.
side=math.ceil(N**(1/3))+1
ids=torch.arange(N,device='cuda',dtype=torch.int64)
pos=ids//2 if args.pairs else ids
coords=torch.stack((torch.zeros_like(ids),4*(pos//(side*side)),4*((pos//side)%side),4*(pos%side)+(ids%2 if args.pairs else 0)),1).int()
del pos
xt=(torch.randn(period,ci,device='cuda')*.05).half();gt=(torch.randn(period,co,device='cuda')*.001).half()
x=xt[ids%period].detach().requires_grad_();g=gt[ids%period]
# Probe gradients across the entire address range while bounding FP16 reduction depth.
probe_rows={0,1,N-2,N-1}
for channels in (ci, co):
 for boundary in (2**31,2**32):
  row=boundary//(channels*2)
  probe_rows.update(r for r in range(row-2,row+2) if 0<=r<N)
# Activate complete adjacent pairs so the independent pair reference remains valid.
probe_rows=sorted({r for v in probe_rows for r in (v//2*2,v//2*2+1) if r<N})
probe=torch.tensor(probe_rows,device='cuda',dtype=torch.long)
active=ids%1024<2
active[probe]=True
g*=active[:,None]
counts=torch.bincount(ids[active]%period,minlength=period).float()
del ids,active
m=sp.SubMConv3d(ci,co,3,bias=False,algo=ConvAlgo.MaskImplicitGemm).cuda().half()
with torch.no_grad():m.weight.normal_(0,.02)
refy=xt.float()@m.weight[:,1,1,1,:].float().T
refdx=gt.float()@m.weight[:,1,1,1,:].float()
# counts counts active upstream-gradient rows, not all input rows.
refdw=gt.float().T@(xt.float()*counts[:,None])
refdw_pos=refdw_neg=None
if args.pairs:
 assert N%2==0
 refy[::2]+=xt[1::2].float()@m.weight[:,1,1,2,:].float().T
 refy[1::2]+=xt[::2].float()@m.weight[:,1,1,0,:].float().T
 refdx[::2]+=gt[1::2].float()@m.weight[:,1,1,0,:].float()
 refdx[1::2]+=gt[::2].float()@m.weight[:,1,1,2,:].float()
 refdw_pos=gt[::2].float().T@(xt[1::2].float()*counts[::2,None])
 refdw_neg=gt[1::2].float().T@(xt[::2].float()*counts[1::2,None])
report={'N':N,'ci':ci,'co':co,'input_bytes':N*ci*2,'output_bytes':N*co*2,'pairs':args.pairs,'extension':ext.__file__,'seed':args.seed,'gradient_pattern':'row % 1024 < 2 plus boundary/tail probe pairs','metrics':{}}
try:
 y=m(sp.SparseConvTensor(x,coords,[side*4]*3,1)).features
 y.backward(g);torch.cuda.synchronize()
 for label,v,ref in [('y',y,refy),('dx',x.grad,refdx)]:
  e2=r2=0.;mx=0.;finite=True
  for lo in range(0,N,65536):
   z=v[lo:lo+65536].float();rr=ref[torch.arange(lo,min(lo+65536,N),device='cuda')%period];
   if label=='dx':
    chunk_ids=torch.arange(lo,min(lo+65536,N),device='cuda')
    rr=rr*((chunk_ids%1024<2)|torch.isin(chunk_ids,probe))[:,None]
   err=z-rr
   e2+=float(err.square().sum());r2+=float(rr.square().sum());mx=max(mx,float(err.abs().max()));finite &= bool(torch.isfinite(z).all())
  report['metrics'][label]={'relative_l2':math.sqrt(e2/max(r2,1e-30)),'max_abs':mx,'finite':finite}
 report['boundary_probes']={}
 report['max_probed_byte_offset']={str(c):max(probe_rows)*c*2 for c in (ci,co)}
 for c in (ci,co):
  for b in (2**31,2**32):
   if (N-1)*c*2>=b:assert any(r*c*2>=b for r in probe_rows)
 for label,v,ref in [('y',y,refy),('dx',x.grad,refdx)]:
  actual=v[probe].float();expected=ref[probe%period]
  row_rel=(actual-expected).norm(dim=1)/expected.norm(dim=1).clamp_min(1e-12)
  report['boundary_probes'][label]={'rows':probe_rows,'relative_l2':row_rel.tolist()}
  assert bool(torch.isfinite(actual).all()) and bool((row_rel<.02).all()),(label,report['boundary_probes'][label])
 w=m.weight.grad.float();err=w[:,1,1,1,:]-refdw
 report['metrics']['dw']={'relative_l2':float(err.norm()/refdw.norm()),'max_abs':float(err.abs().max()),'finite':bool(torch.isfinite(w).all())}
 if args.pairs:
  for label,col,rr in [('dw_pos',2,refdw_pos),('dw_neg',0,refdw_neg)]:
   ee=w[:,1,1,col,:]-rr
   report['metrics'][label]={'relative_l2':float(ee.norm()/rr.norm()),'max_abs':float(ee.abs().max()),'finite':bool(torch.isfinite(ee).all())}
   w[:,1,1,col,:]=0
 w[:,1,1,1,:]=0;report['offcentre_dw_max']=float(w.abs().max())
 report['pass']=all(z['finite'] and z['relative_l2']<.02 for z in report['metrics'].values()) and report['offcentre_dw_max']==0
except Exception as e:
 report['error']=str(e)[:1800];report['pass']=False
report['seconds']=time.time()-start;report['peak_GiB']=torch.cuda.max_memory_allocated()/2**30
print(json.dumps(report),flush=True)
if not report['pass']:sys.exit(1)
