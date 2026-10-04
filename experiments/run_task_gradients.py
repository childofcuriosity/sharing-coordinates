"""Actual LM-loss gradients, natural structured tomography and finite arithmetic."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F

from experiments.train_enhancement import source_hashes
from experiments.run_autograd_tomography import autodiff_response,relative_error,predict,analytic_components
from src.enhancement import effective_forward
from src.language import load_language_checkpoint,load_byte_stream,sample_batch
from src.response_identifiability import pack_basis_matrix,basis_linear_modules
from src.response_tomography import build_tomography_design
from src.response_factor_recovery import recover_factors


def natural_reconstruct(gradients,responses,u):
    x=gradients@u.T
    gperp=gradients-x@u
    rperp=responses-(responses@u.T)@u
    normal=torch.einsum('qld,qmd->lm',gperp,gperp)
    cross=torch.einsum('qld,qmd->lm',rperp,gperp)
    gram=cross@torch.linalg.pinv(normal,rtol=1e-12,hermitian=True)
    y=(responses-gram@gradients)@u.T
    local=torch.stack([y[:,i,:].T@torch.linalg.pinv(x[:,i,:].T,rtol=1e-12)
                       for i in range(x.shape[1])])
    sv=torch.linalg.svdvals(x.transpose(0,1))
    eigen=torch.linalg.eigvalsh(normal).clamp_min(0)
    return gram,local,dict(complement_normal_eigenvalues=eigen.tolist(),
        complement_numerical_rank=int((eigen>1e-12*eigen[-1]).sum()),
        local_singular_values=sv.tolist(),
        local_min_rank=int((sv>1e-12*sv[:,:1]).sum(-1).min()))


def orbit_error(ahat,bhat,a,b):
    candidates=[]
    for perm in itertools.permutations(range(a.shape[1])):
        ids=list(perm)
        ae=relative_error(ahat,a[:,ids]);be=relative_error(bhat,b[ids])
        candidates.append(((ae*ae+be*be)**.5,ae,be,ids))
    best=min(candidates)
    return dict(orbit_relative_error=best[0],router_relative_error=best[1],basis_relative_error=best[2],permutation=best[3])


def run(checkpoint,device):
    start=time.monotonic()
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False
    model,config,meta=load_language_checkpoint(str(checkpoint),device=device)
    if config.tau_end!=1.:
        raise ValueError('This registered experiment uses tau=1')
    model=model.double().eval()
    packed=pack_basis_matrix(model)
    a=model.router.probabilities().detach();b=packed.bases.detach()
    theta=a@b
    design=build_tomography_design(theta,a.shape[1])
    stream_path=Path('data/wikitext-2/train.txt')
    if hashlib.sha256(stream_path.read_bytes()).hexdigest()!=meta['corpus_sha256']['train']:
        raise ValueError('Training corpus mismatch')
    stream=load_byte_stream(str(stream_path),torch.device(device))
    rng=torch.Generator().manual_seed(750000+config.seed)
    gradients=[];losses=[];controls={}
    for index in range(16):
        tokens,targets=sample_batch(stream,2,32,rng)
        variable=theta.detach().requires_grad_()
        logits=effective_forward(model,variable,packed.layout,tokens)
        loss=F.cross_entropy(logits.flatten(0,1),targets.flatten())
        g,=torch.autograd.grad(loss,variable)
        if index==0:
            actual=model(tokens)
            params=[model.router.logits]+[t for _,m in basis_linear_modules(model) for t in (m.weight,m.bias) if t is not None]
            true_grad=torch.autograd.grad(F.cross_entropy(actual.flatten(0,1),targets.flatten()),params)
            z=model.router.logits.detach().requires_grad_();factors=b.detach().requires_grad_()
            dz,db=torch.autograd.grad((z.softmax(-1)@factors*g).sum(),(z,factors))
            controls=dict(forward_max_abs=float((actual-logits).abs().max()),
                router_gradient_relative_error=relative_error(dz,true_grad[0]),
                basis_gradient_relative_error=relative_error(db,torch.cat([v.reshape(a.shape[1],-1) for v in true_grad[1:]],1)))
        gradients.append(g.detach());losses.append(float(loss.detach()))
    gradients=torch.stack(gradients)
    gradient_norms=gradients.flatten(1).norm(dim=1)
    gradients=gradients/gradient_norms[:,None,None]
    responses=torch.stack([autodiff_response(a,b,g) for g in gradients])
    chart_records={}
    eye=torch.eye(a.shape[1],dtype=a.dtype,device=device)
    for name,m in [('permutation',torch.roll(eye,1,1)),('nonpermutation',.65*eye+.35*torch.ones_like(eye)/len(eye))]:
        ca=a@m;cb=torch.linalg.solve(m,b)
        gaps=[]
        for g,r in zip(gradients,responses):
            observed=autodiff_response(ca,cb,g)
            gaps.append(float((observed-r).norm())/max(float(observed.norm()),float(r.norm()),1e-300))
        chart_records[name]=dict(product_relative_error=relative_error(ca@cb,theta),response_relative_gaps=gaps)
    flat=gradients.flatten(1)
    truth_q,truth_t=analytic_components(a,b,design,1.)
    reconstructions=[]
    for q in (1,4,8,12):
        normal=flat[:q]@flat[:q].T
        coefficients=(flat[12:]@flat[:q].T)@torch.linalg.pinv(normal,rtol=1e-12,hermitian=True)
        span_g=coefficients@flat[:q]
        span_r=coefficients@responses[:q].flatten(1)
        gram,local,conditioning=natural_reconstruct(gradients[:q],responses[:q],design.row_basis)
        row=dict(q=q,conditioning=conditioning,
            generic_span_gradient_error=relative_error(span_g,flat[12:]),
            generic_span_response_error=relative_error(span_r,responses[12:].flatten(1)),
            gram_relative_error=relative_error(gram,truth_q),local_relative_error=relative_error(local,truth_t),
            structured_heldout_errors=[relative_error(predict(g,gram,local,design),r) for g,r in zip(gradients[12:],responses[12:])])
        try:
            ah,bh,diagnostics=recover_factors(theta,design.row_basis,gram,local)
            row['recovery']=dict(status='ok',**orbit_error(ah,bh,a,b),diagnostics=diagnostics)
        except (ValueError,RuntimeError) as error:
            row['recovery']=dict(status='failed',error=str(error))
        reconstructions.append(row)
    finite=[]
    raw_g=gradients[0]*gradient_norms[0]
    exact=autodiff_response(a,b,raw_g)
    for dtype in (torch.float64,torch.float32,torch.bfloat16):
        z=a.log().to(dtype).requires_grad_();bb=b.to(dtype).requires_grad_();gg=raw_g.to(dtype)
        base=z.softmax(-1)@bb
        dz,db=torch.autograd.grad((base*gg).sum(),(z,bb))
        for h in (1e-1,1e-2,1e-3,1e-4,1e-5):
            with torch.no_grad():
                after=(z-h*dz).softmax(-1)@(bb-h*db)
                observed=(base.double()-after.double())/h
            finite.append(dict(dtype=str(dtype),step=h,response_relative_error=relative_error(observed,exact),
                               measured_response_norm=float(observed.norm()),exact_response_norm=float(exact.norm())))
    gates=dict(forward=controls['forward_max_abs']<1e-9,
               chain_rule=max(controls['router_gradient_relative_error'],controls['basis_gradient_relative_error'])<1e-8,
               permutation=max(chart_records['permutation']['response_relative_gaps'])<1e-8)
    sources=source_hashes(['experiments/run_task_gradients.py','experiments/train_enhancement.py',
        'experiments/run_autograd_tomography.py','src/enhancement.py','src/response_identifiability.py',
        'src/response_tomography.py','src/response_factor_recovery.py','src/language.py','src/models.py',
        'src/patterns.py','research/ENHANCEMENT_PROTOCOL.md','research/TASK_GRADIENT_RECONSTRUCTION.md'])
    return dict(format='task-gradients-v1',seed=config.seed,source_sha256=sources,
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        training_corpus_sha256=meta['corpus_sha256']['train'],batch_seed=750000+config.seed,
        batch_size=2,sequence_length=32,batch_count=16,controls=controls,gates=gates,
        task_nll=losses,gradient_norms=gradient_norms.tolist(),
        gradient_span_singular_values=torch.linalg.eigvalsh(flat@flat.T).clamp_min(0).sqrt().flip(0).tolist(),
        charts=chart_records,reconstructions=reconstructions,finite_steps=finite,
        device=device,torch_version=torch.__version__,seconds=time.monotonic()-start)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--device',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite a result')
    result=run(args.checkpoint,args.device)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'gates':result['gates'],'seconds':result['seconds']}))
    if not all(result['gates'].values()):
        raise SystemExit(1)
