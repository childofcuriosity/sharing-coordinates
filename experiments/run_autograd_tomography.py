"""Additive, full-packed-map autodiff observations for response tomography."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import torch

from src.language import load_language_checkpoint
from src.response_identifiability import apply_euclidean_response, pack_basis_matrix
from src.response_tomography import build_tomography_design, tomography_probe

ROOT = Path(__file__).resolve().parents[1]


def relative_error(observed, expected):
    norm = float(expected.norm())
    return float((observed - expected).norm()) / max(norm, 1e-300)


def autodiff_response(probabilities, bases, gradient, *, eta_z=1.0, eta_b=1.0, tau=1.0):
    """Compute J diag(eta_z,eta_b) J^T G without an analytic Jacobian."""
    z = (tau * probabilities.detach().log()).requires_grad_(True)
    b = bases.detach().requires_grad_(True)

    def effective(logits, factors):
        return torch.softmax(logits / tau, dim=-1) @ factors

    loss = (effective(z, b) * gradient).sum()
    dz, db = torch.autograd.grad(loss, (z, b))
    _, response = torch.func.jvp(
        effective, (z.detach(), b.detach()),
        (eta_z * dz.detach(), eta_b * db.detach()),
    )
    return response.detach()


def reconstruct(responses, design, eta_b=1.0):
    """Recover Gram and row-space blocks solely from measured outputs."""
    u, v = design.row_basis, design.complement_codes
    gram = responses[0] @ v.T / eta_b
    columns = []
    for index, response in enumerate(responses):
        residual = response - eta_b * gram @ tomography_probe(design, index)
        columns.append(residual @ u.T)
    return gram, torch.stack(columns, dim=-1)


def predict(gradient, gram, local, design, eta_b=1.0):
    coordinates = gradient @ design.row_basis.T
    local_response = torch.einsum('ijk,ik->ij', local, coordinates)
    return eta_b * gram @ gradient + local_response @ design.row_basis


def analytic_components(a, b, design, tau):
    # Separate comparator only; never used by the observation oracle.
    covariance = torch.diag_embed(a) - a.unsqueeze(2) * a.unsqueeze(1)
    reduced_b = b @ design.row_basis.T
    local = reduced_b.T @ (covariance @ covariance) @ reduced_b / tau**2
    return a @ a.T, local


def noise_audit(responses, design, seed):
    generator = torch.Generator(device=responses[0].device).manual_seed(seed)
    gram, local = reconstruct(responses, design)
    records = []
    for scale in (0.0, 1e-10, 1e-8, 1e-6, 1e-4):
        perturbed, eps = [], []
        for response in responses:
            noise = torch.randn(response.shape, generator=generator,
                                device=response.device, dtype=response.dtype)
            noise *= scale * response.norm() / noise.norm()
            perturbed.append(response + noise)
            eps.append(float(noise.norm()))
        q_hat, t_hat = reconstruct(perturbed, design)
        q_error, t_error = float((q_hat-gram).norm()), float((t_hat-local).norm())
        q_bound = eps[0]
        t_bound = math.sqrt((1+math.sqrt(design.depth+1))**2 * eps[0]**2
                           + sum((e+math.sqrt(design.depth)*eps[0])**2 for e in eps[1:]))
        q_slack, t_slack = 1e-8*max(1.,float(gram.norm())), 1e-8*max(1.,float(local.norm()))
        records.append(dict(relative_noise=scale, absolute_noise_norms=eps,
                            gram_absolute_error=q_error, local_absolute_error=t_error,
                            gram_bound=q_bound, local_bound=t_bound,
                            gram_roundoff_slack=q_slack, local_roundoff_slack=t_slack,
                            bounds_pass=q_error<=q_bound+q_slack and t_error<=t_bound+t_slack))
    return records


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(checkpoint, device):
    started = time.monotonic()
    if device.startswith('cuda'):
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, config, _ = load_language_checkpoint(str(checkpoint), device='cpu')
    tau = float(config.tau_end)
    packed = pack_basis_matrix(model, temperature=tau)
    a = torch.softmax(model.router.logits.detach().double()/tau, -1).to(device)
    b = packed.bases.detach().double().to(device)
    del model, packed
    theta = a @ b
    design = build_tomography_design(theta, a.shape[1])
    k = a.shape[1]
    identity = torch.eye(k, dtype=a.dtype, device=device)
    matrices = {'native': identity, 'permutation': torch.roll(identity,1,1),
                'nonpermutation': .65*identity+.35*torch.ones_like(identity)/k}
    results, native_responses = {}, None
    for name, matrix in matrices.items():
        raw_a, chart_b = a @ matrix, torch.linalg.solve(matrix,b)
        chart_a = torch.softmax((tau*raw_a.log())/tau, -1)
        responses, probe_records = [], []
        for index in range(k):
            g = tomography_probe(design,index)
            response = autodiff_response(chart_a,chart_b,g,tau=tau)
            formula = apply_euclidean_response(chart_a,chart_b,g,temperature=tau)
            record = {'index':index,'formula_relative_error':relative_error(response,formula)}
            if native_responses is not None:
                record['relative_gap_from_native'] = float((response-native_responses[index]).norm())/max(float(response.norm()),float(native_responses[index].norm()),1e-300)
            responses.append(response)
            probe_records.append(record)
        gram, local = reconstruct(responses,design)
        expected_gram, expected_local = analytic_components(chart_a,chart_b,design,tau)
        heldout = []
        for seed in (910000+config.seed,920000+config.seed):
            rng = torch.Generator(device=device).manual_seed(seed)
            g = torch.randn(theta.shape,generator=rng,device=device,dtype=a.dtype)
            measured = autodiff_response(chart_a,chart_b,g,tau=tau)
            heldout.append({'seed':seed,'relative_error':relative_error(predict(g,gram,local,design),measured)})
        results[name] = dict(product_relative_error=relative_error(chart_a@chart_b,theta),
                             logit_reencoding_relative_error=relative_error(chart_a,raw_a),
                             probes=probe_records,heldout=heldout,
                             gram_relative_error=relative_error(gram,expected_gram),
                             local_relative_error=relative_error(local,expected_local))
        if name=='native':
            native_responses=responses
    noise = noise_audit(native_responses,design,930000+config.seed)
    gates = {
        'formula_agreement': all(p['formula_relative_error']<=1e-9 for c in results.values() for p in c['probes']),
        'heldout_prediction': all(p['relative_error']<=1e-8 for c in results.values() for p in c['heldout']),
        'product_residual': all(c['product_relative_error']<=1e-10 for c in results.values()),
        'permutation_control': max(p['relative_gap_from_native'] for p in results['permutation']['probes'])<=1e-9,
        'fixed_alternative_separated': max(p['relative_gap_from_native'] for p in results['nonpermutation']['probes'])>1e-3,
        'noise_bounds': all(r['bounds_pass'] for r in noise),
    }
    if device.startswith('cuda'):
        torch.cuda.synchronize(device)
    sources = ['experiments/run_autograd_tomography.py','src/language.py','src/models.py',
               'src/response_identifiability.py','src/response_tomography.py',
               'research/AUTOGRAD_TOMOGRAPHY_PROTOCOL.md']
    return dict(format='autograd-tomography-v1',seed=config.seed,
                dimensions={'L':a.shape[0],'K':k,'D':b.shape[1]},tau=tau,
                eta_z=1.0,eta_b=1.0,dtype=str(a.dtype),device=device,
                measurement='reverse_autograd_gradient_then_forward_mode_jvp_of_full_packed_map',
                checkpoint_sha256=sha256(checkpoint),source_sha256={p:sha256(ROOT/p) for p in sources},
                runtime={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
                         'gpu':torch.cuda.get_device_name(device) if device.startswith('cuda') else None},
                charts=results,noise=noise,gates=gates,all_pass=all(gates.values()),
                seconds=time.monotonic()-started,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else 0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite an existing experiment record')
    result=run(args.checkpoint,args.device)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write('\n')
    print(json.dumps({'output':str(args.output),'gates':result['gates'],'seconds':result['seconds']}))
    if not result['all_pass']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
