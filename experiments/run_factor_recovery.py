"""Prospective checkpoint pilot for factor recovery from autodiff observations."""

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import torch

from experiments.run_autograd_tomography import autodiff_response, reconstruct
from src.language import load_language_checkpoint
from src.response_factor_recovery import recover_factors
from src.response_identifiability import pack_basis_matrix
from src.response_tomography import build_tomography_design, tomography_probe

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(seed, device):
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = ROOT/f'results/checkpoints/language_seed{seed}.pt'
    model, config, _ = load_language_checkpoint(str(checkpoint), device='cpu')
    tau = float(config.tau_end)
    a = (model.router.logits.detach().double()/tau).softmax(-1).to(device)
    b = pack_basis_matrix(model, temperature=tau).bases.detach().double().to(device)
    del model
    theta = a@b
    design = build_tomography_design(theta, a.shape[1])
    responses = [autodiff_response(a, b, tomography_probe(design, i), tau=tau)
                 for i in range(a.shape[1])]
    generator = torch.Generator(device=device).manual_seed(540000+seed)
    records = []
    reduced_truth = b@design.row_basis.T
    for scale in (0., 1e-10, 1e-8, 1e-6, 1e-4):
        noisy, norms = [], []
        for response in responses:
            noise = torch.randn(response.shape, generator=generator, device=device, dtype=response.dtype)
            noise *= scale*response.norm()/noise.norm()
            noisy.append(response+noise)
            norms.append(float(noise.norm()))
        gram, local = reconstruct(noisy, design)
        row = {'relative_noise': scale, 'absolute_noise_norms': norms}
        try:
            ahat, bhat, diagnostics = recover_factors(theta, design.row_basis, gram, local, tau=tau)
            reduced_estimate = bhat@design.row_basis.T
            errors = []
            for perm in itertools.permutations(range(a.shape[1])):
                index = list(perm)
                ae = float((ahat-a[:, index]).norm()/a.norm())
                be = float((reduced_estimate-reduced_truth[index]).norm()/reduced_truth.norm())
                errors.append(((ae*ae+be*be)**.5, ae, be, index))
            best = min(errors)
            row.update(status='ok', orbit_relative_error=best[0],
                       router_relative_error=best[1], basis_relative_error=best[2],
                       evaluation_permutation=best[3], diagnostics=diagnostics)
        except (ValueError, RuntimeError) as error:
            row.update(status='failed', error=str(error))
        records.append(row)
    source_names = ['experiments/run_factor_recovery.py', 'src/response_factor_recovery.py',
                    'experiments/run_autograd_tomography.py', 'src/response_tomography.py',
                    'src/response_identifiability.py', 'src/language.py', 'src/models.py',
                    'research/FACTOR_RECOVERY_PROTOCOL.md']
    return dict(format='factor-recovery-pilot-v1', seed=seed, tau=tau, eta_z=1., eta_b=1.,
                dimensions=dict(L=a.shape[0], K=a.shape[1], D=b.shape[1]),
                weight_seed=520000, noise_seed=540000+seed, device=device,
                torch_version=torch.__version__, checkpoint_sha256=digest(checkpoint),
                source_sha256={name: digest(ROOT/name) for name in source_names},
                records=records, seconds=time.monotonic()-started,
                noiseless_gate=records[0]['status']=='ok' and records[0]['orbit_relative_error']<=1e-5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=range(3), required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite a pilot record')
    record = run(args.seed, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'seed': args.seed, 'noiseless_gate': record['noiseless_gate'],
                      'seconds': record['seconds']}))


if __name__ == '__main__':
    main()
