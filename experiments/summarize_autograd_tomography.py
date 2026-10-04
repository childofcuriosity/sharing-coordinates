"""Validate the additive full-dimensional autodiff records and derive a table."""

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(directory):
    rows, raw_hashes = [], {}
    for seed in range(3):
        path = directory / f'seed{seed}.json'
        record = json.loads(path.read_text())
        if record['format'] != 'autograd-tomography-v1' or record['seed'] != seed:
            raise ValueError(f'Wrong record identity: {path}')
        if record['dimensions'] != {'L': 12, 'K': 4, 'D': 787712}:
            raise ValueError(f'Unexpected checkpoint dimensions: {path}')
        if (record['dtype'] != 'torch.float64' or record['eta_z'] != 1
                or record['eta_b'] != 1 or record['tau'] != 1
                or record['measurement'] != 'reverse_autograd_gradient_then_forward_mode_jvp_of_full_packed_map'):
            raise ValueError(f'Unexpected measurement protocol: {path}')
        checkpoint = ROOT / f'results/checkpoints/language_seed{seed}.pt'
        if record['checkpoint_sha256'] != digest(checkpoint):
            raise ValueError(f'Checkpoint binding mismatch: {path}')
        required = {'experiments/run_autograd_tomography.py', 'src/language.py',
                    'src/models.py', 'src/response_identifiability.py',
                    'src/response_tomography.py', 'research/AUTOGRAD_TOMOGRAPHY_PROTOCOL.md'}
        if set(record['source_sha256']) != required:
            raise ValueError(f'Incomplete source inventory: {path}')
        for name, expected in record['source_sha256'].items():
            if digest(ROOT / name) != expected:
                raise ValueError(f'Generation source mismatch: {name}')
        charts = record['charts']
        if set(charts) != {'native', 'permutation', 'nonpermutation'}:
            raise ValueError(f'Incomplete chart controls: {path}')
        for chart in charts.values():
            if [p['index'] for p in chart['probes']] != list(range(4)):
                raise ValueError(f'Incomplete designed probes: {path}')
            if [p['seed'] for p in chart['heldout']] != [910000+seed, 920000+seed]:
                raise ValueError(f'Incorrect held-out probes: {path}')
        row = {
            'seed': seed,
            'formula_max_relative_error': max(p['formula_relative_error'] for c in charts.values() for p in c['probes']),
            'heldout_max_relative_error': max(p['relative_error'] for c in charts.values() for p in c['heldout']),
            'product_max_relative_error': max(c['product_relative_error'] for c in charts.values()),
            'component_max_relative_error': max(c[key] for c in charts.values() for key in ('gram_relative_error', 'local_relative_error')),
            'permutation_max_relative_gap': max(p['relative_gap_from_native'] for p in charts['permutation']['probes']),
            'nonpermutation_max_relative_gap': max(p['relative_gap_from_native'] for p in charts['nonpermutation']['probes']),
        }
        if not all(math.isfinite(v) and v >= 0 for v in row.values()):
            raise ValueError(f'Nonfinite or negative diagnostics: {path}')
        if (row['formula_max_relative_error'] > 1e-9 or row['heldout_max_relative_error'] > 1e-8
                or row['product_max_relative_error'] > 1e-10 or row['permutation_max_relative_gap'] > 1e-9
                or row['nonpermutation_max_relative_gap'] <= 1e-3):
            raise ValueError(f'Numerical measurement gates failed: {path}')
        noise = record['noise']
        if [r['relative_noise'] for r in noise] != [0, 1e-10, 1e-8, 1e-6, 1e-4]:
            raise ValueError(f'Noise grid differs from protocol: {path}')
        for n in noise:
            eps = n['absolute_noise_norms']
            if len(eps) != 4 or not all(math.isfinite(v) and v >= 0 for v in eps):
                raise ValueError(f'Invalid observation-noise norms: {path}')
            if n['relative_noise'] == 0 and any(eps):
                raise ValueError(f'Nonzero noise at the zero-noise control: {path}')
            # Recompute the proposition from the measured noise norms instead
            # of accepting a potentially corrupted stored bound or pass flag.
            expected_bounds = {
                'gram': eps[0] / record['eta_b'],
                'local': math.sqrt((1 + math.sqrt(13))**2 * eps[0]**2
                                  + sum((e + math.sqrt(12)*eps[0])**2 for e in eps[1:])),
            }
            for key in ('gram', 'local'):
                if not math.isclose(n[key+'_bound'], expected_bounds[key], rel_tol=1e-12, abs_tol=0):
                    raise ValueError(f'Noise bound differs from proposition: {path}')
                numbers = [n[key+'_absolute_error'], n[key+'_bound'], n[key+'_roundoff_slack']]
                if not all(math.isfinite(v) and v >= 0 for v in numbers) or numbers[0] > numbers[1]+numbers[2]:
                    raise ValueError(f'Noise propagation gate failed: {path}')
        rows.append(row)
        raw_hashes[str(path.relative_to(ROOT))] = digest(path)
    return {'format': 'autograd-tomography-summary-v1', 'rows': rows,
            'raw_sha256': raw_hashes, 'scope': 'full packed shared-parameter map; not an end-to-end loss or AdamW trajectory'}


def latex(summary):
    lines = [r'% Generated by experiments/summarize_autograd_tomography.py.',
             r'\begin{table}[H]', r'\centering\small',
             r'\caption{Full-dimensional autodiff response audit. Errors are relative Frobenius norms. Formula and held-out columns take the maximum over native, permutation, and non-permutation charts; each chart uses four designed and two unseen probes.}',
             r'\label{tab:autograd-tomography}',
             r'\begin{tabular}{@{}crrrr@{}}', r'\toprule',
             r'Seed & Formula error & Held-out error & Permutation gap & Non-perm. gap \\', r'\midrule']
    for r in summary['rows']:
        lines.append(f"{r['seed']} & {r['formula_max_relative_error']:.2e} & {r['heldout_max_relative_error']:.2e} & {r['permutation_max_relative_gap']:.2e} & {r['nonpermutation_max_relative_gap']:.4f} " + r'\\')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    summary = summarize(ROOT / 'results/autograd_tomography')
    outputs = {
        ROOT/'results/autograd_tomography/summary.json': json.dumps(summary, indent=2, allow_nan=False)+'\n',
        ROOT/'paper/generated/autograd_tomography_table.tex': latex(summary),
    }
    for path, value in outputs.items():
        if args.check:
            if path.read_bytes() != value.encode('utf-8'):
                raise SystemExit(f'Derived autodiff artifact differs: {path}')
        else:
            with path.open('w', encoding='utf-8', newline='\n') as handle:
                handle.write(value)
    print('verified autodiff tomography records and derived artifacts' if args.check else 'wrote autodiff tomography summary and table')


if __name__ == '__main__':
    main()
