"""Close this extension after both complete campaigns and independent checks."""
import csv
import hashlib
import json
import time
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parents[1]
HISTORICAL_MANIFEST = PARENT/'research/releases/2026-09-29-project-unification/MANIFEST.sha256'

def read(path):
    return json.loads(path.read_text())

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    assert (ROOT/'research/PARENT_MANIFEST.sha256').read_bytes() == HISTORICAL_MANIFEST.read_bytes()
    before = (ROOT/'logs/parent-before.log').read_text()
    after = (ROOT/'logs/parent-after.log').read_text()
    assert before.strip() and after.strip() and before == after
    versions = {}
    for name, base in [('original', ROOT), ('direct_batch_correction', ROOT/'batch-correction')]:
        freeze = read(base/'FREEZE.json')
        subprocess.run([sys.executable, '-c', 'from common import check_freeze; check_freeze()'], cwd=base, check=True)
        execution = read(base/'EXECUTION.json')
        jobs = execution['records']
        assert len(jobs) == 18 and all(x['exit_code'] == 0 for x in jobs)
        selection = read(base/'SELECTION_LOCK.json')
        plan = read(base/'ANALYSIS_PLAN_LOCK.json')
        assert plan['source_sha256'] == sha(base/'analyze.py')
        first_eval = min(x['start_unix'] for x in jobs if x['phase'] == 'evaluate')
        assert freeze['created_unix'] < min(x['start_unix'] for x in jobs)
        assert selection['created_unix'] < first_eval and plan['created_unix'] < first_eval
        assert len(selection['selections']) == 9
        for p, digest in selection['selections'].items():
            assert sha(base/p) == digest
        verification = read(base/'logs/verification.json')
        assert verification['units'] == 9 and verification['sources_and_artifacts_match']
        assert verification['selection_before_evaluation_verified']
        assert verification['parent_publication_pdf_unchanged']
        assert not (base/'logs/verification-errors.log').read_text().strip()
        assert '5 passed' in (base/'logs/tests.log').read_text()
        with (base/'results/all_methods.csv').open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 126
        units = sorted((base/'results').glob('*/evaluation.json'))
        assert len(units) == 9
        recovery_count = 0
        peak = 0.0
        for path in units:
            evaluation = read(path)
            chosen = read(path.parent/'selection.json')
            select_docs = {x['document_sha256'] for x in chosen['data']['documents']}
            test_docs = {x['document_sha256'] for x in evaluation['data']['documents']}
            assert len(select_docs) == 16 and len(test_docs) == 32
            assert not select_docs.intersection(test_docs)
            assert len(evaluation['recovery_batch_indices']) == 64
            for action in evaluation['actions'].values():
                assert set(action['recovery']) == {'sgd', 'adamw'}
                for rec in action['recovery'].values():
                    assert rec['steps'] == 64 and rec['prediction_tokens'] == 32768
                    assert len(rec['training_nll']) == 64
                    assert len(rec['document_nll']) == 32
                    recovery_count += 1
                    peak = max(peak, rec['peak_allocated_gib'])
        summary = read(base/'results/summary.json')
        versions[name] = {
            'units': 9, 'method_optimizer_rows': len(rows), 'synthetic_tests_passed': 5,
            'unique_action_optimizer_recoveries': recovery_count,
            'recovery_steps_each': 64, 'recovery_prediction_tokens_each': 32768,
            'maximum_recovery_peak_allocated_gib': peak,
            'freeze_created_unix': freeze['created_unix'],
            'selection_lock_created_unix': selection['created_unix'],
            'analysis_lock_created_unix': plan['created_unix'],
            'first_evaluation_job_unix': first_eval,
            'verification': {k: v for k, v in verification.items() if k != 'checks'},
            'primary_task_minus_weight': summary['comparisons']['weight'],
            'costs': summary['costs'],
            'makespan_seconds': summary['wall_makespan_seconds'],
            'summed_job_elapsed_seconds': summary['summed_job_elapsed_seconds'],
            'actual_unique_recovery_seconds': summary['actual_unique_recovery_seconds'],
            'artifact_sha256': {p: sha(base/p) for p in [
                'FREEZE.json', 'SELECTION_LOCK.json', 'ANALYSIS_PLAN_LOCK.json',
                'EXECUTION.json', 'results/all_methods.csv', 'results/summary.json',
                'logs/verification.json', 'logs/tests.log', 'verify.py', 'analyze.py']},
        }
    output = {
        'created_unix': time.time(),
        'scope': 'Separate exploratory extension; both versions retained, same nine checkpoints, not eighteen independent replications.',
        'versions': versions,
        'parent_manifest_before_after_equal': True,
        'parent_manifest_sha256': sha(HISTORICAL_MANIFEST),
        'parent_read_only_manifest_check_output': after.strip(),
        'parent_pdf_sha256': sha(PARENT/'paper/main.pdf'),
        'parent_source_archive_sha256': sha(PARENT/'dist/sharing-coordinates-arxiv-draft.tar.gz'),
        'numerical_diagnostic_failure_retained': True,
        'numerical_scope': 'Default FP32 batch-vs-single discrepancy triggered direct-batch math-SDPA correction. Does not claim all FP32 arithmetic equivalences now pass.',
        'interpretation': 'No stable practical advantage over weight clustering in this fixed experiment; no equivalence or population claim.',
        'review': 'scientific-critical-thinking self-assessment, not external peer review',
        'final_study_manifest': 'Written after this validation; independently check with write_manifest.py --check.',
        'report_sha256': sha(ROOT/'REPORT.md'),
    }
    with (ROOT/'VALIDATION.json').open('x') as f:
        json.dump(output, f, indent=2, allow_nan=False)
    print(json.dumps({k: {'units':v['units'], 'recoveries':v['unique_action_optimizer_recoveries']} for k,v in versions.items()}))

if __name__ == '__main__':
    main()
