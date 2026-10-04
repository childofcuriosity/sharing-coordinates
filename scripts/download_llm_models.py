"""Download only public, revision-pinned study models; never bypass gates."""
import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '180')
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def download(row):
    model = row['model']
    print('START', model, flush=True)
    try:
        path = snapshot_download(model, revision=row['revision'],
            cache_dir=str(ROOT/'.cache/llm/hub'), max_workers=3,
            allow_patterns=['*.safetensors', '*.json', 'merges.txt', 'vocab.txt',
                            'tokenizer.model', 'README.md', 'LICENSE*'], token=False)
        result = dict(model=model, revision=row['revision'], path=path, status='downloaded')
    except Exception as exc:
        result = dict(model=model, revision=row['revision'], status='failed',
                      error=type(exc).__name__+': '+str(exc))
    result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    out = ROOT/'.artifact-llm/downloads'
    out.mkdir(parents=True, exist_ok=True)
    (out/(model.replace('/', '--')+'.json')).write_text(json.dumps(result, indent=2)+'\n')
    print('END', model, result['status'], flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--only', nargs='*')
    args = parser.parse_args()
    rows = json.loads((ROOT/'research/LLM_ACCESS_INVENTORY.json').read_text())['models']
    rows = [r for r in rows if r.get('config_status') == 200
            and (not args.only or r['model'] in args.only)]
    priority = {'EleutherAI/pythia-160m':0, 'Qwen/Qwen3-0.6B':1, 'EleutherAI/pythia-1b':2}
    rows.sort(key=lambda r: priority.get(r['model'], 3))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(download, rows))
    if any(r['status'] != 'downloaded' for r in results):
        raise SystemExit(1)
