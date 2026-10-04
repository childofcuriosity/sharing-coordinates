"""Pin small public gradient domains; no benchmark generation claims."""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    out = ROOT/'data/llm-domains'
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for name, repo, branch, file in [
        ('math', 'openai/grade-school-math', 'master', 'grade_school_math/data/train.jsonl'),
        ('code', 'google-research/google-research', 'master', 'mbpp/mbpp.jsonl'),
    ]:
        response = requests.get(f'https://api.github.com/repos/{repo}/commits/{branch}', timeout=60)
        if response.status_code == 403:
            # Public git transport is independent of the unauthenticated API quota.
            remote = subprocess.check_output(['git','ls-remote',
                f'https://github.com/{repo}.git',f'refs/heads/{branch}'],text=True,timeout=90)
            revision = remote.split()[0]
        else:
            response.raise_for_status()
            revision = response.json()['sha']
        url = f'https://raw.githubusercontent.com/{repo}/{revision}/{file}'
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        raw = response.content
        target = out/(name+'.jsonl')
        if target.exists() and target.read_bytes() != raw:
            raise RuntimeError('Refusing changed source: '+str(target))
        target.write_bytes(raw)
        rows = [json.loads(line) for line in raw.decode().splitlines() if line]
        records.append(dict(domain=name,repository=repo,revision=revision,url=url,
                            path=str(target.relative_to(ROOT)),sha256=digest(raw),documents=len(rows)))
    for split in ('train','valid','test'):
        target = ROOT/f'data/wikitext-2/{split}.txt'
        records.append(dict(domain='general',split=split,path=str(target.relative_to(ROOT)),
                            sha256=digest(target.read_bytes())))
    lock = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                purpose='Next-token gradients/losses; no generation accuracy; public pretraining overlap unknown.',
                documents=records)
    target=ROOT/'research/LLM_CORPUS_LOCK.json'
    if target.exists():
        raise RuntimeError('Corpus lock already exists; inspect rather than overwrite')
    target.write_text(json.dumps(lock,indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    main()
