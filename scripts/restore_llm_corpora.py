"""Restore missing corpus bytes from the existing immutable lock, never repin."""
import hashlib
import json
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]


def main():
    lock=json.loads((ROOT/'research/LLM_CORPUS_LOCK.json').read_text())
    for row in lock['documents']:
        path=ROOT/row['path']
        if path.exists():
            payload=path.read_bytes()
        elif 'url' in row:
            response=requests.get(row['url'],timeout=120);response.raise_for_status()
            payload=response.content
        else:
            raise FileNotFoundError('Restore repository-provided corpus: '+str(path))
        if hashlib.sha256(payload).hexdigest()!=row['sha256']:
            raise ValueError('Corpus bytes differ from the frozen lock: '+str(path))
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:f.write(payload)
        print('verified',row['path'])


if __name__=='__main__':main()
