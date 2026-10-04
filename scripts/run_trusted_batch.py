"""Bounded subprocess parallelism; no agents, overwrite, or hidden retry."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from experiments.trusted_protocol import SPLITS,ROOT


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--split',choices=['calibration','evaluation'],required=True)
    args=p.parse_args()
    jobs=[(kind,s) for kind in SPLITS for s in SPLITS[kind][args.split]]
    def run(job):
        kind,seed=job
        target=ROOT/'results/trusted_recovery'/args.split/f'{kind}{seed}'
        log=ROOT/'.artifact-handover'/f'trusted-{args.split}-{kind}{seed}.log'
        build=[sys.executable,'-m','experiments.build_trusted_observations','--kind',kind,
            '--split',args.split,'--seed',str(seed),'--output',str(target)+'.pt']
        if kind=='language':
            build+=['--checkpoint',f'results/trusted_recovery/checkpoints/seed{seed}/checkpoint.pt',
                    '--device',f'cuda:{seed%8}']
        with log.open('x') as handle:
            subprocess.run(build,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)
            subprocess.run([sys.executable,'-m','experiments.run_trusted_recovery',
                '--observations',str(target)+'.pt','--output',str(target)+'.json'],
                cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)
        return str(target)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for value in pool.map(run,jobs): print(value,flush=True)

if __name__=='__main__': main()
