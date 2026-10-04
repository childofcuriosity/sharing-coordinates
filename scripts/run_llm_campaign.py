"""Bounded local orchestration; no training outcomes used for scheduling."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.llm_protocol import CONFIG,MODELS,check_freeze,unit_path


def command(module,model,seed,split,device=None):
    args=[sys.executable,'-m',module,'--model',model,'--seed',str(seed)]
    if module!='experiments.run_llm_decisions':args+=['--split',split]
    if device is not None:args+=['--device',f'cuda:{device}']
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                     HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    stamp=str(time.time_ns())
    log=ROOT/'.artifact-llm'/f'{module.rsplit(".",1)[-1]}-{split}-{model}-{seed}-{stamp}.log'
    with log.open('x') as handle:
        process=subprocess.run(args,cwd=ROOT,env=environment,stdout=handle,stderr=subprocess.STDOUT)
    result=dict(stage=module,model=model,seed=seed,split=split,device=device,
                exit_code=process.returncode,log=str(log.relative_to(ROOT)))
    print(json.dumps(result),flush=True)
    if process.returncode:raise RuntimeError(json.dumps(result))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['gpu','recover','decisions'],required=True)
    parser.add_argument('--split',choices=['calibration','evaluation'],required=True)
    parser.add_argument('--gpus',nargs='+',type=int,default=list(range(8)))
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--models',nargs='*')
    parser.add_argument('--exclude',nargs='*',default=[])
    args=parser.parse_args();check_freeze()
    if args.stage=='recover' and args.split=='evaluation' and not (ROOT/'research/LLM_ACCEPTANCE_POLICY.json').exists():
        raise ValueError('Lock calibration policy before held-out recovery scoring')
    models=args.models or (CONFIG['calibration_models'] if args.split=='calibration' else list(MODELS))
    if args.stage=='decisions':models=[m for m in models if m in CONFIG['decision_models']]
    seeds=CONFIG['calibration_seeds'] if args.split=='calibration' else CONFIG['seeds']
    jobs=[(m,s) for s in seeds for m in models if f'{m}-{s}' not in args.exclude]
    if args.stage=='recover':
        def work(pair):
            m,s=pair
            if (unit_path(m,s,args.split)/'recovery.json').exists():return dict(skipped_completed=f'{m}-{s}')
            return command('experiments.recover_llm_shared',m,s,args.split)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(work,j) for j in jobs]
            failures=[]
            for future in concurrent.futures.as_completed(futures):
                try:future.result()
                except Exception as exc:failures.append(str(exc))
    else:
        pending=queue.Queue()
        for pair in jobs:pending.put(pair)
        def gpu_worker(device):
            failures=[]
            while True:
                try:m,s=pending.get_nowait()
                except queue.Empty:break
                try:
                    out=unit_path(m,s,args.split)
                    if args.stage=='decisions':
                        if not (out/'decisions.json').exists():command('experiments.run_llm_decisions',m,s,args.split,device)
                    else:
                        if not (out/'training.json').exists():command('experiments.train_llm_shared',m,s,args.split,device)
                        if not (out/'observations.json').exists():command('experiments.observe_llm_shared',m,s,args.split,device)
                except Exception as exc:failures.append(str(exc))
                finally:pending.task_done()
            return failures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures=[pool.submit(gpu_worker,d) for d in args.gpus]
            failures=[e for f in futures for e in f.result()]
    if failures:
        print(json.dumps(dict(failures=failures)),flush=True)
        raise SystemExit(1)


if __name__=='__main__':main()
