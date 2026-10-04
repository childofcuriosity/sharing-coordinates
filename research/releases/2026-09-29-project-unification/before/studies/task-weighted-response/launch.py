"""Run each model on a separate GPU; seal all selections before reading test data."""
import concurrent.futures,os,subprocess,time
from common import ROOT,write,sha,check_freeze
MODELS=['pythia160m','qwen06b','smol360m']
PYTHON=str(ROOT.parent/'sharing-coordinates/.cache/llm/venv/bin/python')
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false')

def group(phase,model,device):
    records=[]
    for seed in range(3):
        name=f'{model}-seed{seed}-{phase}';start=time.time()
        with (ROOT/'logs'/f'{name}.log').open('x') as log:
            p=subprocess.run([PYTHON,str(ROOT/'run_study.py'),phase,'--model',model,'--seed',str(seed),'--device',device],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        records.append(dict(phase=phase,model=model,seed=seed,device=device,start_unix=start,elapsed_seconds=time.time()-start,exit_code=p.returncode))
        print(name,'exit',p.returncode,flush=True)
        if p.returncode:break
    return records

def main():
    frozen=check_freeze();start=time.time();all_records=[]
    for phase in ['select','evaluate']:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(group,phase,m,f'cuda:{i}') for i,m in enumerate(MODELS)]
            records=[r for f in futures for r in f.result()]
        all_records+=records
        write(ROOT/f'{phase.upper()}_EXECUTION.json',{'records':records,'freeze_sha256':frozen})
        if len(records)!=9 or any(r['exit_code'] for r in records):raise RuntimeError('Incomplete phase; failures retained; no next phase')
        if phase=='select':
            files=[ROOT/'results'/f'{m}-seed{s}'/'selection.json' for m in MODELS for s in range(3)]
            write(ROOT/'SELECTION_LOCK.json',dict(freeze_sha256=frozen,created_unix=time.time(),selections={str(p.relative_to(ROOT)):sha(p) for p in files}))
            print('All 9 selections sealed; opening evaluation phase',flush=True)
    write(ROOT/'EXECUTION.json',dict(start_unix=start,end_unix=time.time(),makespan_seconds=time.time()-start,per_job_elapsed_seconds=sum(r['elapsed_seconds'] for r in all_records),records=all_records))

if __name__=='__main__':main()
