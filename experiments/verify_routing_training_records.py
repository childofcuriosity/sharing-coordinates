"""Independent projector-cost checks and provenance for the new trajectory records."""
import json
from pathlib import Path
import numpy as np
from src.llm_shared import sha256
from src.routing_geometry import balanced_partitions
from experiments.llm_protocol import ROOT,check_freeze


def main():
    parts=list(balanced_partitions(12,4))
    projectors=np.zeros((len(parts),12,12))
    for q,groups in zip(projectors,parts):
        for g in groups:q[np.ix_(g,g)]=1/3
    flat=projectors.reshape(len(parts),-1)
    checked=0;worst=0.;digests={}
    for path in sorted((ROOT/'results/llm_routing_trajectories').glob('*/trajectory.json')):
        record=json.loads(path.read_text());digests[str(path.relative_to(ROOT))]=sha256(path)
        for name,digest in record['source_sha256'].items():assert sha256(ROOT/name)==digest,name
        assert sha256(ROOT/'research/LLM_ROUTING_TRAJECTORY_PROTOCOL.md')==record['protocol_sha256']
        for name,digest in record['checkpoints'].items():assert sha256(path.parent/name)==digest,name
        steps={0,256}
        for e in record['events']:steps.update([e['step']-1,e['step']])
        for step in sorted(steps):
            r=record['trace'][step];a=np.asarray(r['a']);g=np.asarray(r['gram'])
            assert np.max(np.abs(a.sum(1)-1))<1e-12 and a.min()>0
            for key,metric in [('weight',g),('router',np.eye(4))]:
                moment=a@metric@a.T
                costs=np.trace(moment)-flat@moment.ravel()
                order=np.argsort(costs);best=parts[int(order[0])]
                assert best==tuple(tuple(x) for x in r[key]['partition']),(path,step,key)
                error=abs(float(costs[order[0]])-r[key]['cost']);worst=max(worst,error)
                assert error<1e-9*max(1,abs(r[key]['cost']))
                assert abs(float(costs[order[1]]-costs[order[0]])-r[key]['gap'])<1e-9*max(1,abs(r[key]['cost']))
                checked+=1
    for path in [ROOT/'results/routing_training/training.json',ROOT/'results/llm_local_flow/local_flow.json',ROOT/'results/llm_local_flow_fp64/local_flow.json',ROOT/'results/llm_local_flow_fp64_norm/local_flow.json']:
        record=json.loads(path.read_text());digests[str(path.relative_to(ROOT))]=sha256(path)
        if isinstance(record['source_sha256'],dict):
            for name,digest in record['source_sha256'].items():assert sha256(ROOT/name)==digest
    old=json.loads((ROOT/'research/LLM_VALIDATION.json').read_text())
    for name,digest in old['files'].items():assert sha256(ROOT/name)==digest,name
    check_freeze()
    result=dict(projector_checks=checked,max_absolute_cost_difference=worst,original_llm_hashes_preserved=True,
                source_sha256=sha256(Path(__file__)),input_sha256=digests)
    out=ROOT/'results/routing_training/record_verification.json'
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print('Independent projector objective checks',checked,'maximum error',worst,'original LLM hashes preserved')

if __name__=='__main__':main()
