"""CPU-environment replay plus full-space residual checks on real directions."""
import json
import math
import platform
import time
import torch
import numpy
import scipy
from src.llm_shared import sha256
from src.llm_observations import normalized_queries,observed_frame,compress_on_frame,response_error
from src.response_identifiability import apply_euclidean_response
from src.trusted_recovery import Observation,residual_norm
from experiments.recover_llm_shared import estimate,score,disagreement
from experiments.calibrate_llm_recovery import accepted,SCORES
from experiments.llm_protocol import ROOT,MODELS,CONFIG,unit_path,check_freeze


def main():
    freeze=check_freeze();torch.set_num_threads(1);start=time.monotonic()
    policy_path=ROOT/'research/LLM_ACCEPTANCE_POLICY.json';policy=json.loads(policy_path.read_text())
    records=[];direct=[];inputs={}
    selection=[('designed',0.),('natural_general',1e-4),('natural_general',1e-2)]
    for model in MODELS:
        for seed in CONFIG['seeds']:
            out=unit_path(model,seed,'evaluation')
            data=torch.load(out/'observations.pt',weights_only=True)
            original=json.loads((out/'recovery.json').read_text())
            inputs[str((out/'observations.pt').relative_to(ROOT))]=sha256(out/'observations.pt')
            inputs[str((out/'recovery.json').relative_to(ROOT))]=sha256(out/'recovery.json')
            for family,noise in selection:
                case=next(r for r in data['cases'] if r['family']==family and r['noise']==noise and r['q']==4)
                before=next(r for r in original['records'] if r['family']==family and r['noise']==noise and r['q']==4)
                after=score(estimate(Observation(**case['observation']),Observation(**case['heldout']),
                    case['condition'],case['heldout_condition'],5100000+seed),case['truth'])
                for method in after:
                    one,two=before['methods'][method],after[method]
                    row=dict(model=model,seed=seed,family=family,noise=noise,method=method,
                        status_original=one['status'],status_replay=two['status'])
                    if one['status']==two['status']=='ok':
                        row.update(error_original=one['evaluation']['orbit_relative_error'],
                            error_replay=two['evaluation']['orbit_relative_error'],
                            estimate_orbit_difference=disagreement(torch.tensor(one['estimate']['a'],dtype=torch.double),
                                torch.tensor(one['estimate']['c'],dtype=torch.double),torch.tensor(two['estimate']['a'],dtype=torch.double),
                                torch.tensor(two['estimate']['c'],dtype=torch.double)),
                            acceptance_agreement={s:accepted(one,s,policy['policy'][method][s])==accepted(two,s,policy['policy'][method][s]) for s in SCORES})
                    records.append(row)
            print(model,seed,'replayed',flush=True)
            if seed==0 and model in CONFIG['decision_models']:
                checkpoint=torch.load(out/'checkpoint.pt',weights_only=True)
                a,b=checkpoint['a'],checkpoint['b'];theta=a@b;u,sv=observed_frame(theta,4)
                train=json.loads((out/'training.json').read_text())
                gpath=ROOT/train['gradient_records']['general']['path']
                inputs[str(gpath.relative_to(ROOT))]=sha256(gpath)
                gradients=normalized_queries(torch.load(gpath,weights_only=True)['gradients'].double())[:4]
                # Explicit analytic response, independent of the recorded reverse/JVP measurement.
                measured=torch.stack([apply_euclidean_response(a,b,g) for g in gradients])
                obs,condition=compress_on_frame(theta,gradients,measured,u,0.,sv)
                c=b@u.T
                # Deliberately wrong candidate avoids vacuous near-zero residual comparisons.
                candidate=.8*a+.2/4;cc=c*1.07;bb=cc@u
                predicted=torch.stack([apply_euclidean_response(candidate,bb,g) for g in gradients])
                full_response=float((predicted-measured).norm()/measured.norm())
                full_total=math.hypot(float((candidate@bb-theta).norm()/theta.norm()),full_response)
                compressed_response=response_error(candidate,cc,obs,condition)
                compressed_total=residual_norm(candidate,cc,obs)
                differences=[abs(full_response-compressed_response),abs(full_total-compressed_total)]
                assert max(differences)<1e-10
                direct.append(dict(model=model,full_response=full_response,compressed_response=compressed_response,
                    full_total=full_total,compressed_total=compressed_total,max_absolute_difference=max(differences)))
    result=dict(format='llm-cpu-replay-v1',freeze_sha256=freeze,policy_sha256=sha256(policy_path),input_sha256=inputs,
        source_sha256=sha256(__file__),torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__,
        python=platform.python_version(),records=records,direct_full_space_checks=direct,seconds=time.monotonic()-start,
        scope='81 preselected query cases across all 27 checkpoints, three estimators. Reuses frozen observations; not independent backbone training or a proof.')
    with (ROOT/'results/llm/cpu-replay.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)


if __name__=='__main__':main()
