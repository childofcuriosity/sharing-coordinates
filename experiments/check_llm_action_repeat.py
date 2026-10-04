"""Post hoc same-action repeat to assess Pythia BF16 execution variation."""
import json
import torch
from src.llm_shared import load_model,attach_bank,make_blocks,token_loss,sha256
from src.llm_decisions import refit_bases
from experiments.llm_protocol import ROOT,CONFIG,MODELS,check_freeze,unit_path
from experiments.train_llm_shared import evaluate


def main():
    freeze=check_freeze();torch.set_num_threads(4);torch.cuda.set_device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32=False
    records=[]
    for seed in CONFIG['seeds']:
        torch.manual_seed(seed);out=unit_path('pythia1b',seed,'evaluation')
        original=json.loads((out/'decisions.json').read_text())
        checkpoint=torch.load(out/'checkpoint.pt',weights_only=True,map_location='cuda:0')
        model,tokenizer,_=load_model(MODELS['pythia1b'],'cuda:0')
        bank,_=attach_bank(model,seed,4,64)
        with torch.no_grad():
            bank.logits.copy_(checkpoint['logits']);bank.bases.copy_(checkpoint['bases']);theta=bank.prepare().detach().clone()
        bank.logits.requires_grad_(False)
        labels=torch.tensor(original['actions']['native_router']['labels'])
        bank.hard_routes=torch.nn.functional.one_hot(labels,4).to('cuda:0',dtype=bank.bases.dtype)
        train,_=make_blocks(tokenizer,'general','train','adapt',128)
        blocks={d:make_blocks(tokenizer,d,'test','loss',128,256)[0] for d in CONFIG['evaluation_domains']}
        repeats=[]
        for repeat in range(3):
            with torch.no_grad():bank.bases.copy_(refit_bases(theta,labels,4))
            immediate={d:evaluate(model,bank,x,'cuda:0') for d,x in blocks.items()}
            optimizer=torch.optim.AdamW([bank.bases],lr=CONFIG['decision_lr'],weight_decay=0.)
            losses=[]
            for ids in original['recovery_batch_indices']:
                optimizer.zero_grad(set_to_none=True)
                loss=token_loss(model,train[ids].to('cuda:0'),bank);loss.backward()
                torch.nn.utils.clip_grad_norm_([bank.bases],CONFIG['clip']);optimizer.step()
                losses.append(float(loss.detach()))
            repeats.append(dict(immediate_nll=immediate,training_nll=losses,
                after_recovery_nll={d:evaluate(model,bank,x,'cuda:0') for d,x in blocks.items()}))
            print(seed,repeat,'complete',flush=True)
        records.append(dict(seed=seed,source_sha256=sha256(out/'decisions.json'),repeats=repeats))
        del model,bank;torch.cuda.empty_cache()
    with (ROOT/'results/llm/action-repeat.json').open('x') as f:
        json.dump(dict(freeze_sha256=freeze,source_sha256=sha256(__file__),model='pythia1b',records=records,
            scope='Post hoc diagnostic of identical native labels, tensors, batches and optimizer settings; not an independent structural comparison or a changed primary outcome.'),f,indent=2,allow_nan=False)


if __name__=='__main__':main()
