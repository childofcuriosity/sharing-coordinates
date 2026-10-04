"""Prospective byte-LM training; validation only, final checkpoint, no selection."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from src.language import (LanguageConfig, _seed, _learning_rate, build_language_model,
                          corpus_sha256, load_byte_stream, sample_batch,
                          save_language_checkpoint, evaluate)

ROOT = Path(__file__).resolve().parents[1]


def source_hashes(paths):
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}


def train(seed, steps, device, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _seed(seed)
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = LanguageConfig(seed=seed, router_init='random', dimension=128,
                            hidden_dimension=512, batch_size=16, steps=steps,
                            warmup_steps=300, eval_batches=16, log_every=1000)
    sources = source_hashes(['experiments/train_enhancement.py','src/language.py',
                             'src/models.py','src/patterns.py',
                             'research/ENHANCEMENT_PROTOCOL.md'])
    config.checkpoint_save = str(output/'checkpoint.pt')
    hashes = corpus_sha256(config)
    train_stream = load_byte_stream(config.train_path, torch.device(device))
    valid_stream = load_byte_stream(config.validation_path, torch.device(device))
    model = build_language_model(config).to(device)
    initial = model.router.probabilities().detach().clone()
    weights = [p for n,p in model.named_parameters() if n != 'router.logits']
    optimizer = torch.optim.AdamW([{'params':weights,'weight_decay':.1,'multiplier':1.},
                                 {'params':[model.router.logits],'weight_decay':0.,'multiplier':10.}],lr=config.lr)
    generator = torch.Generator().manual_seed(110000+seed)
    history = []
    started = time.monotonic()
    for step in range(steps):
        model.train()
        for group in optimizer.param_groups:
            group['lr'] = _learning_rate(step,config)*group['multiplier']
        x,y = sample_batch(train_stream,config.batch_size,config.sequence_length,generator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1,256),y.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step==0 or (step+1)%1000==0 or step+1==steps:
            valid = evaluate(model,valid_stream,config,1.,120000+seed)
            movement = float((model.router.probabilities()-initial).abs().sum(-1).mean())
            row = dict(step=step+1,train_bpb=float(loss.detach())/math.log(2),
                       validation_bpb=valid,router_movement=movement,
                       gradient_norm=float(grad_norm),seconds=time.monotonic()-started)
            history.append(row)
            print(json.dumps(row),flush=True)
    final = model.router.probabilities().detach()
    metadata = dict(checkpoint_kind='prospective_final_fixed_step',saved_at_step=steps,
                    initial_probabilities=initial.cpu().tolist(),final_probabilities=final.cpu().tolist(),
                    router_mean_l1_movement=float((final-initial).abs().sum(-1).mean()),
                    history=history,source_sha256=sources,training_precision='bfloat16_autocast',
                    router_lr_multiplier=10.,router_weight_decay=0.)
    save_language_checkpoint(config.checkpoint_save,model,config,hashes,metadata)
    result = dict(format='enhancement-training-v1',config=asdict(config),metadata=metadata,
                  corpus_sha256=hashes,checkpoint_sha256=hashlib.sha256(Path(config.checkpoint_save).read_bytes()).hexdigest(),
                  device=device,torch_version=torch.__version__,cuda_version=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(device),seconds=time.monotonic()-started)
    (output/'training.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--steps',type=int,default=6000)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    train(a.seed,a.steps,a.device,a.output)
