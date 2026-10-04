"""Direct shared additive coordinates on a frozen Hugging Face causal LM.

Only fixed selected output rows are adapted. This is a linear isometric
embedding of the explicitly audited AB module, not a LoRA factorization of B.
"""
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def model_entry(model_id):
    rows=json.loads((ROOT/'research/LLM_ACCESS_INVENTORY.json').read_text())['models']
    row=next(r for r in rows if r['model']==model_id)
    if row.get('config_status')!=200:
        raise ValueError('Model is not publicly accessible under this protocol')
    path=ROOT/'.cache/llm/hub'/('models--'+model_id.replace('/','--'))/'snapshots'/row['revision']
    return row,path


class SharedBank(nn.Module):
    def __init__(self, depth, inputs, outputs, rank=4, channels=64, seed=0, device='cpu'):
        super().__init__()
        if depth<rank or channels>outputs:
            raise ValueError('Insufficient layers or output channels')
        gen=torch.Generator().manual_seed(seed)
        self.depth,self.rank,self.inputs,self.channels=depth,rank,inputs,channels
        self.logits=nn.Parameter(torch.randn(depth,rank,generator=gen,device='cpu').to(device))
        basis=torch.randn(rank,channels*inputs,generator=gen)*(.01/math.sqrt(inputs))
        self.bases=nn.Parameter(basis.to(device))
        ids=torch.linspace(0,outputs-1,channels).round().long()
        if len(ids.unique())!=channels:
            raise ValueError('Repeated output channels')
        self.register_buffer('output_ids',ids.to(device))
        self.theta=None
        self.disabled=False

    def prepare(self, theta=None):
        self.theta=self.logits.softmax(-1)@self.bases if theta is None else theta
        return self.theta

    def inject(self, index, hidden, output):
        if self.disabled:
            return output
        if self.theta is None:
            raise RuntimeError('Call bank.prepare before each forward')
        delta=F.linear(hidden.float(),self.theta[index].reshape(self.channels,self.inputs))
        return output.index_add(-1,self.output_ids,delta.to(output.dtype))


def attach_bank(model, seed, rank=4, channels=64):
    for p in model.parameters():
        p.requires_grad_(False)
    if hasattr(model,'gpt_neox'):
        projections=[(f'gpt_neox.layers.{i}.attention.dense',layer.attention.dense)
                     for i,layer in enumerate(model.gpt_neox.layers)]
    elif hasattr(model,'model') and hasattr(model.model,'layers'):
        projections=[(f'model.layers.{i}.self_attn.o_proj',layer.self_attn.o_proj)
                     for i,layer in enumerate(model.model.layers)]
    else:
        raise ValueError('Unsupported architecture; never silently skip modules')
    if len(projections)%rank:
        count=len(projections)//rank*rank
        indices=torch.linspace(0,len(projections)-1,count).round().long().tolist()
        projections=[projections[i] for i in indices]
    shape=projections[0][1].weight.shape
    if any(module.weight.shape!=shape for _,module in projections):
        raise ValueError('Selected projection shapes differ')
    device=projections[0][1].weight.device
    bank=SharedBank(len(projections),shape[1],shape[0],rank,channels,seed,device)
    model.add_module('sharing_bank',bank)
    for index,(_,module) in enumerate(projections):
        module.register_forward_hook(lambda module,args,output,i=index:bank.inject(i,args[0],output))
    model.config.use_cache=False
    return bank,[name for name,_ in projections]


def token_loss(model, blocks, bank, theta=None):
    bank.prepare(theta)
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=blocks.is_cuda):
        logits=model(input_ids=blocks[:,:-1],use_cache=False).logits
    return F.cross_entropy(logits.float().flatten(0,1),blocks[:,1:].reshape(-1))


def domain_documents(domain, split):
    if domain=='general':
        raw=(ROOT/f'data/wikitext-2/{split}.txt').read_text()
        # Contiguous blank-line-delimited documents; IDs bind the original split.
        docs=[(str(i),x.strip()) for i,x in enumerate(raw.split('\n\n')) if len(x.strip())>=80]
        if split=='test':
            return docs
        return docs
    rows=[json.loads(x) for x in (ROOT/f'data/llm-domains/{domain}.jsonl').read_text().splitlines() if x]
    docs=[]
    for i,row in enumerate(rows):
        if domain=='math':
            text='Question: '+row['question']+'\nAnswer: '+row['answer']
        elif domain=='code':
            text='# '+row['text']+'\n'+row['code']
        else:
            raise ValueError(domain)
        docs.append((str(row.get('task_id',i)),text))
    return docs


def make_blocks(tokenizer, domain, split, role, length=128, limit=4096):
    docs=domain_documents(domain,split)
    if role=='adapt':
        selected=docs
    elif role in ('probe','loss'):
        parity=0 if role=='probe' else 1
        selected=[d for i,d in enumerate(docs) if i%2==parity]
    else:
        raise ValueError(role)
    ids=[]; document_ids=[]
    eos=tokenizer.eos_token_id
    for doc_id,text in selected:
        encoded=tokenizer.encode(text,add_special_tokens=False)
        ids.extend(encoded+([] if eos is None else [eos]))
        document_ids.append(doc_id)
        if len(ids)>=(length+1)*limit:
            break
    count=min(limit,len(ids)//(length+1))
    if count<16:
        raise ValueError('Too few disjoint token blocks')
    blocks=torch.tensor(ids[:count*(length+1)],dtype=torch.long).reshape(count,length+1)
    meta=dict(domain=domain,split=split,role=role,length=length,blocks=count,
              document_ids=document_ids,token_sha256=hashlib.sha256(blocks.numpy().tobytes()).hexdigest())
    return blocks,meta


def load_model(model_id,device):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    entry,path=model_entry(model_id)
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval()
    return model,tokenizer,entry
