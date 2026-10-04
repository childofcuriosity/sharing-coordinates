from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent.parent/'sharing-coordinates'
sys.path.insert(0,str(PARENT))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)

def read(path):return json.loads(Path(path).read_text())

def check_freeze():
    f=read(ROOT/'FREEZE.json')
    for p,d in f['sources'].items():
        assert sha(ROOT/p)==d,('study source changed',p)
    for p,d in f['parent_inputs'].items():
        assert sha(PARENT/p)==d,('parent input changed',p)
    return sha(ROOT/'FREEZE.json')
