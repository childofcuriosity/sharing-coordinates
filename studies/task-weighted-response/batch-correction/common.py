from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parents[2]
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
    migration=read(PARENT/'research/releases/2026-09-29-project-unification/MIGRATION.json')
    for p,d in f['sources'].items():
        current=ROOT/p
        change=migration['changed_files'].get(str(current.relative_to(PARENT)))
        if change:
            assert change['before_sha256']==d,('historical source binding changed',p)
            assert sha(PARENT/change['before_file'])==d,('historical source changed',p)
            assert sha(current)==change['after_sha256'],('migrated source changed',p)
        else:
            assert sha(current)==d,('study source changed',p)
    for p,d in f['parent_inputs'].items():
        assert sha(PARENT/('research/releases/2026-09-29-project-unification/MANIFEST.sha256' if p=='MANIFEST.sha256' else p))==d,('parent input changed',p)
    return sha(ROOT/'FREEZE.json')
