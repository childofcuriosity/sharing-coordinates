"""Inventory this independent extension, including both retained experiment versions."""
import argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    lines=[]
    for f in sorted(ROOT.rglob('*')):
        if not f.is_file() or f.is_symlink() or f==ROOT/'MANIFEST.sha256' or '__pycache__' in f.parts or '.pytest_cache' in f.parts:continue
        h=hashlib.sha256()
        with f.open('rb') as handle:
            for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
        lines.append(f'{h.hexdigest()}  {f.relative_to(ROOT)}')
    content='\n'.join(lines)+'\n';target=ROOT/'MANIFEST.sha256'
    if args.check:
        assert target.read_text()==content,'Independent study inventory differs'
        print('Independent study manifest verified:',len(lines),'files')
    else:target.write_text(content);print('Independent study manifest written:',len(lines),'files')

if __name__=='__main__':main()
