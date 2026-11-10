"""Export public copies without changing locally frozen evidence.

Absolute workspace paths are replaced by '.', and other home-directory prefixes
by '<LOCAL_HOME>'. Checkpoint tensors are preserved; metadata paths are rewritten.
Recorded audit/fingerprint references still describe the original local files.
PUBLICATION_MANIFEST.json maps those originals to their public copies.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import torch

NAMES = ('local', 'benchmark', 'ablation')
TEXT_SUFFIXES = {'.json', '.md', '.log'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(obj):
    h = hashlib.sha256()
    def visit(value, path=''):
        if isinstance(value, torch.Tensor):
            t = value.detach().cpu().contiguous()
            h.update(path.encode()); h.update(str(t.dtype).encode())
            h.update(str(tuple(t.shape)).encode()); h.update(t.numpy().tobytes())
        elif isinstance(value, dict):
            for k in sorted(value, key=str): visit(value[k], path+'/'+str(k))
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value): visit(v, path+'/'+str(i))
    visit(obj)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--destination', type=Path, required=True)
    args = p.parse_args()
    root, dest = args.root.resolve(), args.destination.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    public = root/'research/history_physics_20261110'
    public.mkdir(parents=True, exist_ok=True)
    workspace, home = str(root), str(Path.home())
    def redact(s): return s.replace(workspace, '.').replace(home, '<LOCAL_HOME>')
    def transform(x):
        if isinstance(x, str): return redact(x)
        if isinstance(x, dict): return {k: transform(v) for k,v in x.items()}
        if isinstance(x, list): return [transform(v) for v in x]
        if isinstance(x, tuple): return tuple(transform(v) for v in x)
        return x
    records = []
    for name in NAMES:
        base = root/f'outputs/history_physics_{name}_20261110'
        for source in sorted(base.rglob('*')):
            if not source.is_file(): continue
            relative = source.relative_to(root)
            target = dest/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            record = {'file':relative.as_posix(), 'original_sha256':sha(source)}
            if source.suffix in TEXT_SUFFIXES:
                target.write_text(redact(source.read_text()))
            elif source.suffix == '.pt':
                original = torch.load(source, map_location='cpu', weights_only=False)
                exported = transform(original)
                torch.save(exported, target)
                record['tensor_sha256'] = tensor_sha(original)
                assert tensor_sha(torch.load(target, map_location='cpu', weights_only=False)) == record['tensor_sha256']
            else:
                shutil.copyfile(source, target)
            record.update(published_sha256=sha(target), bytes=target.stat().st_size)
            record['byte_identical'] = record['original_sha256'] == record['published_sha256']
            if source.suffix in {'.json','.md','.png','.pdf'}:
                compact = public/name/source.relative_to(base)
                compact.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(target, compact)
                if compact.suffix == '.md':
                    compact.write_text(compact.read_text().replace('../../experiments/', '../../../experiments/'))
                record['repository_copy'] = compact.relative_to(root).as_posix()
                record['repository_copy_sha256'] = sha(compact)
            records.append(record)
    manifest = dict(scope='Path-redacted publication copies of locally audited synthetic experiments',
        transformation="Workspace absolute paths -> '.'; other home prefixes -> '<LOCAL_HOME>'; checkpoint tensor values unchanged",
        audit_boundary='Embedded recorded hashes/audit PASS statements refer to original local evidence. Published file hashes are mapped below; do not claim the redacted bytes have the original hashes.',
        reproduction='Use the repository source and a fresh output directory for end-to-end re-training/evaluation/auditing. Published snapshots preserve metrics and tensor values but are not in-place canonical rerun directories.',
        files=records)
    (dest/'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (public/'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (dest/'README.txt').write_text('Public history-physics evidence snapshot, 2026-11-10\n\n'
        'Extract into a separate directory to inspect outputs/. Workspace paths in metadata are relative to that root.\n'
        'PUBLICATION_MANIFEST.json records both original local SHA256 and exported SHA256.\n'
        'All checkpoint tensors and numerical NPZ arrays are preserved. File hashes for path-redacted metadata differ.\n'
        'Recorded canonical audits refer to original local bytes. For an end-to-end audit use repository instructions to regenerate in a fresh output directory.\n')
    for f in dest.rglob('*'):
        if f.is_file() and f.suffix in TEXT_SUFFIXES:
            assert workspace.encode() not in f.read_bytes() and home.encode() not in f.read_bytes()
    archive = dest.parent/'history-physics-20261110-evidence.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tf:
        for item in sorted(dest.iterdir()): tf.add(item,arcname=item.name)
    checksum = f'{sha(archive)}  {archive.name}\n'
    (public/'ASSET_SHA256.txt').write_text(checksum)
    archive.with_name(archive.name+'.sha256').write_text(checksum)
    print(json.dumps({'files':len(records),'archive':str(archive),'archive_bytes':archive.stat().st_size,'compact':str(public)}))


if __name__ == '__main__': main()
