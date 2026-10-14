"""Restore only the authorized four-gauge assets from bounded GitHub parts."""
import json,hashlib,zipfile,io,stat
from pathlib import Path

def sha(data):return hashlib.sha256(data).hexdigest()
def main(root):
    root=Path(root).resolve();manifest=json.loads((root/'delivery_manifest.json').read_text())
    parts=[]
    for part in manifest['parts']:
        p=root/part['path'];b=p.read_bytes()
        if len(b)!=part['bytes'] or sha(b)!=part['sha256']:raise RuntimeError('PART_HASH_DIFFERS '+part['path'])
        parts.append(b)
    archive=b''.join(parts)
    if sha(archive)!=manifest['archive_sha256']:raise RuntimeError('ARCHIVE_HASH_DIFFERS')
    assets=(root/'assets').resolve();expected={x['path']:x for x in manifest['members']}
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(expected):raise RuntimeError('EXACT_MEMBER_SET_REQUIRED')
        for name in names:
            info=z.getinfo(name);target=(assets/name).resolve()
            if info.is_dir() or not target.is_relative_to(assets) or stat.S_ISLNK(info.external_attr>>16):
                raise RuntimeError('REGULAR_SAFE_RELATIVE_ASSET_REQUIRED')
            b=z.read(name);row=expected[name]
            if len(b)!=row['bytes'] or sha(b)!=row['sha256']:raise RuntimeError('MEMBER_HASH_DIFFERS '+name)
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if sha(target.read_bytes())!=row['sha256']:raise RuntimeError('EXISTING_ASSET_CHANGED_REFUSE_OVERWRITE')
            else:target.write_bytes(b)
    print(json.dumps(dict(passed=True,restored_members=len(expected),asset_root=str(assets))))
if __name__=='__main__':main(Path(__file__).resolve().parent)
