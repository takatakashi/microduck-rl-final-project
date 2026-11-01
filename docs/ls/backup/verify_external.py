"""Verify the off-instance archive and the three checkpoints used in this project.

Run on the Mac after rsync; this command does not load PyTorch or execute weights.
"""
import argparse
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from microduck_project.artifacts import verify_backup
from microduck_project.common import read_json, write_json, utc_now


def check_required(tar, expected):
    manifest=json.load(tar.extractfile('backup_manifest.json'))
    files={entry['path']:entry for entry in manifest['files']}
    if len(files)!=len(manifest['files']):
        raise ValueError('manifestに重複ファイルがあります')
    results=[]
    for item in expected['checkpoints']:
        name=item['archive_path']
        if name not in files:
            raise ValueError(f'必要な重みがありません: {name}')
        # verify_backup has already compared every member's bytes with this hash.
        if files[name]['sha256']!=item['checkpoint_sha256']:
            raise ValueError(f'実測で使用した重みと不一致: {name}')
        sidecar=str(Path(name).with_suffix('.json'))
        meta=json.load(tar.extractfile(sidecar))
        if meta['checkpoint_sha256']!=item['checkpoint_sha256']:
            raise ValueError(f'重みとsidecarが不一致: {sidecar}')
        for key,value in item['metadata'].items():
            if meta[key]!=value:
                raise ValueError(f'モデル記録が不一致: {sidecar}: {key}')
        source=json.load(tar.extractfile(str(Path(name).parent/'source.json')))
        if source['source_sha256']!=meta['source_sha256']:
            raise ValueError(f'学習ソース記録が不一致: {name}')
        for row in source['files']:
            archived=files.get('project/'+row['path'])
            if archived is None or archived['sha256']!=row['sha256']:
                raise ValueError(f'復元用ソースが学習時と不一致: {row["path"]}')
        for extra in ('run.json','experiment.json','iterations.csv','params/env.yaml','params/agent.yaml'):
            if str(Path(name).parent/extra) not in files:
                raise ValueError(f'復元用の実験記録がありません: {Path(name).parent/extra}')
        lock=files.get('run/backup_dependencies/uv.lock')
        if lock is None or lock['sha256']!=meta['lock_sha256']:
            raise ValueError('保存した上流uv.lockが学習時と一致しません')
        results.append({'file':name,'role':item['role'],'sha256':item['checkpoint_sha256'],'verification':'passed'})
    return results


def verify_external(archive,expected):
    remote=read_json(archive.with_suffix(archive.suffix+'.json'))
    result=verify_backup(archive)
    if remote['verification']!='passed' or remote['sha256']!=result['sha256'] or remote['files']!=result['files']:
        raise ValueError('Brevで作成したアーカイブとMacのコピーが一致しません')
    with tarfile.open(archive,'r:gz') as tar:
        result['required_checkpoints']=check_required(tar,expected)
    result.update(verified_at=utc_now(),copy_hash_matches_brev=True,
                  verification_scope='Archive bytes, required trained checkpoints, metadata, source files and dependency lock. GPU reload was not executed.',
                  storage_location='This command must be run on the Mac after copying off Brev; host placement is confirmed by the operator.')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    args=parser.parse_args()
    expected=read_json(Path(__file__).with_name('expected_checkpoints.json'))
    print('全ファイルのハッシュを検証中です。サイズに応じて時間がかかります。',flush=True)
    result=verify_external(args.archive,expected)
    path=args.archive.with_suffix(args.archive.suffix+'.local-verification.json')
    write_json(path,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
