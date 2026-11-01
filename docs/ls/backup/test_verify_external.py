"""Backup integrity/completeness checks using explicitly synthetic checkpoints."""
from pathlib import Path
import copy
import hashlib
import io
import json
import shutil
import tarfile
import tempfile
import unittest

from verify_external import verify_external
from microduck_project.artifacts import verify_backup


def hashed(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value).encode()


class ExternalBackupTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        weight=b'synthetic fixture; not a trained model'
        lock=b'synthetic dependency lock'
        source=b'synthetic source'
        metadata={'completed_iterations':1001,'source_sha256':'fixture-source','lock_sha256':hashed(lock)}
        self.expected={'checkpoints':[{'archive_path':'run/baseline/model_1000.pt',
            'role':'fixture','checkpoint_sha256':hashed(weight),'metadata':metadata}]}
        self.files={'run/baseline/model_1000.pt':weight,
                    'run/baseline/model_1000.json':encoded({'checkpoint_sha256':hashed(weight),**metadata}),
                    'run/baseline/source.json':encoded({'source_sha256':'fixture-source','files':[
                        {'path':'microduck_project/sample.py','sha256':hashed(source)}]}),
                    'project/microduck_project/sample.py':source,
                    'run/backup_dependencies/uv.lock':lock}
        for name in ['run.json','experiment.json','iterations.csv','params/env.yaml','params/agent.yaml']:
            self.files['run/baseline/'+name]=b'fixture-record'

    def archive(self,files):
        archive=self.root/'source.tar.gz'
        manifest={'schema_version':1,'files':[{'path':name,'sha256':hashed(data),'size_bytes':len(data)}
                                              for name,data in files.items()]}
        with tarfile.open(archive,'w:gz') as tar:
            for name,data in [*files.items(),('backup_manifest.json',encoded(manifest))]:
                member=tarfile.TarInfo(name);member.size=len(data)
                tar.addfile(member,io.BytesIO(data))
        result=verify_backup(archive)
        archive.with_suffix('.gz.json').write_text(json.dumps(result))
        return archive

    def test_copy_passes_only_after_bytes_and_required_model_match(self):
        original=self.archive(self.files)
        copied=self.root/'copied.tar.gz'
        shutil.copyfile(original,copied)
        shutil.copyfile(original.with_suffix('.gz.json'),copied.with_suffix('.gz.json'))
        result=verify_external(copied,self.expected)
        self.assertEqual(result['verification'],'passed')
        self.assertTrue(result['copy_hash_matches_brev'])
        self.assertEqual(len(result['required_checkpoints']),1)

    def test_missing_checkpoint_is_rejected_even_if_archive_is_valid(self):
        files=dict(self.files);del files['run/baseline/model_1000.pt']
        archive=self.archive(files)
        with self.assertRaisesRegex(ValueError,'必要な重み'):verify_external(archive,self.expected)

    def test_wrong_checkpoint_is_rejected_even_if_archive_is_valid(self):
        archive=self.archive(self.files)
        expected=copy.deepcopy(self.expected);expected['checkpoints'][0]['checkpoint_sha256']='different'
        with self.assertRaisesRegex(ValueError,'実測で使用した重み'):verify_external(archive,expected)

    def test_wrong_source_or_lock_is_rejected(self):
        for path in ['project/microduck_project/sample.py','run/backup_dependencies/uv.lock']:
            with self.subTest(path=path):
                files=dict(self.files);files[path]=b'different'
                archive=self.archive(files)
                with self.assertRaises(ValueError):verify_external(archive,self.expected)

    def test_other_archive_sidecar_is_rejected(self):
        archive=self.archive(self.files)
        sidecar=archive.with_suffix('.gz.json')
        data=json.loads(sidecar.read_text());data['sha256']='different'
        sidecar.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'Brevで作成したアーカイブ'):verify_external(archive,self.expected)


if __name__=='__main__':unittest.main()
