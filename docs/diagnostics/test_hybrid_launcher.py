"""Check the system-Python entry point without loading any GPU/controller code."""
from pathlib import Path
import subprocess
import sys
import unittest


class LauncherTests(unittest.TestCase):
    def test_launcher_and_help_before_python312_only_imports(self):
        script = Path(__file__).with_name('hybrid_goal.py').resolve()
        harness = r'''
import enum
import importlib.abc
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
from unittest.mock import patch

# Emulate the missing symbol in the reported Python 3.10 interpreter.
if hasattr(enum, 'StrEnum'):
    del enum.StrEnum

class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'microduck_project.controller', 'torch'}:
            raise ImportError('Runtime imported before dispatch: ' + fullname)
sys.meta_path.insert(0, BlockRuntime())

class Dispatched(Exception):
    pass

script, mode = sys.argv[1:]
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    fake_common = types.ModuleType('microduck_project.common')
    fake_common.verify_upstream = lambda: root / 'vendor'
    def dispatch(executable, arguments, environment):
        assert executable == str(root / 'vendor/.venv/bin/python')
        assert arguments[0] == executable and arguments[1] == script
        assert arguments[-1] == '--worker'
        assert environment['PYTHONPATH'] == str(Path(script).parents[2])
        assert environment['MUJOCO_GL'] == 'egl'
        raise Dispatched()
    sys.argv = [script, '--help'] if mode == 'help' else [script, '--output', str(root/'new-output')]
    if mode == 'fast-validation':
        sys.argv.append('--fast-validation')
    with patch.dict(sys.modules, {'microduck_project.common': fake_common}), patch('os.execve', dispatch), patch.dict(os.environ, {}, clear=True):
        try:
            runpy.run_path(script, run_name='__main__')
        except Dispatched:
            assert mode in ('launch', 'fast-validation')
        except SystemExit as exc:
            assert mode == 'help' and exc.code == 0
        else:
            raise AssertionError('Expected dispatch or help exit')
    assert not (root/'new-output').exists()
'''
        for mode in ('launch', 'help', 'fast-validation'):
            with self.subTest(mode=mode):
                result = subprocess.run([sys.executable, '-c', harness, str(script), mode],
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
