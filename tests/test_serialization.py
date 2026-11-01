from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum, StrEnum, auto
import json
import os
import subprocess
import sys
import unittest

from microduck_project.common import ROOT, digest, jsonable
from microduck_project.config import load_experiment
from microduck_project.runtime import configuration_contract, validate_resume


@dataclass
class ViewerFixture:
    # mjlab 1.3.0 ViewerConfig also contains a nested, ordinary Enum.
    class OriginType(Enum):
        AUTO = auto()
        WORLD = auto()

    origin_type: OriginType = OriginType.AUTO


@dataclass
class EnvironmentFixture:
    viewer: ViewerFixture = field(default_factory=ViewerFixture)


@dataclass
class AgentFixture:
    seed: int = 42
    max_iterations: int = 5
    save_interval: int = 2


class SerializationTests(unittest.TestCase):
    def test_nested_viewer_enum_survives_json_roundtrip(self):
        env = EnvironmentFixture()
        restored = json.loads(json.dumps(jsonable(env), allow_nan=False))
        self.assertEqual(digest(env), digest(restored))
        # Changing run length must still be allowed when resuming this config.
        self.assertEqual(configuration_contract(env, AgentFixture()),
                         configuration_contract(deepcopy(env), AgentFixture(max_iterations=10)))

    def test_changed_enum_is_rejected_when_resuming(self):
        env = EnvironmentFixture()
        agent = AgentFixture()
        experiment = load_experiment(ROOT / "configs/experiments/smoke.json")
        meta = {"experiment": experiment, "configuration_sha256": configuration_contract(env, agent),
                "source_sha256": "source", "lock_sha256": "lock", "completed_iterations": 5}
        resumed = dict(experiment, total_iterations=10)
        validate_resume(meta, resumed, configuration_contract(env, agent), "source", "lock")
        env.viewer.origin_type = ViewerFixture.OriginType.WORLD
        with self.assertRaisesRegex(ValueError, "実効設定"):
            validate_resume(meta, resumed, configuration_contract(env, agent), "source", "lock")

    def test_enum_type_and_value_affect_hash(self):
        first = Enum("Mode", {"AUTO": 1}, module="example.first")
        other_type = Enum("Mode", {"AUTO": 1}, module="example.second")
        changed_value = Enum("Mode", {"AUTO": 2}, module="example.first")
        self.assertNotEqual(digest(first.AUTO), digest(other_type.AUTO))
        self.assertNotEqual(digest(first.AUTO), digest(changed_value.AUTO))

    def test_int_and_string_enums_remain_distinct_from_primitives(self):
        number = IntEnum("NumberMode", {"AUTO": 1})
        text = StrEnum("TextMode", {"AUTO": "auto"})
        for member in (number.AUTO, text.AUTO):
            with self.subTest(member=member):
                self.assertNotEqual(digest(member), digest(member.value))

    def test_enum_hash_is_stable_across_processes(self):
        code = (
            "from enum import Enum\n"
            "from microduck_project.common import digest\n"
            "OriginType = Enum('OriginType', {'AUTO': 1}, "
            "module='mjlab.viewer.viewer_config', qualname='ViewerConfig.OriginType')\n"
            "print(digest({'viewer': {'origin_type': OriginType.AUTO}}))\n"
        )
        hashes = [subprocess.check_output(
            [sys.executable, "-c", code], cwd=ROOT, text=True,
            env={**os.environ, "PYTHONHASHSEED": seed}).strip() for seed in ("1", "2")]
        self.assertEqual(hashes[0], hashes[1])
        self.assertRegex(hashes[0], r"^[0-9a-f]{64}$")

    def test_unsupported_objects_are_not_silently_stringified(self):
        class NotAnEnum:
            name = "AUTO"
            value = 1

        with self.assertRaisesRegex(TypeError, "Cannot serialize NotAnEnum"):
            jsonable(NotAnEnum())
