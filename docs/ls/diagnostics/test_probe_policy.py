"""Regression checks for diagnostic compatibility without GPU dependencies."""
import unittest
from types import SimpleNamespace
from probe_policy import policy_contract


class PolicyContractTests(unittest.TestCase):
    def test_saved_tuple_and_list_select_same_actor(self):
        for groups in [('actor',), ['actor']]:
            with self.subTest(groups=groups):
                result = policy_contract(SimpleNamespace(obs_groups=groups, is_recurrent=False))
                self.assertTrue(result['compatible'])
                self.assertEqual(result['obs_groups'], ['actor'])
                self.assertEqual(result['obs_groups_type'], type(groups).__name__)

    def test_wrong_or_multiple_groups_rejected(self):
        for groups in [('critic',), ['actor', 'critic'], [], 'actor', {'actor'}]:
            with self.subTest(groups=groups):
                self.assertFalse(policy_contract(SimpleNamespace(obs_groups=groups, is_recurrent=False))['compatible'])

    def test_recurrent_model_rejected(self):
        self.assertFalse(policy_contract(SimpleNamespace(obs_groups=('actor',), is_recurrent=True))['compatible'])


if __name__ == '__main__':
    unittest.main()
