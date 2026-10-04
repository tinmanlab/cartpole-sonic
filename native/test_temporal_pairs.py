"""Bounded physical checks; no policy loads or shared-environment writes."""
import unittest
from unittest.mock import patch
import io
import numpy as np
import temporal_pairs as pairs


class TemporalPairsTest(unittest.TestCase):
    def test_declared_cases(self):
        cases = pairs.declared_cases()
        self.assertEqual(len(cases), 12)
        self.assertEqual(cases, pairs.declared_cases())
        self.assertEqual({c.tick for c in cases}, {8, 12, 16})

    def test_physical_pair(self):
        result = pairs.generate_pair(pairs.declared_cases()[0])
        self.assertTrue(result['accepted'], result['reason'])
        self.assertEqual(result['actions'].shape, (2, 28))
        self.assertEqual(result['states'].shape, (2, 29, 4))
        self.assertEqual(result['references'].shape, (2, 8, 4))
        for branch in range(2):
            replay, failed = pairs.simulate(pairs.load_model(), result['case'].x, result['actions'][branch])
            np.testing.assert_array_equal(replay, result['states'][branch])
            self.assertFalse(failed)
        np.testing.assert_array_equal(result['histories'][0], result['histories'][1])
        np.testing.assert_array_equal(result['histories'][0], np.broadcast_to(result['states'][0, 0], (29, 4)))
        np.testing.assert_array_equal(result['states'][:, 0], result['histories'][:, -1])
        self.assertLessEqual(np.abs(result['endpoint_errors']).max(), 1e-7)
        self.assertGreater(result['interior_reference_delta'], 1e-4)
        self.assertGreater(result['first_action_delta'], 1e-5)
        np.testing.assert_array_equal(result['references'], result['states'][:, pairs.OFFSETS])
        np.testing.assert_array_equal(result['actions'][1], -result['actions'][0])

    def test_inputs_and_budget_fail_closed(self):
        for case in [pairs.Case(0., 0, .001), pairs.Case(.26, 8, .001), pairs.Case(0., 8, np.nan)]:
            with self.assertRaises(ValueError):
                pairs.generate_pair(case)
        for budget in [0, 81, True]:
            with self.assertRaises(ValueError):
                pairs.generate_pair(pairs.declared_cases()[0], max_iter=budget)
        result = pairs.generate_pair(pairs.declared_cases()[0], max_candidates=1)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['solver_status'], 2)
        self.assertEqual(result['candidate_count'], 1)
        result = pairs.generate_pair(pairs.declared_cases()[0], max_iter=1)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['solver_status'], 2)
        self.assertIn('iteration_budget_exhausted', result['reason'])
        with self.assertRaises(ValueError):
            pairs.simulate(pairs.load_model(), 0., np.full(28, .251))
        model = pairs.load_model()
        model.opt.timestep = .02
        with self.assertRaises(ValueError):
            pairs.generate_pair(pairs.declared_cases()[0], model=model)

    def test_batched_residual_and_fresh_resets(self):
        original_reset = pairs.mujoco.mj_resetData
        def solver(x0, residual, **kwargs):
            self.assertEqual(kwargs['verbose'], pairs.minimize.Verbosity.SILENT)
            u = np.linspace(-.01, .01, 7)
            batch = residual(np.stack((u, -u, u), axis=1))
            self.assertEqual(batch.shape, (5, 3))
            np.testing.assert_array_equal(batch[:, 0], batch[:, 2])
            np.testing.assert_array_equal(batch[:, 0], residual(u)[:, 0])
            return u, []
        with patch.object(pairs.minimize, 'least_squares', side_effect=solver), patch.object(
                pairs.mujoco, 'mj_resetData', wraps=original_reset) as reset:
            result = pairs.generate_pair(pairs.declared_cases()[0])
            self.assertEqual(reset.call_count, result['candidate_count'] + 3)

    def test_declared_bank_and_numeric_export(self):
        bank = pairs.generate_bank()
        self.assertEqual(bank['summary']['candidates'], 12)
        self.assertEqual(bank['summary']['accepted'] + bank['summary']['rejected'], 12)
        model = pairs.load_model()
        for result in bank['records']:
            self.assertEqual(result['candidate_controls'].shape[0], result['candidate_count'])
            self.assertLessEqual(result['candidate_count'], pairs.MAX_CANDIDATES)
            self.assertTrue(np.all(np.abs(result['actions']) <= pairs.FORCE_BOUND))
            for branch in range(2):
                states, failed = pairs.simulate(model, result['case'].x, result['actions'][branch])
                np.testing.assert_array_equal(states, result['states'][branch])
                self.assertEqual(failed, result['branch_failures'][branch])
            if result['accepted']:
                self.assertLessEqual(np.abs(result['endpoint_errors']).max(), pairs.ENDPOINT_TOL)
                self.assertLessEqual(np.abs(result['target_errors']).max(), pairs.TARGET_TOL)
                self.assertGreater(result['interior_reference_delta'], 1e-4)
                self.assertEqual(result['first_action_informative'], result['first_action_delta'] > 1e-5)
                self.assertFalse(result['branch_failures'].any())
            else:
                self.assertNotEqual(result['reason'], 'accepted')
        archive = io.BytesIO()
        pairs.save_bank(bank, archive)
        archive.seek(0)
        with np.load(archive, allow_pickle=False) as stored:
            for key in stored.files:
                self.assertIn(stored[key].dtype.kind, 'biuf')
            for i, result in enumerate(bank['records']):
                for key in ('states', 'histories', 'actions', 'references', 'candidate_controls',
                            'candidate_residuals', 'candidate_failures', 'solver_controls', 'solver_log'):
                    np.testing.assert_array_equal(stored[f'case_{i}_{key}'], result[key])


if __name__ == '__main__':
    unittest.main()
