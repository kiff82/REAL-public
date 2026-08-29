from __future__ import annotations

import unittest

from examples.structural_contact_demo import run_demo


class StructuralContactDemoTests(unittest.TestCase):
    def test_dynamic_contact_resolves_opposing_local_errors(self) -> None:
        result = run_demo(seed=7)
        self.assertLessEqual(result.base_accuracy, 0.10)
        self.assertLessEqual(result.static_accuracy, 0.40)
        self.assertGreaterEqual(result.real_accuracy, 0.95)
        self.assertGreater(result.real_accuracy, result.static_accuracy)

    def test_four_updates_produce_five_states_and_commit_from_z4(self) -> None:
        result = run_demo(seed=11)
        self.assertEqual(result.updates, 4)
        self.assertEqual([row["state"] for row in result.states], ["z0", "z1", "z2", "z3", "z4"])
        self.assertAlmostEqual(result.states[-1]["accuracy"], result.real_accuracy)

    def test_frozen_decoder_is_not_modified(self) -> None:
        result = run_demo(seed=19)
        self.assertTrue(result.frozen_decoder_unchanged)
        self.assertEqual(result.frozen_decoder_hash_before, result.frozen_decoder_hash_after)


if __name__ == "__main__":
    unittest.main()
