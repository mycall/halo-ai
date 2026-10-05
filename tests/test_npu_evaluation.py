"""Small hand-scored checks for the retrieval evaluation's metric calculations."""
import importlib.util
from pathlib import Path
import math
import unittest

SPEC = importlib.util.spec_from_file_location('npu_evaluation', Path(__file__).resolve().parents[1] / 'tools/halogen_npu_evaluate.py')
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class RetrievalMetricTests(unittest.TestCase):
    def test_perfect_ranking(self):
        result = evaluation.metrics(['a', 'b', 'c'], {'a': 1, 'b': 1})
        self.assertEqual(result['ndcg@10'], 1)
        self.assertEqual(result['recall@1'], .5)
        self.assertEqual(result['recall@10'], 1)
        self.assertEqual(result['mrr@10'], 1)

    def test_missing_positive_and_nonrelevant_first(self):
        result = evaluation.metrics(['noise', 'a'], {'a': 1, 'b': 1})
        self.assertAlmostEqual(result['ndcg@10'], (1 / math.log2(3)) / (1 + 1 / math.log2(3)))
        self.assertEqual(result['recall@10'], .5)
        self.assertEqual(result['mrr@10'], .5)

    def test_reranking_preserves_recall_but_changes_ndcg(self):
        before = evaluation.metrics(['noise', 'a'], {'a': 1})
        after = evaluation.metrics(['a', 'noise'], {'a': 1})
        self.assertEqual(before['recall@10'], after['recall@10'])
        self.assertGreater(after['ndcg@10'], before['ndcg@10'])

    def test_duplicate_ids_are_refused(self):
        with self.assertRaises(ValueError):
            evaluation.metrics(['a', 'a'], {'a': 1})
