import csv
import math
import random
import tempfile
import unittest
from pathlib import Path

from data_processing import DataProcessor
from metrics import MetricsCalculator
from model import ManualKNNClassifier

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_demo_end_to_end_and_reproducible(self):
        first = DataProcessor(ROOT / "dataset/student_performance.csv")
        second = DataProcessor(ROOT / "dataset/student_performance.csv")
        prepared = first.prepare_dataset()
        self.assertEqual(prepared, second.prepare_dataset())
        train, test, train_labels, test_labels = prepared
        self.assertEqual((len(train), len(test)), (90, 30))
        classifier = ManualKNNClassifier(5)
        classifier.fit(train, train_labels)
        predictions = classifier.predict(test)
        matrix = MetricsCalculator().confusion_matrix(test_labels, predictions)
        self.assertEqual(sum(matrix.values()), 30)
        self.assertEqual(matrix, {"TP": 17, "FP": 2, "FN": 1, "TN": 10})

    def test_test_extreme_does_not_fit_scaling_or_imputation(self):
        rows = [[1,10,10,10,"Fail"], [2,20,20,20,"Fail"], [3,30,30,30,"Pass"], [4,40,40,40,"Pass"]]
        order = list(range(4))
        random.Random(42).shuffle(order)
        held_out = order[-1]
        rows[held_out] = [10000, "", 10000, 10000, "Pass"]
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "test.csv"
            with file.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["StudyTime","AttendanceRate","AssignmentCompletion","PreviousGrade","FinalResult"])
                writer.writerows(rows)
            processor = DataProcessor(file)
            _, test, _, _ = processor.prepare_dataset()
            train_rows = [rows[index] for index in order[:-1]]
            self.assertEqual(processor.feature_maxs[0], max(row[0] for row in train_rows))
            self.assertEqual(processor.feature_means[1], sum(row[1] for row in train_rows)/3)
            self.assertGreater(test[0][0], 1)
            low, high = processor.feature_mins[1], processor.feature_maxs[1]
            self.assertAlmostEqual(test[0][1], (processor.feature_means[1]-low)/(high-low))

    def test_constant_feature_and_invalid_input(self):
        processor = DataProcessor("unused")
        self.assertEqual(processor.normalize_features([[2,2,2,2],[2,2,2,2]]), [[0,0,0,0],[0,0,0,0]])
        for row in [[1,2], [math.nan,2,3,4], [1,math.inf,3,4]]:
            with self.assertRaises(ValueError):
                processor.normalize_single_input(row)

    def test_split_keeps_global_random_state_and_validates(self):
        state = random.getstate()
        processor = DataProcessor("unused")
        processor.train_test_split([[1],[2],[3],[4]], ["A","B","C","D"])
        self.assertEqual(state, random.getstate())
        with self.assertRaises(ValueError):
            processor.train_test_split([[1]], ["A"])
        with self.assertRaises(ValueError):
            processor.train_test_split([[1],[2]], ["A","B"], 1)


class ModelAndMetricsTests(unittest.TestCase):
    def test_known_neighbors_and_stable_tie(self):
        classifier = ManualKNNClassifier(3)
        classifier.fit([[0],[1],[2],[9],[10]], ["Fail","Fail","Fail","Pass","Pass"])
        self.assertEqual(classifier.predict_one([0.5]), "Fail")
        tie = ManualKNNClassifier(2)
        tie.fit([[0],[2]], ["Fail","Pass"])
        self.assertEqual(tie.predict_one([1]), "Fail")

    def test_model_invalid_states(self):
        for k in [0,-1,True,1.5]:
            with self.assertRaises(ValueError):
                ManualKNNClassifier(k)
        classifier = ManualKNNClassifier(1)
        with self.assertRaises(ValueError):
            classifier.predict_one([1])
        with self.assertRaises(ValueError):
            classifier.fit([[1],[2,3]], ["Pass","Fail"])
        with self.assertRaises(ValueError):
            ManualKNNClassifier(5).fit([[1]], ["Pass"])
        classifier.fit([[1]], ["Pass"])
        with self.assertRaises(ValueError):
            classifier.predict_one([1,2])

    def test_metrics_known_matrix_and_zero_denominators(self):
        calculator = MetricsCalculator()
        matrix = calculator.confusion_matrix(["Pass","Pass","Fail","Fail"], ["Pass","Fail","Pass","Fail"])
        self.assertEqual(matrix, {"TP":1,"FP":1,"FN":1,"TN":1})
        self.assertTrue(all(value == 0.5 for value in calculator.performance_metrics(matrix).values()))
        self.assertTrue(all(value == 0 for value in calculator.performance_metrics({"TP":0,"FP":0,"FN":0,"TN":0}).values()))

    def test_metrics_reject_silent_truncation_and_unknown_labels(self):
        calculator = MetricsCalculator()
        with self.assertRaises(ValueError):
            calculator.confusion_matrix(["Pass"], [])
        with self.assertRaises(ValueError):
            calculator.confusion_matrix(["Other"], ["Pass"])


if __name__ == "__main__":
    unittest.main()
