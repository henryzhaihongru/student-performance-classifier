"""
Performance metrics module.
This file only calculates the confusion matrix and evaluation metrics.
"""


class MetricsCalculator:
    def __init__(self, positive_label="Pass", negative_label="Fail"):
        # 这里显式指定正类和负类，方便后面计算 TP/FP/FN/TN。
        # Explicitly define positive and negative labels for confusion-matrix calculation.
        self.positive_label = positive_label
        self.negative_label = negative_label

    def confusion_matrix(self, y_true, y_pred):
        if len(y_true) != len(y_pred):
            raise ValueError("Actual and predicted labels must have equal lengths")
        allowed = {self.positive_label, self.negative_label}
        if any(label not in allowed for label in list(y_true) + list(y_pred)):
            raise ValueError("Unexpected classification label")
        tp = fp = fn = tn = 0

        # 逐个比较真实值和预测值，统计四种情况。
        # Compare true and predicted labels one by one and count the four cases.
        for actual, predicted in zip(y_true, y_pred):
            if actual == self.positive_label and predicted == self.positive_label:
                tp += 1
            elif actual == self.negative_label and predicted == self.positive_label:
                fp += 1
            elif actual == self.positive_label and predicted == self.negative_label:
                fn += 1
            elif actual == self.negative_label and predicted == self.negative_label:
                tn += 1

        return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}

    def performance_metrics(self, matrix):
        # 从混淆矩阵中取出四个核心统计量，再计算各种评价指标。
        # Extract TP, FP, FN, and TN from the confusion matrix, then compute evaluation metrics.
        tp = matrix["TP"]
        fp = matrix["FP"]
        fn = matrix["FN"]
        tn = matrix["TN"]

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        # Sensitivity is the same as recall.
        sensitivity = recall
        # Specificity measures correct negative classification.
        specificity = tn / (tn + fp) if (tn + fp) else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

        return {
            # Accuracy: 所有预测中，有多少比例是正确的。
            # Accuracy: the proportion of all predictions that are correct.
            "Accuracy": accuracy,
            # Precision: 预测为 Pass 的样本里，真正 Pass 的比例。
            # Precision: among samples predicted as Pass, how many are actually Pass.
            "Precision": precision,
            # Recall: 实际为 Pass 的样本里，被成功找出来的比例。
            # Recall: among actual Pass samples, how many are correctly identified.
            "Recall": recall,
            "F1-score": f1_score,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
        }
