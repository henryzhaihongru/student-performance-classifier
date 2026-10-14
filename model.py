"""
Manual K-Nearest Neighbors classification model.
This file only contains the KNN classifier.
"""

import math
from collections import Counter

DEFAULT_K = 5
# Odd K reduces tie cases in majority voting.


def explain_model():
    # 返回模型说明信息，便于项目汇报时解释算法设计。
    # Return model description information for presentation and defense.
    return {
        "algorithm": "Manual K-Nearest Neighbors classifier",
        "training": "KNN stores training samples instead of learning weights.",
        "prediction": "Prediction calculates Euclidean distance and uses majority voting.",
        "distance_formula": "sqrt(sum((xi - yi)^2))",
        "why_k_5": "K=5 is a small odd number that reduces tie cases and gives stable majority voting.",
    }


class ManualKNNClassifier:
    def __init__(self, k=DEFAULT_K):
        # k 表示要参考多少个最近邻居。
        # k is the number of nearest neighbors used for voting.
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k
        self.x_train = []
        self.y_train = []

    def fit(self, x_train, y_train):
        """
        KNN does not learn weights like neural networks.
        It stores the training data and uses it later for distance comparison.
        """
        # 所谓“训练”本质就是保存样本。
        # In KNN, training is essentially storing the samples.
        if not x_train or len(x_train) != len(y_train) or self.k > len(x_train):
            raise ValueError("Provide aligned training samples with at least k rows")
        width = len(x_train[0])
        if width == 0 or any(len(row) != width or any(not math.isfinite(value) for value in row) for row in x_train):
            raise ValueError("Training features must have equal dimensions and finite values")
        self.x_train = [list(row) for row in x_train]
        self.y_train = list(y_train)

    def euclidean_distance(self, point1, point2):
        """
        Euclidean distance:
        d = sqrt((x1-y1)^2 + (x2-y2)^2 + ... + (xn-yn)^2)
        """
        total = 0
        for index in range(len(point1)):
            total += (point1[index] - point2[index]) ** 2
        return math.sqrt(total)

    def predict_one(self, sample):
        if not self.x_train:
            raise ValueError("Fit the classifier before predicting")
        if len(sample) != len(self.x_train[0]) or any(not math.isfinite(value) for value in sample):
            raise ValueError("Prediction features must match the trained dimensions and be finite")
        # 先计算待预测样本与每个训练样本之间的距离。
        # First calculate the distance from the new sample to every training sample.
        distances = []
        for index, train_sample in enumerate(self.x_train):
            distance = self.euclidean_distance(sample, train_sample)
            distances.append((distance, self.y_train[index]))

        # 按距离从小到大排序，距离越小表示越相似。
        # Sort by distance from small to large; a smaller distance means greater similarity.
        distances.sort(key=lambda item: item[0])
        nearest_labels = [label for _, label in distances[:self.k]]

        # 取最近的 k 个邻居做多数投票，票数最高的类别就是预测结果。
        # Use majority voting among the nearest k neighbors; the most common label becomes the prediction.
        vote_count = Counter(nearest_labels)
        return vote_count.most_common(1)[0][0]

    def predict(self, x_test):
        # 批量预测测试集中的所有样本。
        # Predict all samples in the test set one by one.
        return [self.predict_one(sample) for sample in x_test]
