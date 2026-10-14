"""
Data processing module for Student Performance Classification System.
This file only handles dataset loading, cleaning, feature extraction,
normalization, and train/test split.
"""

import csv
import random
import math


class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.headers = []
        self.data = []
        # 只保留真正参与预测的四个数值特征。
        # Keep only the four numerical features that are actually used for prediction.
        self.feature_names = [
            "StudyTime",
            "AttendanceRate",
            "AssignmentCompletion",
            "PreviousGrade",
        ]
        # 最终要预测的目标标签。
        # This is the target label that the model tries to predict.
        self.label_name = "FinalResult"
        # 保存归一化时的最小值和最大值，后面预测单个新样本时还要继续用。
        # Store min and max values for normalization so new inputs can use the same scaling rule later.
        self.feature_mins = []
        self.feature_maxs = []
        self.cleaned_row_count = 0
        self.feature_means = []

    def load_data(self):
        """Load CSV file from disk."""
        try:
            # DictReader 会把每一行读成字典，后面按列名取值会更直观。
            # DictReader reads each row as a dictionary, which makes column-based access clearer.
            with open(self.file_path, "r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                self.headers = reader.fieldnames or []
                missing = set(self.feature_names + [self.label_name]) - set(self.headers)
                if missing:
                    raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
                self.data = [row for row in reader if row]
        except FileNotFoundError as error:
            raise FileNotFoundError("Dataset file was not found. Please check the path.") from error

    def handle_missing_values(self):
        """Replace missing feature values with column averages and remove rows with missing labels."""
        if not self.data:
            return

        averages = {}
        # 第一步先计算每一列特征的平均值，后面缺失值用平均值补。
        # First compute the average of each feature column, then use it to fill missing values.
        for feature in self.feature_names:
            values = []
            for row in self.data:
                raw_value = row.get(feature, "")
                try:
                    value = float(raw_value)
                    if math.isfinite(value):
                        values.append(value)
                except (TypeError, ValueError):
                    continue
            if not values:
                raise ValueError(f"Training feature has no valid numbers: {feature}")
            averages[feature] = sum(values) / len(values)
        self.feature_means = [averages[name] for name in self.feature_names]

        cleaned_data = []
        # 第二步真正清洗数据：补全缺失特征，删除无效标签。
        # Then actually clean the dataset: fill missing features and remove invalid labels.
        for row in self.data:
            for feature in self.feature_names:
                raw_value = row.get(feature, "")
                try:
                    row[feature] = float(raw_value)
                    if not math.isfinite(row[feature]):
                        row[feature] = averages[feature]
                except (TypeError, ValueError):
                    # Missing numerical feature values are replaced by column averages to keep the
                    # sample usable while preserving the general feature scale.
                    row[feature] = averages[feature]

            label = (row.get(self.label_name, "") or "").strip()
            # Rows with missing target labels are removed to avoid creating artificial class bias.
            if label in {"Pass", "Fail"}:
                row[self.label_name] = label
                cleaned_data.append(row)

        self.data = cleaned_data
        self.cleaned_row_count = len(self.data)

    def explain_features(self):
        # 这个方法主要服务于 GUI 中的“特征解释”按钮。
        # This method mainly supports the "Feature Explanation" button in the GUI.
        return {
            "StudyTime": "Hours spent studying. It reflects learning effort.",
            "AttendanceRate": "Percentage of classes attended. It reflects class participation.",
            "AssignmentCompletion": "Percentage of assignments completed. It reflects consistency and responsibility.",
            "PreviousGrade": "Previous academic result. It provides historical performance information.",
        }

    def dataset_summary(self):
        # 返回数据集的静态说明信息，方便展示项目背景。
        # Return static dataset information for presentation and explanation.
        return {
            "dataset": "student_performance.csv",
            "project_type": "Classification / Prediction",
            "features": ["StudyTime", "AttendanceRate", "AssignmentCompletion", "PreviousGrade"],
            "label": "FinalResult",
            "classes": ["Pass", "Fail"],
        }

    def extract_features_labels(self):
        """Extract selected features and the label from the dataset."""
        features = []
        labels = []

        # 将字典格式的原始数据，整理成模型更容易使用的：
        # features = [[x1, x2, x3, x4], ...]
        # labels = ["Pass", "Fail", ...]
        # Convert raw dictionary rows into the format the model needs:
        # a feature matrix and a label list.
        for row in self.data:
            try:
                features.append([float(row[feature]) for feature in self.feature_names])
                labels.append(row[self.label_name])
            except (KeyError, TypeError, ValueError):
                continue

        return features, labels

    def normalize_features(self, features):
        """Normalize numerical features using min-max normalization."""
        if not features:
            return []

        # 对每一列找最小值和最大值，供 min-max 归一化使用。
        # Find the min and max of each column for min-max normalization.
        feature_count = len(features[0])
        self.feature_mins = [min(row[index] for row in features) for index in range(feature_count)]
        self.feature_maxs = [max(row[index] for row in features) for index in range(feature_count)]

        normalized_features = []
        for row in features:
            normalized_row = []
            for index, value in enumerate(row):
                min_value = self.feature_mins[index]
                max_value = self.feature_maxs[index]
                if max_value == min_value:
                    # 如果一整列都相同，就没有区分度，统一记成 0。
                    # If all values in one column are identical, that feature has no spread, so map it to 0.
                    normalized_row.append(0)
                else:
                    # min-max 归一化把数值缩放到 [0, 1] 区间，避免大数值特征主导距离。
                    # Min-max normalization scales values into [0, 1] so large-scale features do not dominate distance.
                    normalized_row.append((value - min_value) / (max_value - min_value))
            normalized_features.append(normalized_row)

        return normalized_features

    def normalize_single_input(self, input_features):
        """Normalize one input sample using training data min and max values."""
        if len(self.feature_mins) != 4:
            raise ValueError("Prepare the dataset before predicting")
        if len(input_features) != 4 or any(not math.isfinite(float(value)) for value in input_features):
            raise ValueError("Provide four finite numerical features")
        normalized_input = []
        # 新输入不能重新单独算 min/max，而是必须沿用训练阶段保存下来的缩放规则。
        # A new input must reuse the training-time min/max values instead of computing its own.
        for index, value in enumerate(input_features):
            min_value = self.feature_mins[index]
            max_value = self.feature_maxs[index]
            if max_value == min_value:
                normalized_input.append(0)
            else:
                normalized_input.append((value - min_value) / (max_value - min_value))
        return normalized_input

    def train_test_split(self, features, labels, test_ratio=0.25, seed=42):
        """Split features and labels into training and testing data."""
        # 先把特征和标签打包，打乱后再一起切分，避免错位。
        # Zip features and labels together before shuffling so they stay aligned.
        if len(features) != len(labels) or len(features) < 2 or not 0 < test_ratio < 1:
            raise ValueError("Provide aligned samples and a test_ratio between 0 and 1")
        combined = list(zip(features, labels))
        random.Random(seed).shuffle(combined)

        # 例如 test_ratio=0.25，表示 75% 训练，25% 测试。
        # For example, test_ratio=0.25 means 75% training data and 25% testing data.
        split_index = int(len(combined) * (1 - test_ratio))
        if not 0 < split_index < len(combined):
            raise ValueError("Training and test sets must both be nonempty")
        train_data = combined[:split_index]
        test_data = combined[split_index:]

        x_train = [item[0] for item in train_data]
        y_train = [item[1] for item in train_data]
        x_test = [item[0] for item in test_data]
        y_test = [item[1] for item in test_data]
        return x_train, x_test, y_train, y_test

    def prepare_dataset(self, test_ratio=0.25, seed=42):
        """
        Complete data preparation pipeline:
        1. Load CSV file
        2. Remove invalid labels and split raw samples
        3. Fit missing-value means on training data only
        4. Fit min/max on imputed training data only
        5. Transform test samples with stored training statistics
        """
        # 这个方法把数据预处理完整串起来，主程序只需要调用一次即可。
        # This method chains the entire preprocessing pipeline so the main program only needs one call.
        self.load_data()
        valid_rows = [row for row in self.data if (row.get(self.label_name) or "").strip() in {"Pass", "Fail"}]
        for row in valid_rows:
            row[self.label_name] = row[self.label_name].strip()
        train_rows, test_rows, y_train, y_test = self.train_test_split(
            valid_rows, [row[self.label_name] for row in valid_rows], test_ratio, seed)
        self.data = train_rows
        self.handle_missing_values()
        training_features, _ = self.extract_features_labels()
        x_train = self.normalize_features(training_features)
        x_test = []
        for row in test_rows:
            values = []
            for index, name in enumerate(self.feature_names):
                try:
                    value = float(row.get(name, ""))
                    if not math.isfinite(value):
                        raise ValueError
                except (TypeError, ValueError):
                    value = self.feature_means[index]
                values.append(value)
            x_test.append(self.normalize_single_input(values))
        self.data = valid_rows
        self.cleaned_row_count = len(valid_rows)
        return x_train, x_test, y_train, y_test
