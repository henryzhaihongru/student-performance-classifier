"""
Console demo for classroom presentation and screenshot capture.
"""

from pathlib import Path

from data_processing import DataProcessor
from metrics import MetricsCalculator
from model import DEFAULT_K, ManualKNNClassifier
from visualization import Visualizer


def main():
    # 控制台版本和 GUI 版本使用同一套核心逻辑，
    # 这样可以证明项目不是只做了界面，模型流程本身也能单独运行。
    # The console version and GUI version share the same core logic,
    # which shows that the project is not only a GUI and the model pipeline can run independently.
    processor = DataProcessor(Path(__file__).resolve().parent / "dataset" / "student_performance.csv")
    dataset_info = processor.dataset_summary()
    x_train, x_test, y_train, y_test = processor.prepare_dataset(test_ratio=0.25, seed=42)

    classifier = ManualKNNClassifier(k=DEFAULT_K)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    metrics_calculator = MetricsCalculator()
    matrix = metrics_calculator.confusion_matrix(y_test, y_pred)
    metrics = metrics_calculator.performance_metrics(matrix)

    visualizer = Visualizer(Path(__file__).resolve().parent / "outputs")
    visualizer.plot_metrics(metrics, save_path=visualizer.output_dir / "performance_metrics.png")
    visualizer.plot_confusion_matrix(matrix, save_path=visualizer.output_dir / "confusion_matrix.png")

    # 下面输出的是课堂展示时最常用的信息：数据规模、训练测试划分、指标结果。
    # The output below contains the most useful presentation information: dataset size, split sizes, and metrics.
    print("Student Performance Classification System")
    print(
        "Dataset source: Educational student performance dataset prepared for this course project. "
        "The dataset simulates common academic behavior indicators, including study time, attendance rate, "
        "assignment completion, and previous grade."
    )
    print(f"Dataset size: {len(processor.data)}")
    print(f"Features used: {', '.join(dataset_info['features'])}")
    print(f"Label: {dataset_info['label']}")
    print(f"Train size: {len(x_train)}")
    print(f"Test size: {len(x_test)}")
    print(f"K value: {classifier.k}")
    print("Confusion Matrix:")
    print(f"TP={matrix['TP']}, FP={matrix['FP']}, FN={matrix['FN']}, TN={matrix['TN']}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1-score: {metrics['F1-score']:.4f}")
    print(f"Sensitivity: {metrics['Sensitivity']:.4f}")
    print(f"Specificity: {metrics['Specificity']:.4f}")

    sample_inputs = [
        [15, 92, 95, 88],
        [3, 50, 45, 48],
        [8, 72, 70, 65],
    ]

    print("Sample Predictions:")
    # 再演示几个手动输入样本，让老师看到系统不仅能评估测试集，也能预测新数据。
    # Show several manual input samples so the teacher can see the system predicts new data as well.
    for sample in sample_inputs:
        normalized_sample = processor.normalize_single_input(sample)
        prediction = classifier.predict_one(normalized_sample)
        print(f"  Input {sample} -> {prediction}")


if __name__ == "__main__":
    main()
