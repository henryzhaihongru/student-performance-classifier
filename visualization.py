"""
Visualization module for Student Performance Classification System.
This file only creates and saves charts.
"""

import os

import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        # 如果输出目录不存在，就自动创建，避免保存图表时报错。
        # Create the output directory automatically if it does not exist.
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_metrics(self, metrics, save_path="outputs/performance_metrics.png"):
        # 固定指标顺序，让图表展示更稳定，也更利于讲解。
        # Keep a fixed metric order so the chart is stable and easier to explain.
        metric_order = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "Sensitivity",
            "Specificity",
        ]
        names = [name for name in metric_order if name in metrics]
        values = [metrics[name] for name in names]

        plt.figure(figsize=(10, 5))
        plt.bar(names, values, color="#5B8FF9")
        # 这些指标理论上都在 0 到 1 之间，因此 y 轴固定到 1。
        # These metrics are theoretically between 0 and 1, so the y-axis is fixed at 1.
        plt.ylim(0, 1)
        plt.title("Performance Metrics")
        plt.xlabel("Metric")
        plt.ylabel("Score")

        # 在柱状图顶部标出具体数值，老师查看时更直观。
        # Add exact values on top of bars so the result is easier to read.
        for index, value in enumerate(values):
            plt.text(index, value + 0.02, f"{value:.2f}", ha="center")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_confusion_matrix(self, matrix, save_path="outputs/confusion_matrix.png"):
        # 这里把混淆矩阵整理成 2x2 形式，用热力图展示。
        # Convert the confusion matrix into a 2x2 layout and display it as a heatmap.
        values = [
            [matrix["TP"], matrix["FN"]],
            [matrix["FP"], matrix["TN"]],
        ]

        plt.figure(figsize=(6, 5))
        plt.imshow(values, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xticks([0, 1], ["Predicted Pass", "Predicted Fail"])
        plt.yticks([0, 1], ["Actual Pass", "Actual Fail"])

        # 在每个格子中央写出数量，避免只看颜色不够精确。
        # Put the exact count in each cell so the chart is precise, not only color-based.
        for row_index in range(2):
            for col_index in range(2):
                plt.text(col_index, row_index, str(values[row_index][col_index]), ha="center", va="center", fontsize=14)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
