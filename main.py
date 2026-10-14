"""
Main GUI program.
This file only controls the GUI and workflow calls.
"""

import math
import tkinter as tk
from tkinter import messagebox

from pathlib import Path

from data_processing import DataProcessor
from metrics import MetricsCalculator
from model import DEFAULT_K, ManualKNNClassifier, explain_model
from visualization import Visualizer


class StudentPerformanceApp:
    def __init__(self, root):
        # root 是 Tkinter 的主窗口对象，整个界面都挂在这个窗口上。
        # root is the Tkinter main window object, and the whole GUI is attached to it.
        self.root = root
        self.root.title("Student Performance Classification System")
        self.root.geometry("800x850")
        self.root.minsize(700, 800)

        # 下面四个对象分别负责数据、模型、评价指标、图表。
        # The four objects below handle data, model, metrics, and visualization separately.
        # 这样拆开后，每个文件只做一件事，项目结构更清晰，也方便答辩时逐个解释。
        # This modular design keeps each file focused on one job and makes the project easier to explain.
        self.processor = DataProcessor(Path(__file__).resolve().parent / "dataset" / "student_performance.csv")
        self.classifier = ManualKNNClassifier(k=DEFAULT_K)
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer(Path(__file__).resolve().parent / "outputs")
        self.dataset_info = {}
        self.model_info = explain_model()

        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        self.y_pred = []
        self.matrix = {}
        self.metrics = {}

        # 程序启动时先完成训练和测试，后面 GUI 就可以直接显示指标并接受用户输入预测。
        # The program trains and evaluates first so the GUI can immediately show metrics and accept predictions.
        self.train_and_evaluate_model()
        self.build_gui()

    def train_and_evaluate_model(self):
        """
        Main workflow:
        data processing -> model training -> prediction -> metrics -> visualization
        """
        # 先获取数据集说明信息，主要用于展示或答辩解释。
        # First collect dataset summary information for display and presentation.
        self.dataset_info = self.processor.dataset_summary()

        # Split first; fit imputation and scaling using training samples only.
        self.x_train, self.x_test, self.y_train, self.y_test = self.processor.prepare_dataset(test_ratio=0.25, seed=42)

        # KNN 的“训练”不是学习参数，而是把训练样本保存起来，供后面距离比较使用。
        # In KNN, "training" does not learn weights; it stores training samples for later distance comparison.
        self.classifier.fit(self.x_train, self.y_train)

        # 用测试集做预测，后续再和真实标签比较，计算模型表现。
        # Predict on the test set first, then compare with true labels to evaluate performance.
        self.y_pred = self.classifier.predict(self.x_test)
        self.matrix = self.metrics_calculator.confusion_matrix(self.y_test, self.y_pred)
        self.metrics = self.metrics_calculator.performance_metrics(self.matrix)

        # 将结果保存成图片，便于展示项目成果。
        # Save charts as image files so the results are easy to present.
        self.visualizer.plot_metrics(self.metrics, save_path=self.visualizer.output_dir / "performance_metrics.png")
        self.visualizer.plot_confusion_matrix(self.matrix, save_path=self.visualizer.output_dir / "confusion_matrix.png")

    def build_gui(self):
        # 标题区：让老师一眼看到项目名称和主题。
        # Title section: lets the teacher immediately see the project name and topic.
        title = tk.Label(self.root, text="Student Performance Classification System", font=("Arial", 18, "bold"))
        title.pack(pady=15)

        intro_label = tk.Label(
            self.root,
            text="Manual KNN Classification Demo for Classroom Presentation",
            font=("Arial", 11, "italic"),
        )
        intro_label.pack(pady=2)

        dataset_note = tk.Label(
            self.root,
            text="Dataset Source: Educational student performance dataset prepared for this course project.",
            font=("Arial", 9),
            wraplength=640,
            justify="center",
        )
        dataset_note.pack(pady=4)

        feature_note = tk.Label(
            self.root,
            text="Features: StudyTime, AttendanceRate, AssignmentCompletion, PreviousGrade",
            font=("Arial", 10),
        )
        feature_note.pack(pady=5)

        workflow_note = tk.Label(
            self.root,
            text="Workflow: Data Processing -> KNN Training -> Prediction -> Metrics -> Visualization",
            font=("Arial", 10),
        )
        workflow_note.pack(pady=3)

        # form_frame 用于承载输入框，便于统一布局。
        # form_frame holds the input fields so the layout is easier to manage.
        form_frame = tk.Frame(self.root)
        form_frame.pack(pady=10)

        self.entries = {}
        fields = [
            ("Study Time (hours/week)", "study_time"),
            ("Attendance Rate (%)", "attendance"),
            ("Assignment Completion (%)", "assignment"),
            ("Previous Grade", "previous_grade"),
        ]

        # 循环创建四个输入项，避免重复写四遍相似代码。
        # Create the four input rows in a loop to avoid repeating similar code.
        for row_index, (label_text, key) in enumerate(fields):
            label = tk.Label(form_frame, text=label_text, font=("Arial", 12))
            label.grid(row=row_index, column=0, sticky="w", pady=6, padx=8)

            entry = tk.Entry(form_frame, width=25, font=("Arial", 12))
            entry.grid(row=row_index, column=1, pady=6, padx=8)
            self.entries[key] = entry

        # 预测按钮：读取用户输入并调用模型完成分类。
        # Prediction button: reads user input and calls the model for classification.
        predict_button = tk.Button(
            self.root,
            text="Predict Pass / Fail",
            command=self.predict_student,
            font=("Arial", 12, "bold"),
            width=22,
        )
        predict_button.pack(pady=12)

        # 解释按钮：不是模型必须功能，但对课堂演示很有帮助。
        # Explanation buttons are not required for prediction, but they are useful in a classroom demo.
        explain_button = tk.Button(
            self.root,
            text="Show Feature Explanation",
            command=self.show_feature_explanations,
            font=("Arial", 11),
            width=22,
        )
        explain_button.pack(pady=5)

        workflow_button = tk.Button(
            self.root,
            text="Workflow Explanation",
            command=self.show_workflow_explanation,
            font=("Arial", 11),
            width=22,
        )
        workflow_button.pack(pady=5)

        self.result_label = tk.Label(self.root, text="", font=("Arial", 15, "bold"))
        self.result_label.pack(pady=10)

        # 界面中直接展示测试集上的评价指标，让老师打开 GUI 就能看到结果。
        # Show evaluation metrics directly in the GUI so results are visible as soon as the app opens.
        metrics_label = tk.Label(self.root, text=self.create_metrics_text(), justify="left", font=("Arial", 11))
        metrics_label.pack(pady=10)

        note = tk.Label(self.root, text="Charts are saved in the outputs folder.", font=("Arial", 10))
        note.pack(pady=5)

    def create_metrics_text(self):
        # 这个方法负责把混淆矩阵和指标整理成适合放在界面上的字符串。
        # This method formats the confusion matrix and metrics into GUI-friendly text.
        return (
            "Confusion Matrix:\n"
            f"TP: {self.matrix['TP']}   FP: {self.matrix['FP']}   FN: {self.matrix['FN']}   TN: {self.matrix['TN']}\n\n"
            "Performance Metrics:\n"
            f"Accuracy: {self.metrics['Accuracy']:.2f}\n"
            f"Precision: {self.metrics['Precision']:.2f}\n"
            f"Recall: {self.metrics['Recall']:.2f}\n"
            f"F1-score: {self.metrics['F1-score']:.2f}\n"
            f"Sensitivity: {self.metrics['Sensitivity']:.2f}\n"
            f"Specificity: {self.metrics['Specificity']:.2f}"
        )

    def predict_student(self):
        try:
            # 从四个输入框中读取用户输入，并转成浮点数。
            # Read user input from the four fields and convert values to float.
            study_time = float(self.entries["study_time"].get())
            attendance = float(self.entries["attendance"].get())
            assignment = float(self.entries["assignment"].get())
            previous_grade = float(self.entries["previous_grade"].get())

            # 这里做基础合法性检查，避免负数或超出 100 的百分比直接进入模型。
            # Basic validation is used here so invalid values do not go into the model.
            if not all(math.isfinite(value) for value in [study_time, attendance, assignment, previous_grade]):
                raise ValueError
            if study_time < 0 or attendance < 0 or assignment < 0 or previous_grade < 0:
                raise ValueError
            if attendance > 100 or assignment > 100 or previous_grade > 100:
                raise ValueError

            user_input = [study_time, attendance, assignment, previous_grade]

            # 用户输入必须使用和训练数据相同的归一化规则，
            # 否则距离计算会失真，预测结果也会不可靠。
            # User input must use the same normalization rule as the training data,
            # otherwise the distance calculation will be distorted and prediction becomes unreliable.
            normalized_input = self.processor.normalize_single_input(user_input)
            prediction = self.classifier.predict_one(normalized_input)
            self.result_label.config(text=f"Prediction Result: {prediction}")
        except ValueError:
            messagebox.showerror(
                "Invalid Input",
                "Please enter valid numbers. Percentages and grades should be between 0 and 100.",
            )

    def show_feature_explanations(self):
        # 将特征说明拼接成文本，弹窗展示给用户。
        # Combine feature explanations into one message and show them in a popup.
        explanations = self.processor.explain_features()
        message = "\n\n".join(f"{name}: {text}" for name, text in explanations.items())
        messagebox.showinfo("Feature Explanation", message)

    def show_workflow_explanation(self):
        # 用一句话总结项目流水线，方便课堂展示时快速说明整体过程。
        # Summarize the whole workflow in one sentence for quick classroom explanation.
        message = (
            "Load CSV -> validate labels -> extract features and label -> "
            "split train/test -> fit training imputation and scaling -> train KNN -> predict -> "
            "evaluate -> visualize."
        )
        messagebox.showinfo("Workflow Explanation", message)


if __name__ == "__main__":
    # 只有直接运行 main.py 时才启动图形界面。
    # Start the GUI only when main.py is run directly.
    root = tk.Tk()
    app = StudentPerformanceApp(root)
    root.mainloop()
