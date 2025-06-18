import os
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from torchvision import transforms, datasets
from torchvision.models import resnet50, densenet121, vgg16
from tqdm import tqdm
from prettytable import PrettyTable
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from scipy.stats import chi2
import plotly.graph_objects as go
from plotly.offline import plot
from ResNetEMADropblock import *


class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
        self.fig_format = 'pdf'
        self.fig_dpi = 300

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def calculate_metrics(self):
        matrix = self.matrix
        total = np.sum(matrix)
        metrics = []

        for i in range(self.num_classes):
            TP = matrix[i, i]
            FP = np.sum(matrix[i, :]) - TP
            FN = np.sum(matrix[:, i]) - TP
            TN = total - TP - FP - FN

            precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            recall = TP / (TP + FN) if (TP + FN) != 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            class_acc = (TP + TN) / total if total != 0 else 0

            metrics.append({
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1': f1,
                'accuracy': class_acc
            })

        return metrics

    def summary(self):
        sum_TP = 0
        total = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / total
        print(f"Overall Accuracy: {acc:.4f}")

    def plot(self):
        matrix = self.matrix
        plt.figure(figsize=(10, 8), dpi=self.fig_dpi)
        plt.imshow(matrix, cmap=plt.cm.Blues)
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        plt.yticks(range(self.num_classes), self.labels)
        plt.colorbar()
        plt.xlabel('True Labels', fontsize=12)
        plt.ylabel('Predicted Labels', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14)

        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black",
                         fontsize=8)

        plt.tight_layout()
        plt.savefig(f'confusion_matrix.pdf', format=self.fig_format, dpi=self.fig_dpi, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved as 'confusion_matrix.{self.fig_format}'")


def plot_pca_with_ellipse(features_array, labels_array, class_names, confidence_level=0.9, figsize=(12, 8), dpi=300):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', 'D', '^', 'v']
    ellipse_params = {'alpha': 0.2, 'linewidth': 1.5}
    chi2_val = chi2.ppf(confidence_level, df=2)
    scaling = np.sqrt(chi2_val)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_array)

    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.linewidth'] = 1.5

    for i in range(len(class_names)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        mask = labels_array == i
        data = pca_result[mask]

        ax.scatter(data[:, 0], data[:, 1],
                   c=color, marker=marker, s=80,
                   edgecolors='w', linewidth=0.8,
                   label=f'{class_names[i]} (n={sum(mask)})')

        if len(data) > 1:
            mean = np.mean(data, axis=0)
            cov = np.cov(data.T)
            lambda_, v = np.linalg.eig(cov)
            lambda_ = np.sqrt(lambda_)
            theta = np.degrees(np.arctan2(*v[0][::-1]))

            ell = Ellipse(xy=mean,
                          width=lambda_[0] * 2 * scaling,
                          height=lambda_[1] * 2 * scaling,
                          angle=theta,
                          **ellipse_params)
            ell.set_facecolor(color)
            ax.add_artist(ell)

    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, frameon=False)

    explained_var = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f'PC1 ({explained_var[0]:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({explained_var[1]:.1f}%)', fontsize=12)
    ax.set_title(f'PCA with {confidence_level * 100}% Confidence Ellipses', fontsize=14, pad=15)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.savefig('5Class_PCA_Ellipses11.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("5-class PCA visualization saved as 5Class_PCA_Ellipses.pdf")


def plot_3d_pca(features_array, labels_array, class_names, figsize=(14, 10), dpi=300):
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_array)
    explained_var = pca.explained_variance_ratio_ * 100

    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.axes(projection='3d')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.linewidth'] = 1.5

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', 'D', '^', 'v']

    for i, (color, marker) in enumerate(zip(colors, markers)):
        mask = labels_array == i
        data = pca_result[mask]
        ax.scatter3D(data[:, 0], data[:, 1], data[:, 2],
                     c=color, marker=marker, s=40,
                     edgecolors='w', linewidth=0.5,
                     label=f'{class_names[i]} (n={sum(mask)})')

    ax.set_xlabel(f'PC1 ({explained_var[0]:.1f}%)', fontsize=12, labelpad=10)
    ax.set_ylabel(f'PC2 ({explained_var[1]:.1f}%)', fontsize=12, labelpad=10)
    zlabel = ax.set_zlabel(f'PC3 ({explained_var[2]:.1f}%)', fontsize=12, labelpad=10)

    ax.zaxis.set_rotate_label(False)
    zlabel.set_rotation(90)
    ax.get_zaxis().set_visible(True)

    ax.view_init(elev=20, azim=-45)
    ax.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95), frameon=True, fontsize=10)

    plt.savefig('3D_PCA.pdf', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print("3D PCA visualization saved as 3D_PCA.pdf")


def plot_3d_pca_ball(features_array, labels_array, class_names,
                     figsize=(14, 10), dpi=300, confidence_level=0.95):
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_array)
    explained_var = pca.explained_variance_ratio_ * 100

    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.axes(projection='3d')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.linewidth'] = 1.5

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', 'D', '^', 'v']

    chi2_val = chi2.ppf(confidence_level, df=3)
    light = LightSource(azdeg=135, altdeg=45)

    for i, (color, marker) in enumerate(zip(colors, markers)):
        mask = labels_array == i
        data = pca_result[mask]
        if len(data) < 4:
            continue

        ax.scatter3D(data[:, 0], data[:, 1], data[:, 2],
                     c=color, marker=marker, s=40,
                     edgecolors='w', linewidth=0.5,
                     label=f'{class_names[i]} (n={sum(mask)})')

        mu = np.mean(data, axis=0)
        cov = np.cov(data.T)
        eig_val, eig_vec = np.linalg.eigh(cov)
        order = eig_val.argsort()[::-1]
        eig_val, eig_vec = eig_val[order], eig_vec[:, order]
        radii = np.sqrt(chi2_val * eig_val)

        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

        ellipsoid = (eig_vec @ np.stack([x, y, z], axis=2).reshape(-1, 3).T).T
        ellipsoid = ellipsoid.reshape(x.shape + (3,)) + mu

        shade_factor = 0.2
        facecolor = color if len(color) != 5 else color + '80'
        if len(facecolor) == 7: facecolor += '80'

        ax.plot_surface(
            ellipsoid[..., 0], ellipsoid[..., 1], ellipsoid[..., 2],
            rstride=2, cstride=2,
            color=facecolor,
            shade=True,
            lightsource=light,
            alpha=0.2,
            edgecolor='none',
            zorder=0
        )

    ax.set_xlabel(f'PC1 ({explained_var[0]:.1f}%)', fontsize=12, labelpad=10)
    ax.set_ylabel(f'PC2 ({explained_var[1]:.1f}%)', fontsize=12, labelpad=10)
    zlabel = ax.set_zlabel(f'PC3 ({explained_var[2]:.1f}%)', fontsize=12, labelpad=10)

    ax.zaxis.set_rotate_label(False)
    zlabel.set_rotation(90)
    ax.get_zaxis().set_visible(True)

    ax.view_init(elev=7, azim=85)
    ax.set_box_aspect((1, 1, 1))

    ax.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95), frameon=True, fontsize=10, framealpha=0.9)

    plt.savefig('3D_PCA_ball.pdf', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()


def plot_interactive_3d_pca(features_array, labels_array, class_names):
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_array)

    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (color, class_name) in enumerate(zip(colors, class_names)):
        mask = labels_array == i
        data = pca_result[mask]
        fig.add_trace(go.Scatter3d(
            x=data[:, 0],
            y=data[:, 1],
            z=data[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=color,
                opacity=0.8,
                line=dict(width=0.5, color='white')
            ),
            name=f'{class_name} (n={len(data)})'
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)',
            zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2] * 100:.1f}%)'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(x=0.05, y=0.95, bgcolor='rgba(255,255,255,0.9)'),
        title='Interactive 3D PCA Visualization'
    )

    plot(fig, filename='3D_PCA_Interactive.html', auto_open=False)
    print("Interactive 3D PCA saved as 3D_PCA_Interactive.html")


def plot_roc_curve(y_true, y_score, class_names, figsize=(10, 8), dpi=300, format='pdf'):
    plt.figure(figsize=figsize, dpi=dpi)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.linewidth'] = 1.5

    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')

    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro Average (AUC = {roc_auc["macro"]:.3f})',
             color='navy', linestyle=':', linewidth=3)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, labelpad=10)
    plt.ylabel('True Positive Rate', fontsize=14, labelpad=10)
    plt.title('ROC Curves', fontsize=16, pad=20)
    plt.legend(loc="lower right", prop={'size': 12}, frameon=False)

    plt.savefig('ROC_Curve.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("ROC curve saved as ROC_Curve.pdf")
    return roc_auc  # 返回每个类别的AUC


def run_evaluation(round_idx, data_transform, image_path, model_weight_path, class_indict, labels,
                   confidence_threshold):
    """执行单轮评估并返回各类指标及AUC"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Round {round_idx + 1}/{num_rounds}, Device: {device}")

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"), transform=data_transform)

    # 为每轮设置不同的随机种子，使数据加载顺序不同
    validate_loader = torch.utils.data.DataLoader(
        validate_dataset,
        batch_size=batch_size,
        shuffle=True,  # 开启随机打乱
        num_workers=2,
        generator=torch.Generator().manual_seed(round_idx)  # 设置随机种子
    )

    num_classes = 5
    net = ResNet50WithDropBlockAndEMA(num_classes=num_classes)
    assert os.path.exists(model_weight_path), f"cannot find {model_weight_path} file"
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.to(device)

    confusion = ConfusionMatrix(num_classes=num_classes, labels=labels)

    # 低置信度样本保存目录（每轮单独保存）
    low_confidence_dir = f"low_confidence_samples_round_{round_idx + 1}"
    if not os.path.exists(low_confidence_dir):
        os.makedirs(low_confidence_dir)

    low_confidence_samples = []
    all_features = []
    all_labels = []
    all_probs = []

    net.eval()
    with torch.no_grad():
        for batch_idx, val_data in enumerate(tqdm(validate_loader)):
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))

            features = outputs
            if features.ndim == 1:
                features = features.reshape(1, -1)
            all_features.append(features)

            outputs_softmax = torch.softmax(outputs, dim=1)
            all_probs.append(outputs_softmax.cpu().numpy())
            all_labels.append(val_labels.cpu().numpy())

            confidences, preds = torch.max(outputs_softmax, dim=1)
            preds_np = preds.cpu().numpy()
            confidences_np = confidences.cpu().numpy()
            labels_np = val_labels.cpu().numpy()

            batch_size_current = val_images.size(0)
            global_indices = [batch_idx * batch_size + i for i in range(batch_size_current)]

            for i in range(batch_size_current):
                if confidences_np[i] < confidence_threshold:
                    sample_idx = global_indices[i]
                    file_path = validate_dataset.samples[sample_idx][0]
                    true_label = class_indict[str(labels_np[i])]
                    pred_label = class_indict[str(preds_np[i])]
                    is_correct = "Correct" if true_label == pred_label else "Incorrect"

                    low_confidence_samples.append({
                        "file_path": file_path,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "confidence": float(confidences_np[i]),
                        "is_correct": is_correct
                    })

                    image_name = f"{true_label}_{pred_label}_{confidences_np[i]:.2f}_{is_correct}_" + os.path.basename(
                        file_path)
                    save_path = os.path.join(low_confidence_dir, image_name)
                    shutil.copy(file_path, save_path)

            confusion.update(preds_np, labels_np)

    labels_array = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    if round_idx == num_rounds - 1:
        roc_auc = plot_roc_curve(all_labels, all_probs, labels)
    else:
        # 非最后一轮也计算AUC但不绘图
        n_classes = len(labels)
        y_true_bin = label_binarize(all_labels, classes=range(n_classes))
        roc_auc = {}
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
            roc_auc[i] = auc(fpr, tpr)

    print(
        f"\nRound {round_idx + 1}: Found {len(low_confidence_samples)} low-confidence samples (threshold={confidence_threshold}):")
    if low_confidence_samples and round_idx == num_rounds - 1:
        table = PrettyTable()
        table.field_names = ["File Path", "True Label", "Pred Label", "Confidence", "Is Correct"]
        for sample in low_confidence_samples:
            table.add_row([
                os.path.basename(sample["file_path"]),
                sample["true_label"],
                sample["pred_label"],
                f"{sample['confidence']:.4f}",
                sample["is_correct"]
            ])
        print(table)

        csv_path = f"low_confidence_samples_round_{round_idx + 1}.csv"
        with open(csv_path, "w") as f:
            f.write("file_path,true_label,pred_label,confidence,is_correct\n")
            for sample in low_confidence_samples:
                f.write(f"{sample['file_path']},{sample['true_label']},"
                        f"{sample['pred_label']},{sample['confidence']:.4f},"
                        f"{sample['is_correct']}\n")
        print(f"Saved details to {csv_path}")
    elif round_idx == num_rounds - 1:
        print("No low-confidence samples found")

    if round_idx == num_rounds - 1:
        confusion.plot()

    # 返回各类评估指标和AUC
    return confusion.calculate_metrics(), roc_auc


if __name__ == '__main__':
    num_rounds = 5  # 运行轮次
    batch_size = 16
    confidence_threshold = 0.85  # 置信度阈值

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))
    image_path = os.path.join(data_root, "data")
    assert os.path.exists(image_path), "data path {} does not exist.".format(image_path)

    model_weight_path = "best_finetune.pth"
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)

    json_label_path = './class_indices_LZW.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    with open(json_label_path, 'r') as f:
        class_indict = json.load(f)

    labels = [label for _, label in class_indict.items()]
    num_classes = len(labels)

    # 存储每轮的评估指标和AUC
    all_metrics = []
    all_auc = []

    # 运行评估
    for round_idx in range(num_rounds):
        print(f"\n===== Starting Round {round_idx + 1}/{num_rounds} =====")
        round_metrics, round_auc = run_evaluation(
            round_idx,
            data_transform,
            image_path,
            model_weight_path,
            class_indict,
            labels,
            confidence_threshold
        )
        all_metrics.append(round_metrics)
        all_auc.append(round_auc)
        print(f"===== Round {round_idx + 1} completed =====")

    # 计算每类指标的平均值和标准差
    metrics_names = ['accuracy', 'precision', 'recall', 'specificity', 'f1']
    class_metrics = {label: {metric: [] for metric in metrics_names} for label in labels}
    class_auc = {label: [] for label in labels}  # 存储每类AUC

    for round_idx, round_metrics in enumerate(all_metrics):
        round_auc = all_auc[round_idx]
        for class_idx, class_label in enumerate(labels):
            for metric in metrics_names:
                class_metrics[class_label][metric].append(round_metrics[class_idx][metric])
            # 存储AUC
            class_auc[class_label].append(round_auc[class_idx])

    # 生成统计结果表格
    table = PrettyTable()
    table.field_names = ["Class", "Metric"] + [f"Round {i + 1}" for i in range(num_rounds)] + ["Mean", "Std"]

    for class_label in labels:
        # 添加原有指标
        for metric in metrics_names:
            values = class_metrics[class_label][metric]
            mean = np.mean(values)
            std = np.std(values)
            row = [class_label, metric.capitalize()]
            row += [f"{v:.4f}" for v in values]
            row += [f"{mean:.4f}", f"{std:.4f}"]
            table.add_row(row)

        # 添加AUC指标
        auc_values = class_auc[class_label]
        auc_mean = np.mean(auc_values)
        auc_std = np.std(auc_values)
        row = [class_label, "AUC"]
        row += [f"{v:.4f}" for v in auc_values]
        row += [f"{auc_mean:.4f}", f"{auc_std:.4f}"]
        table.add_row(row)

    # 添加总体平均行
    overall_mean = {metric: [] for metric in metrics_names}
    overall_std = {metric: [] for metric in metrics_names}
    overall_auc_mean = []
    overall_auc_std = []

    for metric in metrics_names:
        all_values = []
        for class_label in labels:
            all_values.extend(class_metrics[class_label][metric])
        overall_mean[metric] = np.mean(all_values)
        overall_std[metric] = np.std(all_values)

    # 计算总体AUC
    all_auc_values = []
    for class_label in labels:
        all_auc_values.extend(class_auc[class_label])
    overall_auc_mean = np.mean(all_auc_values)
    overall_auc_std = np.std(all_auc_values)

    for metric in metrics_names:
        row = ["Overall", metric.capitalize()]
        row += ["-"] * num_rounds
        row += [f"{overall_mean[metric]:.4f}", f"{overall_std[metric]:.4f}"]
        table.add_row(row)

    # 添加总体AUC行
    row = ["Overall", "AUC"]
    row += ["-"] * num_rounds
    row += [f"{overall_auc_mean:.4f}", f"{overall_auc_std:.4f}"]
    table.add_row(row)

    print("\n===== Evaluation Metrics Statistics =====")
    print(table)

    final_table = PrettyTable()
    final_table.field_names = ["Metric", "Mean", "Std Dev", "Min", "Max"]
    final_table.align["Metric"] = "l"
    final_table.align["Mean"] = "r"
    final_table.align["Std Dev"] = "r"
    final_table.align["Min"] = "r"
    final_table.align["Max"] = "r"

    # 计算总体准确率
    overall_acc = [np.mean([class_metrics[class_label]['accuracy'][i] for class_label in labels])
                   for i in range(num_rounds)]
    final_table.add_row([
        "Overall Accuracy",
        f"{np.mean(overall_acc):.4f}",
        f"{np.std(overall_acc):.4f}",
        f"{min(overall_acc):.4f}",
        f"{max(overall_acc):.4f}"
    ])

    # 计算宏平均指标
    for metric in ['precision', 'recall', 'specificity', 'f1']:
        macro_values = [np.mean([class_metrics[class_label][metric][i] for class_label in labels])
                        for i in range(num_rounds)]
        final_table.add_row([
            f"Macro {metric.capitalize()}",
            f"{np.mean(macro_values):.4f}",
            f"{np.std(macro_values):.4f}",
            f"{min(macro_values):.4f}",
            f"{max(macro_values):.4f}"
        ])

    # 计算类平均准确率
    class_avg_acc = [np.mean([class_metrics[class_label]['accuracy'][i] for class_label in labels])
                     for i in range(num_rounds)]
    final_table.add_row([
        "Class-Average Accuracy",
        f"{np.mean(class_avg_acc):.4f}",
        f"{np.std(class_avg_acc):.4f}",
        f"{min(class_avg_acc):.4f}",
        f"{max(class_avg_acc):.4f}"
    ])

    # 计算宏平均AUC
    macro_auc_values = [np.mean([class_auc[class_label][i] for class_label in labels])
                        for i in range(num_rounds)]
    final_table.add_row([
        "Macro AUC",
        f"{np.mean(macro_auc_values):.4f}",
        f"{np.std(macro_auc_values):.4f}",
        f"{min(macro_auc_values):.4f}",
        f"{max(macro_auc_values):.4f}"
    ])

    print(final_table)