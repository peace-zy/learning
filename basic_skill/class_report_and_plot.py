import pandas as pd
from collections import OrderedDict
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import csv

def get_data(execl_file):
    with open(execl_file, "rb") as f:
        df = pd.read_excel(f, header=[0, 1], sheet_name="Sheet1")
        #print(df.columns)
    data_dict = {}
    used_headers = None
    for index, row in df.iterrows():
        if used_headers is None:
            used_headers = list(row.标签.keys())
        image_file = row.图片类型.values[0] + "/" + row.图片名称.values[0]
        if row.标签.isna().all():
            continue
        data_dict[image_file] = OrderedDict({})
        for label, value in row.标签.items():
            n_v = 0 # 0: 无法确认, 1: 是, 2: 否
            if value == "是":
                n_v = 1
            elif value == "否":
                n_v = 2
            data_dict[image_file][label] = n_v
    return data_dict, used_headers
def main():
    gt_execl_file = "非标户型-标签标注-V1-GT.xlsx"
    pred_execl_file = "非标户型-标签标注-V1.xlsx"
    gt_dict, used_headers = get_data(gt_execl_file)
    pred_dict, _ = get_data(pred_execl_file)
    y_true = []
    y_pred = []
    for pre_f, pre_info in pred_dict.items():
        if pre_f not in gt_dict:
            continue
        y_true.append(list(gt_dict[pre_f].values()))
        y_pred.append(list(pre_info.values()))
    #print(y_true, y_pred)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    #print(y_true.shape)
    error_f = open("error.csv", "w")
    error_writer = csv.writer(error_f, delimiter="\t")
    error_writer.writerow(["标签", "准确率", "错误分类"])
    class_names = ["Uncertain", "Yes", "No"]
    for i in range(y_true_arr.shape[1]):
        y_true = y_true_arr[:, i]
        y_pred = y_pred_arr[:, i]
        used_labels = np.unique(np.concatenate((y_true, y_pred)))
        used_class_names = [class_names[idx] for idx in used_labels]
        class_report = classification_report(y_true, y_pred, labels=used_labels,
                                             target_names=used_class_names, zero_division=np.nan,
                                             output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)
        print(f"标签-{used_headers[i]}")
        print("<class_report>")
        print(class_report)
        print("<conf_matrix>")
        print(conf_matrix)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=used_class_names,
                    yticklabels=used_class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(f"confusion_matrix_{used_headers[i]}.png")
        misclassified_samples = {}

        errors = np.where(y_pred != y_true)[0]
        # 按真实类别分组错误数据
        for index in errors:
            true_label = y_true[index]
            predicted_label = y_pred[index]
            key = f"{class_names[true_label]}->{class_names[predicted_label]}"
            if key not in misclassified_samples:
                misclassified_samples[key] = []
            misclassified_samples[key].append(index)
        print(misclassified_samples)
        # 打印按类别分组的错误分类信息
        #error_f.write(f"标签-{used_headers[i]}\t{round(class_report['accuracy'] * 100, 2)}")
        error_info = []
        for key, misclassifications in misclassified_samples.items():
            print(f"{key}\t{misclassifications}")
            #error_f.write(f"\t{key}\t{misclassifications}")
            error_info += [f"{key}", f"{misclassifications}"]
        #error_f.write("\n")
        error_writer.writerow([f"标签-{used_headers[i]}", f"{round(class_report['accuracy'] * 100, 2)}"] + error_info)
        print("-" * 100)
    error_f.close()
    return

if __name__ == "__main__":
    main()
