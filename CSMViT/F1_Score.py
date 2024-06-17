import os
import json
import torch
from PIL import Image
from torchvision import transforms
from tqdm import trange
from model import mobile_vit_xx_small as create_model
import matplotlib.pyplot as plt

directory = r"D:\deep-learning\deep-learning-for-image-processing-master\data_set\cell_data\test"


def detect(i, threshold):
    fm = 0
    fz = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_size = 640
    data_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    imgs_root = os.path.join(directory, i)
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' does not exist."
    img_path_list = [os.path.join(imgs_root, img) for img in os.listdir(imgs_root) if img.endswith(".png")]

    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."
    with open(json_path, "r") as json_file:
        class_indict = json.load(json_file)

    model = create_model(num_classes=2).to(device)
    weights_path = "weights/cmvit.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' does not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()
    batch_size = 4

    with torch.no_grad():
        for ids in trange(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' does not exist."
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)

            batch_img = torch.stack(img_list, dim=0)
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            for idx, cla in enumerate(classes):
                img_path = img_path_list[ids * batch_size + idx]
                predicted_class = class_indict[str(cla.numpy())]
                fm += 1
                if float(predict[idx][1].item()) >= threshold:
                    fz += 1
    return fm, fz


def count():
    threshold = 0.5  # 可以根据需要调整阈值
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    total_images_class_0 = 0
    total_images_class_1 = 0

    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            fm, fz = detect(folder, threshold)
            if folder == '1':
                TP += fz
                FN += fm - fz
                total_images_class_1 += fm
            if folder == '0':
                FP += fz
                TN += fm - fz
                total_images_class_0 += fm

    # 计算每个类别的精确率和召回率
    precision_class_0 = TN / (TN + FN) if (TN + FN) > 0 else 0
    precision_class_1 = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall_class_0 = TN / (TN + FP) if (TN + FP) > 0 else 0
    recall_class_1 = TP / (TP + FN) if (TP + FN) > 0 else 0

    # 计算每个类别的F1-score
    f1_class_0 = 2 * (precision_class_0 * recall_class_0) / (precision_class_0 + recall_class_0) if (
                                                                                                                precision_class_0 + recall_class_0) > 0 else 0
    f1_class_1 = 2 * (precision_class_1 * recall_class_1) / (precision_class_1 + recall_class_1) if (
                                                                                                                precision_class_1 + recall_class_1) > 0 else 0

    # 计算宏平均F1-score
    total_f1_score = (f1_class_0 + f1_class_1) / 2

    # 计算总体的敏感性
    overall_sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0

    print(f"Total F1-score: {total_f1_score:.3f}")
    print(f"Overall Sensitivity: {overall_sensitivity:.3f}")


if __name__ == '__main__':
    count()
