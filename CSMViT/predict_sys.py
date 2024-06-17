import os
import json
import PIL
import torch
from PIL import Image
from tqdm import trange
from torchvision import transforms
from model import CMViT_Config as create_model
from PIL import Image, ImageFilter
from histolab.slide import Slide
from histolab.tiler import GridTiler
from histolab.masks import TissueMask
from histolab.filters.image_filters import (
    ApplyMaskImage,
    GreenPenFilter,
    Invert,
    OtsuThreshold,
    RgbToGrayscale,
)
from histolab.filters.morphological_filters import RemoveSmallHoles, RemoveSmallObjects
import time
import re

# 遍历的文件夹路径
directory = r'E:\PTC-ln\png\processed_rgb'
# 设置保存路径
BASE_PATH = 'E:/PTC-ln/png'
PROCESS_PATH_CUSTOM = os.path.join(BASE_PATH, 'processed_rgb')


def detect(img_folder):
    num = []
    pro = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_size = 640
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # 指向需要遍历预测的图像文件夹
    imgs_root = f"E:\PTC-ln\png\processed_rgb/{img_folder}"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".png")]

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = create_model(num_classes=2).to(device)
    # load model weights
    weights_path = "weights/best_data_4.14_1.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    batch_size = 8  # 每次预测时将多少张图片打包成一个batch

    with torch.no_grad():
        for ids in trange(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)

            # batch img
            # 将img_list列表中的所有图像打包成一个batch
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            i = 0
            for idx, cla in enumerate(classes):
                img_path = img_path_list[ids * batch_size + idx]
                predicted_class = class_indict[str(cla.numpy())]
                # 仅保存预测类别到num数组中
                num.append(predicted_class)
                if predicted_class == '1':
                    pro.append(predict[i][cla].item())  # 将class为1的概率添加到pro数组中
                i += 1
    return num, pro


def cut(your_tiff_image_path, img_folder):
    # 创建 Slide 对象
    custom_slide = Slide(your_tiff_image_path, processed_path=PROCESS_PATH_CUSTOM)

    # 打印 Slide 信息
    print(f"Slide name: {custom_slide.name}")
    print(f"Levels: {custom_slide.levels}")
    print(f"Dimensions at level 0: {custom_slide.dimensions}")
    print(f"Dimensions at level 1: {custom_slide.level_dimensions(level=1)}")
    print(f"Dimensions at level 2: {custom_slide.level_dimensions(level=2)}")

    # 创建 GridTiler 对象
    grid_tiles_extractor = GridTiler(
        tile_size=(640, 640),
        level=0,
        check_tissue=False,
        pixel_overlap=0,
        prefix=f"{img_folder}/",
        suffix=".png"
    )
    # 创建组织 Slide 对象和掩膜

    mask = TissueMask(
        RgbToGrayscale(),
        OtsuThreshold(),
        ApplyMaskImage(custom_slide.thumbnail),
        GreenPenFilter(),
        RgbToGrayscale(),
        Invert(),
        OtsuThreshold(),
        RemoveSmallHoles(),
        RemoveSmallObjects(),
    )
    custom_slide.locate_mask(mask, scale_factor=64)
    # 切割并保存图片
    grid_tiles_extractor.locate_tiles(
        slide=custom_slide,
        scale_factor=64,
        alpha=256,
        outline="#046C4C",
    )
    # 保存切割的图片
    grid_tiles_extractor.extract(custom_slide, mask)
    #  分割示意图部分
    pic, grid = grid_tiles_extractor.locate_tiles(
        slide=custom_slide,
        extraction_mask=mask,
        scale_factor=64,
        alpha=256,
        outline="#046C4C", )
    pic_list.append(pic)

    # print("列表的形状:", [len(sublist) for sublist in pic_list])

    # print(pic_list)
    grid.save(f"E:\PTC-ln\png\grid\{img_folder}.png")
    return pic_list, grid


def all(pic_list, img_folder, num_p):
    num, pro = detect(img_folder)
    j = 0
    n = 0
    ill_1 = []
    # print(pro)
    img = Image.open(f"E:\PTC-ln\png\grid\{img_folder}.png")
    img_i = Image.open(f"E:\PTC-ln\png\grid\{img_folder}.png")
    draw = PIL.ImageDraw.Draw(img)
    if pro:
        # print("num:",len(num))
        for i in range(len(num)):
            if num[i] == '1':
                if float(pro[n]) >= float(0.5):
                    x1 = y1 = x2 = y2 = 0
                    number = re.findall(r'\d+', pic_list[num_p][i])
                    # print("all:",num_p)
                    x1 = int(number[0])
                    y1 = int(number[1])
                    x2 = int(number[2])
                    y2 = int(number[3])
                    draw.rectangle(((x1, y1), (x2, y2)), outline='red', width=10)
                    j = j + 1
                    i += 1
                    n = n + 1
                else:
                    n = n + 1
                    i = i + 1
            else:
                i += 1
    if j > 2:
        print(f'有{j}处可疑')
        print('该图为阳性')
        ill_1.append(img_folder)
        img.save(rf'E:\PTC-ln\png\ROC\0.5\test-{img_folder}.png')
    else:
        print('该图为阴性')
        # img.save(rf'E:\PTC-ln\png\red_grid\test-{img_folder}.png')
        img.save(rf'E:\PTC-ln\png\ROC\0.5\test-{img_folder}.png')
    return ill_1, j


def save_list():
    with open(full_path, 'w', encoding='utf-8') as file:
        for item in pic_list:
            # 将列表项转换为字符串并写入文件，每个项后面跟一个换行符
            file.write("%s\n" % item)


def read_list(num_p):
    with open(full_path, 'r', encoding='utf-8') as file:
        # 读取文件的每一行，并去除每行末尾的换行符
        lines = file.readlines()
        pic = [line.strip().split(']') for line in lines]
        # 将每一行分割成一个列表，这里假设元素之间用逗号分隔
        pic_list = [line[0].strip().split('),') for line in pic]
    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            img_folder = folder
            print(img_folder)
            a, b = all(pic_list, img_folder, num_p)
            # print("for:",num_p)
            num_p = num_p + 1
            if b > 2:
                ill.append(a)


if __name__ == '__main__':
    start_time = time.time()

    tiff_path = r'E:\PTC-ln\to_tiff\300'
    ill = []
    num_p = 0
    pic_list = []
    full_path = r'E:\PTC-ln\png\list\my_list1.txt'  # save coordinate
    for root, dirs, files in os.walk(tiff_path):
        for file in files:
            img_folder = file[:-5]
            print(img_folder)
            your_tiff_image_path = f"E:/PTC-ln/to_tiff/300/{img_folder}.tiff"
            pic_list, grid = cut(your_tiff_image_path, img_folder)
    save_list()
    read_list(num_p)
    print(ill)
    print(len(ill))
    # 再次获取当前时间
    end_time = time.time()
    # 计算并打印运行时间
    print("程序运行时间: ", end_time - start_time, "秒")
