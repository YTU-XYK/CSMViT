from PIL import Image
import os
from tqdm import tqdm

def convert_rgba_to_rgb(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in tqdm(os.listdir(input_folder)):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 检查文件是否为图像文件
        if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # 打开图像
            image = Image.open(input_path)

            # 检查图像是否包含Alpha通道
            if 'A' in image.getbands():
                # 将RGBA格式的图像转换为RGB格式
                rgb_image = Image.new("RGB", image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3])
            else:
                # 图像不包含Alpha通道，直接保存为RGB格式
                rgb_image = image.convert('RGB')

            # 保存转换后的图像
            rgb_image.save(output_path)

            # print(f"Converted {filename} to RGB format")


if __name__ == "__main__":
    directory = r"D:\deep-learning\deep-learning-for-image-processing-master\data_set\cell_data\add_5.13\add"
    for root, dirs, files in os.walk(directory):
        for file in dirs:
            print(file)
            input_folder = fr"D:\deep-learning\deep-learning-for-image-processing-master\data_set\cell_data\add_5.13\add\{file}"
            output_folder = fr"D:\deep-learning\deep-learning-for-image-processing-master\data_set\cell_data\add_5.13\add_rgb\{file}"

            convert_rgba_to_rgb(input_folder, output_folder)
