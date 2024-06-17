import torch
from torch import nn
from model import mobile_vit_xx_small as create_model
from torchviz import make_dot
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

model_dict = torch.load(
    r"D:\deep-learning\deep-learning-for-image-processing-master\pytorch_classification\MobileViT\weights"
    r"\best_model640-train.pth")
model = create_model(num_classes=2).to(device)
model.load_state_dict(model_dict)
model = model.eval().to(device)

x = torch.randn(1, 3, 640, 640).to(device)
y = model(x)
make_dot(y, params=dict(model.named_parameters()))
dot = make_dot(y, params=dict(model.named_parameters()))
dot.format = 'png'
dot.render("mobile_vit_graph")  # 保存模型图为PNG文件
plt.show()