import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_table
from model import mobile_vit_xx_small as create_model

# 选择一个示例模型
model = create_model(num_classes=2).to('cuda:0')

# 创建一个示例输入
input_tensor = torch.randn(1, 3, 640, 640).to('cuda:0')

# 计算FLOPs
flop_analyzer = FlopCountAnalysis(model, input_tensor)
flops = flop_analyzer.total()

# 转换为Python内置整数类型
flops = int(flops)

# 转换为GFLOPs
gflops = flops / 1e9

# 打印FLOPs和GFLOPs
print(f"Total FLOPs: {flops}")
print(f"Total GFLOPs: {gflops:.3f}")
print(flop_count_table(flop_analyzer))
