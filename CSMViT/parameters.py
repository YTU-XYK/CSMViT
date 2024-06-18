from model import mobile_vit_xx_small as create_model

model = create_model(num_classes=2).to(0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 使用示例
total_params = count_parameters(model)
print(f"Total trainable parameters: {total_params}")
