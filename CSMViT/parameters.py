from model import mobile_vit_xx_small as create_model

model = create_model(num_classes=2).to(0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 使用示例
total_params = count_parameters(model)
print(f"Total trainable parameters: {total_params}")
'''
    Flow_Attention   202390  
    model  xx_small  951666
    model  small    4938914
    model  x_small  1933618
    New_Backbone     649537
    Flow_Block       511505
    NEW_Backbone_GAM 728377
    NEW_Inception   1039781
    MyAttention     1045047
'''
