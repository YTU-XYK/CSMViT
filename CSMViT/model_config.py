def get_config(mode: str = "CMViT_Config") -> dict:
    if mode == "CMViT_Config":
        mv2_exp_mult = 2  # 将特征图的通道数翻多少倍
        config = {
            "layer1": {
                "out_channels": 24,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,    # block堆叠几次
                "stride": 2,
                "block_type": "mv2",
            },
            "layer2": {
                "out_channels": 24,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,    # 第一个下采样，后面两个不下采样
                "stride": 2,
                "block_type": "mv2",
            },
            "layer3": {  # 28x28
                "out_channels": 48,
                "transformer_channels": 80,
                "ffn_dim": 128,
                "transformer_blocks": 2,    # l = 2
                "patch_h": 2,  # 8,
                "patch_w": 2,  # 8,
                "stride": 2,   # 第一个mv2结构的下采样
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer4": {  # 14x14
                "out_channels": 96,
                "transformer_channels": 96,
                "ffn_dim": 160,
                "transformer_blocks": 4,
                "patch_h": 2,  # 4,
                "patch_w": 2,  # 4,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "last_layer_exp_factor": 4,
            "cls_dropout": 0.1
        }
    else:
        raise NotImplementedError

    for k in ["layer1", "layer2", "layer3", "layer4"]:
        config[k].update({"dropout": 0.1, "ffn_dropout": 0.0, "attn_dropout": 0.0})

    return config
