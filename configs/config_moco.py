norm_cfg = dict(type="BN", requires_grad=True)
# pretrain_path = 'open-mmlab://resnet50'    # Please set the path to pretrained weights for Quick Tuning
pretrain_path = "torchvision://resnet50"  # Please set the path to pretrained weights for Quick Tuning

model = dict(
    type="EncoderDecoder",
    # pretrained=pretrain_path,
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint=pretrain_path),
        contract_dilation=False,
    ),
    decode_head=dict(
        type="FCNHead",
        num_convs=0,
        in_channels=2048,
        in_index=3,
        channels=2048,
        num_classes=2,
        norm_cfg=norm_cfg,
    ),
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
