norm_cfg = dict(type='SyncBN', requires_grad=True)
pretrain_path = None  # Update this path

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='VisionTransformer',
        img_size=224,
        patch_size=16,
        in_channels=3,  # Number of input channels
        embed_dims=768,  # Note the change from `embed_dim` to `embed_dims`
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cls_token=True,
        output_cls_token=False,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        patch_norm=False,
        final_norm=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrain_path),
    ),
    decode_head=dict(
    type='UPerHead',
    in_channels=[768],  # Multiple output channels
    in_index=[0],  # Indices for the selected layers
    pool_scales=(1, 2, 3, 6),
    channels=512,
    dropout_ratio=0.1,
    num_classes=2,
    norm_cfg=norm_cfg,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,  # Should match the output feature dimensions
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,  # Change to match your dataset
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
    ),
    # Model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)




optimizer = dict(
    type='AdamW',
    lr=3e-4,  # Starting learning rate, adjust as needed
    betas=(0.9, 0.95),
    weight_decay=0.05,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# Learning rate scheduler
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0)