default_scope = 'mmpretrain'
custom_imports = dict(imports='mm_redecology', allow_failed_imports=False)

# model configs
qwen_cfg_model_name = '/mnt/ali-sh-1/dataset/zeus/cache/modelscope/hub/qwen/Qwen2-VL-7B-Instruct/'
# init_cfg = dict(
#     type='Pretrained',
#     checkpoint='work_dirs/audit-mid-3img_241105_qwen2-1.5b_siglipso/epoch_8_fp32.pth',
# )

init_cfg =  dict()
data_preprocessor = dict(
    type='MultiImgClsDataPreprocessor',
    mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
    std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
    to_rgb=True,
    pad_value=0,
    non_blocking=True,
)

# data configs
batch_size_per_gpu = 2
num_workers = 8
persistent_workers = True

dataset_type = 'NewsMultiLabelDataset_jsonl'


level1_categories = [
    "Politics", "Environment", "Economy", "Safety", "Culture",
    "Entertainment", "Technology", "Education", "Health",
    "Sport", "Daily Life", "Infrastructure",
]



level2_categories = [
    # 0. Politics  (34)
    "US Election", "England Election", "Wales Election", "Scotland Election",
    "Northern Ireland Election", "South Africa Election", "French Election",
    "European Election", "Indian Election", "Parliament", "Assembly",
    "National Defense Ministry", "Justice Ministry", "Foreign Affairs Ministry",
    "Education Ministry", "Transport Ministry", "Agriculture Ministry",
    "Health Ministry", "Customs", "Finance Ministry", "Cabinet Office",
    "Interior Ministry", "Prosecutors Office", "Executive Branch",
    "US Congress", "Courts", "Immigration Policy", "United Nations",
    "European Union", "IMF", "WHO", "NATO", "WTO", "LGBT",

    # 1. Environment (29)
    "Extreme Weather", "Flood", "Drought", "Heatwave", "Storm", "Earthquake",
    "Hurricane", "Climate Change", "Land Animals", "Birds", "Plants",
    "Marine Life", "Woodlands", "Wetlands", "Trees", "Ocean", "River",
    "Forest", "National Park", "Wind Energy", "Solar Energy",
    "Hydrogen Energy", "Energy Storage", "Recycling", "Waste Management",
    "Carbon Management", "Green Economy", "Minerals", "Pollution",

    # 2. Economy (20)
    "Brexit Impact", "Inflation", "Recession", "Global Trade", "Stocks",
    "Bonds", "Exchange Rates", "Cost Of Living", "Personal Finance",
    "Mortgages", "Poverty", "Unemployment", "Employment", "Wages",
    "Gender Pay Gap", "Taxation", "Cryptocurrency", "Startups",
    "Supply And Demand", "Revenue",

    # 3. Safety (22)
    "Cyber Attacks", "Surveillance", "Espionage", "Fraud", "Theft",
    "Smuggling", "Human Trafficking", "Domestic Violence", "Sexual Violence",
    "Sexual Harassment", "Murder", "Arson", "Hate Crime", "Drug Trafficking",
    "Money Laundering", "Financial Crime", "Identity Theft", "Judicial System",
    "Parole Board", "Human Rights", "Terrorism", "War",

    # 4. Culture (12)
    "Classical Music", "Opera", "Ballet", "Contemporary Art", "Street Art",
    "Painting", "Sculpture", "Museum", "Antique Collection",
    "Archaeological Excavation", "Historic Site Restoration", "Human History",

    # 5. Entertainment (20)
    "Pop Music", "Hip Hop Music", "Live Music", "African Music",
    "Music Festival", "Film", "Bollywood", "Hollywood", "Oscar Awards",
    "TV Series", "Reality Show", "Performance", "Jazz", "Dance", "Comedy",
    "Drama", "Musical Theatre", "Pantomime", "Video Game", "Photography",

    # 6. Technology (34)   ← 少了 Nanotechnology
    "Artificial Intelligence", "Robotics", "Electric Vehicles",
    "Renewable Energy Tech", "Hydrogen Vehicles", "Battery",
    "Semiconductor Chips", "Quantum Computing", "5G", "Virtual Reality",
    "Augmented Reality", "Genetic Engineering", "3D Printing", "Drone",
    "Aerospace Tech", "Mars", "Moon", "Space Tourism", "ISS", "Astronaut",
    "Astronomy", "Universe", "Planetary Science", "Comet", "Biology",
    "Physics", "Chemistry", "Earth Science", "Genetics", "Medical Research",
    "Climate Science", "Marine Science", "Computer Science", "Blockchain",

    # 7. Education (20)
    "Primary School", "Secondary School", "Continuing Education",
    "Higher Education", "University", "Apprenticeship", "Student Finance",
    "Teacher", "Exam", "Education Policy", "School Meals", "Special Education",
    "Early Childhood Education", "Homeschooling", "Parenting", "Child Health",
    "Children's Rights", "Child Poverty", "Adolescent Mental Health",
    "Social Media Impact",

    # 8. Health (51)  
    "Mental Health", "Cancer", "Dementia", "Diabetes", "Heart Disease",
    "Stroke", "Obesity", "Eating Disorders", "Depression", "Anxiety",
    "Parkinson's Disease", "ADHD", "Autism", "Down Syndrome",
    "Cerebral Palsy", "COVID 19", "Monkeypox", "Measles", "Influenza",
    "Ebola", "Respiratory Disease", "Sepsis", "HIV AIDS", "Hepatitis",
    "Fertility", "Pregnancy", "Childbirth", "Menopause", "Vaccination",
    "Blood Donation", "Organ Donation", "Transplant", "Medical Cannabis",
    "Prescription Drugs", "Nutrition", "Alcoholism", "Smoking", "Drug Abuse",
    "Misogyny", "Learning Disabilities", "Pharmaceutical Industry",
    "Nursing", "Midwifery", "Dentistry", "Nursing Home", "Elderly Care",
    "Dermatological Disease", "Chronic Disease", "Genetic Disease",
    "Cardiovascular Disease", "Gastrointestinal Disease",

    # 9. Sport (34)
    "Archery", "Athletics", "Badminton", "Basketball", "Boxing", "Canoeing",
    "Cycling", "Diving", "Equestrian", "Fencing", "Field Hockey",
    "Football (Soccer)", "Golf", "Gymnastics", "Judo", "Rowing", "Rugby",
    "Sailing", "Shooting", "Skateboarding", "Sport Climbing", "Surfing",
    "Swimming", "Table Tennis", "Taekwondo", "Tennis", "Triathlon",
    "Volleyball", "Weightlifting", "Wrestling", "Billiards", "Motor Racing",
    "Baseball", "Esports",

    # 10. Daily Life (12)
    "Rescue Operations", "Safety Incidents", "Social Security",
    "Population Aging", "Gender Equality", "Consumer Shopping", "Cuisine",
    "Family Life", "Transportation", "Charitable Welfare", "Commemorations",
    "Festival Activities",

    # 11. Infrastructure (15)
    "Highway", "Railway", "Port", "Power System", "Water Conservancy",
    "Commercial Development", "Airport Development", "Smart City",
    "Medical Facilities", "Educational Infrastructure",
    "Cultural And Sports Facilities", "Community Service Facilities",
    "Government Agency Infrastructure",
    "Environmental Protection Infrastructure",
    "Disaster Prevention And Reconstruction",
]

# 3️⃣ child_to_parent（len = 303）
child_to_parent = (
    [0]*34   +   # Politics
    [1]*29   +   # Environment
    [2]*20   +   # Economy
    [3]*22   +   # Safety
    [4]*12   +   # Culture
    [5]*20   +   # Entertainment
    [6]*34   +   # Technology
    [7]*20   +   # Education
    [8]*51   +   # Health
    [9]*34   +   # Sport
    [10]*12  +   # Daily Life
    [11]*15      # Infrastructure
)


metainfo = dict(level1_categories=level1_categories, level2_categories=level2_categories)
num_classes_1 = len(level1_categories)
num_classes_2 = len(level2_categories)

indices = None

img_size = (384, 384)
num_imgs = 5

# model
model = dict(
    init_cfg=init_cfg,
    type='NewsIMGQWEN2TEXTFeatFusionClassifier',  # 使用 Qwen 模型的分类器
    freeze_backbone=False,
    data_preprocessor=data_preprocessor,
    is_pooling_feats=False,
    backbone=dict(
        type='HFSiglipVision',  # 使用 Qwen 的视觉骨干
        hf_pretrain_name='/root/.cache/modelscope/hub/AI-ModelScope/siglip-so400m-patch14-384/',
        lora_cfg=dict(
            r=32,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
        ),
        init_cfg=None,  # 不使用预训练权重初始化
        vision_project=dict(type='torch.nn.Linear', in_features=1152, out_features=1024),
    ),
    text_backbone=dict(
        type='Qwen2Text05B',  # 使用 Qwen 的文本骨干
        hf_pretrain_name=qwen_cfg_model_name,
        lora_cfg=dict(
            r=32,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
        ),
        init_cfg=dict(),
    ),
    neck=dict(
        type='TransformerFusionNeckForQwen',  # 保持与 Qwen 兼容的融合颈部
        embed_dims=1024,
        text_hidden_dims=1536,
        num_modality=2,
        with_cls_token=True,
        num_encoder_layers=3,
    ),
    head1=dict(
        type='MultiLabelLinearClsHead',  # 一级分类头
        num_classes=num_classes_1,  # 一级类目的数量
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, use_sigmoid=True),
        label_key='labels_level1',  # 指定一级标签
        prefix='level1',  # 添加前缀
    ),
    head2=dict(
        type='MultiLabelLinearClsHead',  # 二级分类头
        num_classes=num_classes_2,  # 二级类目的数量
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, use_sigmoid=True),
        label_key='labels_level2',  # 指定二级标签
        prefix='level2',  # 添加前缀
    ),
    pgvr_cfg=dict(
        type='PrototypeGuidedVisualRouting',
        num_classes=12,
        feat_dim=1024, 
        use_cosine=True,
        loss_weight=1
        
    ),
    consistency_loss_cfg=dict(
        type='HierConsistencyLoss',
        child_to_parent=child_to_parent,
        reduction='mean',
        loss_weight=1
    ),
)

# dataset
# load_train_pipeline = [
#     dict(type='LoadImageFromUrl', to_float32=True, ignore_empty=True, mean_rgb=data_preprocessor['mean']),
#     dict(type='Resize', scale=img_size, interpolation='bicubic'),
#     dict(type='RandomFlip', prob=0.5, direction='horizontal'),
# ]
# load_test_pipeline = [
#     dict(type='LoadImageFromUrl', to_float32=True, ignore_empty=True, mean_rgb=data_preprocessor['mean']),
#     dict(type='Resize', scale=img_size, interpolation='bicubic'),
# ]


load_train_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        ignore_empty=True
    ),
    dict(type='Resize', scale=img_size, interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
]

load_test_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        ignore_empty=True
    ),
    dict(type='Resize', scale=img_size, interpolation='bicubic'),
]


post_pipeline = [
    dict(type='PackInputs', algorithm_keys=[
        'input_main_string',
        'img_attn_masks',
        'filename',
        'labels_level1',
        'labels_level2',
        ]
         ),
]

train_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    drop_last=False,
    pin_memory=True,
    prefetch_factor=4,
    dataset=dict(
        type=dataset_type,
        data_root='/mnt/ali-sh-1/usr/qiming2/news_tags_filter_multimodal',  # 图片目录的根
        ann_file='/mnt/ali-sh-1/usr/qiming2/news_tags_filter_multimodal/merged_output_20250328_local_train.jsonl',
        pipeline=post_pipeline,
        level1_categories=level1_categories,
        level2_categories=level2_categories,
        num_imgs = num_imgs,
        use_img=True,
        main_feat_cfg=dict(
            max_text_length=1024,
            tokenizer=qwen_cfg_model_name,  # 使用 Qwen 的分词器
        ),
        load_pipeline=load_train_pipeline,
        test_mode=False
    ),
)

val_dataloader = dict(
    batch_size=batch_size_per_gpu*2,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=False),
    drop_last=False,
    pin_memory=True,
    prefetch_factor=4,
    dataset=dict(
        type=dataset_type,
        data_root='/mnt/ali-sh-1/usr/qiming2/news_tags_filter_multimodal/news_figures',  # 图片目录的根
        ann_file='/mnt/ali-sh-1/usr/qiming2/news_tags_filter_multimodal/merged_output_20250328_local_test.jsonl',
        pipeline=post_pipeline,
        level1_categories=level1_categories,
        level2_categories=level2_categories,
        num_imgs = num_imgs,
        use_img=True,
        main_feat_cfg=dict(
            max_text_length=1024,
            tokenizer=qwen_cfg_model_name,
        ),
        load_pipeline=load_test_pipeline,
        test_mode=True
    ),
)

work_dir = 'work_dirs/news-qwen_1024_3img_250510_qwen2-1.5b_siglipso_pgvr_hc'
val_evaluator = [
    dict(
        type='TwoLevelF1Metric',  # 使用双层 F1 评估指标
        num_classes_level1=num_classes_1,
        num_classes_level2=num_classes_2
    )
]

test_dataloader = val_dataloader
test_evaluator = val_evaluator

# scheduler
base_lr = 8e-5
max_epochs = 40  # 可以根据需要调整

## optimizer
optim_wrapper = dict(
    type='DeepSpeedOptimWrapper',  # 使用 DeepSpeed 优化器
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=0.05
    ),
)

## learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=base_lr / 100,
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr*0.01,
        begin=1,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True
    ),
]

## train\val\test setting
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# runtime
runner_type = 'FlexibleRunner'

## deepspeed setting
strategy = dict(
    type='DeepSpeedStrategy',
    bf16=dict(
        enabled=True,
    ),
    inputs_to_half=['inputs'],
    zero_optimization=dict(
        stage=2,
        allgather_partitions=True,
        allgather_bucket_size=2e8,
        reduce_scatter=True,
        reduce_bucket_size='auto',
        overlap_comm=True,
        contiguous_gradients=True,
    ),
)

## configure hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=4,
        by_epoch=True,
        save_last=True,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False),
)
custom_hooks = [
    dict(
        type='SetDropoutRateHook',
        drop_img_rate=0.2,
        drop_token_rate=0.2,
        drop_extra_rate=0.2,
        drop_modality_rate=0.2,
    ),
]

## configure environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

## set visualizer
# vis_backends = [
#     dict(type='LocalVisBackend'),
#     dict(type='WandbVisBackend', init_kwargs=dict(project='eco_exps', group='high_exposure', name='{{fileBasenameNoExtension}}', resume='auto')),
# ]
# visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

## set log level
log_level = 'INFO'

## set load or resume
load_from = None
resume = False

## set random
randomness = dict(seed=None, deterministic=False)
