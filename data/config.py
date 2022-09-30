# datasets
OPIXray_train = {
    "model_classes": ('Folding_Knife', 'Straight_Knife', 'Scissor', 'Utility_Knife', 'Multi-tool_Knife'),
    "dataset_root": "../datasets/OPIXray/train",
    "imagesetfile": "../datasets/OPIXray/train/train_knife.txt"
}

OPIXray_test = {
    "model_classes": ('Folding_Knife', 'Straight_Knife', 'Scissor', 'Utility_Knife', 'Multi-tool_Knife'),
    "dataset_root": "../datasets/OPIXray/test",
    "imagesetfile": "../datasets/OPIXray/test/test_knife.txt"
}

HiXray_train = {
    "model_classes": ('Portable_Charger_1', 'Portable_Charger_2', 'Water', 'Laptop', 
                      'Mobile_Phone', 'Tablet', 'Cosmetic', 'Nonmetallic_Lighter'),
    "dataset_root": "../datasets/HiXray/train",
    "imagesetfile": "../datasets/HiXray/train/train_knife.txt"
}

HiXray_test = {
    "model_classes": ('Portable_Charger_1', 'Portable_Charger_2', 'Water', 'Laptop', 
                      'Mobile_Phone', 'Tablet', 'Cosmetic', 'Nonmetallic_Lighter'),
    "dataset_root": "../datasets/HiXray/test",
    "imagesetfile": "../datasets/HiXray/test/test_knife_1.txt"
}

XAD_train = {
    "model_classes": ('scissor', 'tszhediedao', 'tsbingdao', 'yktsdaopian'),
    "dataset_root": "../datasets/XAD/train",
    "imagesetfile": "../datasets/XAD/train/train_knife.txt"
}

XAD_test= {
    "model_classes": ('scissor', 'tszhediedao', 'tsbingdao', 'yktsdaopian'),
    "dataset_root": "../datasets/XAD/test",
    "imagesetfile": "../datasets/XAD/test/test_knife.txt"
}

# models
original = {
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 1000000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'SSD_original',
}

DOAM = {
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 1000000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'DOAM',
}

LIM = {
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 1000000,
    'feature_maps': [75, 38, 19,10, 5],
    'min_dim': 300,
    'steps': [4,8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2], [2, 3], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'LIM',
}

FasterRCNN = {
    'min_size': 300,
    'max:size': 500,
    'rpn_sigma': 3.,
    'roi_sigma': 1.,
    'weight_decay': 5e-4,
    'lr_decay': 0.1,
    'lr': 1e-3,
    'use_adam': False
}