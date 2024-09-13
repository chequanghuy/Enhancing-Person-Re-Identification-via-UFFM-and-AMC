# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline, build_transformer


def build_model(cfg, num_classes, camera_num, view_num, baseline):
    if baseline == "bot":
        model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    else:
        model = build_transformer(num_classes, camera_num, view_num, cfg)
    return model
