# encoding: utf-8

import logging

import torch
import torch.nn as nn
from ignite.engine import Engine
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from utils.reid_metric import R1_mAP
from utils.coeff_calc import Coeff
from torch import nn

def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
    

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) #if torch.cuda.device_count() >= 1 else data
            feat = model(data)
           
            return feat, pids, camids

    engine = Engine(_inference)
    for name, metric in metrics.items():
        metric.attach(engine, name)
        qbar = ProgressBar(bar_format='EvalFeatures:{percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]')
        qbar.attach(engine)

    return engine

def create_supervised_coeff(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
    

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, ids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, ids, camids

    engine = Engine(_inference)
    for name, metric in metrics.items():
        metric.attach(engine, name)
        qbar = ProgressBar(bar_format='TrainFeatures: {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]')
        qbar.attach(engine)

    return engine


def inference(
        cfg,
        args,
        model,
        val_loader,
        train_loader,
        num_query,
        val_set
):
    device = cfg.MODEL.DEVICE
    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")
    alpha, beta, theta = 0.0, 0.0, 0.0
    if not args.uffm_only:
        print("Create trainer for coeff...")

        coeff_trainer = create_supervised_coeff(model, metrics={'Coeff': Coeff(num_query, feat_norm=cfg.TEST.FEAT_NORM, \
                                                        n_data=args.n_triple, rand_seed=args.seed)},
                                                device=device)
        coeff_trainer.run(train_loader)
        score, score_pos, score_neg, alpha, beta, theta = coeff_trainer.state.metrics['Coeff']

        del coeff_trainer
        logger.info('Coefficent Result:')
        logger.info("Score: {:.1%}".format(score))
        logger.info("Alpha value: {}".format(alpha))
        logger.info("Beta value: {}".format(beta))
        logger.info("Theta value: {}".format(theta))

    print("Create evaluator")
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(cfg, args, num_query, alpha, beta, theta, max_rank=50, \
                            feat_norm=cfg.TEST.FEAT_NORM, \
                            k=args.k)},
                                            device=device)

    evaluator.run(val_loader)
    
    cmc, mAP = evaluator.state.metrics['r1_mAP']
    del evaluator

    logger.info('Validation Results:')
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
