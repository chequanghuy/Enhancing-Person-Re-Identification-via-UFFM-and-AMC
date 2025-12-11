# encoding: utf-8

import math
import numpy as np
import torch
from ignite.metrics import Metric
from collections import defaultdict
from data.datasets.eval_reid import eval_func
from tqdm import tqdm
from tqdm.contrib import tzip
import os
from .rank_cylib.rank_cy import evaluate_cy
def get_euclidean(x, y, **kwargs):
    m = x.shape[0]
    n = y.shape[0]
    distmat = (
        torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n)
        + torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    distmat.addmm_(x, y.t(),alpha=1, beta=-2)
    return distmat

def cosine_similarity(
    x_norm: torch.Tensor, y_norm: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """
    Computes cosine similarity between two tensors.
    Value == 1 means the same vector
    Value == 0 means perpendicular vectors
    """
    # x_n, y_n = x.norm(dim=1)[:, None], y.norm(dim=1)[:, None]
    # x_norm = x / torch.max(x_n, eps * torch.ones_like(x_n))
    # y_norm = y / torch.max(y_n, eps * torch.ones_like(y_n))
    sim_mt = torch.mm(x_norm, y_norm.transpose(0, 1))
    return sim_mt


def get_cosine(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Computes cosine distance between two tensors.
    The cosine distance is the inverse cosine similarity
    -> cosine_distance = abs(-cosine_distance) to make it
    similar in behaviour to euclidean distance
    """
    sim_mt = cosine_similarity(x, y, eps)
    return torch.abs(1 - sim_mt).clamp(min=eps)


def get_dist_func(func_name="euclidean"):
    if func_name == "cosine":
        dist_func = get_cosine
    elif func_name == "euclidean":
        dist_func = get_euclidean
    #print(f"Using {func_name} as distance function during evaluation")
    return dist_func

class R1_mAP(Metric):
    def __init__(self, cfg, args, num_query, alpha, beta, theta, max_rank=50, feat_norm='yes', k=5):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.k = k
        self.cfg = cfg
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.uffm_only = args.uffm_only
        self.re_rank = args.re_rank

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def CCE(self, q_cam, g_camids):
        q_cam = torch.tensor(q_cam)
        g_camids = torch.tensor(g_camids)

        x = torch.tensor(0.0)
        alpha = torch.tensor(self.theta, dtype=x.dtype)
        
        cce = torch.where(q_cam == g_camids, alpha, x)
        cce = cce.view(1, -1)

        return cce

    def kwf_rerank(self, q_camids, g_camids, q_ids, g_ids, qf, gf):
        k1 = 100
        k2 = 6
    

        camid2idx = defaultdict(list)
        index_g = []
        for idx, camid in enumerate(g_camids):
            index_g.append(idx)
            camid2idx[camid].append(idx)

        # camid2idx_q = defaultdict(list)
        # for idx, camid in enumerate(q_camids):
        #     camid2idx_q[camid].append(idx)
        unique_camids_q = sorted(np.unique(q_camids))
        num_camid_q = len(unique_camids_q)
        simmat = torch.tensor([]).cuda()
        qf, gf = qf.cuda(), gf.cuda()

        for i in range(len(q_ids)):
            # print(f'\r{i}', end='')
            curr_camid = q_camids[i]
            ind_qcamid = camid2idx[curr_camid] # index of gallery feature has camid = q_camid
            ind_exq = np.setdiff1d(index_g, ind_qcamid) #index of gallery exp camid = q_camid
            gf_new = gf[ind_exq]
            sim_qg = torch.mm(qf[i].view(1, -1), gf.t())
            qg_argsort =  torch.argsort(1-sim_qg, dim = 1)
            #arange_1 = torch.arange(sim_qg.size(0)).unsqueeze(1)
            qg_arg_argsort = torch.argsort(qg_argsort, dim = 1)

            qg_argsort_tp = qg_argsort[:, :k1].squeeze()
            qg_argsort_tail = qg_argsort[:, k1:].squeeze()
            gf_top = gf[qg_argsort_tp]
            gf_tail = gf[qg_argsort_tail]
            sim_ggtop = torch.mm(gf_top, gf_new.t())
            ggtop_argsort = torch.argsort(1-sim_ggtop, axis = 1)
            ggtop_as_tp = ggtop_argsort[:, :k2]

            arange_ = torch.arange(sim_ggtop.size(0)).unsqueeze(1)
            sim_gg_topk = sim_ggtop[arange_, ggtop_as_tp] #top_k sim
            if self.re_rank == 'uniform':
                weight = torch.ones_like(sim_gg_topk) / k2
            elif self.re_rank == 'sim':
                weight = sim_gg_topk / torch.sum(sim_gg_topk, dim=-1).view(-1, 1)
            elif self.re_rank == 'inv_dist_pow':
                weight = (1/(1 - sim_gg_topk)**2) / torch.sum(1/(1 - sim_gg_topk)**2, dim=-1).view(-1, 1)
            elif self.re_rank == 'exp_decay':
                weight = torch.exp(-(1-sim_gg_topk)) / torch.sum(torch.exp(-(1-sim_gg_topk)), dim=-1).view(-1, 1)
            gf_topk =  gf_new[ggtop_as_tp]
            gf_topk_p = gf_topk.permute(0, 2, 1)
            centroid_g_top = torch.bmm(gf_topk_p, weight.unsqueeze(-1)).squeeze()
            centroid_g = torch.cat((centroid_g_top, gf_tail), dim = 0)
            centroid_g = centroid_g[qg_arg_argsort.squeeze()]
            del sim_qg, sim_ggtop
            sim_qg = torch.mm(qf[i].view(1, -1), gf.t())
            sim_centroid = torch.mm(qf[i].view(1, -1), centroid_g.t())
            simmat = torch.cat((simmat, sim_centroid.view(1, -1)), dim = 0)


        return simmat

    def calculate_similarity(self, q_camids, g_camids, q_ids, g_ids, qf, gf, k):
        # Mapping camera IDs to their corresponding gallery indices
        camid2idx = defaultdict(list)
        index_g = [] 
        for idx, camid in enumerate(g_camids):
            index_g.append(idx)
            camid2idx[camid].append(idx)

        # Get unique camera IDs for the queries and their count
        unique_camids_q = sorted(np.unique(q_camids))
        num_camid_q = len(unique_camids_q)

        # Initialize tensors to store similarity matrices and a dictionary for uncertain multi-view features
        simmat = torch.tensor([]).cuda()
        dict_umvf = {}
        cce = {}

        # Compute similarity between all gallery features
        sim_gg = torch.mm(gf, gf.t())

        # Loop through each unique camera ID in the query
        for indx in unique_camids_q:
            # Find indices of gallery features with the same camera ID as the current query
            ind_qcamid = camid2idx[indx] 
            
            # Exclude gallery features with the same camera ID as the query
            ind_exq = np.setdiff1d(index_g, ind_qcamid)  
            gf_new = gf[ind_exq]  
            sim_gg_exq = sim_gg[:, ind_exq]  
            sim_gg_argsort = torch.argsort(1 - sim_gg_exq, dim=1)
            sim_gg_argtopk = sim_gg_argsort[:, :k]  

            # Calculate weights for the top-k gallery features
            sim_gg_topk = sim_gg_exq[torch.arange(sim_gg_exq.size(0)).unsqueeze(1), sim_gg_argtopk] 
            sum_sim_topk = torch.sum(sim_gg_topk, dim=-1)
            weight = sim_gg_topk / sum_sim_topk.view(-1, 1)
            
            # Calculate the centroid of the top-k gallery features
            gf_topk = gf_new[sim_gg_argtopk]
            gf_topk_p = gf_topk.permute(0, 2, 1)
            centroid_g = torch.bmm(gf_topk_p, weight.unsqueeze(-1)).squeeze()
            # Store centroid and camera ID's contextual cross-entropy value
            cce[indx] = self.CCE(indx, g_camids)
            dict_umvf[indx] = centroid_g
            print(f"\rNumber of Camera ID done: {indx + 1}/{num_camid_q}", end="")

        del sim_gg

        # Compute similarity using UFFM + AMC for each query or just using UFFM
        if not self.uffm_only:
            print('\nUsing UFFM + AMC...')
            for i in tqdm(range(len(q_ids)), desc="CalcSimilarity: "):
                curr_camid = q_camids[i]  # Camera ID of the current query
                sim_qg = torch.mm(qf[i].view(1, -1), gf.t())
                centroid_g = dict_umvf[curr_camid]
                sim_centroid = torch.mm(qf[i].view(1, -1), centroid_g.t())
                fusion = self.alpha * sim_qg + self.beta * sim_centroid + cce[curr_camid].cuda()
                simmat = torch.cat((simmat, fusion.view(1, -1)), dim=0)
        else:
            for i in tqdm(range(len(q_ids)), desc="CalcSimilarity: "):
                curr_camid = q_camids[i]  # Camera ID of the current query
                centroid_g = dict_umvf[curr_camid]
                sim_centroid = torch.mm(qf[i].view(1, -1), centroid_g.t())
                simmat = torch.cat((simmat, sim_centroid.view(1, -1)), dim=0)
        del dict_umvf

        return simmat

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
            print("\nThe test feature is normalized")
  
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        print("Calculating similarity...")
        if self.re_rank != 'none':
            simmat = self.kwf_rerank(q_camids, g_camids, q_pids, g_pids, qf, gf)
        else:   
            simmat = self.calculate_similarity(q_camids, g_camids, q_pids, g_pids, qf, gf, k=self.k)
        dismat = 1 - simmat
        cmc, all_AP, _ = evaluate_cy(dismat.cpu(), q_pids, g_pids, q_camids, g_camids, max_rank=50, use_metric_cuhk03=False)
        mAP = np.mean(all_AP)

        return cmc, mAP
