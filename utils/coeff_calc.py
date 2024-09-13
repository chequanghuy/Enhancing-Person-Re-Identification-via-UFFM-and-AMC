# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import math
import numpy as np
import torch
from ignite.metrics import Metric
from collections import defaultdict
import torch.nn.functional as F
import random
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

class Coeff(Metric):
    def __init__(self, num_query, feat_norm='yes', n_data=1000, rand_seed = 0):
        super(Coeff, self).__init__()
        self.num_query = num_query
        self.feat_norm = feat_norm
        self.n_data = n_data
        self.rand_seed = rand_seed
        self.cce = True

    def reset(self):
        self.feats = []
        self.ids = []
        self.camids = []

    def update(self, output):
        feat, id, camid = output
        self.feats.append(feat)
        self.ids.extend(np.asarray(id))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
            print("The train feature is normalized")
        f_ids = np.asarray(self.ids)
        f_camids = np.asarray(self.camids)

        metric = "cosine"
        np.random.seed(self.rand_seed)

        positives = []
        negatives = []

        for ii in tqdm(range(self.n_data), desc="PositiveData: "):
            pid = np.unique(f_ids)
            m = np.random.choice(pid, 1)[0]
            index_arr = np.where(f_ids == m)[0]
            np.random.shuffle(index_arr)
            
            index_ch = index_arr[0]
            point1 = feats[index_ch].view(1, -1)
            point2 = feats[index_arr[1]].view(1, -1)
            centers = torch.mean(feats[index_arr[2:]], 0)
            
            cce = 1.0 if f_camids[index_ch] == f_camids[index_arr[1]] else 0.0

            centers = centers.view(1, -1)

            cos_sim = float(torch.matmul(point1, point2.t())[0][0])
            cos_sim_c = float(torch.matmul(point1, centers.t())[0][0])
  
            positives.append([cos_sim, cos_sim_c, cce])
    

        for ii in tqdm(range(self.n_data), desc="NegativeData: "):
            pid = np.unique(f_ids)
            m, n = np.random.choice(pid, 2)
            index_arr_m = np.where(f_ids == m)[0]
            index_arr_n = np.where(f_ids == n)[0]

            np.random.shuffle(index_arr_m)
            np.random.shuffle(index_arr_n)
            
            index_ch = index_arr_m[0]


            point1 = feats[index_ch].view(1, -1)
            point2 = feats[index_arr_n[0]].view(1, -1)

            centers = torch.mean(feats[index_arr_n[1:]], 0)
            
            cce = 1.0 if f_camids[index_ch] == f_camids[index_arr_n[0]] else 0.0

            centers = centers.view(1, -1)

            cos_sim = float(torch.matmul(point1, point2.t())[0][0])
            cos_sim_c = float(torch.matmul(point1, centers.t())[0][0])
            negatives.append([cos_sim, cos_sim_c, cce])
      
        Y = np.concatenate(
            (np.ones(len(positives)), np.full(len(negatives), -1)), axis=0)
        X = np.concatenate((positives, negatives), axis=0)

        reg = LinearRegression()
        model = reg.fit(X, Y)
        coef = model.coef_
        
        score = model.score(X, Y)
        score_pos = model.score(positives, np.ones(len(positives)))
        score_neg = model.score(negatives, np.zeros(len(negatives)))

        alpha = coef[0]
        beta = coef[1]
        theta = coef[2]

        return score, score_pos, score_neg, alpha, beta, theta