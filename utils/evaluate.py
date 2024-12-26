import torch
import numpy as np

import pandas as pd
import scipy.stats as stats


def top1_margin(label, pred):
    pred_opt_order, pred_index = torch.sort(pred, dim=-1, descending=True)
    opt_label_order, label_index = torch.sort(label, dim=-1, descending=True)
    all_label = opt_label_order.shape[0]
    top1_cor = 0
    for i in range(opt_label_order.shape[0]):
        if opt_label_order[i][0] < 0.05 and pred_opt_order[i][0] < 0.05:
            top1_cor += 1
        elif pred_index[i][0] == label_index[i][0]:
            top1_cor += 1
        elif opt_label_order[i][0] - label[i][pred_index[i][0]] < 0.01:
            top1_cor += 1 
    
    return top1_cor / (1.0 * all_label)

def mrr(pred, label):
    mrr_res = 0
    
    for i, lab in enumerate(label):
        mrr_res += (1 / (pred[i].index(lab[0])+1))
    return mrr_res / len(label)

def extended_tau_2(list_a, list_b, all_label):
    """ Calculate the extended Kendall tau from two lists. """
    if len(list_a) < len(list_b):
        remaining_elements = set(all_label) - set(list_a) - set(list_b)
        while len(list_a) < len(list_b) and remaining_elements:
            list_a.append(remaining_elements.pop())
        # for i in range(len(list_b) - len(list_a)):
        #     list_a.append((set(all_label) - set(list_a) - set(list_b)).pop())
    if len(list_a) == 0 and len(list_a) == len(list_b):
        return 1.0
    if len(list_b) == 0:
        return 0.0
    ranks = join_ranks(create_rank(list_a), create_rank(list_b)).fillna(12)
    dummy_df = pd.DataFrame([{'rank_a': 12, 'rank_b': 12} for i in range(2*len(list_a)-len(ranks))])
    total_df = pd.concat([ranks, dummy_df])
    return scale_tau(len(list_a), stats.kendalltau(total_df['rank_a'], total_df['rank_b'])[0])

def scale_tau(length, value):
    """ Scale an extended tau correlation such that it falls in [-1, +1]. """
    n_0 = 2*length*(2*length-1)
    n_a = length*(length-1)
    n_d = n_0 - n_a
    min_tau = (2.*n_a - n_0) / (n_d)
    return 2*(value-min_tau)/(1-min_tau) - 1

def create_rank(a):
    """ Convert an ordered list to a DataFrame with ranks. """
    return pd.DataFrame(
                zip(a, range(len(a))),
                columns=['key', 'rank'])\
            .set_index('key')

def join_ranks(rank_a, rank_b):
    """ Join two rank DataFrames. """
    return rank_a.join(rank_b, lsuffix='_a', rsuffix='_b', how='outer')

def evaluate_tau(label_list, pred_list):
    tau_res = []
    all_label = list(range(12))
    for i in range(len(label_list)):
        tau_res.append(extended_tau_2(label_list[i], pred_list[i], all_label))
    return torch.tensor(tau_res).mean().item()







