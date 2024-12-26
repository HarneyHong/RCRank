import copy
import os
import torch
import torch.nn as nn
from torch import optim

import numpy as np
import pandas as pd
import json
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import scipy.stats as stats

from model.modules.FuseModel.CrossTransformer import CrossTransformer
from model.modules.FuseModel.Attention import MultiHeadedAttention
from model.modules.rcrank_model import GateComDiffPretrainModel
from model.modules.TSModel.ts_model import CustomConvAutoencoder

from model.loss.loss import ThresholdLoss 
from model.loss.loss import MarginLoss, ListnetLoss, ListMleLoss
import random
from utils.evaluate import evaluate_tau


cross_model = {"CrossTransformer": CrossTransformer}

attn_model = {"MultiHeadedAttention": MultiHeadedAttention}
model_dict = {"GateComDiffPretrainModel": GateComDiffPretrainModel}

margin_loss_types = {"ListnetLoss": ListnetLoss, "MarginLoss": MarginLoss, "ListMleLoss": ListMleLoss}

 

def test(model, test_dataloader, tokenizer, device, test_len, epoch, model_name, train_dataset, opt_threshold, select_model, batch_size, para_args, best_me_num, model_path, best_model_path,args):
    label_list = []
    pred_list = []
    test_idx = 0
    test_len = test_len
    MSE_loss = 0
    right_label_all = 0
    top1_valid_sum = 0
    top1_valid_num = 0

    test_pred_opt = None
    model.eval()

    for index, input1 in enumerate(test_dataloader):
        sql, plan, time, log, multilabel, opt_label = input1["query"], input1["plan"], input1["timeseries"], input1["log"], input1["multilabel"], input1["opt_label"]
        sql = tokenizer(sql, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        if select_model == "cross_attn_no_plan":
            plan = None
        else:
            plan['x'] = plan['x'].squeeze(1).to(device).to(torch.float32)
            plan["attn_bias"] = plan["attn_bias"].squeeze(1).to(device)
            plan["rel_pos"] = plan["rel_pos"].squeeze(1).to(device)
            plan["heights"] = plan["heights"].squeeze(1).to(device)
        
        time = time.to(device)
        log = log.to(device)
        multilabel = multilabel.to(device)
        
        pred_label, pred_opt_raw = model(sql, plan, time, log)
        
        pred_opt = pred_opt_raw.detach()
        pred_opt_raw = pred_opt_raw.detach()


        pred_opt_raw = pred_opt_raw.to("cpu")
        pred_opt = pred_opt.to("cpu")
        multilabel = multilabel.to("cpu")
        pred_label = pred_label.to("cpu")
        pred_opt = (pred_opt * (train_dataset.opt_labels_train_std + 1e-6) + train_dataset.opt_labels_train_mean)
        
        duration = input1["duration"]
        opt_min_duration = (duration * opt_threshold).unsqueeze(1)
        pred_multilabel = pred_opt.gt(opt_threshold).nonzero()
        sorted_time_index = torch.argsort(pred_opt, dim=1, descending=True)

        opt_label = input1["ori_opt_label"]
        opt_label = (opt_label * (train_dataset.opt_labels_train_std+ 1e-6) + train_dataset.opt_labels_train_mean)
        opt_label_m = opt_label
        label_multilabel = opt_label.gt(opt_threshold).nonzero()
        label_sorted_time_index = torch.argsort(opt_label, dim=1, descending=True)
        
        if para_args.pred_type == "multilabel":
            
            multilabel_true = torch.where(opt_label > opt_threshold, 1, 0)
            multilabel_pred = torch.where(pred_label > 0.5, 1, 0)
            right_label = torch.where(multilabel_true == multilabel_pred, 1, 0).sum()
            right_label_all += right_label
        else:
            multilabel_true = torch.where(opt_label > opt_threshold, 1, 0)
            multilabel_pred = torch.where(pred_opt > opt_threshold, 1, 0)
            right_label = torch.where(multilabel_true == multilabel_pred, 1, 0).sum()
            right_label_all += right_label


        if test_pred_opt is None:
            test_pred_opt = pred_opt
        else:
            test_pred_opt = torch.cat((test_pred_opt, pred_opt), dim=0)

        start_row = 0
        label_list.append([])
        
        kk_i = 0
        for row, col in label_multilabel:
            if row == start_row:
                label_list[batch_size * test_idx + row].append(label_sorted_time_index[row][kk_i].item())
                kk_i += 1
            else:
                kk_i = 0
                while row != start_row:
                    start_row += 1
                    label_list.append([])
                label_list[batch_size * test_idx + row].append(label_sorted_time_index[row][kk_i].item())
                kk_i += 1

        len_data = (batch_size * (test_idx+1) if batch_size * (test_idx+1) < test_len else test_len)
        label_len = len(label_list)
        if label_len < len_data:
            for i in range(len_data - label_len):
                label_list.append([])

        start_row = 0
        pred_list.append([])

        kk_i = 0
        
        for row, col in pred_multilabel:
            if row == start_row:
                pred_list[batch_size * test_idx + row].append(sorted_time_index[row][kk_i].item())
                kk_i += 1
            else:
                kk_i = 0
                while row != start_row:
                    start_row += 1
                    pred_list.append([])
                pred_list[batch_size * test_idx + row].append(sorted_time_index[row][kk_i].item())
                kk_i += 1
        
        pred_len = len(pred_list)
        if pred_len < len_data:
            for i in range(len_data - pred_len):
                pred_list.append([])
        
        test_idx += 1
        
        for i in range(sorted_time_index.shape[0]):
            if pred_opt[i][sorted_time_index[i][0]] > opt_threshold:
                top1_valid_num += 1
                if opt_label[i][sorted_time_index[i][0]] > 0:
                    top1_valid_sum += opt_label[i][sorted_time_index[i][0]]

        MSE_loss += torch.pow(pred_opt_raw - opt_label_m, 2).mean(-1).sum().item()
        torch.cuda.empty_cache()

    all_right_cnt = 0
    for i in range(len(label_list)):
        if label_list[i] == pred_list[i]:
            all_right_cnt += 1
            
    top_1_cor = 0
    lab_cor = 0 
    top_1_label = {}
    top_1_pred = {}
    top_1_cor_l = {}
    for i, v in enumerate(label_list):
        if len(label_list[i]) == 0:
            top_1_label[0] = top_1_label.get(0, 0) + 1
        else:
            top_1_label[label_list[i][0]+1] = top_1_label.get(label_list[i][0]+1, 0) + 1
            
        if len(pred_list[i]) == 0:
            top_1_pred[0] = top_1_pred.get(0, 0) + 1
        else:
            top_1_pred[pred_list[i][0]+1] = top_1_pred.get(pred_list[i][0]+1, 0) + 1
            
        if len(label_list[i]) == 0 and len(pred_list[i]) == 0: 
            top_1_cor += 1
            top_1_cor_l[0] = top_1_cor_l.get(0, 0) + 1
            lab_cor += 1
        
        elif  len(label_list[i]) != 0 and len(pred_list[i]) != 0: 
            if label_list[i][0] == pred_list[i][0]:
                top_1_cor += 1
                top_1_cor_l[label_list[i][0]+1] = top_1_cor_l.get(label_list[i][0]+1, 0) + 1
            if len(label_list[i]) == len(pred_list[i]) and len(set(label_list[i]) & set(pred_list[i])) == len(label_list[i]):
                lab_cor += 1


    pred_dict = {}
    for i, v in enumerate(pred_list):
        pred_dict[str(v)] = pred_dict.get(str(v), 0) + 1
    
    tau = evaluate_tau(label_list, pred_list)
    me_num = (top_1_cor / float(test_len)) + tau
    if me_num > best_me_num:
        best_model_path = model_path
        torch.save(model.state_dict(), "/".join(model_path.split("/")[:-1]) + "/best_model.pt")
    torch.save(model.state_dict(), model_path)
    if me_num > best_me_num:
        return me_num, best_model_path
    return best_me_num, best_model_path


def train(select_model, use_fuse_model, train_dataloader, test_dataloader, valid_dataloader, train_len, test_len, valid_len, train_dataset, betas = (0.9, 0.999), lr = 0.0003, batch_size = 8, epoch = 100, l_input_dim = 13,
        t_input_dim = 9,l_hidden_dim = 64, t_hidden_dim = 64, input_dim = 12, emb_dim = 32, fuse_num_layers = 3,
        fuse_ffn_dim = 128, fuse_head_size = 4, dropout = 0.1, opt_threshold = 0.1, model_path_dir=None, model_name=None,use_metrics=True, use_log=True, use_margin_loss=False, use_label_loss=False, use_weight_loss=False, use_threshold_loss=False, margin_loss_type="MarginLoss", multi_head="all_cross", plan_args=None, para_args=None, attn_model_name="MultiHeadedAttention", cross_model_name="CrossTransformer", margin_loss_margin=0.01,config=None, seed=42):
   
    if not os.path.exists(model_path_dir):
        os.mkdir(model_path_dir)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    device = plan_args.device

    tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")
    mul_label_loss_fn = nn.BCELoss(reduction="mean")
    opt_label_loss_fn = nn.MSELoss(reduction="mean")
    loss_margin = margin_loss_types[margin_loss_type](margin=margin_loss_margin)
    loss_ts = ThresholdLoss(threshold=para_args.std_threshold)
    

    print("start train")

    sql_model = BertModel.from_pretrained("./bert-base-uncased")
    time_model = CustomConvAutoencoder()

    fuse_model = None
    multihead_attn_modules_cross_attn = nn.ModuleList(
            [MultiHeadedAttention(fuse_head_size, emb_dim, dropout=dropout, use_metrics=use_metrics, use_log=use_log)
            for _ in range(fuse_num_layers)])
    fuse_model = CrossTransformer(num_layers=fuse_num_layers, d_model=emb_dim, heads=fuse_head_size, d_ff=fuse_ffn_dim, dropout=dropout, attn_modules=multihead_attn_modules_cross_attn)
    
    r_attn_model = nn.ModuleList(
        [attn_model[attn_model_name](fuse_head_size, emb_dim, dropout=dropout, use_metrics=use_metrics, use_log=use_log)
        for _ in range(int(fuse_num_layers))])
    rootcause_cross_model = cross_model[cross_model_name](num_layers=int(fuse_num_layers), d_model=emb_dim, heads=fuse_head_size, d_ff=fuse_ffn_dim, dropout=dropout, attn_modules=r_attn_model)
    model = model_dict[model_name](t_input_dim, l_input_dim, l_hidden_dim, t_hidden_dim, emb_dim, device=device, plan_args=plan_args, sql_model=sql_model, cross_model=fuse_model, time_model=time_model, rootcause_cross_model=rootcause_cross_model)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr, betas)
    best_me_num = 0
    best_model_path = ""
    for i in range(epoch):
        print(f'start epoch {i}')
        epoch_loss = 0
        opt.zero_grad()
        import time as timeutil
        start_time = timeutil.time()
        for index, input1 in enumerate(train_dataloader):
            opt.zero_grad()
            sql, plan, time, log, multilabel, opt_label = input1["query"], input1["plan"], input1["timeseries"], input1["log"], input1["multilabel"], input1["opt_label"]

            sql = tokenizer(sql, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            if select_model == "cross_attn_no_plan":
                plan = None
            else:
                plan['x'] = plan['x'].squeeze(1).to(device).to(torch.float32)
                plan["attn_bias"] = plan["attn_bias"].squeeze(1).to(device)
                plan["rel_pos"] = plan["rel_pos"].squeeze(1).to(device)
                plan["heights"] = plan["heights"].squeeze(1).to(device)
            
            time = time.to(device)
            log = log.to(device)
            multilabel = multilabel.to(device)
            opt_label = opt_label.to(device)

            pred_label, pred_opt = model(sql, plan, time, log)
            
            if use_margin_loss:
                margin_loss = loss_margin(pred_opt, opt_label)

            opt_label_loss = opt_label_loss_fn(pred_opt, opt_label)

            loss = opt_label_loss
            
            if use_margin_loss:
                loss += (margin_loss*para_args.margin_weight)
            if use_threshold_loss:
                ts_loss = loss_ts(pred_opt, opt_label)
                loss += (ts_loss*para_args.ts_weight)
            if use_label_loss:
                mul_label_loss = mul_label_loss_fn(pred_label, multilabel.to(torch.float32))
                loss += (mul_label_loss*para_args.mul_label_weight)
                
            if use_weight_loss:
                errors = torch.abs(pred_opt - opt_label)
                weights = 1 / (1 + errors)
                weights = weights.detach()
                loss = weights * loss
            
            epoch_loss += loss.item()
            loss.backward()
            opt.step()
        print(i, epoch_loss / train_len)
        end_time = timeutil.time()
        print("============diag during time==========: ", end_time - start_time)
        
        
        model_path = model_path_dir + f"/{i}.pt"
        if (i+1) % 5 == 0 or i == 0 or i == epoch-1:
            if valid_dataloader is not None:
                best_me_num, best_model_path = test(model, valid_dataloader, tokenizer, device, valid_len, epoch, model_name, train_dataset, opt_threshold, select_model, batch_size, para_args, best_me_num, model_path, best_model_path,args=config)
            else:
                test(model, test_dataloader, tokenizer, device, test_len, epoch, model_name, train_dataset, opt_threshold, select_model, batch_size, para_args, best_me_num, model_path,args=config)
    
    torch.save(model.state_dict(), model_path)
    
    model.load_state_dict(torch.load(model_path_dir + "/best_model.pt"))
     
     
    label_list = []
    pred_list = []
    test_idx = 0
    test_len = test_len
    MSE_loss = 0
    top_1_cor = 0
    right_label_all = 0
    all_right_cnt = 0
    rank_pred = []
    rank_true = []

    test_pred_opt = None
    model.eval()
    top1_valid_sum = 0
    top1_valid_num = 0
    
    sqlss = {"sql":[], "predrootcause": []}
    
    preds_opt_all = []
    labels_opt_all = []
    res_data = {
        "index": [],
        "sql": [],
        "pred_opt": [],
        "pred_rank": [],
        "label_rank": [],
        "multilabel": []
    }
    
    for index, input1 in enumerate(test_dataloader):
        sqls, plan, time, log, multilabel, opt_label, indexes = input1["query"], input1["plan"], input1["timeseries"], input1["log"], input1["multilabel"], input1["opt_label"], input1["indexes"]
        
        sqlss["sql"].extend(sqls)
        sql = tokenizer(sqls, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        if select_model == "cross_attn_no_plan":
            plan = None
        else:
            plan['x'] = plan['x'].squeeze(1).to(device).to(torch.float32)
            plan["attn_bias"] = plan["attn_bias"].squeeze(1).to(device)
            plan["rel_pos"] = plan["rel_pos"].squeeze(1).to(device)
            plan["heights"] = plan["heights"].squeeze(1).to(device)
            
        time = time.to(device)
        log = log.to(device)
        multilabel = multilabel.to(device)
        
        
        if model_name == "CommonSpecialModel":
        
            pred_label, pred_opt_raw, share_sql_emb, share_plan_emb, private_sql_emb, private_plan_emb = model(sql, plan, time, log)
        elif model_name == "TopConstractModel" or model_name == "GateContrastCommonAttnModel":
            pred_label, pred_opt_raw, sql_plan_global_emb, logit_scale = model(sql, plan, time, log)
        else:
            pred_label, pred_opt_raw = model(sql, plan, time, log)    
            
        
        pred_opt = pred_opt_raw.detach()
        pred_opt_raw = pred_opt_raw.detach()

        pred_opt_raw = pred_opt_raw.to("cpu")
        pred_opt = pred_opt.to("cpu")
        multilabel = multilabel.to("cpu")
        pred_label = pred_label.to("cpu")
        pred_opt = (pred_opt * (train_dataset.opt_labels_train_std + 1e-6) + train_dataset.opt_labels_train_mean)
                
        duration = input1["duration"]
        opt_min_duration = (duration * opt_threshold).unsqueeze(1)
        pred_multilabel = pred_opt.gt(opt_threshold).nonzero()
        sorted_time_index = torch.argsort(pred_opt, dim=1, descending=True)

        opt_label = input1["ori_opt_label"]
        opt_label = (opt_label * (train_dataset.opt_labels_train_std + 1e-6) + train_dataset.opt_labels_train_mean)
        opt_label_m = opt_label
        label_multilabel = opt_label.gt(opt_threshold).nonzero()
        label_sorted_time_index = torch.argsort(opt_label, dim=1, descending=True)
        
        preds_opt_all.extend(pred_opt.tolist())
        labels_opt_all.extend(opt_label.tolist())
            
        
        if para_args.pred_type == "multilabel":
            
            multilabel_true = torch.where(opt_label > opt_threshold, 1, 0)
            multilabel_pred = torch.where(pred_label > 0.5, 1, 0)
            right_label = torch.where(multilabel_true == multilabel_pred, 1, 0).sum()
            right_label_all += right_label
        else:
            multilabel_true = torch.where(opt_label > opt_threshold, 1, 0)
            multilabel_pred = torch.where(pred_opt > opt_threshold, 1, 0)
            right_label = torch.where(multilabel_true == multilabel_pred, 1, 0).sum()
            right_label_all += right_label


        for k_i in range(len(pred_opt)):
            p_rank = torch.empty_like(sorted_time_index[k_i])
            p_rank[sorted_time_index[k_i]] = torch.arange(len(pred_opt[k_i]))
            rank_pred.append(p_rank.tolist())
            
            t_rank = torch.empty_like(label_sorted_time_index[k_i])
            t_rank[label_sorted_time_index[k_i]] = torch.arange(len(opt_label[k_i]))
            rank_true.append(t_rank.tolist())
            
            if label_sorted_time_index[k_i][0] == sorted_time_index[k_i][0]: top_1_cor += 1 
            if all(label_sorted_time_index[k_i][:3] == sorted_time_index[k_i][:3]): all_right_cnt += 1 
            top1_valid_sum += opt_label[k_i][sorted_time_index[k_i][0]]
            
        MSE_loss += torch.pow(pred_opt_raw - opt_label_m, 2).mean(-1).sum().item()
        torch.cuda.empty_cache()
        
    
    res_path = model_path_dir + "/res.txt"
    with open(res_path, "w") as f:
        Kendall = stats.kendalltau(np.array(rank_true), np.array(rank_pred))[0]
        top1acc = top_1_cor / float(test_len)
        vacc = right_label_all / test_len / 9 
        mse = MSE_loss / float(test_len)
        mcacc = all_right_cnt / float(test_len)
        top1IR = top1_valid_sum / float(test_len)

        
        print("V-ACC: ", vacc, file=f)
        print("Top1-ACC: ", top1acc, file=f)
        print("MSE: ", mse, file=f)
        print("MC-ACC: ", mcacc, file=f)
        print("Tau: ", Kendall, file=f)
        print("Top1-IR: ", top1IR, file=f)
        
        
        print("V-ACC: ", vacc)
        print("Top1-ACC: ", top1acc)
        print("MSE: ", mse)
        print("MC-ACC: ", mcacc)
        print("Tau: ", Kendall)
        print("Top1-IR: ", top1IR)
