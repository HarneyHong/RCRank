import torch
from torch.utils import data


from model.modules.QueryFormer.utils import *

import json

class Tensor_Opt_modal_dataset(data.Dataset):
    '''
        sample: [feature, label]
    '''
    def __init__(self, df, device, train=True, encoding=None, tokenizer=None, train_dataset=None):
        super().__init__()
        self.encoding = encoding
        self.treeNodes = []

        samples_list = df.values.tolist()
        samples_data = []

        if train_dataset is None:
            logs = []
            timeseries = []
            querys = []
            for i, samples in enumerate(samples_list):
                querys.append(samples[0])
                
                logs.append(torch.tensor(samples[2]))
                timeseries.append(torch.tensor(samples[3]))
            

            logs = torch.stack(logs, dim=0)
            self.logs_train_mean = logs.mean(dim=0)
            self.logs_train_std = logs.std(dim=0)

            timeseries = torch.stack(timeseries, dim=0)
            self.timeseries_train_mean = timeseries.mean(dim=[0, 2])
            self.timeseries_train_std = timeseries.std(dim=[0, 2])

        else:
            querys = df["query"].values.tolist()

            self.logs_train_mean = train_dataset.logs_train_mean
            self.logs_train_std = train_dataset.logs_train_std
            self.timeseries_train_mean = train_dataset.timeseries_train_mean
            self.timeseries_train_std = train_dataset.timeseries_train_std

        for i, samples in enumerate(samples_list):
            query_val = samples[0]
            plan_val = samples[1]
            log_val = samples[2]
            ts_val = samples[3]
            multilabel_val = samples[4]
            duration_val = samples[5]
            case_label_val = samples[6] if len(samples) > 6 else "positive"

            sam = {
                "query": query_val,
                "plan": plan_val,
                "log": (torch.tensor(log_val) - self.logs_train_mean) / (self.logs_train_std + 1e-6),
                "timeseries": (torch.tensor(ts_val) - self.timeseries_train_mean.unsqueeze(1)) / (self.timeseries_train_std.unsqueeze(1) + 1e-6),
                "multilabel": torch.tensor(eval(multilabel_val)) if isinstance(multilabel_val, str) else torch.tensor(multilabel_val),
                "duration": duration_val,
                "case_label": case_label_val,
            }
            samples_data.append(sam)

        self.samples = samples_data
        self.device = device
        self.train = train

    def __getitem__(self,index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)
    
    
