import copy
import random

import more_itertools as miter
import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataloader import default_collate

np.random.seed(42)


class DataLoader_personalized:

    def __init__(self, config, data, random_state=None):
        self.random_state = random_state
        self.data_name = config['data_name']
        self.batch_size = config["batch_size"]
        self.shuffle = config["shuffle"]
        self.collate_fn = default_collate
        self.metric = config["metric"]

        self.num_questions = config.num_items
        self.num_nongradable_items = config.num_nongradable_items

        self.seed = config['seed']

        self.validation_split = config["validation_split"]
        self.mode = config["mode"]

        self.min_seq_len = config["min_seq_len"] if "min_seq_len" in config else None
        self.max_seq_len = config["max_seq_len"] if "max_seq_len" in config else None
        self.stride = config["max_seq_len"] if "max_seq_len" in config else None

        self.init_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
        }

        self.generate_train_test_data(data)
    
        if self.metric == 'rmse':
            self.train_data = TensorDataset(torch.Tensor(self.train_data_q).long(),
                                            torch.Tensor(self.train_data_a).float(),
                                            torch.Tensor(self.train_data_l).long(),
                                            torch.Tensor(self.train_data_d).long(),
                                            torch.Tensor(self.train_data_s).long(),
                                            torch.Tensor(self.train_target_answers).float(),
                                            torch.Tensor(self.train_target_masks).bool(),
                                            torch.Tensor(self.train_target_masks_l).bool())

            self.test_data = TensorDataset(torch.Tensor(self.test_data_q).long(), torch.Tensor(self.test_data_a).float(),
                                           torch.Tensor(self.test_data_l).long(), torch.Tensor(self.test_data_d).long(),
                                           torch.Tensor(self.test_data_s).long(),
                                           torch.Tensor(self.test_target_answers).float(),
                                           torch.Tensor(self.test_target_masks).bool(),
                                           torch.Tensor(self.test_target_masks_l).bool())

        else:
            self.train_data = TensorDataset(torch.Tensor(self.train_data_q).long(),
                                            torch.Tensor(self.train_data_a).long(),
                                            torch.Tensor(self.train_data_l).long(),
                                            torch.Tensor(self.train_data_d).long(),
                                            torch.Tensor(self.train_data_s).long(),
                                            torch.Tensor(self.train_target_answers).long(),
                                            torch.Tensor(self.train_target_masks).bool(),
                                            torch.Tensor(self.train_target_masks_l).bool())


            self.test_data = TensorDataset(torch.Tensor(self.test_data_q).long(), torch.Tensor(self.test_data_a).long(),
                                            torch.Tensor(self.test_data_l).long(), torch.Tensor(self.test_data_d).long(),
                                            torch.Tensor(self.test_data_s).long(),
                                           torch.Tensor(self.test_target_answers).long(),
                                            torch.Tensor(self.test_target_masks).bool(),
                                           torch.Tensor(self.test_target_masks_l).bool())

        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size)
        self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size)

    def generate_train_test_data(self, data):
        # merging 800-students train set with 200-students test set to obtain the whole 1000 students recordings
        q_records = data["traindata"]["q_data"] + data["testdata"]["q_data"]
        a_records = data["traindata"]["a_data"] + data["testdata"]["a_data"]
        l_records = data["traindata"]["l_data"] + data["testdata"]["l_data"]
        d_records = data["traindata"]["d_data"] + data["testdata"]["d_data"]

        self.num_students = len(q_records)

        train_data_q, train_data_a, train_data_l, train_data_d, train_data_s = self.ML_BH_ExtDataset(q_records,
                                                                                                       a_records,
                                                                                                       l_records,
                                                                                                       d_records,
                                                                                                       self.max_seq_len,
                                                                                                       stride=self.stride)

        # exclude those student with only one activity, to prevent stratified train_test_split from error
        indices = np.where(np.unique(train_data_s, return_counts=True)[1] == 1)[0]
        mask = np.array([True if i in indices else False for i in train_data_s])
        not_mask = np.logical_not(mask)
        train_data_q_ = train_data_q[mask]
        train_data_a_ = train_data_a[mask]
        train_data_l_ = train_data_l[mask]
        train_data_d_ = train_data_d[mask]
        train_data_s_ = train_data_s[mask]
        train_data_q = train_data_q[not_mask]
        train_data_a = train_data_a[not_mask]
        train_data_l = train_data_l[not_mask]
        train_data_d = train_data_d[not_mask]
        train_data_s = train_data_s[not_mask]

        train_data_q, self.test_data_q, \
            train_data_a, self.test_data_a, \
            train_data_l, self.test_data_l, \
            train_data_d, self.test_data_d, \
            train_data_s, self.test_data_s = train_test_split(
                train_data_q, train_data_a, train_data_l, train_data_d, train_data_s, test_size=self.validation_split, stratify=train_data_s, random_state=self.random_state)

        # adding excluded one-activity students to train set only
        self.train_data_q = np.concatenate([train_data_q, train_data_q_], 0)
        self.train_data_a = np.concatenate([train_data_a, train_data_a_], 0)
        self.train_data_l = np.concatenate([train_data_l, train_data_l_], 0)
        self.train_data_d = np.concatenate([train_data_d, train_data_d_], 0)
        self.train_data_s = np.concatenate([train_data_s, train_data_s_], 0)

        self.train_target_answers = np.copy(self.train_data_a)
        self.train_target_masks = (self.train_data_q != 0)
        self.train_target_masks_l = (self.train_data_l != 0)
        self.test_target_answers = np.copy(self.test_data_a)
        self.test_target_masks = (self.test_data_q != 0)
        self.test_target_masks_l = (self.test_data_l != 0)

    def ML_BH_ExtDataset(self, q_records, a_records, l_records, d_records,
                                           max_seq_len,
                                           stride):

        s_data = []
        q_data = []
        a_data = []
        l_data = []
        d_data = []
        for index in range(self.num_students):
            q_list = q_records[index]
            a_list = a_records[index]
            l_list = l_records[index]
            d_list = d_records[index]
            q_list.insert(0, 0)
            a_list.insert(0, 2)
            l_list.insert(0, 0)
            d_list.insert(0, 0)

            q_list.insert(0, 0)
            a_list.insert(0, 2)
            l_list.insert(0, 0)
            d_list.insert(0, 0)
            q_patches = list(miter.windowed(q_list, max_seq_len, fillvalue=0, step=stride-2))
            a_patches = list(miter.windowed(a_list, max_seq_len, fillvalue=2, step=stride-2))
            l_patches = list(miter.windowed(l_list, max_seq_len, fillvalue=0, step=stride-2))
            d_patches = list(miter.windowed(d_list, max_seq_len, fillvalue=0, step=stride-2))

            q_data.extend(q_patches)
            a_data.extend(a_patches)
            l_data.extend(l_patches)
            d_data.extend(d_patches)

            s_data.extend([index] * len(q_patches))

        return np.array(q_data), np.array(a_data), np.array(l_data), np.array(d_data), np.array(s_data)
    
    def get_test_splits(self, num_splits):
        skf = KFold(n_splits=num_splits, shuffle=True, random_state=self.random_state)
        for _, test_index in skf.split(self.test_data, np.zeros(self.test_data_q.shape[0])):
            test_split = self.test_data[test_index]
            test_split = TensorDataset(*test_split)
            yield DataLoader(test_split, batch_size=self.batch_size)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_q, data_a, data_l, data_d, data_s, target_answers, target_masks, target_masks_l, train_split):
        self.data_q = data_q
        self.data_a = data_a
        self.data_l = data_l
        self.data_d = data_d
        self.data_s = data_s
        self.target_answers = target_answers
        self.target_masks = target_masks
        self.target_masks_l = target_masks_l

        self.train_split = train_split

    def __getitem__(self, index):
        data_q, data_a, data_l, data_d, data_s, target_answers, target_masks, target_masks_l = self.data_q[index], self.data_a[index], self.data_l[index], self.data_d[index], self.data_s[index], self.target_answers[index], self.target_masks[index], self.target_masks_l[index]

        if self.train_split:
            for i in range(data_q.shape[0]):
                if random.random() > 0.5:
                    indices = np.random.randint(0, data_q.shape[1], int(0.10 * data_q.shape[1]))
                    data_q[i][indices] = 0
                    data_a[i][indices] = 2
                    data_l[i][indices] = 0
                    data_d[i][indices] = 0
                    target_answers[i][indices] = 2
                    target_masks[i][indices] = False
                    target_masks_l[i][indices] = False

        return data_q, data_a, data_l, data_d, data_s, target_answers, target_masks, target_masks_l

    def __len__(self):
        return self.data_q.shape[0]


class CustomBatchSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, sequences, batch_size):
        self.batch_size = batch_size

        self.init_budgets = np.array(list(map(len, sequences)))
        self.init_sequences = sequences

    def initialize(self):
        self.budgets = copy.deepcopy(self.init_budgets)
        self.sequences = copy.deepcopy(self.init_sequences)

    def __iter__(self):
        self.initialize()
        batch = []
        for _ in range(self.__len__()):
            existing_ids = np.where(self.budgets > 0)[0]
            indices = np.random.choice(existing_ids, size=min(self.batch_size, len(existing_ids)), replace=False)

            for idx in indices:
                self.budgets[idx] -= 1
                seq = self.sequences[idx].pop(0)
                batch.append(seq)
            
            yield batch
            batch = []
    
    def __len__(self):
        return np.ceil(self.init_budgets.sum() / self.batch_size).astype(int)


class DataLoader_personalized_stateful:

    def __init__(self, config, data, random_state=None):
        self.random_state = random_state
        self.data_name = config['data_name']
        self.batch_size = config["batch_size"]
        self.shuffle = config["shuffle"]
        self.collate_fn = default_collate
        self.metric = config["metric"]

        self.num_questions = config.num_items
        self.num_nongradable_items = config.num_nongradable_items

        self.seed = config['seed']

        self.validation_split = config["validation_split"]
        self.mode = config["mode"]

        self.min_seq_len = config["min_seq_len"] if "min_seq_len" in config else None
        self.max_seq_len = config["max_seq_len"] if "max_seq_len" in config else None
        self.stride = config["max_seq_len"] if "max_seq_len" in config else None

        q_records = data["traindata"]["q_data"] + data["testdata"]["q_data"]
        a_records = data["traindata"]["a_data"] + data["testdata"]["a_data"]
        l_records = data["traindata"]["l_data"] + data["testdata"]["l_data"]
        d_records = data["traindata"]["d_data"] + data["testdata"]["d_data"]

        self.num_students = len(q_records)

        (train_data_q, train_data_a, train_data_l, train_data_d, train_data_s, train_idx_data), \
            (test_data_q, test_data_a, test_data_l, test_data_d, test_data_s, test_idx_data) = self.ML_BH_ExtDataset(q_records,
                                                                                                       a_records,
                                                                                                       l_records,
                                                                                                       d_records,
                                                                                                       self.max_seq_len,
                                                                                                       stride=self.stride)

        train_target_answers = np.copy(train_data_a)
        train_target_masks = (train_data_q != 0)
        train_target_masks_l = (train_data_l != 0)
        test_target_answers = np.copy(test_data_a)
        test_target_masks = (test_data_q != 0)
        test_target_masks_l = (test_data_l != 0)

        train_dataset = Dataset(train_data_q, train_data_a, train_data_l, train_data_d, train_data_s, train_target_answers, train_target_masks, train_target_masks_l, train_split=True)
        
        sampler = CustomBatchSampler(train_idx_data, batch_size=self.batch_size)
        self.train_loader = DataLoader(train_dataset, sampler=sampler, collate_fn=lambda batch: tuple(map(torch.tensor, batch[0])))

        self.test_dataset = Dataset(test_data_q, test_data_a, test_data_l, test_data_d, test_data_s, test_target_answers, test_target_masks, test_target_masks_l, train_split=False)
        sampler = CustomBatchSampler(test_idx_data, batch_size=self.batch_size)
        self.test_loader = DataLoader(self.test_dataset, sampler=sampler, collate_fn=lambda batch: tuple(map(torch.tensor, batch[0])))

        self.test_idx_data = test_idx_data

    def ML_BH_ExtDataset(self, q_records, a_records, l_records, d_records,
                                           max_seq_len,
                                           stride):

        s_data = []
        q_data = []
        a_data = []
        l_data = []
        d_data = []
        idx_data = []
        start_idx = 0
        test_s_data = []
        test_q_data = []
        test_a_data = []
        test_l_data = []
        test_d_data = []
        test_idx_data = []
        test_start_idx = 0
        for index in range(self.num_students):
            q_list = q_records[index]
            a_list = a_records[index]
            l_list = l_records[index]
            d_list = d_records[index]

            q_list.insert(0, 0)
            a_list.insert(0, 2)
            l_list.insert(0, 0)
            d_list.insert(0, 0)

            q_list.insert(0, 0)
            a_list.insert(0, 2)
            l_list.insert(0, 0)
            d_list.insert(0, 0)

            q_patches = list(miter.windowed(q_list, max_seq_len, fillvalue=0, step=stride-2))
            a_patches = list(miter.windowed(a_list, max_seq_len, fillvalue=2, step=stride-2))
            l_patches = list(miter.windowed(l_list, max_seq_len, fillvalue=0, step=stride-2))
            d_patches = list(miter.windowed(d_list, max_seq_len, fillvalue=0, step=stride-2))

            if len(q_patches) > 1:
                train_q_patches, test_q_patches, train_a_patches, test_a_patches, train_l_patches, test_l_patches, train_d_patches, test_d_patches = train_test_split(q_patches, a_patches, l_patches, d_patches, test_size=self.validation_split, shuffle=False, stratify=None)
            else:
                train_q_patches, train_a_patches, train_l_patches, train_d_patches = q_patches, a_patches, l_patches, d_patches
                test_q_patches, test_a_patches, test_l_patches, test_d_patches = [], [], [], []

            q_data.extend(train_q_patches)
            a_data.extend(train_a_patches)
            l_data.extend(train_l_patches)
            d_data.extend(train_d_patches)
            s_data.extend([index] * len(train_q_patches))

            end_idx = start_idx + len(train_q_patches)
            idx_data.append(list(range(start_idx, end_idx)))
            start_idx = end_idx

            test_q_data.extend(test_q_patches)
            test_a_data.extend(test_a_patches)
            test_l_data.extend(test_l_patches)
            test_d_data.extend(test_d_patches)
            test_s_data.extend([index] * len(test_q_patches))

            if len(test_q_patches) > 0:
                test_end_idx = test_start_idx + len(test_q_patches)
                test_idx_data.append(list(range(test_start_idx, test_end_idx)))
                test_start_idx = test_end_idx

        return (np.array(q_data), np.array(a_data), np.array(l_data), np.array(d_data), np.array(s_data), idx_data), \
                (np.array(test_q_data), np.array(test_a_data), np.array(test_l_data), np.array(test_d_data), np.array(test_s_data), test_idx_data)

    def get_test_splits(self, num_splits):
        skf = KFold(n_splits=num_splits, shuffle=True, random_state=self.random_state)
        for _, test_index in skf.split(self.test_idx_data, np.zeros(len(self.test_idx_data))):
            test_idx_data = [self.test_idx_data[idx] for idx in test_index]
            sampler = CustomBatchSampler(test_idx_data, batch_size=self.batch_size)
            test_loader = DataLoader(self.test_dataset, sampler=sampler, collate_fn=lambda batch: tuple(map(torch.tensor, batch[0])))
            yield test_loader
