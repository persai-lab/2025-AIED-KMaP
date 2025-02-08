import torch
from easydict import EasyDict
import pickle
from dataloader.dataloader import KTBM_DataLoader_personalized_stateful

from trainer.trainer import trainer_mat_pers 
from trainer.trainer import trainer_mat_nopers 
from trainer.trainer import trainer_pers_nomat 

from trainer.trainer_relwork import trainer_material_KTBM 
from trainer.trainer_relwork import trainer_material_sasrec 
from trainer.trainer_relwork import trainer_material_lstm_material 
from trainer.trainer_relwork import trainer_material_tedcn 
from trainer.trainer_relwork import trainer_material_kobem 
from trainer.trainer_relwork import trainer_material_lstm_type 
from trainer.trainer_relwork import trainer_material_saint 
from trainer.trainer_relwork import trainer_material_gmkt 
from trainer.trainer_relwork import trainer_material_dkvmn 

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind


def circle_points(r, n):
    """
    generate evenly distributed unit divide vectors for two tasks
    """
    circles = []
    for r, n_ in zip(r, n):
        t = np.linspace(0, 0.5 * np.pi, n_)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    return circles

ednet_config = {
    "data_name": 'ednet',
    "model_name": "KTBM",

    "mode": 'test',
    "fold": 1,
    "metric": 'auc',
    "shuffle": True,

    "cuda": True,
    "gpu_device": 0,
    "seed": 1024,

    "n_tasks": 2,
    "num_pref": 5,  # number of dividing vector

    "min_seq_len": 2,
    "max_seq_len": 100,  # the max step of RNN model
    "batch_size": 64,
    "learning_rate": 0.1,#0.01,
    "max_epoch": 200,#70,
    "validation_split": 0.2,
    "top_k_metrics": 5,
    "num_eval_negative_sampling": 99,
    "num_train_negative_sampling": 5,
    "num_clusters": 3,
    "student_learning_rate": 0.1,#0.01,

    "embedding_size_q": 64,
    "embedding_size_a": 32,
    "embedding_size_l": 32,
    "embedding_size_d": 32,#16,
    "embedding_size_s": 32,
    "embedding_size_q_behavior": 32,#16,
    "embedding_size_l_behavior": 32,#16,
    "num_concepts": 8,
    "key_dim": 32,
    "value_dim": 32,
    "summary_dim": 32,
    "behavior_map_size": 32,
    "behavior_hidden_size": 32,
    "behavior_summary_fc": 32,
    "embedding_size_latent": 32,

    'weight_type': 0.1,

    "init_std": 0.2,
    "max_grad_norm": 10,

    "save_checkpoint_every": 3, # save epochs
    "optimizer": 'adam',
    "epsilon": 0.1,
    "beta1": 0.9,
    "beta2": 0.999,
    "weight_decay": 1e-5,#0.05,
}
gmkt_ednet_config = {
    "data_name": 'ednet',
    "model_name": "KTBM",

    "mode": 'test',
    "fold": 1,
    "metric": 'auc',
    "shuffle": True,

    "cuda": True,
    "gpu_device": 0,
    "seed": 1024,

    "n_tasks": 2,
    "num_pref": 5,  # number of dividing vector

    "min_seq_len": 2,
    "max_seq_len": 100,  # the max step of RNN model
    "batch_size": 64,
    "learning_rate": 0.1,#0.01,
    "max_epoch": 200,#70,
    "validation_split": 0.2,
    "top_k_metrics": 5,
    "num_eval_negative_sampling": 99,
    "num_train_negative_sampling": 5,
    "num_clusters": 3,
    "student_learning_rate": 0.1,#0.01,

    # "embedding_size_q": 64,
    # "embedding_size_a": 32,
    "embedding_size_l": 32,
    "embedding_size_d": 32,#16,
    "embedding_size_s": 32,
    "embedding_size_q_behavior": 32,#16,
    "embedding_size_l_behavior": 32,#16,
    # "num_concepts": 8,
    # "key_dim": 32,
    # "value_dim": 32,
    # "summary_dim": 32,
    "behavior_map_size": 32,
    "behavior_hidden_size": 32,
    "behavior_summary_fc": 32,
    "embedding_size_latent": 32,

    "lambda_q": 0.5,
    "lambda_l": 0.5,
    "k_neighbors": 10,
    "weight_material": 0.01,
    "embedding_size_q": 32,
    "embedding_size_a": 32,
    "num_concepts": 8,
    "key_dim": 32,
    "value_dim": 32,
    "summary_dim": 32,

    'weight_type': 0.1,

    "init_std": 0.2,
    "max_grad_norm": 10,

    "save_checkpoint_every": 3, # save epochs
    "optimizer": 'adam',
    "epsilon": 0.1,
    "beta1": 0.9,
    "beta2": 0.999,
    "weight_decay": 1e-5,#0.05,
}
ktbm_ednet_config = {
    "data_name": 'ednet',
    "model_name": "KTBM",

    "mode": 'test',
    "fold": 1,
    "metric": 'auc',
    "shuffle": True,

    "cuda": True,
    "gpu_device": 0,
    "seed": 1024,

    "n_tasks": 2,
    "num_pref": 5,  # number of dividing vector

    "min_seq_len": 2,
    "max_seq_len": 100,  # the max step of RNN model
    "batch_size": 64,
    "learning_rate": 0.1,#0.01,
    "max_epoch": 200,#70,
    "validation_split": 0.2,
    "top_k_metrics": 5,
    "num_eval_negative_sampling": 99,
    "num_train_negative_sampling": 5,
    "num_clusters": 3,
    "student_learning_rate": 0.1,#0.01,

    "embedding_size_q": 64,
    "embedding_size_a": 32,
    "embedding_size_l": 32,
    "embedding_size_d": 16,
    "embedding_size_s": 32,
    "embedding_size_q_behavior": 16,
    "embedding_size_l_behavior": 16,
    "num_concepts": 8,
    "key_dim": 32,
    "value_dim": 32,
    "summary_dim": 32,
    "behavior_map_size": 32,
    "behavior_hidden_size": 32,
    "behavior_summary_fc": 32,
    "embedding_size_latent": 32,

    'weight_type': 0.1,

    "init_std": 0.2,
    "max_grad_norm": 10,

    "save_checkpoint_every": 3, # save epochs
    "optimizer": 'adam',
    "epsilon": 0.1,
    "beta1": 0.9,
    "beta2": 0.999,
    "weight_decay": 1e-5,#0.05,
}
junyi_config = {
    "data_name": 'Junyi2063',
    "model_name": 'KTBM',
    # "model_name": 'MANN',

    "mode": 'test',
    "fold": 1,
    "metric": 'auc',
    "shuffle": True,

    "n_tasks": 2,
    "num_pref": 5,  # number of dividing vector

    "cuda": True,
    "gpu_device": 0,
    "seed": 1024,

    "min_seq_len": 2,
    "max_seq_len": 100,  # the max step of RNN model
    "batch_size": 64,
    "learning_rate": 0.1,
    "max_epoch": 200,#70,
    "validation_split": 0.2,
    "top_k_metrics": 5,
    "num_eval_negative_sampling": 99,
    "num_train_negative_sampling": 5,
    "num_clusters": 3,
    "student_learning_rate": 0.01,

    "embedding_size_q": 64,
    "embedding_size_a": 32,
    "embedding_size_l": 32,
    "embedding_size_d": 32,#16,
    "embedding_size_s": 32,
    "embedding_size_q_behavior": 32,#16,
    "embedding_size_l_behavior": 32,#16,
    "num_concepts": 8,
    "key_dim": 32,
    "value_dim": 32,
    "summary_dim": 32,
    "behavior_map_size": 32,
    "behavior_hidden_size": 32,
    "behavior_summary_fc": 32,
    "embedding_size_latent": 32,

    # "lambda_q": 0.5,
    # "lambda_l": 0.5,
    # "k_neighbors": 10,
    # "weight_material": 0.01,
    # "embedding_size_q": 32,
    # "embedding_size_a": 32,
    # "num_concepts": 8,
    # "key_dim": 32,
    # "value_dim": 32,
    # "summary_dim": 32,

    'weight_type': 0.15,

    "init_std": 0.2,
    "max_grad_norm": 50,

    "save_checkpoint_every": 3, # save epochs
    "optimizer": 'adam',
    "epsilon": 0.1,
    "beta1": 0.9,
    "beta2": 0.999,
    "weight_decay": 1e-5,#0.05,
}
gmkt_junyi_config = {
    "data_name": 'Junyi2063',
    "model_name": 'KTBM',
    # "model_name": 'MANN',

    "mode": 'test',
    "fold": 1,
    "metric": 'auc',
    "shuffle": True,

    "n_tasks": 2,
    "num_pref": 5,  # number of dividing vector

    "cuda": True,
    "gpu_device": 0,
    "seed": 1024,

    "min_seq_len": 2,
    "max_seq_len": 100,  # the max step of RNN model
    "batch_size": 64,
    "learning_rate": 0.1,
    "max_epoch": 200,#70,
    "validation_split": 0.2,
    "top_k_metrics": 5,
    "num_eval_negative_sampling": 99,
    "num_train_negative_sampling": 5,

    # "embedding_size_q": 64,
    # "embedding_size_a": 32,
    "embedding_size_l": 32,
    "embedding_size_d": 32,#16,
    "embedding_size_s": 32,
    "embedding_size_q_behavior": 32,#16,
    "embedding_size_l_behavior": 32,#16,
    # "num_concepts": 8,
    # "key_dim": 32,
    # "value_dim": 32,
    # "summary_dim": 32,
    "behavior_map_size": 32,
    "behavior_hidden_size": 32,
    "behavior_summary_fc": 32,
    "embedding_size_latent": 32,

    "lambda_q": 0.5,
    "lambda_l": 0.5,
    "k_neighbors": 10,
    "weight_material": 0.01,
    "embedding_size_q": 32,
    "embedding_size_a": 32,
    "num_concepts": 8,
    "key_dim": 32,
    "value_dim": 32,
    "summary_dim": 32,

    'weight_type': 0.15,

    "init_std": 0.2,
    "max_grad_norm": 50,

    "save_checkpoint_every": 3, # save epochs
    "optimizer": 'adam',
    "epsilon": 0.1,
    "beta1": 0.9,
    "beta2": 0.999,
    "weight_decay": 1e-5,#0.05,
}
ktbm_junyi_config = {
    "data_name": 'Junyi2063',
    "model_name": 'KTBM',
    # "model_name": 'MANN',

    "mode": 'test',
    "fold": 1,
    "metric": 'auc',
    "shuffle": True,

    "n_tasks": 2,
    "num_pref": 5,  # number of dividing vector

    "cuda": True,
    "gpu_device": 0,
    "seed": 1024,

    "min_seq_len": 2,
    "max_seq_len": 100,  # the max step of RNN model
    "batch_size": 64,
    "learning_rate": 0.1,
    "max_epoch": 200,#70,
    "validation_split": 0.2,
    "top_k_metrics": 5,
    "num_eval_negative_sampling": 99,
    "num_train_negative_sampling": 5,

    "embedding_size_q": 64,
    "embedding_size_a": 32,
    "embedding_size_l": 32,
    "embedding_size_d": 16,
    "embedding_size_s": 32,
    "embedding_size_q_behavior": 16,
    "embedding_size_l_behavior": 16,
    "num_concepts": 8,
    "key_dim": 32,
    "value_dim": 32,
    "summary_dim": 32,
    "behavior_map_size": 32,
    "behavior_hidden_size": 32,
    "behavior_summary_fc": 32,
    "embedding_size_latent": 32,

    # "lambda_q": 0.5,
    # "lambda_l": 0.5,
    # "k_neighbors": 10,
    # "weight_material": 0.01,
    # "embedding_size_q": 32,
    # "embedding_size_a": 32,
    # "num_concepts": 8,
    # "key_dim": 32,
    # "value_dim": 32,
    # "summary_dim": 32,

    'weight_type': 0.15,

    "init_std": 0.2,
    "max_grad_norm": 50,

    "save_checkpoint_every": 3, # save epochs
    "optimizer": 'adam',
    "epsilon": 0.1,
    "beta1": 0.9,
    "beta2": 0.999,
    "weight_decay": 1e-5,#0.05,
}

def evaluation(exp, data_loader, num_splits):
    exp_results = []
    for test_loader in data_loader.get_test_splits(num_splits):
        results = exp.validate_for_hypo_test(test_loader)
        exp_results.append(results)
    
    exp_results = np.array(exp_results)

    print("Evaluation avg metric:", exp_results.mean(0))

    return exp_results

def prepare_config(config):
    config = EasyDict(config)
    ref_vec = torch.tensor(circle_points([1], [config.num_pref])[0]).float()
    data = pickle.load(open('data/{}/train_val_test_{}.pkl'.format(config.data_name, config.fold), 'rb'))
    config.num_items = data['num_items_Q']
    config.num_nongradable_items = data['num_items_L']
    config.num_users = data['num_users']
    print(config)
    pref_idx = 2

    return config, data, ref_vec, pref_idx

def ednet():     
    ednet_config["run_problems_model"] = True  
    config, data, ref_vec, pref_idx = prepare_config(ednet_config)
    ednet_config["run_problems_model"] = False  
    config_l, data_l, _, _ = prepare_config(ednet_config)

    data_loader = KTBM_DataLoader_personalized_stateful(config, data, random_state=42)

    exp = trainer_mat_pers(config, data)
    exp.model.load_state_dict(torch.load("checkpoints_ednet_proposed_mat_pers/checkpoint-0099.pt"))
    results_mat_pers = evaluation(exp, data_loader, 5)

    exp = trainer_mat_nopers(config, data)
    exp.model.load_state_dict(torch.load("checkpoints_ednet_proposed_mat_nopers/checkpoint-0099.pt"))
    results_mat_nopers = evaluation(exp, data_loader, 5)

    exp = trainer_pers_nomat(config, data)
    exp.model.load_state_dict(torch.load("checkpoints_ednet_proposed_pers_nomat/checkpoint-0099.pt"))
    results_pers_nomat = evaluation(exp, data_loader, 5)

    exp = trainer_material_lstm_material(config, data)
    exp.model.load_state_dict(torch.load("checkpoints_ednet_lstm_material/checkpoint-0099_problem.pt"))
    results_lstm_q = evaluation(exp, data_loader, 5)
    exp = trainer_material_lstm_material(config_l, data_l)
    exp.model.load_state_dict(torch.load("checkpoints_ednet_lstm_material/checkpoint-0099_lecture.pt"))
    results_lstm_l = evaluation(exp, data_loader, 5)

    exp = trainer_material_lstm_type(config, data)
    exp.model.load_state_dict(torch.load("checkpoints_ednet_lstm_type/checkpoint-0099_problem.pt"))
    results_lstm_type = evaluation(exp, data_loader, 5)

    exp = trainer_material_dkvmn(config, data)
    exp.model.load_state_dict(torch.load("checkpoints_ednet_dkvmn/checkpoint-0099.pt"))
    results_dkvmn = evaluation(exp, data_loader, 5)

    gmkt_ednet_config["run_problems_model"] = True  
    config_, data_, _, _ = prepare_config(gmkt_ednet_config)
    exp = trainer_material_gmkt(config_, data_)
    exp.model.load_state_dict(torch.load("checkpoints_ednet_gmkt/checkpoint-0018.pt"))
    results_gmkt = evaluation(exp, data_loader, 5)

    exp = trainer_material_saint(config, data)
    exp.model.load_state_dict(torch.load("checkpoints_ednet_saint/checkpoint-0099.pt"))
    results_saint = evaluation(exp, data_loader, 5)

    config_, data_, _, _ = prepare_config(ktbm_ednet_config)
    exp = trainer_material_KTBM(config_, data_, ref_vec, pref_idx)
    exp.model.load_state_dict(torch.load("checkpoints_ednet_ktbm/checkpoint-0099.pt"))
    results_ktbm = evaluation(exp, data_loader, 5)

    exp = trainer_material_sasrec(config, data)
    exp.model.load_state_dict(torch.load("checkpoints_ednet_sasrec/checkpoint-0072_problem.pt"))
    results_sasrec_q = evaluation(exp, data_loader, 5)
    exp = trainer_material_sasrec(config_l, data_l)
    exp.model.load_state_dict(torch.load("checkpoints_ednet_sasrec/checkpoint-0099_lecture.pt"))
    results_sasrec_l = evaluation(exp, data_loader, 5)

    exp = trainer_material_tedcn(config, data)
    exp.model.load_state_dict(torch.load("checkpoints_ednet_tedcn/checkpoint-0099_problem.pt"))
    results_tedcn_q = evaluation(exp, data_loader, 5)
    exp = trainer_material_tedcn(config_l, data_l)
    exp.model.load_state_dict(torch.load("checkpoints_ednet_tedcn/checkpoint-0099_lecture.pt"))
    results_tedcn_l = evaluation(exp, data_loader, 5)

    print("performance")
    print("dkvmn", ttest_ind(results_mat_pers[:, 0], results_dkvmn[:, 0], alternative='greater').pvalue)
    print("saint", ttest_ind(results_mat_pers[:, 0], results_saint[:, 0], alternative='greater').pvalue)
    print("gmkt", ttest_ind(results_mat_pers[:, 0], results_gmkt[:, 0], alternative='greater').pvalue)
    print("ktbm", ttest_ind(results_mat_pers[:, 0], results_ktbm[:, 0], alternative='greater').pvalue)

    print("type")
    print("lstm", ttest_ind(results_mat_pers[:, 1], results_lstm_type[:, 0], alternative='greater').pvalue)
    print("gmkt", ttest_ind(results_mat_pers[:, 1], results_gmkt[:, 1], alternative='greater').pvalue)
    print("ktbm", ttest_ind(results_mat_pers[:, 1], results_ktbm[:, 1], alternative='greater').pvalue)

    print("Q - HR")
    print("lstm", ttest_ind(results_mat_pers[:, 2], results_lstm_q[:, 0], alternative='greater').pvalue)
    print("sasrec", ttest_ind(results_mat_pers[:, 2], results_sasrec_q[:, 0], alternative='greater').pvalue)
    print("tedcn", ttest_ind(results_mat_pers[:, 2], results_tedcn_q[:, 0], alternative='greater').pvalue)

    print("Q - NDCG")
    print("lstm", ttest_ind(results_mat_pers[:, 3], results_lstm_q[:, 1], alternative='greater').pvalue)
    print("sasrec", ttest_ind(results_mat_pers[:, 3], results_sasrec_q[:, 1], alternative='greater').pvalue)
    print("tedcn", ttest_ind(results_mat_pers[:, 3], results_tedcn_q[:, 1], alternative='greater').pvalue)

    print("Q - MRR")
    print("lstm", ttest_ind(results_mat_pers[:, 4], results_lstm_q[:, 2], alternative='greater').pvalue)
    print("sasrec", ttest_ind(results_mat_pers[:, 4], results_sasrec_q[:, 2], alternative='greater').pvalue)
    print("tedcn", ttest_ind(results_mat_pers[:, 4], results_tedcn_q[:, 2], alternative='greater').pvalue)

    print("L - HR")
    print("lstm", ttest_ind(results_mat_pers[:, 5], results_lstm_l[:, 0], alternative='greater').pvalue)
    print("sasrec", ttest_ind(results_mat_pers[:, 5], results_sasrec_l[:, 0], alternative='greater').pvalue)
    print("tedcn", ttest_ind(results_mat_pers[:, 5], results_tedcn_l[:, 0], alternative='greater').pvalue)

    print("L - NDCG")
    print("lstm", ttest_ind(results_mat_pers[:, 6], results_lstm_l[:, 1], alternative='greater').pvalue)
    print("sasrec", ttest_ind(results_mat_pers[:, 6], results_sasrec_l[:, 1], alternative='greater').pvalue)
    print("tedcn", ttest_ind(results_mat_pers[:, 6], results_tedcn_l[:, 1], alternative='greater').pvalue)

    print("L - MRR")
    print("lstm", ttest_ind(results_mat_pers[:, 7], results_lstm_l[:, 2], alternative='greater').pvalue)
    print("sasrec", ttest_ind(results_mat_pers[:, 7], results_sasrec_l[:, 2], alternative='greater').pvalue)
    print("tedcn", ttest_ind(results_mat_pers[:, 7], results_tedcn_l[:, 2], alternative='greater').pvalue)

def Junyi2063():
    junyi_config["run_problems_model"] = True  
    config, data, ref_vec, pref_idx = prepare_config(junyi_config)
    junyi_config["run_problems_model"] = False  
    config_l, data_l, _, _ = prepare_config(junyi_config)

    data_loader = KTBM_DataLoader_personalized_stateful(config, data, random_state=42)

    exp = trainer_mat_pers(config, data)
    exp.model.load_state_dict(torch.load("checkpoints_Junyi2063_proposed_mat_pers/checkpoint-0099.pt"))
    results_mat_pers = evaluation(exp, data_loader, 5)

    exp = trainer_mat_nopers(config, data)
    exp.model.load_state_dict(torch.load("checkpoints_Junyi2063_proposed_mat_nopers/checkpoint-0099.pt"))
    results_mat_nopers = evaluation(exp, data_loader, 5)

    exp = trainer_pers_nomat(config, data)
    exp.model.load_state_dict(torch.load("checkpoints_Junyi2063_proposed_pers_nomat/checkpoint-0099.pt"))
    results_pers_nomat = evaluation(exp, data_loader, 5)

    exp = trainer_material_lstm_material(config, data)
    exp.model.load_state_dict(torch.load("checkpoints_Junyi2063_lstm_material/checkpoint-0099_problem.pt"))
    results_lstm_q = evaluation(exp, data_loader, 5)
    exp = trainer_material_lstm_material(config_l, data_l)
    exp.model.load_state_dict(torch.load("checkpoints_Junyi2063_lstm_material/checkpoint-0099_lecture.pt"))
    results_lstm_l = evaluation(exp, data_loader, 5)

    exp = trainer_material_lstm_type(config, data)
    exp.model.load_state_dict(torch.load("checkpoints_Junyi2063_lstm_type/checkpoint-0099_problem.pt"))
    results_lstm_type = evaluation(exp, data_loader, 5)

    exp = trainer_material_dkvmn(config, data)
    exp.model.load_state_dict(torch.load("checkpoints_Junyi2063_dkvmn/checkpoint-0099.pt"))
    results_dkvmn = evaluation(exp, data_loader, 5)

    gmkt_junyi_config["run_problems_model"] = True  
    config_, data_, _, _ = prepare_config(gmkt_junyi_config)
    exp = trainer_material_gmkt(config_, data_)
    exp.model.load_state_dict(torch.load("checkpoints_Junyi2063_gmkt/checkpoint-0069.pt"))
    results_gmkt = evaluation(exp, data_loader, 5)

    exp = trainer_material_saint(config, data)
    exp.model.load_state_dict(torch.load("checkpoints_Junyi2063_saint/checkpoint-0099.pt"))
    results_saint = evaluation(exp, data_loader, 5)

    config_, data_, _, _ = prepare_config(ktbm_junyi_config)
    exp = trainer_material_KTBM(config_, data_, ref_vec, pref_idx)
    exp.model.load_state_dict(torch.load("checkpoints_Junyi2063_ktbm/checkpoint-0099.pt"))
    results_ktbm = evaluation(exp, data_loader, 5)

    exp = trainer_material_sasrec(config, data)
    exp.model.load_state_dict(torch.load("checkpoints_Junyi2063_sasrec/checkpoint-0099_problem.pt"))
    results_sasrec_q = evaluation(exp, data_loader, 5)
    exp = trainer_material_sasrec(config_l, data_l)
    exp.model.load_state_dict(torch.load("checkpoints_Junyi2063_sasrec/checkpoint-0099_lecture.pt"))
    results_sasrec_l = evaluation(exp, data_loader, 5)

    exp = trainer_material_tedcn(config, data)
    exp.model.load_state_dict(torch.load("checkpoints_Junyi2063_tedcn/checkpoint-0099_problem.pt"))
    results_tedcn_q = evaluation(exp, data_loader, 5)
    exp = trainer_material_tedcn(config_l, data_l)
    exp.model.load_state_dict(torch.load("checkpoints_Junyi2063_tedcn/checkpoint-0099_lecture.pt"))
    results_tedcn_l = evaluation(exp, data_loader, 5)

    print("performance")
    print("dkvmn", ttest_ind(results_mat_pers[:, 0], results_dkvmn[:, 0], alternative='greater').pvalue)
    print("saint", ttest_ind(results_mat_pers[:, 0], results_saint[:, 0], alternative='greater').pvalue)
    print("gmkt", ttest_ind(results_mat_pers[:, 0], results_gmkt[:, 0], alternative='greater').pvalue)
    print("ktbm", ttest_ind(results_mat_pers[:, 0], results_ktbm[:, 0], alternative='greater').pvalue)

    print("type")
    print("lstm", ttest_ind(results_mat_pers[:, 1], results_lstm_type[:, 0], alternative='greater').pvalue)
    print("gmkt", ttest_ind(results_mat_pers[:, 1], results_gmkt[:, 1], alternative='greater').pvalue)
    print("ktbm", ttest_ind(results_mat_pers[:, 1], results_ktbm[:, 1], alternative='greater').pvalue)

    print("Q - HR")
    print("lstm", ttest_ind(results_mat_pers[:, 2], results_lstm_q[:, 0], alternative='greater').pvalue)
    print("sasrec", ttest_ind(results_mat_pers[:, 2], results_sasrec_q[:, 0], alternative='greater').pvalue)
    print("tedcn", ttest_ind(results_mat_pers[:, 2], results_tedcn_q[:, 0], alternative='greater').pvalue)

    print("Q - NDCG")
    print("lstm", ttest_ind(results_mat_pers[:, 3], results_lstm_q[:, 1], alternative='greater').pvalue)
    print("sasrec", ttest_ind(results_mat_pers[:, 3], results_sasrec_q[:, 1], alternative='greater').pvalue)
    print("tedcn", ttest_ind(results_mat_pers[:, 3], results_tedcn_q[:, 1], alternative='greater').pvalue)

    print("Q - MRR")
    print("lstm", ttest_ind(results_mat_pers[:, 4], results_lstm_q[:, 2], alternative='greater').pvalue)
    print("sasrec", ttest_ind(results_mat_pers[:, 4], results_sasrec_q[:, 2], alternative='greater').pvalue)
    print("tedcn", ttest_ind(results_mat_pers[:, 4], results_tedcn_q[:, 2], alternative='greater').pvalue)

    print("L - HR")
    print("lstm", ttest_ind(results_mat_pers[:, 5], results_lstm_l[:, 0], alternative='greater').pvalue)
    print("sasrec", ttest_ind(results_mat_pers[:, 5], results_sasrec_l[:, 0], alternative='greater').pvalue)
    print("tedcn", ttest_ind(results_mat_pers[:, 5], results_tedcn_l[:, 0], alternative='greater').pvalue)

    print("L - NDCG")
    print("lstm", ttest_ind(results_mat_pers[:, 6], results_lstm_l[:, 1], alternative='greater').pvalue)
    print("sasrec", ttest_ind(results_mat_pers[:, 6], results_sasrec_l[:, 1], alternative='greater').pvalue)
    print("tedcn", ttest_ind(results_mat_pers[:, 6], results_tedcn_l[:, 1], alternative='greater').pvalue)

    print("L - MRR")
    print("lstm", ttest_ind(results_mat_pers[:, 7], results_lstm_l[:, 2], alternative='greater').pvalue)
    print("sasrec", ttest_ind(results_mat_pers[:, 7], results_sasrec_l[:, 2], alternative='greater').pvalue)
    print("tedcn", ttest_ind(results_mat_pers[:, 7], results_tedcn_l[:, 2], alternative='greater').pvalue)

if __name__== '__main__':
    # ednet()
    Junyi2063()