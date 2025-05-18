import pickle
from easydict import EasyDict

# from trainer.trainer import trainer_KMaP as trainer
# from trainer.trainer import trainer_KMaP_M as trainer
from trainer.trainer import trainer_KMaP_P as trainer


def experiment(config):
    config = EasyDict(config)
    data = pickle.load(open('data/{}/train_val_test_{}.pkl'.format(config.data_name, config.fold), 'rb'))

    config.num_items = data['num_items_Q']
    config.num_nongradable_items = data['num_items_L']
    config.num_users = data['num_users']

    print(config)

    exp_trainner = trainer(config, data)
    exp_trainner.train()

ednet_config = {
    "data_name": 'ednet',
    "model_name": "KMaP",

    "mode": 'test',
    "fold": 1,
    "metric": 'auc',
    "shuffle": True,

    "cuda": True,
    "gpu_device": 0,
    "seed": 42,

    "min_seq_len": 2,
    "max_seq_len": 100,  # the max step of RNN model
    "batch_size": 64,
    "learning_rate": 0.1,
    "max_epoch": 100,
    "validation_split": 0.2,
    "top_k_metrics": 5,
    "num_eval_negative_sampling": 99,
    "num_train_negative_sampling": 5,
    "num_clusters": 3,
    "student_learning_rate": 0.01,

    "embedding_size_q": 64,
    "embedding_size_a": 32,
    "embedding_size_l": 32,
    "embedding_size_d": 32,
    "embedding_size_s": 32,
    "embedding_size_q_behavior": 32,
    "embedding_size_l_behavior": 32,
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
    "weight_decay": 1e-5,
}
junyi_config = {
    "data_name": 'Junyi2063',
    "model_name": 'KMaP',

    "mode": 'test',
    "fold": 1,
    "metric": 'auc',
    "shuffle": True,

    "n_tasks": 2,
    "num_pref": 5,  # number of dividing vector

    "cuda": True,
    "gpu_device": 0,
    "seed": 42,

    "min_seq_len": 2,
    "max_seq_len": 100,  # the max step of RNN model
    "batch_size": 64,
    "learning_rate": 0.01,
    "max_epoch": 100,
    "validation_split": 0.2,
    "top_k_metrics": 5,
    "num_eval_negative_sampling": 99,
    "num_train_negative_sampling": 5,
    "num_clusters": 3,
    "student_learning_rate": 0.01,

    "embedding_size_q": 64,
    "embedding_size_a": 32,
    "embedding_size_l": 32,
    "embedding_size_d": 32,
    "embedding_size_s": 32,
    "embedding_size_q_behavior": 32,
    "embedding_size_l_behavior": 32,
    "num_concepts": 8,
    "key_dim": 32,
    "value_dim": 32,
    "summary_dim": 32,
    "behavior_map_size": 32,
    "behavior_hidden_size": 32,
    "behavior_summary_fc": 32,
    "embedding_size_latent": 32,
    
    'weight_type': 0.15,

    "init_std": 0.2,
    "max_grad_norm": 50,

    "save_checkpoint_every": 3, # save epochs
    "optimizer": 'adam',
    "epsilon": 0.1,
    "beta1": 0.9,
    "beta2": 0.999,
    "weight_decay": 1e-5,
}

if __name__== '__main__':
    experiment(ednet_config)
    experiment(junyi_config)