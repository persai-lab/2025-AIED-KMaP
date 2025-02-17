import logging
import os
import warnings

import numpy as np
import torch
import tqdm
from sklearn import metrics
from torch import nn
from torch.backends import cudnn

from dataloader.dataloader import (KTBM_DataLoader_personalized,
                                   KTBM_DataLoader_personalized_stateful)
from model.KTBM import KTBM_mat_nopers, KTBM_mat_pers, KTBM_pers_nomat
from model.modules import SoftKmeans
from utils.metrics import Metrics

warnings.filterwarnings("ignore")
cudnn.benchmark = True


class trainer_mat_pers:

    def __init__(self, config, data):
        self.config = config
        self.logger = logging.getLogger("trainer")
        self.metric = config.metric

        self.manual_seed = config.seed
        self.device = torch.device("cpu")

        self.current_epoch = 1

        self.task_train_losses = []
        self.task_test_losses = []
        self.train_evals = []
        self.test_evals = []

        self.init_model(data)

        self.metrics = Metrics(config.top_k_metrics)

        print('==>>> total number of trainable parameters: {}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        print('==>>> total number of parameters: {}'.format(sum(p.numel() for p in self.model.parameters())))
        print('==>>> total trainning batch number: {}'.format(len(self.data_loader.train_loader)))
        print('==>>> total testing batch number: {}'.format(len(self.data_loader.test_loader)))

        self.mse_criterion = nn.MSELoss(reduction='mean')
        self.bce_criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.ce_criterion = nn.CrossEntropyLoss(reduction='mean')
        
        self.init_optimizer()

        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            self.model = self.model.to(self.device)
            self.mse_criterion = self.mse_criterion.to(self.device)
            self.bce_criterion = self.bce_criterion.to(self.device)
            self.ce_criterion = self.ce_criterion.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print("Program will run on *****GPU-CUDA***** ")
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")
            print("Program will run on *****CPU*****\n")

    def init_model(self, data):
        self.data_loader = KTBM_DataLoader_personalized_stateful(self.config, data, random_state=self.config.seed)
        self.config.num_students = self.data_loader.num_students
        self.model = KTBM_mat_pers(self.config)
        self.model.initialize()

        self.kmeans = SoftKmeans(self.config.num_clusters)

    def init_optimizer(self):
        if self.config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                             lr=self.config.learning_rate,
                                             momentum=self.config.momentum,
                                             weight_decay=self.config.weight_decay)
        elif self.config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                              lr=self.config.learning_rate,
                                              betas=(self.config.beta1, self.config.beta2),
                                              eps=self.config.epsilon,
                                              weight_decay=self.config.weight_decay)

        self.student_optimizer = torch.optim.Adam(self.model.s_embed_matrix.parameters(), lr=self.config.student_learning_rate)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=3,
            min_lr=1e-5,
            factor=0.5,
            verbose=True
        )
        self.student_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.student_optimizer,
            mode='min',
            patience=3,
            min_lr=0.01,
            factor=0.7,
            verbose=True
        )

    def train(self):
        for epoch in range(1, self.config.max_epoch + 1):
            print("=" * 50 + "Epoch {}".format(epoch) + "=" * 50)
            self.train_one_epoch()
            self.validate()
            self.current_epoch += 1

    def ntxent_loss(self, anchor, positive, negatives, eps=1e-8):
        pos_sim = -torch.nn.functional.pairwise_distance(anchor, positive, p=2)
        pos_sim = torch.exp(pos_sim)

        neg_sim_total = 0
        for negative in negatives:
            neg_sim = -torch.nn.functional.pairwise_distance(anchor, negative, p=2)
            neg_sim = torch.exp(neg_sim)
            neg_sim_total += neg_sim

        loss = neg_sim_total / (pos_sim + eps) # = exp(ntxent-loss)
        loss = loss.mean()

        return loss

    def calculate_metrics(self, target_masks_q, target_masks_l, next_pos_embed_q, next_neg_embeds_q, pred_embed_q, contrastive_pos_q, contrastive_neg_q, pred_dec_q, next_dec_q, pred_dec_gt_q, next_dec_gt_q, next_pos_embed_l, next_neg_embeds_l, pred_embed_l, contrastive_pos_l, contrastive_neg_l, pred_dec_l, next_dec_l, pred_dec_gt_l, next_dec_gt_l, num_neg_sampling, evaluation=False):
        flattened_target_masks_q = target_masks_q.flatten()
        flattened_target_masks_l = target_masks_l.flatten()

        # calculate contrastive loss
        contrastive_ = torch.concat([contrastive_neg_.unsqueeze(2) for contrastive_neg_ in contrastive_neg_q], 2)
        contrastive_ = torch.concat([contrastive_pos_q.unsqueeze(2), contrastive_], 2)
        contrastive_preds_q = contrastive_.flatten(0, 1)
        contrastive_labels_q = torch.zeros(contrastive_preds_q.shape[0],).to(torch.long)
        contrastive_loss_q = self.ce_criterion(contrastive_preds_q[flattened_target_masks_q], contrastive_labels_q[flattened_target_masks_q])

        contrastive_ = torch.concat([contrastive_neg_.unsqueeze(2) for contrastive_neg_ in contrastive_neg_l], 2)
        contrastive_ = torch.concat([contrastive_pos_l.unsqueeze(2), contrastive_], 2)
        contrastive_preds_l = contrastive_.flatten(0, 1)
        contrastive_labels_l = torch.zeros(contrastive_preds_l.shape[0],).to(torch.long)
        contrastive_loss_l = self.ce_criterion(contrastive_preds_l[flattened_target_masks_l], contrastive_labels_l[flattened_target_masks_l])

        # calculate triplet/ntxent loss
        pred_embed_ = pred_embed_q.permute(0, 2, 1).flatten(0, 1)[flattened_target_masks_q]
        next_pos_embed_ = next_pos_embed_q.permute(0, 2, 1).flatten(0, 1)[flattened_target_masks_q]
        next_neg_embeds_ = [next_neg_embeds_q[idx].permute(0, 2, 1).flatten(0, 1)[flattened_target_masks_q] for idx in range(num_neg_sampling)]
        triplet_loss_q = self.ntxent_loss(pred_embed_, next_pos_embed_, next_neg_embeds_)

        pred_embed_ = pred_embed_l.permute(0, 2, 1).flatten(0, 1)[flattened_target_masks_l]
        next_pos_embed_ = next_pos_embed_l.permute(0, 2, 1).flatten(0, 1)[flattened_target_masks_l]
        next_neg_embeds_ = [next_neg_embeds_l[idx].permute(0, 2, 1).flatten(0, 1)[flattened_target_masks_l] for idx in range(num_neg_sampling)]
        triplet_loss_l = self.ntxent_loss(pred_embed_, next_pos_embed_, next_neg_embeds_)

        # calculate reconstruction loss
        pred_dec_q = pred_dec_q.permute(0, 2, 1).flatten(0, 1)[flattened_target_masks_q]
        next_dec_q = next_dec_q.permute(0, 2, 1).flatten(0, 1)[flattened_target_masks_q]
        pred_dec_gt_q = pred_dec_gt_q.permute(0, 2, 1).flatten(0, 1)[flattened_target_masks_q]
        next_dec_gt_q = next_dec_gt_q.permute(0, 2, 1).flatten(0, 1)[flattened_target_masks_q]
        reconstruction_loss_q = self.mse_criterion(torch.concat([pred_dec_q, next_dec_q]), torch.concat([next_dec_gt_q, pred_dec_gt_q]))

        pred_dec_l = pred_dec_l.permute(0, 2, 1).flatten(0, 1)[flattened_target_masks_l]
        next_dec_l = next_dec_l.permute(0, 2, 1).flatten(0, 1)[flattened_target_masks_l]
        pred_dec_gt_l = pred_dec_gt_l.permute(0, 2, 1).flatten(0, 1)[flattened_target_masks_l]
        next_dec_gt_l = next_dec_gt_l.permute(0, 2, 1).flatten(0, 1)[flattened_target_masks_l]
        reconstruction_loss_l = self.mse_criterion(torch.concat([pred_dec_l, next_dec_l]), torch.concat([next_dec_gt_l, pred_dec_gt_l]))

        if evaluation:
            features = torch.concat([next_pos_embed_q.unsqueeze(0), next_neg_embeds_q], 0)
            features = features.permute(1, 3, 0, 2).flatten(0, 1)
            preds = pred_embed_q.unsqueeze(0).repeat(num_neg_sampling+1, 1, 1, 1)
            preds = preds.permute(1, 3, 0, 2).flatten(0, 1)
            targets = torch.zeros(*features.shape[:2])
            targets[:, 0] = 1
            dist = torch.nn.functional.pairwise_distance(features, preds, p=2)
            sims = -dist

            hitratio_q, ndcg_q, mrr_q = self.metrics.calculate_all_with_target(sims[flattened_target_masks_q], targets[flattened_target_masks_q])

            features = torch.concat([next_pos_embed_l.unsqueeze(0), next_neg_embeds_l], 0)
            features = features.permute(1, 3, 0, 2).flatten(0, 1)
            preds = pred_embed_l.unsqueeze(0).repeat(num_neg_sampling+1, 1, 1, 1)
            preds = preds.permute(1, 3, 0, 2).flatten(0, 1)
            targets = torch.zeros(*features.shape[:2])
            targets[:, 0] = 1
            dist = torch.nn.functional.pairwise_distance(features, preds, p=2)
            sims = -dist

            hitratio_l, ndcg_l, mrr_l = self.metrics.calculate_all_with_target(sims[flattened_target_masks_l], targets[flattened_target_masks_l])

            return {"contrastive_loss": (contrastive_loss_q + contrastive_loss_l) / 2, 
                    "ntxent_loss": (triplet_loss_q + triplet_loss_l) / 2, 
                    "reconstruction_loss": (reconstruction_loss_q + reconstruction_loss_l) / 2,
                    "hitratio_q": hitratio_q,
                    "ndcg_q": ndcg_q,
                    "mrr_q": mrr_q,
                    "hitratio_l": hitratio_l,
                    "ndcg_l": ndcg_l,
                    "mrr_l": mrr_l,
                    "contrastive_preds": torch.concat([torch.argmax(contrastive_preds_q, 1), torch.argmax(contrastive_preds_l, 1)]),
                    "contrastive_labels": torch.concat([contrastive_labels_q, contrastive_labels_l])
                }
    
        return {"contrastive_loss": (contrastive_loss_q + contrastive_loss_l) / 2, 
                "ntxent_loss": (triplet_loss_q + triplet_loss_l) / 2, 
                "reconstruction_loss": (reconstruction_loss_q + reconstruction_loss_l) / 2,
                }

    def train_one_epoch(self):
        self.model.initialize_states()
        self.model.train()

        student_embeddings = []
        behavior_embeddings = []
        student_ids = []
        for data in tqdm.tqdm(self.data_loader.train_loader):
            q_list, a_list, l_list, d_list, s_list, target_answers_list, target_masks_list, target_masks_l_list = data

            q_list = q_list.to(self.device)
            a_list = a_list.to(self.device)
            l_list = l_list.to(self.device)
            d_list = d_list.to(self.device)
            s_list = s_list.to(self.device)
            target_answers_list = target_answers_list.to(self.device)
            target_masks_list = target_masks_list.to(self.device)
            target_masks_l_list = target_masks_l_list.to(self.device)

            self.optimizer.zero_grad()
            output, output_type, \
                contrastive_pos_q, contrastive_neg_q, \
                batch_next_repr_neg_q, batch_pred_repr_q, batch_next_repr_q, \
                batch_pred_repr_dec_q, batch_next_repr_dec_q, \
                batch_pred_repr_dec_gt_q, batch_next_repr_dec_gt_q, \
                contrastive_pos_l, contrastive_neg_l, \
                batch_next_repr_neg_l, batch_pred_repr_l, batch_next_repr_l, \
                batch_pred_repr_dec_l, batch_next_repr_dec_l, \
                batch_pred_repr_dec_gt_l, batch_next_repr_dec_gt_l = self.model(q_list, a_list, l_list, d_list, s_list)

            label = torch.masked_select(target_answers_list[:, 2:], target_masks_list[:, 2:])
            label_type = torch.masked_select(d_list[:, 2:], (target_masks_list + target_masks_l_list)[:, 2:])

            output = torch.masked_select(output, target_masks_list[:, 2:])
            loss_q = self.bce_criterion(output.float(), label.float())

            output_type = torch.masked_select(output_type, (target_masks_list + target_masks_l_list)[:, 2:])
            loss_type = self.bce_criterion(output_type.float(), label_type.float())

            metrics_q = self.calculate_metrics(target_masks_list[:, 2:], target_masks_l_list[:, 2:], \
                                               batch_next_repr_q, batch_next_repr_neg_q, batch_pred_repr_q, contrastive_pos_q, contrastive_neg_q, batch_pred_repr_dec_q, batch_next_repr_dec_q, batch_pred_repr_dec_gt_q, batch_next_repr_dec_gt_q, \
                                                batch_next_repr_l, batch_next_repr_neg_l, batch_pred_repr_l, contrastive_pos_l, contrastive_neg_l, batch_pred_repr_dec_l, batch_next_repr_dec_l, batch_pred_repr_dec_gt_l, batch_next_repr_dec_gt_l, \
                                                self.config.num_train_negative_sampling)

            task_loss = torch.stack([loss_q, loss_type, metrics_q["contrastive_loss"], metrics_q["ntxent_loss"], metrics_q["reconstruction_loss"]])
            task_loss.sum().backward()

            new_embedding_weights, behavior_embedding = self.get_new_student_embeddings(s_list)
            student_embeddings.append(new_embedding_weights)
            behavior_embeddings.append(behavior_embedding)
            student_ids.append(s_list.detach())

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        student_embeddings = torch.concat(student_embeddings)
        behavior_embeddings = torch.concat(behavior_embeddings)
        student_ids = torch.concat(student_ids)

        self.update_student_embeddings(student_embeddings, behavior_embeddings, student_ids)

    def get_new_student_embeddings(self, s_list):
        weights = self.model.s_embed_matrix(s_list)

        s_list_oh = torch.nn.functional.one_hot(s_list, self.config.num_students).float()
        gradients = s_list_oh @ self.model.s_embed_matrix.weight.grad
        
        lr = self.scheduler.get_last_lr()[-1]
        new_embedding_weights = weights - lr * gradients
        
        self.model.s_embed_matrix.weight.grad = None

        behavior_embedding = s_list_oh @ self.model.stateful_hidden_states

        return new_embedding_weights, behavior_embedding
    
    def update_student_embeddings(self, embeddings, behavior_embeddings, student_ids):
        loss, (unique_students, cluster_assignments) = self.kmeans.apply_constraints(embeddings, student_ids, behavior_embeddings)
        loss.backward()
        self.student_optimizer.step()
        self.student_scheduler.step(loss.detach().item())

    def validate(self):
        self.train_loss = 0
        self.train_loss_type = 0
        self.train_loss_contrastive = 0
        self.train_loss_triplet = 0
        train_elements = 0
        train_elements_type = 0
        train_elements_contrastive = 0
        train_elements_triplet = 0

        self.test_loss = 0
        self.test_loss_type = 0
        self.test_loss_contrastive = 0
        self.test_loss_triplet = 0
        test_elements = 0
        test_elements_type = 0
        test_elements_contrastive = 0
        test_elements_triplet = 0

        self.model.initialize_states()
        self.model.eval()
        with torch.no_grad():
            self.train_output_all = []
            self.train_output_type_all = []
            self.train_contrastive_all = []
            self.train_label_all = []
            self.train_label_type_all = []
            self.train_label_contrastive_all = []
            self.train_q_hitratios = []
            self.train_q_ndcgs = []
            self.train_q_mrrs = []
            self.train_l_hitratios = []
            self.train_l_ndcgs = []
            self.train_l_mrrs = []

            self.test_output_all = []
            self.test_output_type_all = []
            self.test_contrastive_all = []
            self.test_label_all = []
            self.test_label_type_all = []
            self.test_label_contrastive_all = []
            self.test_q_hitratios = []
            self.test_q_ndcgs = []
            self.test_q_mrrs = []
            self.test_l_hitratios = []
            self.test_l_ndcgs = []
            self.test_l_mrrs = []

            for data in tqdm.tqdm(self.data_loader.train_loader):
                q_list, a_list, l_list, d_list, s_list, target_answers_list, target_masks_list, target_masks_l_list = data
                q_list = q_list.to(self.device)
                a_list = a_list.to(self.device)
                l_list = l_list.to(self.device)
                d_list = d_list.to(self.device)
                s_list = s_list.to(self.device)
                target_answers_list = target_answers_list.to(self.device)
                target_masks_list = target_masks_list.to(self.device)
                target_masks_l_list = target_masks_l_list.to(self.device)

                output, output_type, \
                    contrastive_pos_q, contrastive_neg_q, \
                    batch_next_repr_neg_q, batch_pred_repr_q, batch_next_repr_q, \
                    batch_pred_repr_dec_q, batch_next_repr_dec_q, \
                    batch_pred_repr_dec_gt_q, batch_next_repr_dec_gt_q, \
                    contrastive_pos_l, contrastive_neg_l, \
                    batch_next_repr_neg_l, batch_pred_repr_l, batch_next_repr_l, \
                    batch_pred_repr_dec_l, batch_next_repr_dec_l, \
                    batch_pred_repr_dec_gt_l, batch_next_repr_dec_gt_l = self.model(q_list, a_list, l_list, d_list, s_list, evaluation=True)

                label = torch.masked_select(target_answers_list[:, 2:], target_masks_list[:, 2:])
                label_type = torch.masked_select(d_list[:, 2:], (target_masks_list + target_masks_l_list)[:, 2:])

                output = torch.masked_select(output, target_masks_list[:, 2:])
                loss_q = self.bce_criterion(output.float(), label.float())

                output_type = torch.masked_select(output_type, (target_masks_list + target_masks_l_list)[:, 2:])
                loss_type = self.bce_criterion(output_type.float(), label_type.float())

                metrics_q = self.calculate_metrics(target_masks_list[:, 2:], target_masks_l_list[:, 2:], \
                                               batch_next_repr_q, batch_next_repr_neg_q, batch_pred_repr_q, contrastive_pos_q, contrastive_neg_q, batch_pred_repr_dec_q, batch_next_repr_dec_q, batch_pred_repr_dec_gt_q, batch_next_repr_dec_gt_q, \
                                                batch_next_repr_l, batch_next_repr_neg_l, batch_pred_repr_l, contrastive_pos_l, contrastive_neg_l, batch_pred_repr_dec_l, batch_next_repr_dec_l, batch_pred_repr_dec_gt_l, batch_next_repr_dec_gt_l, \
                                                self.config.num_eval_negative_sampling, evaluation=True)

                contrastive_loss = metrics_q["contrastive_loss"] + metrics_q["reconstruction_loss"]
                ntxent_loss = metrics_q["ntxent_loss"]

                self.train_loss += loss_q.item()
                train_elements += torch.tensor(1)
                self.train_loss_type += loss_type.item()
                train_elements_type += torch.tensor(1)
                self.train_loss_contrastive += contrastive_loss.item()
                train_elements_contrastive += torch.tensor(1)
                self.train_loss_triplet += ntxent_loss.item()
                train_elements_triplet += torch.tensor(1)

                self.train_output_all.extend(output.tolist())
                self.train_output_type_all.extend(output_type.tolist())
                self.train_contrastive_all.extend(metrics_q["contrastive_preds"])
                self.train_label_all.extend(label.tolist())
                self.train_label_type_all.extend(label_type.tolist())
                self.train_label_contrastive_all.extend(metrics_q["contrastive_labels"])

                self.train_q_hitratios.append(metrics_q["hitratio_q"])
                self.train_q_ndcgs.append(metrics_q["ndcg_q"])
                self.train_q_mrrs.append(metrics_q["mrr_q"])

                self.train_l_hitratios.append(metrics_q["hitratio_l"])
                self.train_l_ndcgs.append(metrics_q["ndcg_l"])
                self.train_l_mrrs.append(metrics_q["mrr_l"])
                
            for data in tqdm.tqdm(self.data_loader.test_loader):
                q_list, a_list, l_list, d_list, s_list, target_answers_list, target_masks_list, target_masks_l_list = data
                q_list = q_list.to(self.device)
                a_list = a_list.to(self.device)
                l_list = l_list.to(self.device)
                d_list = d_list.to(self.device)
                s_list = s_list.to(self.device)
                target_answers_list = target_answers_list.to(self.device)
                target_masks_list = target_masks_list.to(self.device)
                target_masks_l_list = target_masks_l_list.to(self.device)

                output, output_type, \
                    contrastive_pos_q, contrastive_neg_q, \
                    batch_next_repr_neg_q, batch_pred_repr_q, batch_next_repr_q, \
                    batch_pred_repr_dec_q, batch_next_repr_dec_q, \
                    batch_pred_repr_dec_gt_q, batch_next_repr_dec_gt_q, \
                    contrastive_pos_l, contrastive_neg_l, \
                    batch_next_repr_neg_l, batch_pred_repr_l, batch_next_repr_l, \
                    batch_pred_repr_dec_l, batch_next_repr_dec_l, \
                    batch_pred_repr_dec_gt_l, batch_next_repr_dec_gt_l = self.model(q_list, a_list, l_list, d_list, s_list, evaluation=True)

                label = torch.masked_select(target_answers_list[:, 2:], target_masks_list[:, 2:])
                label_type = torch.masked_select(d_list[:, 2:], (target_masks_list + target_masks_l_list)[:, 2:])

                output = torch.masked_select(output, target_masks_list[:, 2:])
                loss_q = self.bce_criterion(output.float(), label.float())

                output_type = torch.masked_select(output_type, (target_masks_list + target_masks_l_list)[:, 2:])
                loss_type = self.bce_criterion(output_type.float(), label_type.float())

                metrics_q = self.calculate_metrics(target_masks_list[:, 2:], target_masks_l_list[:, 2:], \
                                               batch_next_repr_q, batch_next_repr_neg_q, batch_pred_repr_q, contrastive_pos_q, contrastive_neg_q, batch_pred_repr_dec_q, batch_next_repr_dec_q, batch_pred_repr_dec_gt_q, batch_next_repr_dec_gt_q, \
                                                batch_next_repr_l, batch_next_repr_neg_l, batch_pred_repr_l, contrastive_pos_l, contrastive_neg_l, batch_pred_repr_dec_l, batch_next_repr_dec_l, batch_pred_repr_dec_gt_l, batch_next_repr_dec_gt_l, \
                                                self.config.num_eval_negative_sampling, evaluation=True)

                contrastive_loss = metrics_q["contrastive_loss"] + metrics_q["reconstruction_loss"]
                ntxent_loss = metrics_q["ntxent_loss"]

                self.test_loss += loss_q.item()
                test_elements += torch.tensor(1)
                self.test_loss_type += loss_type.item()
                test_elements_type += torch.tensor(1)
                self.test_loss_contrastive += contrastive_loss.item()
                test_elements_contrastive += torch.tensor(1)
                self.test_loss_triplet += ntxent_loss.item()
                test_elements_triplet += torch.tensor(1)

                self.test_output_all.extend(output.tolist())
                self.test_output_type_all.extend(output_type.tolist())
                self.test_contrastive_all.extend(metrics_q["contrastive_preds"])
                self.test_label_all.extend(label.tolist())
                self.test_label_type_all.extend(label_type.tolist())
                self.test_label_contrastive_all.extend(metrics_q["contrastive_labels"])

                self.test_q_hitratios.append(metrics_q["hitratio_q"])
                self.test_q_ndcgs.append(metrics_q["ndcg_q"])
                self.test_q_mrrs.append(metrics_q["mrr_q"])

                self.test_l_hitratios.append(metrics_q["hitratio_l"])
                self.test_l_ndcgs.append(metrics_q["ndcg_l"])
                self.test_l_mrrs.append(metrics_q["mrr_l"])
                
            self.train_output_all = np.array(self.train_output_all).squeeze()
            self.train_label_all = np.array(self.train_label_all).squeeze()
            self.train_output_type_all = np.array(self.train_output_type_all).squeeze()
            self.train_label_type_all = np.array(self.train_label_type_all).squeeze()
            self.train_contrastive_all = np.array(self.train_contrastive_all).squeeze()
            self.train_label_contrastive_all = np.array(self.train_label_contrastive_all).squeeze()

            self.test_output_all = np.array(self.test_output_all).squeeze()
            self.test_label_all = np.array(self.test_label_all).squeeze()
            self.test_output_type_all = np.array(self.test_output_type_all).squeeze()
            self.test_label_type_all = np.array(self.test_label_type_all).squeeze()
            self.test_contrastive_all = np.array(self.test_contrastive_all).squeeze()
            self.test_label_contrastive_all = np.array(self.test_label_contrastive_all).squeeze()

            train_evals = np.stack(
                [metrics.roc_auc_score(self.train_label_all, self.train_output_all),
                    metrics.roc_auc_score(self.train_label_type_all, self.train_output_type_all),
                    metrics.accuracy_score(self.train_label_contrastive_all, self.train_contrastive_all),
                    sum(self.train_q_hitratios) / len(self.train_q_hitratios),
                    sum(self.train_q_ndcgs) / len(self.train_q_ndcgs),
                    sum(self.train_q_mrrs) / len(self.train_q_mrrs),
                    sum(self.train_l_hitratios) / len(self.train_l_hitratios),
                    sum(self.train_l_ndcgs) / len(self.train_l_ndcgs),
                    sum(self.train_l_mrrs) / len(self.train_l_mrrs),
                    ])
            test_evals = np.stack(
                [metrics.roc_auc_score(self.test_label_all, self.test_output_all),
                    metrics.roc_auc_score(self.test_label_type_all, self.test_output_type_all),
                    metrics.accuracy_score(self.test_label_contrastive_all, self.test_contrastive_all),
                    sum(self.test_q_hitratios) / len(self.test_q_hitratios),
                    sum(self.test_q_ndcgs) / len(self.test_q_ndcgs),
                    sum(self.test_q_mrrs) / len(self.test_q_mrrs),
                    sum(self.test_l_hitratios) / len(self.test_l_hitratios),
                    sum(self.test_l_ndcgs) / len(self.test_l_ndcgs),
                    sum(self.test_l_mrrs) / len(self.test_l_mrrs),
                    ])

            average_train_loss = torch.stack([self.train_loss / train_elements, self.train_loss_type / train_elements_type, self.train_loss_contrastive / train_elements_contrastive, self.train_loss_triplet / train_elements_triplet])
            average_test_loss = torch.stack([self.test_loss / test_elements, self.test_loss_type / test_elements_type, self.test_loss_contrastive / test_elements_contrastive, self.test_loss_triplet / test_elements_triplet])

        self.task_train_losses.append(average_train_loss.data.cpu().numpy())
        self.train_evals.append(train_evals)
        self.task_test_losses.append(average_test_loss.data.cpu().numpy())
        self.test_evals.append(test_evals)

        self.save_weights()

        print('{}/{}: train_loss={}, train_acc={}, test_loss={}, test_acc={}'.format(
            self.current_epoch, self.config.max_epoch, self.task_train_losses[-1], self.train_evals[-1],
            self.task_test_losses[-1], self.test_evals[-1]))

        self.scheduler.step(np.sum(self.task_train_losses[-1]))

    def save_weights(self):
        if not os.path.isdir(f"checkpoints_{self.config.data_name}_proposed_mat_pers"):
            os.makedirs((f"checkpoints_{self.config.data_name}_proposed_mat_pers"))

        if self.current_epoch % self.config.save_checkpoint_every == 0:
            torch.save(self.model.cpu().state_dict(), os.path.join(f"checkpoints_{self.config.data_name}_proposed_mat_pers", "checkpoint-{:04}.pt".format(self.current_epoch)))

    def validate_for_hypo_test(self, test_loader):
        self.model.initialize_states()
        self.model.eval()
        with torch.no_grad():
            self.test_output_all = []
            self.test_output_type_all = []
            self.test_contrastive_all = []
            self.test_label_all = []
            self.test_label_type_all = []
            self.test_label_contrastive_all = []
            self.test_q_hitratios = []
            self.test_q_ndcgs = []
            self.test_q_mrrs = []
            self.test_l_hitratios = []
            self.test_l_ndcgs = []
            self.test_l_mrrs = []

            for data in tqdm.tqdm(test_loader):
                q_list, a_list, l_list, d_list, s_list, target_answers_list, target_masks_list, target_masks_l_list = data
                q_list = q_list.to(self.device)
                a_list = a_list.to(self.device)
                l_list = l_list.to(self.device)
                d_list = d_list.to(self.device)
                s_list = s_list.to(self.device)
                target_answers_list = target_answers_list.to(self.device)
                target_masks_list = target_masks_list.to(self.device)
                target_masks_l_list = target_masks_l_list.to(self.device)

                output, output_type, \
                    contrastive_pos_q, contrastive_neg_q, \
                    batch_next_repr_neg_q, batch_pred_repr_q, batch_next_repr_q, \
                    batch_pred_repr_dec_q, batch_next_repr_dec_q, \
                    batch_pred_repr_dec_gt_q, batch_next_repr_dec_gt_q, \
                    contrastive_pos_l, contrastive_neg_l, \
                    batch_next_repr_neg_l, batch_pred_repr_l, batch_next_repr_l, \
                    batch_pred_repr_dec_l, batch_next_repr_dec_l, \
                    batch_pred_repr_dec_gt_l, batch_next_repr_dec_gt_l = self.model(q_list, a_list, l_list, d_list, s_list, evaluation=True)

                label = torch.masked_select(target_answers_list[:, 2:], target_masks_list[:, 2:])
                label_type = torch.masked_select(d_list[:, 2:], (target_masks_list + target_masks_l_list)[:, 2:])

                output = torch.masked_select(output, target_masks_list[:, 2:])
                output_type = torch.masked_select(output_type, (target_masks_list + target_masks_l_list)[:, 2:])
                metrics_q = self.calculate_metrics(target_masks_list[:, 2:], target_masks_l_list[:, 2:], \
                                               batch_next_repr_q, batch_next_repr_neg_q, batch_pred_repr_q, contrastive_pos_q, contrastive_neg_q, batch_pred_repr_dec_q, batch_next_repr_dec_q, batch_pred_repr_dec_gt_q, batch_next_repr_dec_gt_q, \
                                                batch_next_repr_l, batch_next_repr_neg_l, batch_pred_repr_l, contrastive_pos_l, contrastive_neg_l, batch_pred_repr_dec_l, batch_next_repr_dec_l, batch_pred_repr_dec_gt_l, batch_next_repr_dec_gt_l, \
                                                self.config.num_eval_negative_sampling, evaluation=True)

                self.test_output_all.extend(output.tolist())
                self.test_output_type_all.extend(output_type.tolist())
                self.test_label_all.extend(label.tolist())
                self.test_label_type_all.extend(label_type.tolist())

                self.test_q_hitratios.append(metrics_q["hitratio_q"])
                self.test_q_ndcgs.append(metrics_q["ndcg_q"])
                self.test_q_mrrs.append(metrics_q["mrr_q"])

                self.test_l_hitratios.append(metrics_q["hitratio_l"])
                self.test_l_ndcgs.append(metrics_q["ndcg_l"])
                self.test_l_mrrs.append(metrics_q["mrr_l"])

            self.test_output_all = np.array(self.test_output_all).squeeze()
            self.test_label_all = np.array(self.test_label_all).squeeze()
            self.test_output_type_all = np.array(self.test_output_type_all).squeeze()
            self.test_label_type_all = np.array(self.test_label_type_all).squeeze()

            if self.metric == "rmse":
                test_evals = np.stack(
                    [np.sqrt(metrics.mean_squared_error(self.test_label_all, self.test_output_all)),
                     metrics.roc_auc_score(self.test_label_type_all, self.test_output_type_all),
                     ])
            elif self.metric == "auc":
                test_evals = np.stack(
                    [metrics.roc_auc_score(self.test_label_all, self.test_output_all),
                     metrics.roc_auc_score(self.test_label_type_all, self.test_output_type_all),
                     sum(self.test_q_hitratios) / len(self.test_q_hitratios),
                     sum(self.test_q_ndcgs) / len(self.test_q_ndcgs),
                     sum(self.test_q_mrrs) / len(self.test_q_mrrs),
                     sum(self.test_l_hitratios) / len(self.test_l_hitratios),
                     sum(self.test_l_ndcgs) / len(self.test_l_ndcgs),
                     sum(self.test_l_mrrs) / len(self.test_l_mrrs),
                     ])

        return test_evals


class trainer_mat_nopers(trainer_mat_pers):

    def init_model(self, data):
        self.data_loader = KTBM_DataLoader_personalized(self.config, data, random_state=self.config.seed)
        # self.data_loader = KTBM_DataLoader_personalized_stateful(self.config, data, random_state=self.config.seed)
        self.config.num_students = self.data_loader.num_students
        self.model = KTBM_mat_nopers(self.config)
        self.model.initialize()

    def init_optimizer(self):
        if self.config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                             lr=self.config.learning_rate,
                                             momentum=self.config.momentum,
                                             weight_decay=self.config.weight_decay)
        elif self.config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                              lr=self.config.learning_rate,
                                              betas=(self.config.beta1, self.config.beta2),
                                              eps=self.config.epsilon,
                                              weight_decay=self.config.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=3,
            min_lr=1e-5,
            factor=0.5,
            verbose=True
        )

    def train_one_epoch(self):
        self.model.train()
        for data in tqdm.tqdm(self.data_loader.train_loader):
            q_list, a_list, l_list, d_list, s_list, target_answers_list, target_masks_list, target_masks_l_list = data
            q_list = q_list.to(self.device)
            a_list = a_list.to(self.device)
            l_list = l_list.to(self.device)
            d_list = d_list.to(self.device)
            s_list = s_list.to(self.device)
            target_answers_list = target_answers_list.to(self.device)
            target_masks_list = target_masks_list.to(self.device)
            target_masks_l_list = target_masks_l_list.to(self.device)

            self.optimizer.zero_grad()
            output, output_type, \
                contrastive_pos_q, contrastive_neg_q, \
                batch_next_repr_neg_q, batch_pred_repr_q, batch_next_repr_q, \
                batch_pred_repr_dec_q, batch_next_repr_dec_q, \
                batch_pred_repr_dec_gt_q, batch_next_repr_dec_gt_q, \
                contrastive_pos_l, contrastive_neg_l, \
                batch_next_repr_neg_l, batch_pred_repr_l, batch_next_repr_l, \
                batch_pred_repr_dec_l, batch_next_repr_dec_l, \
                batch_pred_repr_dec_gt_l, batch_next_repr_dec_gt_l = self.model(q_list, a_list, l_list, d_list, s_list)

            label = torch.masked_select(target_answers_list[:, 2:], target_masks_list[:, 2:])
            label_type = torch.masked_select(d_list[:, 2:], (target_masks_list + target_masks_l_list)[:, 2:])

            output = torch.masked_select(output, target_masks_list[:, 2:])
            loss_q = self.bce_criterion(output.float(), label.float())

            output_type = torch.masked_select(output_type, (target_masks_list + target_masks_l_list)[:, 2:])
            loss_type = self.bce_criterion(output_type.float(), label_type.float())

            metrics_q = self.calculate_metrics(target_masks_list[:, 2:], target_masks_l_list[:, 2:], \
                                               batch_next_repr_q, batch_next_repr_neg_q, batch_pred_repr_q, contrastive_pos_q, contrastive_neg_q, batch_pred_repr_dec_q, batch_next_repr_dec_q, batch_pred_repr_dec_gt_q, batch_next_repr_dec_gt_q, \
                                                batch_next_repr_l, batch_next_repr_neg_l, batch_pred_repr_l, contrastive_pos_l, contrastive_neg_l, batch_pred_repr_dec_l, batch_next_repr_dec_l, batch_pred_repr_dec_gt_l, batch_next_repr_dec_gt_l, \
                                                self.config.num_train_negative_sampling)

            task_loss = torch.stack([loss_q, loss_type, metrics_q["contrastive_loss"], metrics_q["ntxent_loss"], metrics_q["reconstruction_loss"]])

            (task_loss.sum()).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

    def validate(self):
        self.train_loss = 0
        self.train_loss_type = 0
        self.train_loss_contrastive = 0
        self.train_loss_triplet = 0
        train_elements = 0
        train_elements_type = 0
        train_elements_contrastive = 0
        train_elements_triplet = 0

        self.test_loss = 0
        self.test_loss_type = 0
        self.test_loss_contrastive = 0
        self.test_loss_triplet = 0
        test_elements = 0
        test_elements_type = 0
        test_elements_contrastive = 0
        test_elements_triplet = 0

        self.model.eval()
        with torch.no_grad():
            self.train_output_all = []
            self.train_output_type_all = []
            self.train_contrastive_all = []
            self.train_label_all = []
            self.train_label_type_all = []
            self.train_label_contrastive_all = []
            self.train_q_hitratios = []
            self.train_q_ndcgs = []
            self.train_q_mrrs = []
            self.train_l_hitratios = []
            self.train_l_ndcgs = []
            self.train_l_mrrs = []

            self.test_output_all = []
            self.test_output_type_all = []
            self.test_contrastive_all = []
            self.test_label_all = []
            self.test_label_type_all = []
            self.test_label_contrastive_all = []
            self.test_q_hitratios = []
            self.test_q_ndcgs = []
            self.test_q_mrrs = []
            self.test_l_hitratios = []
            self.test_l_ndcgs = []
            self.test_l_mrrs = []

            for data in tqdm.tqdm(self.data_loader.train_loader):
                q_list, a_list, l_list, d_list, s_list, target_answers_list, target_masks_list, target_masks_l_list = data
                q_list = q_list.to(self.device)
                a_list = a_list.to(self.device)
                l_list = l_list.to(self.device)
                d_list = d_list.to(self.device)
                s_list = s_list.to(self.device)
                target_answers_list = target_answers_list.to(self.device)
                target_masks_list = target_masks_list.to(self.device)
                target_masks_l_list = target_masks_l_list.to(self.device)

                output, output_type, \
                    contrastive_pos_q, contrastive_neg_q, \
                    batch_next_repr_neg_q, batch_pred_repr_q, batch_next_repr_q, \
                    batch_pred_repr_dec_q, batch_next_repr_dec_q, \
                    batch_pred_repr_dec_gt_q, batch_next_repr_dec_gt_q, \
                    contrastive_pos_l, contrastive_neg_l, \
                    batch_next_repr_neg_l, batch_pred_repr_l, batch_next_repr_l, \
                    batch_pred_repr_dec_l, batch_next_repr_dec_l, \
                    batch_pred_repr_dec_gt_l, batch_next_repr_dec_gt_l = self.model(q_list, a_list, l_list, d_list, s_list, evaluation=True)

                label = torch.masked_select(target_answers_list[:, 2:], target_masks_list[:, 2:])
                label_type = torch.masked_select(d_list[:, 2:], (target_masks_list + target_masks_l_list)[:, 2:])

                output = torch.masked_select(output, target_masks_list[:, 2:])
                loss_q = self.bce_criterion(output.float(), label.float())

                output_type = torch.masked_select(output_type, (target_masks_list + target_masks_l_list)[:, 2:])
                loss_type = self.bce_criterion(output_type.float(), label_type.float())

                metrics_q = self.calculate_metrics(target_masks_list[:, 2:], target_masks_l_list[:, 2:], \
                                               batch_next_repr_q, batch_next_repr_neg_q, batch_pred_repr_q, contrastive_pos_q, contrastive_neg_q, batch_pred_repr_dec_q, batch_next_repr_dec_q, batch_pred_repr_dec_gt_q, batch_next_repr_dec_gt_q, \
                                                batch_next_repr_l, batch_next_repr_neg_l, batch_pred_repr_l, contrastive_pos_l, contrastive_neg_l, batch_pred_repr_dec_l, batch_next_repr_dec_l, batch_pred_repr_dec_gt_l, batch_next_repr_dec_gt_l, \
                                                self.config.num_eval_negative_sampling, evaluation=True)

                contrastive_loss = metrics_q["contrastive_loss"] + metrics_q["reconstruction_loss"]
                ntxent_loss = metrics_q["ntxent_loss"]

                self.train_loss += loss_q.item()
                train_elements += torch.tensor(1)
                self.train_loss_type += loss_type.item()
                train_elements_type += torch.tensor(1)
                self.train_loss_contrastive += contrastive_loss.item()
                train_elements_contrastive += torch.tensor(1)
                self.train_loss_triplet += ntxent_loss.item()
                train_elements_triplet += torch.tensor(1)

                self.train_output_all.extend(output.tolist())
                self.train_output_type_all.extend(output_type.tolist())
                self.train_contrastive_all.extend(metrics_q["contrastive_preds"])
                self.train_label_all.extend(label.tolist())
                self.train_label_type_all.extend(label_type.tolist())
                self.train_label_contrastive_all.extend(metrics_q["contrastive_labels"])

                self.train_q_hitratios.append(metrics_q["hitratio_q"])
                self.train_q_ndcgs.append(metrics_q["ndcg_q"])
                self.train_q_mrrs.append(metrics_q["mrr_q"])

                self.train_l_hitratios.append(metrics_q["hitratio_l"])
                self.train_l_ndcgs.append(metrics_q["ndcg_l"])
                self.train_l_mrrs.append(metrics_q["mrr_l"])
                
            for data in tqdm.tqdm(self.data_loader.test_loader):
                q_list, a_list, l_list, d_list, s_list, target_answers_list, target_masks_list, target_masks_l_list = data
                q_list = q_list.to(self.device)
                a_list = a_list.to(self.device)
                l_list = l_list.to(self.device)
                d_list = d_list.to(self.device)
                s_list = s_list.to(self.device)
                target_answers_list = target_answers_list.to(self.device)
                target_masks_list = target_masks_list.to(self.device)
                target_masks_l_list = target_masks_l_list.to(self.device)

                output, output_type, \
                    contrastive_pos_q, contrastive_neg_q, \
                    batch_next_repr_neg_q, batch_pred_repr_q, batch_next_repr_q, \
                    batch_pred_repr_dec_q, batch_next_repr_dec_q, \
                    batch_pred_repr_dec_gt_q, batch_next_repr_dec_gt_q, \
                    contrastive_pos_l, contrastive_neg_l, \
                    batch_next_repr_neg_l, batch_pred_repr_l, batch_next_repr_l, \
                    batch_pred_repr_dec_l, batch_next_repr_dec_l, \
                    batch_pred_repr_dec_gt_l, batch_next_repr_dec_gt_l = self.model(q_list, a_list, l_list, d_list, s_list, evaluation=True)

                label = torch.masked_select(target_answers_list[:, 2:], target_masks_list[:, 2:])
                label_type = torch.masked_select(d_list[:, 2:], (target_masks_list + target_masks_l_list)[:, 2:])

                output = torch.masked_select(output, target_masks_list[:, 2:])
                loss_q = self.bce_criterion(output.float(), label.float())

                output_type = torch.masked_select(output_type, (target_masks_list + target_masks_l_list)[:, 2:])
                loss_type = self.bce_criterion(output_type.float(), label_type.float())

                metrics_q = self.calculate_metrics(target_masks_list[:, 2:], target_masks_l_list[:, 2:], \
                                               batch_next_repr_q, batch_next_repr_neg_q, batch_pred_repr_q, contrastive_pos_q, contrastive_neg_q, batch_pred_repr_dec_q, batch_next_repr_dec_q, batch_pred_repr_dec_gt_q, batch_next_repr_dec_gt_q, \
                                                batch_next_repr_l, batch_next_repr_neg_l, batch_pred_repr_l, contrastive_pos_l, contrastive_neg_l, batch_pred_repr_dec_l, batch_next_repr_dec_l, batch_pred_repr_dec_gt_l, batch_next_repr_dec_gt_l, \
                                                self.config.num_eval_negative_sampling, evaluation=True)

                contrastive_loss = metrics_q["contrastive_loss"] + metrics_q["reconstruction_loss"]
                ntxent_loss = metrics_q["ntxent_loss"]

                self.test_loss += loss_q.item()
                test_elements += torch.tensor(1)
                self.test_loss_type += loss_type.item()
                test_elements_type += torch.tensor(1)
                self.test_loss_contrastive += contrastive_loss.item()
                test_elements_contrastive += torch.tensor(1)
                self.test_loss_triplet += ntxent_loss.item()
                test_elements_triplet += torch.tensor(1)

                self.test_output_all.extend(output.tolist())
                self.test_output_type_all.extend(output_type.tolist())
                self.test_contrastive_all.extend(metrics_q["contrastive_preds"])
                self.test_label_all.extend(label.tolist())
                self.test_label_type_all.extend(label_type.tolist())
                self.test_label_contrastive_all.extend(metrics_q["contrastive_labels"])

                self.test_q_hitratios.append(metrics_q["hitratio_q"])
                self.test_q_ndcgs.append(metrics_q["ndcg_q"])
                self.test_q_mrrs.append(metrics_q["mrr_q"])

                self.test_l_hitratios.append(metrics_q["hitratio_l"])
                self.test_l_ndcgs.append(metrics_q["ndcg_l"])
                self.test_l_mrrs.append(metrics_q["mrr_l"])
                
            self.train_output_all = np.array(self.train_output_all).squeeze()
            self.train_label_all = np.array(self.train_label_all).squeeze()
            self.train_output_type_all = np.array(self.train_output_type_all).squeeze()
            self.train_label_type_all = np.array(self.train_label_type_all).squeeze()
            self.train_contrastive_all = np.array(self.train_contrastive_all).squeeze()
            self.train_label_contrastive_all = np.array(self.train_label_contrastive_all).squeeze()

            self.test_output_all = np.array(self.test_output_all).squeeze()
            self.test_label_all = np.array(self.test_label_all).squeeze()
            self.test_output_type_all = np.array(self.test_output_type_all).squeeze()
            self.test_label_type_all = np.array(self.test_label_type_all).squeeze()
            self.test_contrastive_all = np.array(self.test_contrastive_all).squeeze()
            self.test_label_contrastive_all = np.array(self.test_label_contrastive_all).squeeze()

            train_evals = np.stack(
                [metrics.roc_auc_score(self.train_label_all, self.train_output_all),
                    metrics.roc_auc_score(self.train_label_type_all, self.train_output_type_all),
                    metrics.accuracy_score(self.train_label_contrastive_all, self.train_contrastive_all),
                    sum(self.train_q_hitratios) / len(self.train_q_hitratios),
                    sum(self.train_q_ndcgs) / len(self.train_q_ndcgs),
                    sum(self.train_q_mrrs) / len(self.train_q_mrrs),
                    sum(self.train_l_hitratios) / len(self.train_l_hitratios),
                    sum(self.train_l_ndcgs) / len(self.train_l_ndcgs),
                    sum(self.train_l_mrrs) / len(self.train_l_mrrs),
                    ])
            test_evals = np.stack(
                [metrics.roc_auc_score(self.test_label_all, self.test_output_all),
                    metrics.roc_auc_score(self.test_label_type_all, self.test_output_type_all),
                    metrics.accuracy_score(self.test_label_contrastive_all, self.test_contrastive_all),
                    sum(self.test_q_hitratios) / len(self.test_q_hitratios),
                    sum(self.test_q_ndcgs) / len(self.test_q_ndcgs),
                    sum(self.test_q_mrrs) / len(self.test_q_mrrs),
                    sum(self.test_l_hitratios) / len(self.test_l_hitratios),
                    sum(self.test_l_ndcgs) / len(self.test_l_ndcgs),
                    sum(self.test_l_mrrs) / len(self.test_l_mrrs),
                    ])

            average_train_loss = torch.stack([self.train_loss / train_elements, self.train_loss_type / train_elements_type, self.train_loss_contrastive / train_elements_contrastive, self.train_loss_triplet / train_elements_triplet])
            average_test_loss = torch.stack([self.test_loss / test_elements, self.test_loss_type / test_elements_type, self.test_loss_contrastive / test_elements_contrastive, self.test_loss_triplet / test_elements_triplet])

        self.task_train_losses.append(average_train_loss.data.cpu().numpy())
        self.train_evals.append(train_evals)
        self.task_test_losses.append(average_test_loss.data.cpu().numpy())
        self.test_evals.append(test_evals)

        self.save_weights()

        print('{}/{}: train_loss={}, train_acc={}, test_loss={}, test_acc={}'.format(
            self.current_epoch, self.config.max_epoch, self.task_train_losses[-1], self.train_evals[-1],
            self.task_test_losses[-1], self.test_evals[-1]))

        self.scheduler.step(np.sum(self.task_train_losses[-1]))

    def save_weights(self):
        if not os.path.isdir(f"checkpoints_{self.config.data_name}_proposed_mat_nopers"):
            os.makedirs((f"checkpoints_{self.config.data_name}_proposed_mat_nopers"))

        if self.current_epoch % self.config.save_checkpoint_every == 0:
            torch.save(self.model.cpu().state_dict(), os.path.join(f"checkpoints_{self.config.data_name}_proposed_mat_nopers", "checkpoint-{:04}.pt".format(self.current_epoch)))

    def validate_for_hypo_test(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            self.test_output_all = []
            self.test_output_type_all = []
            self.test_contrastive_all = []
            self.test_label_all = []
            self.test_label_type_all = []
            self.test_label_contrastive_all = []
            self.test_q_hitratios = []
            self.test_q_ndcgs = []
            self.test_q_mrrs = []
            self.test_l_hitratios = []
            self.test_l_ndcgs = []
            self.test_l_mrrs = []

            for data in tqdm.tqdm(test_loader):
                q_list, a_list, l_list, d_list, s_list, target_answers_list, target_masks_list, target_masks_l_list = data
                q_list = q_list.to(self.device)
                a_list = a_list.to(self.device)
                l_list = l_list.to(self.device)
                d_list = d_list.to(self.device)
                s_list = s_list.to(self.device)
                target_answers_list = target_answers_list.to(self.device)
                target_masks_list = target_masks_list.to(self.device)
                target_masks_l_list = target_masks_l_list.to(self.device)

                output, output_type, \
                    contrastive_pos_q, contrastive_neg_q, \
                    batch_next_repr_neg_q, batch_pred_repr_q, batch_next_repr_q, \
                    batch_pred_repr_dec_q, batch_next_repr_dec_q, \
                    batch_pred_repr_dec_gt_q, batch_next_repr_dec_gt_q, \
                    contrastive_pos_l, contrastive_neg_l, \
                    batch_next_repr_neg_l, batch_pred_repr_l, batch_next_repr_l, \
                    batch_pred_repr_dec_l, batch_next_repr_dec_l, \
                    batch_pred_repr_dec_gt_l, batch_next_repr_dec_gt_l = self.model(q_list, a_list, l_list, d_list, s_list, evaluation=True)

                label = torch.masked_select(target_answers_list[:, 2:], target_masks_list[:, 2:])
                label_type = torch.masked_select(d_list[:, 2:], (target_masks_list + target_masks_l_list)[:, 2:])

                output = torch.masked_select(output, target_masks_list[:, 2:])
                output_type = torch.masked_select(output_type, (target_masks_list + target_masks_l_list)[:, 2:])
                metrics_q = self.calculate_metrics(target_masks_list[:, 2:], target_masks_l_list[:, 2:], \
                                               batch_next_repr_q, batch_next_repr_neg_q, batch_pred_repr_q, contrastive_pos_q, contrastive_neg_q, batch_pred_repr_dec_q, batch_next_repr_dec_q, batch_pred_repr_dec_gt_q, batch_next_repr_dec_gt_q, \
                                                batch_next_repr_l, batch_next_repr_neg_l, batch_pred_repr_l, contrastive_pos_l, contrastive_neg_l, batch_pred_repr_dec_l, batch_next_repr_dec_l, batch_pred_repr_dec_gt_l, batch_next_repr_dec_gt_l, \
                                                self.config.num_eval_negative_sampling, evaluation=True)

                self.test_output_all.extend(output.tolist())
                self.test_output_type_all.extend(output_type.tolist())
                self.test_label_all.extend(label.tolist())
                self.test_label_type_all.extend(label_type.tolist())

                self.test_q_hitratios.append(metrics_q["hitratio_q"])
                self.test_q_ndcgs.append(metrics_q["ndcg_q"])
                self.test_q_mrrs.append(metrics_q["mrr_q"])

                self.test_l_hitratios.append(metrics_q["hitratio_l"])
                self.test_l_ndcgs.append(metrics_q["ndcg_l"])
                self.test_l_mrrs.append(metrics_q["mrr_l"])

            self.test_output_all = np.array(self.test_output_all).squeeze()
            self.test_label_all = np.array(self.test_label_all).squeeze()
            self.test_output_type_all = np.array(self.test_output_type_all).squeeze()
            self.test_label_type_all = np.array(self.test_label_type_all).squeeze()

            if self.metric == "rmse":
                test_evals = np.stack(
                    [np.sqrt(metrics.mean_squared_error(self.test_label_all, self.test_output_all)),
                     metrics.roc_auc_score(self.test_label_type_all, self.test_output_type_all),
                     ])
            elif self.metric == "auc":
                test_evals = np.stack(
                    [metrics.roc_auc_score(self.test_label_all, self.test_output_all),
                     metrics.roc_auc_score(self.test_label_type_all, self.test_output_type_all),
                     sum(self.test_q_hitratios) / len(self.test_q_hitratios),
                     sum(self.test_q_ndcgs) / len(self.test_q_ndcgs),
                     sum(self.test_q_mrrs) / len(self.test_q_mrrs),
                     sum(self.test_l_hitratios) / len(self.test_l_hitratios),
                     sum(self.test_l_ndcgs) / len(self.test_l_ndcgs),
                     sum(self.test_l_mrrs) / len(self.test_l_mrrs),
                     ])

        return test_evals


class trainer_pers_nomat(trainer_mat_pers):

    def init_model(self, data):
        self.data_loader = KTBM_DataLoader_personalized_stateful(self.config, data, random_state=self.config.seed)
        self.config.num_students = self.data_loader.num_students
        self.model = KTBM_pers_nomat(self.config)
        self.model.initialize()

        self.kmeans = SoftKmeans(self.config.num_clusters)

    def train_one_epoch(self):
        self.model.initialize_states()
        self.model.train()

        student_embeddings = []
        behavior_embeddings = []
        student_ids = []
        for data in tqdm.tqdm(self.data_loader.train_loader):
            q_list, a_list, l_list, d_list, s_list, target_answers_list, target_masks_list, target_masks_l_list = data
            
            q_list = q_list.to(self.device)
            a_list = a_list.to(self.device)
            l_list = l_list.to(self.device)
            d_list = d_list.to(self.device)
            s_list = s_list.to(self.device)
            target_answers_list = target_answers_list.to(self.device)
            target_masks_list = target_masks_list.to(self.device)
            target_masks_l_list = target_masks_l_list.to(self.device)

            self.optimizer.zero_grad()
            output, output_type = self.model(q_list, a_list, l_list, d_list, s_list)

            label = torch.masked_select(target_answers_list[:, 2:], target_masks_list[:, 2:])
            label_type = torch.masked_select(d_list[:, 2:], (target_masks_list + target_masks_l_list)[:, 2:])

            output = torch.masked_select(output, target_masks_list[:, 2:])
            loss_q = self.bce_criterion(output.float(), label.float())

            output_type = torch.masked_select(output_type, (target_masks_list + target_masks_l_list)[:, 2:])
            loss_type = self.bce_criterion(output_type.float(), label_type.float())

            task_loss = torch.stack([loss_q, loss_type])
            task_loss.sum().backward()

            new_embedding_weights, behavior_embedding = self.get_new_student_embeddings(s_list)
            student_embeddings.append(new_embedding_weights)
            behavior_embeddings.append(behavior_embedding)
            student_ids.append(s_list.detach())

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        student_embeddings = torch.concat(student_embeddings)
        behavior_embeddings = torch.concat(behavior_embeddings)
        student_ids = torch.concat(student_ids)

        self.update_student_embeddings(student_embeddings, behavior_embeddings, student_ids)

    def validate(self):
        self.train_loss = 0
        self.train_loss_type = 0
        train_elements = 0
        train_elements_type = 0

        self.test_loss = 0
        self.test_loss_type = 0
        test_elements = 0
        test_elements_type = 0

        self.model.initialize_states()
        self.model.eval()
        with torch.no_grad():
            self.train_output_all = []
            self.train_output_type_all = []
            self.train_label_all = []
            self.train_label_type_all = []

            self.test_output_all = []
            self.test_output_type_all = []
            self.test_label_all = []
            self.test_label_type_all = []

            for data in tqdm.tqdm(self.data_loader.train_loader):
                q_list, a_list, l_list, d_list, s_list, target_answers_list, target_masks_list, target_masks_l_list = data
                q_list = q_list.to(self.device)
                a_list = a_list.to(self.device)
                l_list = l_list.to(self.device)
                d_list = d_list.to(self.device)
                s_list = s_list.to(self.device)
                target_answers_list = target_answers_list.to(self.device)
                target_masks_list = target_masks_list.to(self.device)
                target_masks_l_list = target_masks_l_list.to(self.device)

                output, output_type = self.model(q_list, a_list, l_list, d_list, s_list, evaluation=True)

                label = torch.masked_select(target_answers_list[:, 2:], target_masks_list[:, 2:])
                label_type = torch.masked_select(d_list[:, 2:], (target_masks_list + target_masks_l_list)[:, 2:])

                output = torch.masked_select(output, target_masks_list[:, 2:])
                loss_q = self.bce_criterion(output.float(), label.float())

                output_type = torch.masked_select(output_type, (target_masks_list + target_masks_l_list)[:, 2:])
                loss_type = self.bce_criterion(output_type.float(), label_type.float())

                self.train_loss += loss_q.item()
                train_elements += torch.tensor(1)
                self.train_loss_type += loss_type.item()
                train_elements_type += torch.tensor(1)

                self.train_output_all.extend(output.tolist())
                self.train_output_type_all.extend(output_type.tolist())
                self.train_label_all.extend(label.tolist())
                self.train_label_type_all.extend(label_type.tolist())

            for data in tqdm.tqdm(self.data_loader.test_loader):
                q_list, a_list, l_list, d_list, s_list, target_answers_list, target_masks_list, target_masks_l_list = data
                q_list = q_list.to(self.device)
                a_list = a_list.to(self.device)
                l_list = l_list.to(self.device)
                d_list = d_list.to(self.device)
                s_list = s_list.to(self.device)
                target_answers_list = target_answers_list.to(self.device)
                target_masks_list = target_masks_list.to(self.device)
                target_masks_l_list = target_masks_l_list.to(self.device)

                output, output_type = self.model(q_list, a_list, l_list, d_list, s_list, evaluation=True)

                label = torch.masked_select(target_answers_list[:, 2:], target_masks_list[:, 2:])
                label_type = torch.masked_select(d_list[:, 2:], (target_masks_list + target_masks_l_list)[:, 2:])

                output = torch.masked_select(output, target_masks_list[:, 2:])
                loss_q = self.bce_criterion(output.float(), label.float())

                output_type = torch.masked_select(output_type, (target_masks_list + target_masks_l_list)[:, 2:])
                loss_type = self.bce_criterion(output_type.float(), label_type.float())

                self.test_loss += loss_q.item()
                test_elements += torch.tensor(1)
                self.test_loss_type += loss_type.item()
                test_elements_type += torch.tensor(1)

                self.test_output_all.extend(output.tolist())
                self.test_output_type_all.extend(output_type.tolist())
                self.test_label_all.extend(label.tolist())
                self.test_label_type_all.extend(label_type.tolist())

            self.train_output_all = np.array(self.train_output_all).squeeze()
            self.train_label_all = np.array(self.train_label_all).squeeze()
            self.train_output_type_all = np.array(self.train_output_type_all).squeeze()
            self.train_label_type_all = np.array(self.train_label_type_all).squeeze()
            
            self.test_output_all = np.array(self.test_output_all).squeeze()
            self.test_label_all = np.array(self.test_label_all).squeeze()
            self.test_output_type_all = np.array(self.test_output_type_all).squeeze()
            self.test_label_type_all = np.array(self.test_label_type_all).squeeze()
            
            train_evals = np.stack(
                [metrics.roc_auc_score(self.train_label_all, self.train_output_all),
                    metrics.roc_auc_score(self.train_label_type_all, self.train_output_type_all),
                    ])
            test_evals = np.stack(
                [metrics.roc_auc_score(self.test_label_all, self.test_output_all),
                    metrics.roc_auc_score(self.test_label_type_all, self.test_output_type_all),
                    ])

            average_train_loss = torch.stack([self.train_loss / train_elements, self.train_loss_type / train_elements_type])
            average_test_loss = torch.stack([self.test_loss / test_elements, self.test_loss_type / test_elements_type])

        self.task_train_losses.append(average_train_loss.data.cpu().numpy())
        self.train_evals.append(train_evals)
        self.task_test_losses.append(average_test_loss.data.cpu().numpy())
        self.test_evals.append(test_evals)

        self.save_weights()

        print('{}/{}: train_loss={}, train_acc={}, test_loss={}, test_acc={}'.format(
            self.current_epoch, self.config.max_epoch, self.task_train_losses[-1], self.train_evals[-1],
            self.task_test_losses[-1], self.test_evals[-1]))

        self.scheduler.step(np.sum(self.task_train_losses[-1]))

    def save_weights(self):
        if not os.path.isdir(f"checkpoints_{self.config.data_name}_proposed_pers_nomat"):
            os.makedirs((f"checkpoints_{self.config.data_name}_proposed_pers_nomat"))

        if self.current_epoch % self.config.save_checkpoint_every == 0:
            torch.save(self.model.cpu().state_dict(), os.path.join(f"checkpoints_{self.config.data_name}_proposed_pers_nomat", "checkpoint-{:04}.pt".format(self.current_epoch)))

    def validate_for_hypo_test(self, test_loader):
        self.model.initialize_states()
        self.model.eval()
        with torch.no_grad():
            self.test_output_all = []
            self.test_output_type_all = []
            self.test_contrastive_all = []
            self.test_label_all = []
            self.test_label_type_all = []
            self.test_label_contrastive_all = []

            for data in tqdm.tqdm(test_loader):
                q_list, a_list, l_list, d_list, s_list, target_answers_list, target_masks_list, target_masks_l_list = data
                q_list = q_list.to(self.device)
                a_list = a_list.to(self.device)
                l_list = l_list.to(self.device)
                d_list = d_list.to(self.device)
                s_list = s_list.to(self.device)
                target_answers_list = target_answers_list.to(self.device)
                target_masks_list = target_masks_list.to(self.device)
                target_masks_l_list = target_masks_l_list.to(self.device)

                output, output_type = self.model(q_list, a_list, l_list, d_list, s_list, evaluation=True)

                label = torch.masked_select(target_answers_list[:, 2:], target_masks_list[:, 2:])
                label_type = torch.masked_select(d_list[:, 2:], (target_masks_list + target_masks_l_list)[:, 2:])

                output = torch.masked_select(output, target_masks_list[:, 2:])
                output_type = torch.masked_select(output_type, (target_masks_list + target_masks_l_list)[:, 2:])

                self.test_output_all.extend(output.tolist())
                self.test_output_type_all.extend(output_type.tolist())
                self.test_label_all.extend(label.tolist())
                self.test_label_type_all.extend(label_type.tolist())

            self.test_output_all = np.array(self.test_output_all).squeeze()
            self.test_label_all = np.array(self.test_label_all).squeeze()
            self.test_output_type_all = np.array(self.test_output_type_all).squeeze()
            self.test_label_type_all = np.array(self.test_label_type_all).squeeze()

            if self.metric == "rmse":
                test_evals = np.stack(
                    [np.sqrt(metrics.mean_squared_error(self.test_label_all, self.test_output_all)),
                     metrics.roc_auc_score(self.test_label_type_all, self.test_output_type_all),
                     ])
            elif self.metric == "auc":
                test_evals = np.stack(
                    [metrics.roc_auc_score(self.test_label_all, self.test_output_all),
                     metrics.roc_auc_score(self.test_label_type_all, self.test_output_type_all)
                     ])

        return test_evals