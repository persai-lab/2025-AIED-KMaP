import random
import numpy as np
import torch
import torch.nn as nn
from model.modules import MultiHeadAttentionModule
from utils.metrics import Metrics
import torch.nn.functional as F


class KTBM_mat_pers(nn.Module):

    def __init__(self, config):
        super(KTBM_mat_pers, self).__init__()

        self.is_cuda = torch.cuda.is_available()
        self.cuda = self.is_cuda & config.cuda
        if self.cuda:
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cuda")
        else:
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cpu")
        self.metric = config.metric
        self.config = config
        
        self.metrics = Metrics(config.top_k_metrics)

        self.num_eval_negative_sampling = config.num_eval_negative_sampling
        self.num_train_negative_sampling = config.num_train_negative_sampling

        # initialize the dim size hyper parameters
        self.num_questions = config.num_items
        self.num_nongradable_items = config.num_nongradable_items
        self.embedding_size_q = config.embedding_size_q
        self.embedding_size_a = config.embedding_size_a
        self.embedding_size_l = config.embedding_size_l
        self.embedding_size_d = config.embedding_size_d
        self.embedding_size_s = config.embedding_size_s
        self.embedding_size_q_behavior = config.embedding_size_q_behavior
        self.embedding_size_l_behavior = config.embedding_size_l_behavior
        self.embedding_size_latent = config.embedding_size_latent

        self.num_concepts = config.num_concepts
        self.key_dim = config.key_dim
        self.value_dim = config.value_dim
        self.summary_dim = config.summary_dim
        self.init_std = config.init_std
        self.num_students = config.num_students

        self.behavior_summary_fc = config.behavior_summary_fc
        self.behavior_map_size = config.behavior_map_size
        self.behavior_hidden_size = config.behavior_hidden_size

        # initialize the activiate functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(-1)

    def initialize(self):
        self.init_embeddings_module()
        self.init_knowledge_module()
        self.init_behavior_module()
        self.init_material_pred_module()

    def init_embeddings_module(self):
        self.q_embed_matrix = nn.Embedding(num_embeddings=self.num_questions + 1, embedding_dim=self.embedding_size_q, padding_idx=0)
        self.l_embed_matrix = nn.Embedding(num_embeddings=self.num_nongradable_items + 1, embedding_dim=self.embedding_size_l, padding_idx=0)

        self.q_corr_weight_matrix = nn.Sequential(
            nn.Embedding(num_embeddings=self.num_questions + 1, embedding_dim=self.num_concepts, padding_idx=0),
            nn.Softmax(-1)
        )
        self.l_corr_weight_matrix = nn.Sequential(
            nn.Embedding(num_embeddings=self.num_nongradable_items + 1, embedding_dim=self.num_concepts, padding_idx=0),
            nn.Softmax(-1)
        )

        if self.metric == "rmse":
            self.a_embed_matrix = nn.Linear(1, self.embedding_size_a)
        else:
            self.a_embed_matrix = nn.Embedding(3, self.embedding_size_a, padding_idx=2)

        self.d_embed_matrix = nn.Embedding(3, self.embedding_size_d, padding_idx=2)
        self.s_embed_matrix = nn.Embedding(num_embeddings=self.num_students, embedding_dim=self.embedding_size_s)

        self.q_behavior_embed_matrix = nn.Embedding(num_embeddings=self.num_questions + 1, embedding_dim=self.embedding_size_q_behavior, padding_idx=0)
        self.l_behavior_embed_matrix = nn.Embedding(num_embeddings=self.num_nongradable_items + 1, embedding_dim=self.embedding_size_l_behavior, padding_idx=0)

    def init_knowledge_module(self):
        self.erase_E_Q = nn.Linear(self.embedding_size_q + self.embedding_size_a, self.value_dim, bias=True)
        self.erase_E_L = nn.Linear(self.embedding_size_l, self.value_dim, bias=True)
        self.erase_E_bh = nn.Linear(self.behavior_hidden_size, self.value_dim, bias=False)
        self.erase_E_stu = nn.Linear(self.embedding_size_s, self.value_dim, bias=True)

        self.add_D_Q = nn.Linear(self.embedding_size_q + self.embedding_size_a, self.value_dim, bias=True)
        self.add_D_L = nn.Linear(self.embedding_size_l, self.value_dim, bias=True)
        self.add_D_bh = nn.Linear(self.behavior_hidden_size, self.value_dim, bias=False)
        self.add_D_stu = nn.Linear(self.embedding_size_s, self.value_dim, bias=False)

        self.transition_proj_M = nn.Linear(2*self.embedding_size_d, self.value_dim, bias=True)

        self.linear_out = nn.Sequential(
            nn.Linear(self.embedding_size_q + self.value_dim + self.behavior_hidden_size, self.summary_dim),
            nn.Tanh(),
            nn.Linear(self.summary_dim, 1),
        ) 

    def init_behavior_module(self):
        # initialize the LSTM layers
        self.behavior_mapQ = nn.Linear(self.embedding_size_q_behavior + self.embedding_size_d, self.behavior_map_size, bias=True)
        self.behavior_mapL = nn.Linear(self.embedding_size_l_behavior + self.embedding_size_d, self.behavior_map_size, bias=True)

        self.sum_knowledge2behavior = nn.Linear(self.num_concepts, 1, bias=True)

        self.W_i = nn.Linear(self.behavior_map_size, self.behavior_hidden_size, bias=True)
        self.W_i_knowledge = nn.Linear(self.value_dim, self.behavior_hidden_size, bias=True)
        self.W_ih = nn.Linear(self.behavior_hidden_size, self.behavior_hidden_size, bias=False)
        self.W_i_stu = nn.Linear(self.embedding_size_s, self.behavior_hidden_size, bias=True)


        self.W_g = nn.Linear(self.behavior_map_size, self.behavior_hidden_size, bias=True)
        self.W_g_knowledge = nn.Linear(self.value_dim, self.behavior_hidden_size, bias=True)
        self.W_gh = nn.Linear(self.behavior_hidden_size, self.behavior_hidden_size, bias=False)
        self.W_g_stu = nn.Linear(self.embedding_size_s, self.behavior_hidden_size, bias=True)


        self.W_f = nn.Linear(self.behavior_map_size, self.behavior_hidden_size, bias=True)
        self.W_f_knowledge = nn.Linear(self.value_dim, self.behavior_hidden_size, bias=True)
        self.W_fh = nn.Linear(self.behavior_hidden_size, self.behavior_hidden_size, bias=False)
        self.W_f_stu = nn.Linear(self.embedding_size_s, self.behavior_hidden_size, bias=True)


        self.W_o = nn.Linear(self.behavior_map_size, self.behavior_hidden_size, bias=True)
        self.W_o_knowledge = nn.Linear(self.value_dim, self.behavior_hidden_size, bias=True)
        self.W_oh = nn.Linear(self.behavior_hidden_size, self.behavior_hidden_size, bias=False)
        self.W_o_stu = nn.Linear(self.embedding_size_s, self.behavior_hidden_size, bias=True)


        self.attn_type_pred = MultiHeadAttentionModule(self.behavior_hidden_size, heads=4, dim_head=32, dropout=0.1)
        self.behavior_prefrence = nn.Linear(self.behavior_hidden_size, 1, bias=True)

    def initialize_states(self):
        self.stateful_hidden_states = torch.zeros(self.num_students, self.behavior_hidden_size)
        self.stateful_cell_states = torch.zeros(self.num_students, self.behavior_hidden_size)
        self.stateful_value_matrix = torch.Tensor(self.num_students, self.num_concepts, self.value_dim).to(self.device)
        nn.init.normal_(self.stateful_value_matrix, mean=0., std=self.init_std)

    def init_material_pred_module(self):
        self.encoder_q = nn.Sequential(
            nn.Linear(self.behavior_hidden_size+self.embedding_size_q_behavior+3*(self.value_dim+self.num_concepts), 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, self.embedding_size_latent),
        )
        self.decoder_q = nn.Sequential(
            nn.Linear(2*self.embedding_size_l_behavior+self.embedding_size_q_behavior+4*(self.value_dim+self.num_concepts)+self.embedding_size_latent, 64),
            nn.Tanh(),
            nn.Linear(64, self.embedding_size_q_behavior),
        )
        self.encoder_l = nn.Sequential(
            nn.Linear(self.behavior_hidden_size+self.embedding_size_l_behavior+3*(self.value_dim+self.num_concepts), 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, self.embedding_size_latent),
        )
        self.decoder_l = nn.Sequential(
            nn.Linear(2*self.embedding_size_l_behavior+self.embedding_size_q_behavior+4*(self.value_dim+self.num_concepts)+self.embedding_size_latent, 64),
            nn.Tanh(),
            nn.Linear(64, self.embedding_size_q_behavior),
        )

        self.q_contrastive_layer = nn.Sequential(
            nn.Linear(2*self.embedding_size_latent, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.l_contrastive_layer = nn.Sequential(
            nn.Linear(2*self.embedding_size_latent, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def get_negative_samples(self, tensor_, lower_bound, upper_bound, num):
        tensor = tensor_.clone().detach().cpu()

        new_tensor = []
        for idx in range(tensor.shape[0]):
            forbidden_values = set(tensor[idx].unique().tolist())
            all_values = list(set(range(lower_bound, upper_bound)) - forbidden_values)
            new_tensor.append(torch.tensor(random.choices(all_values, k=num)).unsqueeze(0))

        tensor = torch.concat(new_tensor)
        tensor = tensor.to(self.device)

        return tensor

    def problem_encoder(self, behavioral_hidden_states, q_b, rc_q, corr_w_q, rc_q1, corr_w_q1, rc_l, corr_w_l):
        features = torch.concat([
            behavioral_hidden_states, 
            q_b, 
            rc_q, 
            corr_w_q, 
            rc_q1, 
            corr_w_q1, 
            rc_l, 
            corr_w_l,
        ], -1)
        enc_out = self.encoder_q(features)

        return enc_out, q_b
    
    def problem_decoder(self, enc_out, rc_q, corr_w_q, l_b, rc_l, corr_w_l, q_b1, rc_q1, corr_w_q1, l_b1, rc_l1, corr_w_l1):
        """
        Reference: 
        Unsupervised Speech Representation Learning for Behavior Modeling using Triplet Enhanced Contextualized Networks
        https://www.sciencedirect.com/science/article/abs/pii/S0885230821000334?fr=RR-2&ref=pdf_download&rr=8cb56a32cc7309b3
        """
        
        features = torch.concat([
            enc_out,
            rc_q, 
            corr_w_q, 
            l_b, 
            rc_l, 
            corr_w_l, 
            q_b1, 
            rc_q1, 
            corr_w_q1, 
            l_b1, 
            rc_l1, 
            corr_w_l1
        ], -1)
        features_dec = self.decoder_q(features)

        return features_dec.unsqueeze(2)
    
    def lecture_encoder(self, behavioral_hidden_states, l_b, rc_l, corr_w_l, rc_l1, corr_w_l1, rc_q, corr_w_q):
        features = torch.concat([
            behavioral_hidden_states, 
            l_b, 
            rc_l, 
            corr_w_l, 
            rc_l1, 
            corr_w_l1, 
            rc_q, 
            corr_w_q,
        ], -1)
        enc_out = self.encoder_l(features)
        
        return enc_out, l_b
    
    def lecture_decoder(self, enc_out, rc_l, corr_w_l, q_b, rc_q, corr_w_q, l_b1, rc_l1, corr_w_l1, q_b1, rc_q1, corr_w_q1):
        features = torch.concat([
            enc_out,
            rc_l, 
            corr_w_l, 
            q_b, 
            rc_q, 
            corr_w_q, 
            l_b1, 
            rc_l1, 
            corr_w_l1, 
            q_b1, 
            rc_q1, 
            corr_w_q1
        ], -1)
        features_dec = self.decoder_l(features)

        return features_dec.unsqueeze(2)
    
    def forward(self, q_data, a_data, l_data, d_data, s_data, evaluation=False):        
        num_negative_sampling = self.num_eval_negative_sampling if evaluation else self.num_train_negative_sampling

        batch_size, seq_len = q_data.size(0), q_data.size(1)

        # inintialize h0 and m0 and value matrix
        self.h = self.stateful_hidden_states[s_data].detach().clone()
        self.m = self.stateful_cell_states[s_data].detach().clone()
        self.value_matrix = self.stateful_value_matrix[s_data].detach().clone()

        # get embeddings of learning material and response
        q_embed_data = self.q_embed_matrix(q_data.long())

        if self.metric == 'rmse':
            a_data = torch.unsqueeze(a_data, dim=2)
            a_embed_data = self.a_embed_matrix(a_data)
        else:
            a_embed_data = self.a_embed_matrix(a_data)

        l_embed_data = self.l_embed_matrix(l_data)
        d_embed_data = self.d_embed_matrix(d_data)
        s_embed_data = self.s_embed_matrix(s_data)
        q_behavior_embed_data = self.q_behavior_embed_matrix(q_data)
        l_behavior_embed_data = self.l_behavior_embed_matrix(l_data)

        q_corr_weight = self.q_corr_weight_matrix(q_data.long())
        l_corr_weight = self.l_corr_weight_matrix(l_data.long())

        # split the data seq into chunk and process each question sequentially, and get embeddings of each learning material
        sliced_q_embed_data = torch.chunk(q_embed_data, seq_len, dim=1)
        sliced_a_embed_data = torch.chunk(a_embed_data, seq_len, dim=1)
        sliced_l_embed_data = torch.chunk(l_embed_data, seq_len, dim=1)
        sliced_d_embed_data = torch.chunk(d_embed_data, seq_len, dim=1)
        sliced_q_behavior_embed_data = torch.chunk(q_behavior_embed_data, seq_len, dim=1)
        sliced_l_behavior_embed_data = torch.chunk(l_behavior_embed_data, seq_len, dim=1)
        sliced_d_data = torch.chunk(d_data, seq_len, dim=1)
        sliced_q_corr_weight = torch.chunk(q_corr_weight, seq_len, dim=1)
        sliced_l_corr_weight = torch.chunk(l_corr_weight, seq_len, dim=1)

        batch_pred, batch_pred_type = [], []
        batch_pred_repr_q, batch_next_repr_q, batch_next_repr_neg_q = [], [], []
        batch_pred_repr_dec_q, batch_next_repr_dec_q = [], []
        batch_pred_repr_dec_gt_q, batch_next_repr_dec_gt_q = [], []
        batch_pred_repr_l, batch_next_repr_l, batch_next_repr_neg_l = [], [], []
        batch_pred_repr_dec_l, batch_next_repr_dec_l = [], []
        batch_pred_repr_dec_gt_l, batch_next_repr_dec_gt_l = [], []

        for i in range(1, seq_len - 1):
            # embedding layer, get material embeddings and neighbors embeddings for each time step t
            q = sliced_q_embed_data[i].squeeze(1)  # (batch_size, emebeding_size_q)
            a = sliced_a_embed_data[i].squeeze(1)
            l = sliced_l_embed_data[i].squeeze(1)
            d = sliced_d_embed_data[i].squeeze(1)
            d_1 = sliced_d_embed_data[i - 1].squeeze(1)
            q_b = sliced_q_behavior_embed_data[i].squeeze(1)
            l_b = sliced_l_behavior_embed_data[i].squeeze(1)
            d_t = sliced_d_data[i]
            d_t_1 = sliced_d_data[i - 1]
            q_b_next = sliced_q_behavior_embed_data[i + 1].squeeze(1)  # (batch_size, key_dim)
            l_b_next = sliced_l_behavior_embed_data[i + 1].squeeze(1)
            correlation_weight_q_prev = sliced_q_corr_weight[i - 1].squeeze(1)
            correlation_weight_l_prev = sliced_l_corr_weight[i - 1].squeeze(1)
            correlation_weight_q_curr = sliced_q_corr_weight[i].squeeze(1)
            correlation_weight_l_curr = sliced_l_corr_weight[i].squeeze(1)
            correlation_weight_q_next = sliced_q_corr_weight[i + 1].squeeze(1)
            correlation_weight_l_next = sliced_l_corr_weight[i + 1].squeeze(1)

            #update knowledge state
            self.knowledge_MANN(q, a, l, d_t, d_t_1, s_embed_data, correlation_weight_q_curr, correlation_weight_l_curr, d, d_1)

            #update behavior prefrence
            self.behavior_LSTM(q_b, d, l_b, d_t, s_embed_data) 

            #predict type
            type_attn = self.attn_type_pred(
                s_embed_data.unsqueeze(1), 
                torch.cat([q_b.unsqueeze(1), a.unsqueeze(1), l_b.unsqueeze(1), self.h.unsqueeze(1)], dim = 1)
            ).mean(1)
            batch_sliced_pred_type = self.behavior_prefrence(type_attn)
            batch_pred_type.append(batch_sliced_pred_type)

            #predict response
            q_next = sliced_q_embed_data[i + 1].squeeze(1)  # (batch_size, key_dim)
            read_content_next = self.read(correlation_weight_q_next, d_t)
            batch_sliced_pred = self.linear_out(torch.cat([read_content_next, q_next, self.h], dim = 1))
            batch_pred.append(batch_sliced_pred)

            #predict next learning material
            read_content_q_prev = self.read(correlation_weight_q_prev, d_t, lecture_next=False)
            read_content_q_curr = self.read(correlation_weight_q_curr, d_t, lecture_next=False)
            read_content_q_next = self.read(correlation_weight_q_next, d_t, lecture_next=False)
            read_content_l_prev = self.read(correlation_weight_l_prev, d_t, lecture_next=True)
            read_content_l_curr = self.read(correlation_weight_l_curr, d_t, lecture_next=True)
            read_content_l_next = self.read(correlation_weight_l_next, d_t, lecture_next=True)

            # encode current question & reconstruct next question embedding
            enc_out, dec_gt = self.problem_encoder(
                self.h, 
                q_b,
                read_content_q_curr,
                correlation_weight_q_curr,
                read_content_q_prev,
                correlation_weight_q_prev,
                read_content_l_prev,
                correlation_weight_l_prev,
            )
            dec_recon = self.problem_decoder(
                enc_out, 
                read_content_q_next,
                correlation_weight_q_next,
                l_b_next,
                read_content_l_next,
                correlation_weight_l_next,
                q_b,
                read_content_q_curr,
                correlation_weight_q_curr,
                l_b,
                read_content_l_curr,
                correlation_weight_l_curr
            )
            batch_pred_repr_q.append(enc_out.unsqueeze(2))
            batch_pred_repr_dec_q.append(dec_recon)
            batch_pred_repr_dec_gt_q.append(dec_gt.unsqueeze(2))

            # encode next question & reconstruct current question embedding
            enc_out, dec_gt = self.problem_encoder(
                self.h,
                q_b_next,
                read_content_q_next,
                correlation_weight_q_next,
                read_content_q_curr,
                correlation_weight_q_curr,
                read_content_l_curr,
                correlation_weight_l_curr,
            )
            dec_recon = self.problem_decoder(
                enc_out, 
                read_content_q_curr,
                correlation_weight_q_curr,
                l_b,
                read_content_l_curr,
                correlation_weight_l_curr,
                q_b_next,
                read_content_q_next,
                correlation_weight_q_next,
                l_b_next,
                read_content_l_next,
                correlation_weight_l_next
            )
            batch_next_repr_q.append(enc_out.unsqueeze(2))
            batch_next_repr_dec_q.append(dec_recon)
            batch_next_repr_dec_gt_q.append(dec_gt.unsqueeze(2))
            
            q_next_neg_indices = self.get_negative_samples(q_data[:, i+1].unsqueeze(1), 0, self.num_questions+1, num_negative_sampling)
            q_b_next_neg = self.q_behavior_embed_matrix(q_next_neg_indices)
            correlation_weight_q_next_neg = self.q_corr_weight_matrix(q_next_neg_indices)
            read_content_q_next_neg = torch.concat([self.read(correlation_weight_q_next_neg[:, idx], d_t, lecture_next=False).unsqueeze(1) for idx in range(num_negative_sampling)], 1)

            # encode next negative question
            enc_out, _ = self.problem_encoder(
                self.h.unsqueeze(1).repeat(1, num_negative_sampling, 1),
                q_b_next_neg,
                read_content_q_next_neg,
                correlation_weight_q_next_neg,
                read_content_q_curr.unsqueeze(1).repeat(1, num_negative_sampling, 1),
                correlation_weight_q_curr.unsqueeze(1).repeat(1, num_negative_sampling, 1),
                read_content_l_curr.unsqueeze(1).repeat(1, num_negative_sampling, 1),
                correlation_weight_l_curr.unsqueeze(1).repeat(1, num_negative_sampling, 1),
            )
            batch_next_repr_neg_q.append(enc_out.unsqueeze(3).permute(1, 0, 2, 3))

            # encode current lecture & reconstruct next lecture embedding
            enc_out, dec_gt = self.lecture_encoder(
                self.h, 
                l_b,
                read_content_l_curr,
                correlation_weight_l_curr,
                read_content_l_prev,
                correlation_weight_l_prev,
                read_content_q_prev,
                correlation_weight_q_prev,
            )
            dec_recon = self.lecture_decoder(
                enc_out, 
                read_content_l_next,
                correlation_weight_l_next,
                q_b_next,
                read_content_q_next,
                correlation_weight_q_next,
                l_b,
                read_content_l_curr,
                correlation_weight_l_curr,
                q_b,
                read_content_q_curr,
                correlation_weight_q_curr
            )
            batch_pred_repr_l.append(enc_out.unsqueeze(2))
            batch_pred_repr_dec_l.append(dec_recon)
            batch_pred_repr_dec_gt_l.append(dec_gt.unsqueeze(2))

            # encode next lecture & reconstruct current lecture embedding
            enc_out, dec_gt = self.lecture_encoder(
                self.h, 
                l_b_next,
                read_content_l_next,
                correlation_weight_l_next,
                read_content_l_curr,
                correlation_weight_l_curr,
                read_content_q_curr,
                correlation_weight_q_curr,
            )
            dec_recon = self.lecture_decoder(
                enc_out, 
                read_content_l_curr,
                correlation_weight_l_curr,
                q_b,
                read_content_q_curr,
                correlation_weight_q_curr,
                l_b_next,
                read_content_l_next,
                correlation_weight_l_next,
                q_b_next,
                read_content_q_next,
                correlation_weight_q_next
            )
            batch_next_repr_l.append(enc_out.unsqueeze(2))
            batch_next_repr_dec_l.append(dec_recon)
            batch_next_repr_dec_gt_l.append(dec_gt.unsqueeze(2))

            l_next_neg_indices = self.get_negative_samples(l_data[:, i+1].unsqueeze(1), 0, self.num_nongradable_items+1, num_negative_sampling)
            l_b_next_neg = self.l_behavior_embed_matrix(l_next_neg_indices)
            correlation_weight_l_next_neg = self.l_corr_weight_matrix(l_next_neg_indices)
            read_content_l_next_neg = torch.concat([self.read(correlation_weight_l_next_neg[:, idx], d_t, lecture_next=True).unsqueeze(1) for idx in range(num_negative_sampling)], 1)

            # encode next negative lecture
            enc_out, _ = self.lecture_encoder(
                self.h.unsqueeze(1).repeat(1, num_negative_sampling, 1),
                l_b_next_neg,
                read_content_l_next_neg,
                correlation_weight_l_next_neg,
                read_content_l_curr.unsqueeze(1).repeat(1, num_negative_sampling, 1),
                correlation_weight_l_curr.unsqueeze(1).repeat(1, num_negative_sampling, 1),
                read_content_q_curr.unsqueeze(1).repeat(1, num_negative_sampling, 1),
                correlation_weight_q_curr.unsqueeze(1).repeat(1, num_negative_sampling, 1),
            )
            batch_next_repr_neg_l.append(enc_out.unsqueeze(3).permute(1, 0, 2, 3))

        self.stateful_hidden_states[s_data] = self.h.detach().clone()
        self.stateful_cell_states[s_data] = self.m.detach().clone()
        self.stateful_value_matrix[s_data] = self.value_matrix.detach().clone()

        batch_pred = torch.cat(batch_pred, dim=-1)
        batch_pred_type = torch.cat(batch_pred_type, dim=-1)

        batch_pred_repr_q = torch.cat(batch_pred_repr_q, dim=-1)
        batch_next_repr_q = torch.cat(batch_next_repr_q, dim=-1)
        batch_next_repr_neg_q = torch.cat(batch_next_repr_neg_q, dim=-1)
        batch_pred_repr_dec_q = torch.cat(batch_pred_repr_dec_q, dim=-1)
        batch_next_repr_dec_q = torch.cat(batch_next_repr_dec_q, dim=-1)
        batch_pred_repr_dec_gt_q = torch.cat(batch_pred_repr_dec_gt_q, dim=-1)
        batch_next_repr_dec_gt_q = torch.cat(batch_next_repr_dec_gt_q, dim=-1)

        batch_pred_repr_l = torch.cat(batch_pred_repr_l, dim=-1)
        batch_next_repr_l = torch.cat(batch_next_repr_l, dim=-1)
        batch_next_repr_neg_l = torch.cat(batch_next_repr_neg_l, dim=-1)
        batch_pred_repr_dec_l = torch.cat(batch_pred_repr_dec_l, dim=-1)
        batch_next_repr_dec_l = torch.cat(batch_next_repr_dec_l, dim=-1)
        batch_pred_repr_dec_gt_l = torch.cat(batch_pred_repr_dec_gt_l, dim=-1)
        batch_next_repr_dec_gt_l = torch.cat(batch_next_repr_dec_gt_l, dim=-1)
        
        contrastive_pos_q = self.q_contrastive_layer(torch.concat([batch_pred_repr_q, batch_next_repr_q], 1).permute(0, 2, 1)).squeeze(2)
        contrastive_neg_q = [self.q_contrastive_layer(torch.concat([batch_pred_repr_q, batch_next_repr_neg_q[idx]], 1).permute(0, 2, 1)).squeeze(2) for idx in range(num_negative_sampling)]

        contrastive_pos_l = self.l_contrastive_layer(torch.concat([batch_pred_repr_l, batch_next_repr_l], 1).permute(0, 2, 1)).squeeze(2)
        contrastive_neg_l = [self.l_contrastive_layer(torch.concat([batch_pred_repr_l, batch_next_repr_neg_l[idx]], 1).permute(0, 2, 1)).squeeze(2) for idx in range(num_negative_sampling)]

        return (batch_pred, batch_pred_type, 
                contrastive_pos_q, contrastive_neg_q, 
                batch_next_repr_neg_q, batch_pred_repr_q, batch_next_repr_q,
                batch_pred_repr_dec_q, batch_next_repr_dec_q,
                batch_pred_repr_dec_gt_q, batch_next_repr_dec_gt_q,
                contrastive_pos_l, contrastive_neg_l, 
                batch_next_repr_neg_l, batch_pred_repr_l, batch_next_repr_l,
                batch_pred_repr_dec_l, batch_next_repr_dec_l,
                batch_pred_repr_dec_gt_l, batch_next_repr_dec_gt_l,
            )
    
    def knowledge_MANN(self, q, a, l, d_t, d_t_1, s_embed_data, correlation_weight_q, correlation_weight_l, d, d_1):
        qa = torch.cat([q, a], dim=1)
        correlation_weight = (1 - d_t) * correlation_weight_q + d_t * correlation_weight_l
        self.value_matrix = self.write(correlation_weight, qa, l, d_t, d_t_1, s_embed_data, d, d_1)

    def write(self, correlation_weight, qa_embed, l_embed, d_t, d_t_1, s_embed_data, d, d_1):
        """
                write function is to update memory based on the interaction
                value_matrix: (batch_size, memory_size, memory_state_dim)
                correlation_weight: (batch_size, memory_size)
                qa_embedded: (batch_size, memory_state_dim)
                """
        batch_size = self.value_matrix.size(0)

        erase_vector = (1-d_t) * self.erase_E_Q(qa_embed) + d_t*self.erase_E_L(l_embed) + self.erase_E_stu(s_embed_data) + self.erase_E_bh(self.h) # (batch_size, value_dim)
        erase_vector = self.sigmoid(erase_vector)

        add_vector = (1-d_t)*self.add_D_Q(qa_embed) + d_t*self.add_D_L(l_embed) + self.add_D_stu(s_embed_data) + self.add_D_bh(self.h)   # (batch_size, value_dim)
        add_vector = self.tanh(add_vector)

        erase_reshaped = erase_vector.reshape(batch_size, 1, self.value_dim)
        cw_reshaped = correlation_weight.reshape(batch_size, self.num_concepts, 1)  # the multiplication is to generate weighted erase vector for each memory cell, therefore, the size is (batch_size, num_concepts, value_dim)
        erase_mul = erase_reshaped * cw_reshaped

        transition = self.transition_proj_M(torch.concat([d_1, d], -1)).reshape(-1, 1, self.value_dim)
        memory_after_erase = self.tanh(transition) * self.value_matrix * (1 - erase_mul)

        add_reshaped = add_vector.reshape(batch_size, 1, self.value_dim)  # the multiplication is to generate weighted add vector for each memory cell therefore, the size is (batch_size, num_concepts, value_dim)
        add_memory = add_reshaped * cw_reshaped
        updated_memory = memory_after_erase + add_memory

        return updated_memory

    def behavior_LSTM(self, q, d, l, d_t, s_embed_data):
        qd = torch.cat([q, d], dim = 1)
        ld = torch.cat([l, d], dim = 1)
        maped_embedded = (1 - d_t)*self.behavior_mapQ(qd) + d_t*self.behavior_mapL(ld)
        sumed_knowledge = self.sum_knowledge2behavior(torch.transpose(self.value_matrix, 1,2)).squeeze(2)

        i = self.sigmoid(self.W_i(maped_embedded) + self.W_ih(self.h) + self.W_i_knowledge(sumed_knowledge) + self.W_i_stu(s_embed_data))
        g = self.tanh(self.W_g(maped_embedded) + self.W_gh(self.h) + self.W_g_knowledge(sumed_knowledge) + self.W_g_stu(s_embed_data))
        f = self.sigmoid(self.W_f(maped_embedded) + self.W_fh(self.h) + self.W_f_knowledge(sumed_knowledge) + self.W_f_stu(s_embed_data))
        o = self.sigmoid(self.W_o(maped_embedded) + self.W_oh(self.h) + self.W_o_knowledge(sumed_knowledge) + self.W_o_stu(s_embed_data))

        self.m = f * self.m + i * g
        self.h = o * self.tanh(self.m)

    def read(self, correlation_weight, d_t, lecture_next=False):
        """
        read function is to read a student's knowledge level on part of concepts covered by a
        target question.
        we could view value_matrix as the latent representation of a student's knowledge
        in terms of all possible concepts.

        value_matrix: (batch_size, num_concepts, concept_embedding_dim)
        correlation_weight: (batch_size, num_concepts)
        """
        batch_size = self.value_matrix.size(0)

        d_1 = self.d_embed_matrix(d_t)
        if lecture_next:
            d = self.d_embed_matrix(torch.ones_like(d_t))
        else:
            d = self.d_embed_matrix(torch.zeros_like(d_t))
        transition = self.transition_proj_M(torch.concat([d_1, d], -1)).reshape(-1, 1, self.value_dim)
        value_matrix_reshaped = self.tanh(transition) * self.value_matrix
        
        value_matrix_reshaped = value_matrix_reshaped.reshape(batch_size * self.num_concepts, self.value_dim)
        correlation_weight_reshaped = correlation_weight.reshape(batch_size * self.num_concepts, 1)
        # a (10,3) * b (10,1) = c (10, 3)is every row vector of a multiplies the row scalar of b
        # the multiplication below is to scale the memory embedding by the correlation weight
        rc = value_matrix_reshaped * correlation_weight_reshaped
        read_content = rc.reshape(batch_size, self.num_concepts, self.value_dim)
        read_content = torch.sum(read_content, dim=1)  # sum over all concepts

        return read_content


class KTBM_mat_nopers(nn.Module):

    def __init__(self, config):
        super(KTBM_mat_nopers, self).__init__()

        self.is_cuda = torch.cuda.is_available()
        self.cuda = self.is_cuda & config.cuda
        if self.cuda:
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cuda")
        else:
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cpu")

        self.metric = config.metric
        self.config = config
        
        self.metrics = Metrics(config.top_k_metrics)

        self.num_eval_negative_sampling = config.num_eval_negative_sampling
        self.num_train_negative_sampling = config.num_train_negative_sampling

        # initialize the dim size hyper parameters
        self.num_questions = config.num_items
        self.num_nongradable_items = config.num_nongradable_items
        self.embedding_size_q = config.embedding_size_q
        self.embedding_size_a = config.embedding_size_a
        self.embedding_size_l = config.embedding_size_l
        self.embedding_size_d = config.embedding_size_d
        self.embedding_size_q_behavior = config.embedding_size_q_behavior
        self.embedding_size_l_behavior = config.embedding_size_l_behavior
        self.embedding_size_latent = config.embedding_size_latent

        self.num_concepts = config.num_concepts
        self.key_dim = config.key_dim
        self.value_dim = config.value_dim
        self.summary_dim = config.summary_dim
        self.init_std = config.init_std
        self.num_students = config.num_students

        self.behavior_summary_fc = config.behavior_summary_fc
        self.behavior_map_size = config.behavior_map_size
        self.behavior_hidden_size = config.behavior_hidden_size

        # initialize the activiate functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(-1)

    def initialize(self):
        self.init_embeddings_module()
        self.init_knowledge_module()
        self.init_behavior_module()
        self.init_material_pred_module()

    def init_embeddings_module(self):
        self.q_embed_matrix = nn.Embedding(num_embeddings=self.num_questions + 1, embedding_dim=self.embedding_size_q, padding_idx=0)
        self.l_embed_matrix = nn.Embedding(num_embeddings=self.num_nongradable_items + 1, embedding_dim=self.embedding_size_l, padding_idx=0)

        self.q_corr_weight_matrix = nn.Sequential(
            nn.Embedding(num_embeddings=self.num_questions + 1, embedding_dim=self.num_concepts, padding_idx=0),
            nn.Softmax(-1)
        )
        self.l_corr_weight_matrix = nn.Sequential(
            nn.Embedding(num_embeddings=self.num_nongradable_items + 1, embedding_dim=self.num_concepts, padding_idx=0),
            nn.Softmax(-1)
        )

        if self.metric == "rmse":
            self.a_embed_matrix = nn.Linear(1, self.embedding_size_a)
        else:
            self.a_embed_matrix = nn.Embedding(3, self.embedding_size_a, padding_idx=2)

        self.d_embed_matrix = nn.Embedding(3, self.embedding_size_d, padding_idx=2)

        self.q_behavior_embed_matrix = nn.Embedding(num_embeddings=self.num_questions + 1, embedding_dim=self.embedding_size_q_behavior, padding_idx=0)
        self.l_behavior_embed_matrix = nn.Embedding(num_embeddings=self.num_nongradable_items + 1, embedding_dim=self.embedding_size_l_behavior, padding_idx=0)

    def init_knowledge_module(self):
        self.value_matrix_init = torch.Tensor(self.num_concepts, self.value_dim).to(self.device)
        nn.init.normal_(self.value_matrix_init, mean=0., std=self.init_std)

        self.erase_E_Q = nn.Linear(self.embedding_size_q + self.embedding_size_a, self.value_dim, bias=True)
        self.erase_E_L = nn.Linear(self.embedding_size_l, self.value_dim, bias=True)
        self.erase_E_bh = nn.Linear(self.behavior_hidden_size, self.value_dim, bias=False)

        self.add_D_Q = nn.Linear(self.embedding_size_q + self.embedding_size_a, self.value_dim, bias=True)
        self.add_D_L = nn.Linear(self.embedding_size_l, self.value_dim, bias=True)
        self.add_D_bh = nn.Linear(self.behavior_hidden_size, self.value_dim, bias=False)

        self.transition_proj_M = nn.Linear(2*self.embedding_size_d, self.value_dim, bias=True)

        self.linear_out = nn.Sequential(
            nn.Linear(self.embedding_size_q + self.value_dim + self.behavior_hidden_size, self.summary_dim),
            nn.Tanh(),
            nn.Linear(self.summary_dim, 1),
        ) 

    def init_behavior_module(self):
        # initialize the LSTM layers
        self.behavior_mapQ = nn.Linear(self.embedding_size_q_behavior + self.embedding_size_d, self.behavior_map_size, bias=True)
        self.behavior_mapL = nn.Linear(self.embedding_size_l_behavior + self.embedding_size_d, self.behavior_map_size, bias=True)
        self.behavior_mapKnowledge = nn.Linear(self.value_dim, self.num_concepts, bias=True)

        self.sum_knowledge2behavior = nn.Linear(self.num_concepts, 1, bias=True)

        self.W_i = nn.Linear(self.behavior_map_size, self.behavior_hidden_size, bias=True)
        self.W_i_knowledge = nn.Linear(self.value_dim, self.behavior_hidden_size, bias=True)
        self.W_ih = nn.Linear(self.behavior_hidden_size, self.behavior_hidden_size, bias=False)


        self.W_g = nn.Linear(self.behavior_map_size, self.behavior_hidden_size, bias=True)
        self.W_g_knowledge = nn.Linear(self.value_dim, self.behavior_hidden_size, bias=True)
        self.W_gh = nn.Linear(self.behavior_hidden_size, self.behavior_hidden_size, bias=False)


        self.W_f = nn.Linear(self.behavior_map_size, self.behavior_hidden_size, bias=True)
        self.W_f_knowledge = nn.Linear(self.value_dim, self.behavior_hidden_size, bias=True)
        self.W_fh = nn.Linear(self.behavior_hidden_size, self.behavior_hidden_size, bias=False)


        self.W_o = nn.Linear(self.behavior_map_size, self.behavior_hidden_size, bias=True)
        self.W_o_knowledge = nn.Linear(self.value_dim, self.behavior_hidden_size, bias=True)
        self.W_oh = nn.Linear(self.behavior_hidden_size, self.behavior_hidden_size, bias=False)


        self.behavior_prefrence = nn.Linear(self.behavior_hidden_size + self.embedding_size_q_behavior + self.embedding_size_a + self.embedding_size_l_behavior, self.behavior_summary_fc, bias=True)
        self.behavior_out_type = nn.Linear(self.behavior_summary_fc, 1, bias=True)
    
    def init_material_pred_module(self):
        self.encoder_q = nn.Sequential(
            nn.Linear(self.behavior_hidden_size+self.embedding_size_q_behavior+3*(self.value_dim+self.num_concepts), 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, self.embedding_size_latent),
        )
        self.decoder_q = nn.Sequential(
            nn.Linear(2*self.embedding_size_l_behavior+self.embedding_size_q_behavior+4*(self.value_dim+self.num_concepts)+self.embedding_size_latent, 64),
            nn.Tanh(),
            nn.Linear(64, self.embedding_size_q_behavior),
        )
        self.encoder_l = nn.Sequential(
            nn.Linear(self.behavior_hidden_size+self.embedding_size_l_behavior+3*(self.value_dim+self.num_concepts), 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, self.embedding_size_latent),
        )
        self.decoder_l = nn.Sequential(
            nn.Linear(2*self.embedding_size_l_behavior+self.embedding_size_q_behavior+4*(self.value_dim+self.num_concepts)+self.embedding_size_latent, 64),
            nn.Tanh(),
            nn.Linear(64, self.embedding_size_q_behavior),
        )

        self.q_contrastive_layer = nn.Sequential(
            nn.Linear(2*self.embedding_size_latent, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.l_contrastive_layer = nn.Sequential(
            nn.Linear(2*self.embedding_size_latent, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def get_negative_samples(self, tensor_, lower_bound, upper_bound, num):
        tensor = tensor_.clone().detach().cpu()

        new_tensor = []
        for idx in range(tensor.shape[0]):
            forbidden_values = set(tensor[idx].unique().tolist())
            all_values = list(set(range(lower_bound, upper_bound)) - forbidden_values)
            new_tensor.append(torch.tensor(random.choices(all_values, k=num)).unsqueeze(0))

        tensor = torch.concat(new_tensor)
        tensor = tensor.to(self.device)

        return tensor

    def problem_encoder(self, behavioral_hidden_states, q_b, rc_q, corr_w_q, rc_q1, corr_w_q1, rc_l, corr_w_l):
        features = torch.concat([
            behavioral_hidden_states, 
            q_b, 
            rc_q, 
            corr_w_q, 
            rc_q1, 
            corr_w_q1, 
            rc_l, 
            corr_w_l
        ], -1)
        enc_out = self.encoder_q(features)

        return enc_out, q_b
    
    def problem_decoder(self, enc_out, rc_q, corr_w_q, l_b, rc_l, corr_w_l, q_b1, rc_q1, corr_w_q1, l_b1, rc_l1, corr_w_l1):
        """
        Reference: 
        Unsupervised Speech Representation Learning for Behavior Modeling using Triplet Enhanced Contextualized Networks
        https://www.sciencedirect.com/science/article/abs/pii/S0885230821000334?fr=RR-2&ref=pdf_download&rr=8cb56a32cc7309b3
        """
        
        features = torch.concat([
            enc_out,
            rc_q, 
            corr_w_q, 
            l_b, 
            rc_l, 
            corr_w_l, 
            q_b1, 
            rc_q1, 
            corr_w_q1, 
            l_b1, 
            rc_l1, 
            corr_w_l1
        ], -1)
        features_dec = self.decoder_q(features)

        return features_dec.unsqueeze(2)
    
    def lecture_encoder(self, behavioral_hidden_states, l_b, rc_l, corr_w_l, rc_l1, corr_w_l1, rc_q, corr_w_q):
        features = torch.concat([
            behavioral_hidden_states, 
            l_b, 
            rc_l, 
            corr_w_l, 
            rc_l1, 
            corr_w_l1, 
            rc_q, 
            corr_w_q
        ], -1)
        enc_out = self.encoder_l(features)
        
        return enc_out, l_b
    
    def lecture_decoder(self, enc_out, rc_l, corr_w_l, q_b, rc_q, corr_w_q, l_b1, rc_l1, corr_w_l1, q_b1, rc_q1, corr_w_q1):
        features = torch.concat([
            enc_out,
            rc_l, 
            corr_w_l, 
            q_b, 
            rc_q, 
            corr_w_q, 
            l_b1, 
            rc_l1, 
            corr_w_l1, 
            q_b1, 
            rc_q1, 
            corr_w_q1
        ], -1)
        features_dec = self.decoder_l(features)

        return features_dec.unsqueeze(2)
    
    def forward(self, q_data, a_data, l_data, d_data, s_data, evaluation=False):
        '''
           get output of the model
           :param q_data: (batch_size, seq_len) question indexes/ids of each learning interaction, 0 represent paddings
           :param a_data: (batch_size, seq_len) student performance of each learning interaction, 2 represent paddings
           :param l_data: (batch_size, seq_len) non-assessed material indexes/ids of each learning interaction, 0 represent paddings
           :param d_data: (batch_size, seq_len) material type of each learning interaction, 0: question 1:non-assessed material
           :return:
       '''
        
        num_negative_sampling = self.num_eval_negative_sampling if evaluation else self.num_train_negative_sampling

        batch_size, seq_len = q_data.size(0), q_data.size(1)

        # inintialize h0 and m0 and value matrix
        self.h = torch.zeros(batch_size, self.behavior_hidden_size).to(self.device)
        self.m = torch.zeros(batch_size, self.behavior_hidden_size).to(self.device)
        self.value_matrix = self.value_matrix_init.clone().repeat(batch_size, 1, 1)

        # get embeddings of learning material and response
        q_embed_data = self.q_embed_matrix(q_data.long())

        if self.metric == 'rmse':
            a_data = torch.unsqueeze(a_data, dim=2)
            a_embed_data = self.a_embed_matrix(a_data)
        else:
            a_embed_data = self.a_embed_matrix(a_data)

        l_embed_data = self.l_embed_matrix(l_data)
        d_embed_data = self.d_embed_matrix(d_data)
        q_behavior_embed_data = self.q_behavior_embed_matrix(q_data)
        l_behavior_embed_data = self.l_behavior_embed_matrix(l_data)

        q_corr_weight = self.q_corr_weight_matrix(q_data.long())
        l_corr_weight = self.l_corr_weight_matrix(l_data.long())

        # split the data seq into chunk and process each question sequentially, and get embeddings of each learning material
        sliced_q_embed_data = torch.chunk(q_embed_data, seq_len, dim=1)
        sliced_a_embed_data = torch.chunk(a_embed_data, seq_len, dim=1)
        sliced_l_embed_data = torch.chunk(l_embed_data, seq_len, dim=1)
        sliced_d_embed_data = torch.chunk(d_embed_data, seq_len, dim=1)
        sliced_q_behavior_embed_data = torch.chunk(q_behavior_embed_data, seq_len, dim=1)
        sliced_l_behavior_embed_data = torch.chunk(l_behavior_embed_data, seq_len, dim=1)
        sliced_d_data = torch.chunk(d_data, seq_len, dim=1)
        sliced_q_corr_weight = torch.chunk(q_corr_weight, seq_len, dim=1)
        sliced_l_corr_weight = torch.chunk(l_corr_weight, seq_len, dim=1)

        batch_pred, batch_pred_type = [], []
        batch_pred_repr_q, batch_next_repr_q, batch_next_repr_neg_q = [], [], []
        batch_pred_repr_dec_q, batch_next_repr_dec_q = [], []
        batch_pred_repr_dec_gt_q, batch_next_repr_dec_gt_q = [], []
        batch_pred_repr_l, batch_next_repr_l, batch_next_repr_neg_l = [], [], []
        batch_pred_repr_dec_l, batch_next_repr_dec_l = [], []
        batch_pred_repr_dec_gt_l, batch_next_repr_dec_gt_l = [], []

        for i in range(1, seq_len - 1):
            # embedding layer, get material embeddings and neighbors embeddings for each time step t
            q = sliced_q_embed_data[i].squeeze(1)  # (batch_size, emebeding_size_q)
            a = sliced_a_embed_data[i].squeeze(1)
            l = sliced_l_embed_data[i].squeeze(1)
            d = sliced_d_embed_data[i].squeeze(1)
            d_1 = sliced_d_embed_data[i - 1].squeeze(1)
            q_b = sliced_q_behavior_embed_data[i].squeeze(1)
            l_b = sliced_l_behavior_embed_data[i].squeeze(1)
            d_t = sliced_d_data[i]
            d_t_1 = sliced_d_data[i - 1]
            q_b_next = sliced_q_behavior_embed_data[i + 1].squeeze(1)  # (batch_size, key_dim)
            l_b_next = sliced_l_behavior_embed_data[i + 1].squeeze(1)
            correlation_weight_q_prev = sliced_q_corr_weight[i - 1].squeeze(1)
            correlation_weight_l_prev = sliced_l_corr_weight[i - 1].squeeze(1)
            correlation_weight_q_curr = sliced_q_corr_weight[i].squeeze(1)
            correlation_weight_l_curr = sliced_l_corr_weight[i].squeeze(1)
            correlation_weight_q_next = sliced_q_corr_weight[i + 1].squeeze(1)
            correlation_weight_l_next = sliced_l_corr_weight[i + 1].squeeze(1)

            #update knowledge state
            self.knowledge_MANN(q, a, l, d_t, d_t_1, correlation_weight_q_curr, correlation_weight_l_curr, d, d_1)

            #update behavior prefrence
            self.behavior_LSTM(q_b, d, l_b, d_t) 

            #predict type
            prefrence_type = self.behavior_prefrence(torch.cat([q_b, a, l_b, self.h], dim = 1))
            batch_sliced_pred_type= self.behavior_out_type(prefrence_type)
            batch_pred_type.append(batch_sliced_pred_type)

            #predict response
            q_next = sliced_q_embed_data[i + 1].squeeze(1)  # (batch_size, key_dim)
            read_content_next = self.read(correlation_weight_q_next, d_t)
            batch_sliced_pred = self.linear_out(torch.cat([read_content_next, q_next, self.h], dim = 1))
            batch_pred.append(batch_sliced_pred)

            #predict next learning material
            read_content_q_prev = self.read(correlation_weight_q_prev, d_t, lecture_next=False)
            read_content_q_curr = self.read(correlation_weight_q_curr, d_t, lecture_next=False)
            read_content_q_next = self.read(correlation_weight_q_next, d_t, lecture_next=False)
            read_content_l_prev = self.read(correlation_weight_l_prev, d_t, lecture_next=True)
            read_content_l_curr = self.read(correlation_weight_l_curr, d_t, lecture_next=True)
            read_content_l_next = self.read(correlation_weight_l_next, d_t, lecture_next=True)

            # encode current question & reconstruct next question embedding
            enc_out, dec_gt = self.problem_encoder(
                self.h, 
                q_b,
                read_content_q_curr,
                correlation_weight_q_curr,
                read_content_q_prev,
                correlation_weight_q_prev,
                read_content_l_prev,
                correlation_weight_l_prev,
            )
            dec_recon = self.problem_decoder(
                enc_out, 
                read_content_q_next,
                correlation_weight_q_next,
                l_b_next,
                read_content_l_next,
                correlation_weight_l_next,
                q_b,
                read_content_q_curr,
                correlation_weight_q_curr,
                l_b,
                read_content_l_curr,
                correlation_weight_l_curr
            )
            batch_pred_repr_q.append(enc_out.unsqueeze(2))
            batch_pred_repr_dec_q.append(dec_recon)
            batch_pred_repr_dec_gt_q.append(dec_gt.unsqueeze(2))

            # encode next question & reconstruct current question embedding
            enc_out, dec_gt = self.problem_encoder(
                self.h,
                q_b_next,
                read_content_q_next,
                correlation_weight_q_next,
                read_content_q_curr,
                correlation_weight_q_curr,
                read_content_l_curr,
                correlation_weight_l_curr,
            )
            dec_recon = self.problem_decoder(
                enc_out, 
                read_content_q_curr,
                correlation_weight_q_curr,
                l_b,
                read_content_l_curr,
                correlation_weight_l_curr,
                q_b_next,
                read_content_q_next,
                correlation_weight_q_next,
                l_b_next,
                read_content_l_next,
                correlation_weight_l_next
            )
            batch_next_repr_q.append(enc_out.unsqueeze(2))
            batch_next_repr_dec_q.append(dec_recon)
            batch_next_repr_dec_gt_q.append(dec_gt.unsqueeze(2))
            
            q_next_neg_indices = self.get_negative_samples(q_data[:, i+1].unsqueeze(1), 0, self.num_questions+1, num_negative_sampling)
            q_b_next_neg = self.q_behavior_embed_matrix(q_next_neg_indices)
            correlation_weight_q_next_neg = self.q_corr_weight_matrix(q_next_neg_indices)
            read_content_q_next_neg = torch.concat([self.read(correlation_weight_q_next_neg[:, idx], d_t, lecture_next=False).unsqueeze(1) for idx in range(num_negative_sampling)], 1)

            # encode next negative question
            enc_out, _ = self.problem_encoder(
                self.h.unsqueeze(1).repeat(1, num_negative_sampling, 1),
                q_b_next_neg,
                read_content_q_next_neg,
                correlation_weight_q_next_neg,
                read_content_q_curr.unsqueeze(1).repeat(1, num_negative_sampling, 1),
                correlation_weight_q_curr.unsqueeze(1).repeat(1, num_negative_sampling, 1),
                read_content_l_curr.unsqueeze(1).repeat(1, num_negative_sampling, 1),
                correlation_weight_l_curr.unsqueeze(1).repeat(1, num_negative_sampling, 1),
            )
            batch_next_repr_neg_q.append(enc_out.unsqueeze(3).permute(1, 0, 2, 3))

            # encode current lecture & reconstruct next lecture embedding
            enc_out, dec_gt = self.lecture_encoder(
                self.h, 
                l_b,
                read_content_l_curr,
                correlation_weight_l_curr,
                read_content_l_prev,
                correlation_weight_l_prev,
                read_content_q_prev,
                correlation_weight_q_prev,
            )
            dec_recon = self.lecture_decoder(
                enc_out, 
                read_content_l_next,
                correlation_weight_l_next,
                q_b_next,
                read_content_q_next,
                correlation_weight_q_next,
                l_b,
                read_content_l_curr,
                correlation_weight_l_curr,
                q_b,
                read_content_q_curr,
                correlation_weight_q_curr
            )
            batch_pred_repr_l.append(enc_out.unsqueeze(2))
            batch_pred_repr_dec_l.append(dec_recon)
            batch_pred_repr_dec_gt_l.append(dec_gt.unsqueeze(2))

            # encode next lecture & reconstruct current lecture embedding
            enc_out, dec_gt = self.lecture_encoder(
                self.h, 
                l_b_next,
                read_content_l_next,
                correlation_weight_l_next,
                read_content_l_curr,
                correlation_weight_l_curr,
                read_content_q_curr,
                correlation_weight_q_curr,
            )
            dec_recon = self.lecture_decoder(
                enc_out, 
                read_content_l_curr,
                correlation_weight_l_curr,
                q_b,
                read_content_q_curr,
                correlation_weight_q_curr,
                l_b_next,
                read_content_l_next,
                correlation_weight_l_next,
                q_b_next,
                read_content_q_next,
                correlation_weight_q_next
            )
            batch_next_repr_l.append(enc_out.unsqueeze(2))
            batch_next_repr_dec_l.append(dec_recon)
            batch_next_repr_dec_gt_l.append(dec_gt.unsqueeze(2))

            l_next_neg_indices = self.get_negative_samples(l_data[:, i+1].unsqueeze(1), 0, self.num_nongradable_items+1, num_negative_sampling)
            l_b_next_neg = self.l_behavior_embed_matrix(l_next_neg_indices)
            correlation_weight_l_next_neg = self.l_corr_weight_matrix(l_next_neg_indices)
            read_content_l_next_neg = torch.concat([self.read(correlation_weight_l_next_neg[:, idx], d_t, lecture_next=True).unsqueeze(1) for idx in range(num_negative_sampling)], 1)

            # encode next negative lecture
            enc_out, _ = self.lecture_encoder(
                self.h.unsqueeze(1).repeat(1, num_negative_sampling, 1),
                l_b_next_neg,
                read_content_l_next_neg,
                correlation_weight_l_next_neg,
                read_content_l_curr.unsqueeze(1).repeat(1, num_negative_sampling, 1),
                correlation_weight_l_curr.unsqueeze(1).repeat(1, num_negative_sampling, 1),
                read_content_q_curr.unsqueeze(1).repeat(1, num_negative_sampling, 1),
                correlation_weight_q_curr.unsqueeze(1).repeat(1, num_negative_sampling, 1),
            )
            batch_next_repr_neg_l.append(enc_out.unsqueeze(3).permute(1, 0, 2, 3))

        batch_pred = torch.cat(batch_pred, dim=-1)
        batch_pred_type = torch.cat(batch_pred_type, dim=-1)

        batch_pred_repr_q = torch.cat(batch_pred_repr_q, dim=-1)
        batch_next_repr_q = torch.cat(batch_next_repr_q, dim=-1)
        batch_next_repr_neg_q = torch.cat(batch_next_repr_neg_q, dim=-1)
        batch_pred_repr_dec_q = torch.cat(batch_pred_repr_dec_q, dim=-1)
        batch_next_repr_dec_q = torch.cat(batch_next_repr_dec_q, dim=-1)
        batch_pred_repr_dec_gt_q = torch.cat(batch_pred_repr_dec_gt_q, dim=-1)
        batch_next_repr_dec_gt_q = torch.cat(batch_next_repr_dec_gt_q, dim=-1)

        batch_pred_repr_l = torch.cat(batch_pred_repr_l, dim=-1)
        batch_next_repr_l = torch.cat(batch_next_repr_l, dim=-1)
        batch_next_repr_neg_l = torch.cat(batch_next_repr_neg_l, dim=-1)
        batch_pred_repr_dec_l = torch.cat(batch_pred_repr_dec_l, dim=-1)
        batch_next_repr_dec_l = torch.cat(batch_next_repr_dec_l, dim=-1)
        batch_pred_repr_dec_gt_l = torch.cat(batch_pred_repr_dec_gt_l, dim=-1)
        batch_next_repr_dec_gt_l = torch.cat(batch_next_repr_dec_gt_l, dim=-1)
        
        contrastive_pos_q = self.q_contrastive_layer(torch.concat([batch_pred_repr_q, batch_next_repr_q], 1).permute(0, 2, 1)).squeeze(2)
        contrastive_neg_q = [self.q_contrastive_layer(torch.concat([batch_pred_repr_q, batch_next_repr_neg_q[idx]], 1).permute(0, 2, 1)).squeeze(2) for idx in range(num_negative_sampling)]

        contrastive_pos_l = self.l_contrastive_layer(torch.concat([batch_pred_repr_l, batch_next_repr_l], 1).permute(0, 2, 1)).squeeze(2)
        contrastive_neg_l = [self.l_contrastive_layer(torch.concat([batch_pred_repr_l, batch_next_repr_neg_l[idx]], 1).permute(0, 2, 1)).squeeze(2) for idx in range(num_negative_sampling)]

        return (batch_pred, batch_pred_type, 
                contrastive_pos_q, contrastive_neg_q, 
                batch_next_repr_neg_q, batch_pred_repr_q, batch_next_repr_q,
                batch_pred_repr_dec_q, batch_next_repr_dec_q,
                batch_pred_repr_dec_gt_q, batch_next_repr_dec_gt_q,
                contrastive_pos_l, contrastive_neg_l, 
                batch_next_repr_neg_l, batch_pred_repr_l, batch_next_repr_l,
                batch_pred_repr_dec_l, batch_next_repr_dec_l,
                batch_pred_repr_dec_gt_l, batch_next_repr_dec_gt_l
            )
    
    def knowledge_MANN(self, q, a, l, d_t, d_t_1, correlation_weight_q, correlation_weight_l, d, d_1):
        qa = torch.cat([q, a], dim=1)
        correlation_weight = (1 - d_t) * correlation_weight_q + d_t * correlation_weight_l
        self.value_matrix = self.write(correlation_weight, qa, l, d_t, d_t_1, d, d_1)

    def write(self, correlation_weight, qa_embed, l_embed, d_t, d_t_1, d, d_1):
        """
                write function is to update memory based on the interaction
                value_matrix: (batch_size, memory_size, memory_state_dim)
                correlation_weight: (batch_size, memory_size)
                qa_embedded: (batch_size, memory_state_dim)
                """
        batch_size = self.value_matrix.size(0)

        erase_vector = (1-d_t) * self.erase_E_Q(qa_embed) + d_t*self.erase_E_L(l_embed) + self.erase_E_bh(self.h) # (batch_size, value_dim)
        erase_vector = self.sigmoid(erase_vector)

        add_vector = (1-d_t)*self.add_D_Q(qa_embed) + d_t*self.add_D_L(l_embed) + self.add_D_bh(self.h)   # (batch_size, value_dim)
        add_vector = self.tanh(add_vector)

        erase_reshaped = erase_vector.reshape(batch_size, 1, self.value_dim)
        cw_reshaped = correlation_weight.reshape(batch_size, self.num_concepts, 1)  # the multiplication is to generate weighted erase vector for each memory cell, therefore, the size is (batch_size, num_concepts, value_dim)
        erase_mul = erase_reshaped * cw_reshaped

        transition = self.transition_proj_M(torch.concat([d_1, d], -1)).reshape(-1, 1, self.value_dim)
        memory_after_erase = self.tanh(transition) * self.value_matrix * (1 - erase_mul)

        add_reshaped = add_vector.reshape(batch_size, 1, self.value_dim)  # the multiplication is to generate weighted add vector for each memory cell therefore, the size is (batch_size, num_concepts, value_dim)
        add_memory = add_reshaped * cw_reshaped
        updated_memory = memory_after_erase + add_memory

        return updated_memory

    def behavior_LSTM(self, q, d, l, d_t):
        qd = torch.cat([q, d], dim = 1)
        ld = torch.cat([l, d], dim = 1)
        maped_embedded = (1 - d_t)*self.behavior_mapQ(qd) + d_t*self.behavior_mapL(ld)
        sumed_knowledge = self.sum_knowledge2behavior(torch.transpose(self.value_matrix, 1,2)).squeeze(2)

        i = self.sigmoid(self.W_i(maped_embedded) + self.W_ih(self.h) + self.W_i_knowledge(sumed_knowledge))
        g = self.tanh(self.W_g(maped_embedded) + self.W_gh(self.h) + self.W_g_knowledge(sumed_knowledge))
        f = self.sigmoid(self.W_f(maped_embedded) + self.W_fh(self.h) + self.W_f_knowledge(sumed_knowledge))
        o = self.sigmoid(self.W_o(maped_embedded) + self.W_oh(self.h) + self.W_o_knowledge(sumed_knowledge))

        self.m = f * self.m + i * g
        self.h = o * self.tanh(self.m)

    def read(self, correlation_weight, d_t, lecture_next=False):
        """
        read function is to read a student's knowledge level on part of concepts covered by a
        target question.
        we could view value_matrix as the latent representation of a student's knowledge
        in terms of all possible concepts.

        value_matrix: (batch_size, num_concepts, concept_embedding_dim)
        correlation_weight: (batch_size, num_concepts)
        """
        batch_size = self.value_matrix.size(0)

        d_1 = self.d_embed_matrix(d_t)
        if lecture_next:
            d = self.d_embed_matrix(torch.ones_like(d_t))
        else:
            d = self.d_embed_matrix(torch.zeros_like(d_t))
        transition = self.transition_proj_M(torch.concat([d_1, d], -1)).reshape(-1, 1, self.value_dim)
        value_matrix_reshaped = self.tanh(transition) * self.value_matrix
        
        value_matrix_reshaped = value_matrix_reshaped.reshape(batch_size * self.num_concepts, self.value_dim)
        correlation_weight_reshaped = correlation_weight.reshape(batch_size * self.num_concepts, 1)
        # a (10,3) * b (10,1) = c (10, 3)is every row vector of a multiplies the row scalar of b
        # the multiplication below is to scale the memory embedding by the correlation weight
        rc = value_matrix_reshaped * correlation_weight_reshaped
        read_content = rc.reshape(batch_size, self.num_concepts, self.value_dim)
        read_content = torch.sum(read_content, dim=1)  # sum over all concepts

        return read_content
    

class KTBM_pers_nomat(nn.Module):

    def __init__(self, config):
        super(KTBM_pers_nomat, self).__init__()

        self.is_cuda = torch.cuda.is_available()
        self.cuda = self.is_cuda & config.cuda
        if self.cuda:
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cuda")
        else:
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cpu")

        self.metric = config.metric
        self.config = config
        
        self.metrics = Metrics(config.top_k_metrics)

        # initialize the dim size hyper parameters
        self.num_questions = config.num_items
        self.num_nongradable_items = config.num_nongradable_items
        self.embedding_size_q = config.embedding_size_q
        self.embedding_size_a = config.embedding_size_a
        self.embedding_size_l = config.embedding_size_l
        self.embedding_size_d = config.embedding_size_d
        self.embedding_size_s = config.embedding_size_s
        self.embedding_size_q_behavior = config.embedding_size_q_behavior
        self.embedding_size_l_behavior = config.embedding_size_l_behavior
        self.embedding_size_latent = config.embedding_size_latent

        self.num_concepts = config.num_concepts
        self.key_dim = config.key_dim
        self.value_dim = config.value_dim
        self.summary_dim = config.summary_dim
        self.init_std = config.init_std
        self.num_students = config.num_students

        self.behavior_summary_fc = config.behavior_summary_fc
        self.behavior_map_size = config.behavior_map_size
        self.behavior_hidden_size = config.behavior_hidden_size

        # initialize the activiate functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(-1)

    def initialize(self):
        self.init_embeddings_module()
        self.init_knowledge_module()
        self.init_behavior_module()

    def init_embeddings_module(self):
        self.q_embed_matrix = nn.Embedding(num_embeddings=self.num_questions + 1, embedding_dim=self.embedding_size_q, padding_idx=0)
        self.l_embed_matrix = nn.Embedding(num_embeddings=self.num_nongradable_items + 1, embedding_dim=self.embedding_size_l, padding_idx=0)

        self.q_corr_weight_matrix = nn.Sequential(
            nn.Embedding(num_embeddings=self.num_questions + 1, embedding_dim=self.num_concepts, padding_idx=0),
            nn.Softmax(-1)
        )
        self.l_corr_weight_matrix = nn.Sequential(
            nn.Embedding(num_embeddings=self.num_nongradable_items + 1, embedding_dim=self.num_concepts, padding_idx=0),
            nn.Softmax(-1)
        )

        if self.metric == "rmse":
            self.a_embed_matrix = nn.Linear(1, self.embedding_size_a)
        else:
            self.a_embed_matrix = nn.Embedding(3, self.embedding_size_a, padding_idx=2)

        self.d_embed_matrix = nn.Embedding(3, self.embedding_size_d, padding_idx=2)
        self.s_embed_matrix = nn.Embedding(num_embeddings=self.num_students, embedding_dim=self.embedding_size_s)

        self.q_behavior_embed_matrix = nn.Embedding(num_embeddings=self.num_questions + 1, embedding_dim=self.embedding_size_q_behavior, padding_idx=0)
        self.l_behavior_embed_matrix = nn.Embedding(num_embeddings=self.num_nongradable_items + 1, embedding_dim=self.embedding_size_l_behavior, padding_idx=0)

    def init_knowledge_module(self):
        self.erase_E_Q = nn.Linear(self.embedding_size_q + self.embedding_size_a, self.value_dim, bias=True)
        self.erase_E_L = nn.Linear(self.embedding_size_l, self.value_dim, bias=True)
        self.erase_E_bh = nn.Linear(self.behavior_hidden_size, self.value_dim, bias=False)
        self.erase_E_stu = nn.Linear(self.embedding_size_s, self.value_dim, bias=True)

        self.add_D_Q = nn.Linear(self.embedding_size_q + self.embedding_size_a, self.value_dim, bias=True)
        self.add_D_L = nn.Linear(self.embedding_size_l, self.value_dim, bias=True)
        self.add_D_bh = nn.Linear(self.behavior_hidden_size, self.value_dim, bias=False)
        self.add_D_stu = nn.Linear(self.embedding_size_s, self.value_dim, bias=False)

        self.transition_proj_M = nn.Linear(2*self.embedding_size_d, self.value_dim, bias=True)

        self.linear_out = nn.Sequential(
            nn.Linear(self.embedding_size_q + self.value_dim + self.behavior_hidden_size, self.summary_dim),
            nn.Tanh(),
            nn.Linear(self.summary_dim, 1),
        ) 

    def init_behavior_module(self):
        # initialize the LSTM layers
        self.behavior_mapQ = nn.Linear(self.embedding_size_q_behavior + self.embedding_size_d, self.behavior_map_size, bias=True)
        self.behavior_mapL = nn.Linear(self.embedding_size_l_behavior + self.embedding_size_d, self.behavior_map_size, bias=True)

        self.sum_knowledge2behavior = nn.Linear(self.num_concepts, 1, bias=True)

        self.W_i = nn.Linear(self.behavior_map_size, self.behavior_hidden_size, bias=True)
        self.W_i_knowledge = nn.Linear(self.value_dim, self.behavior_hidden_size, bias=True)
        self.W_ih = nn.Linear(self.behavior_hidden_size, self.behavior_hidden_size, bias=False)
        self.W_i_stu = nn.Linear(self.embedding_size_s, self.behavior_hidden_size, bias=True)


        self.W_g = nn.Linear(self.behavior_map_size, self.behavior_hidden_size, bias=True)
        self.W_g_knowledge = nn.Linear(self.value_dim, self.behavior_hidden_size, bias=True)
        self.W_gh = nn.Linear(self.behavior_hidden_size, self.behavior_hidden_size, bias=False)
        self.W_g_stu = nn.Linear(self.embedding_size_s, self.behavior_hidden_size, bias=True)


        self.W_f = nn.Linear(self.behavior_map_size, self.behavior_hidden_size, bias=True)
        self.W_f_knowledge = nn.Linear(self.value_dim, self.behavior_hidden_size, bias=True)
        self.W_fh = nn.Linear(self.behavior_hidden_size, self.behavior_hidden_size, bias=False)
        self.W_f_stu = nn.Linear(self.embedding_size_s, self.behavior_hidden_size, bias=True)


        self.W_o = nn.Linear(self.behavior_map_size, self.behavior_hidden_size, bias=True)
        self.W_o_knowledge = nn.Linear(self.value_dim, self.behavior_hidden_size, bias=True)
        self.W_oh = nn.Linear(self.behavior_hidden_size, self.behavior_hidden_size, bias=False)
        self.W_o_stu = nn.Linear(self.embedding_size_s, self.behavior_hidden_size, bias=True)


        self.attn_type_pred = MultiHeadAttentionModule(self.behavior_hidden_size, heads=4, dim_head=32, dropout=0.1)
        self.behavior_prefrence = nn.Linear(self.behavior_hidden_size, 1, bias=True)

    def initialize_states(self):
        self.stateful_hidden_states = torch.zeros(self.num_students, self.behavior_hidden_size)
        self.stateful_cell_states = torch.zeros(self.num_students, self.behavior_hidden_size)
        self.stateful_value_matrix = torch.Tensor(self.num_students, self.num_concepts, self.value_dim).to(self.device)
        nn.init.normal_(self.stateful_value_matrix, mean=0., std=self.init_std)
    
    def forward(self, q_data, a_data, l_data, d_data, s_data, evaluation=False):
        '''
           get output of the model
           :param q_data: (batch_size, seq_len) question indexes/ids of each learning interaction, 0 represent paddings
           :param a_data: (batch_size, seq_len) student performance of each learning interaction, 2 represent paddings
           :param l_data: (batch_size, seq_len) non-assessed material indexes/ids of each learning interaction, 0 represent paddings
           :param d_data: (batch_size, seq_len) material type of each learning interaction, 0: question 1:non-assessed material
           :return:
       '''
        
        batch_size, seq_len = q_data.size(0), q_data.size(1)

        # inintialize h0 and m0 and value matrix
        self.h = self.stateful_hidden_states[s_data].detach().clone()
        self.m = self.stateful_cell_states[s_data].detach().clone()
        self.value_matrix = self.stateful_value_matrix[s_data].detach().clone()

        # get embeddings of learning material and response
        q_embed_data = self.q_embed_matrix(q_data.long()) # (b, 100) -> (b, 100, 32)

        if self.metric == 'rmse':
            a_data = torch.unsqueeze(a_data, dim=2)
            a_embed_data = self.a_embed_matrix(a_data)
        else:
            a_embed_data = self.a_embed_matrix(a_data)

        l_embed_data = self.l_embed_matrix(l_data)
        d_embed_data = self.d_embed_matrix(d_data)
        s_embed_data = self.s_embed_matrix(s_data)
        q_behavior_embed_data = self.q_behavior_embed_matrix(q_data)
        l_behavior_embed_data = self.l_behavior_embed_matrix(l_data)

        q_corr_weight = self.q_corr_weight_matrix(q_data.long())
        l_corr_weight = self.l_corr_weight_matrix(l_data.long())

        # split the data seq into chunk and process each question sequentially, and get embeddings of each learning material
        sliced_q_embed_data = torch.chunk(q_embed_data, seq_len, dim=1)
        sliced_a_embed_data = torch.chunk(a_embed_data, seq_len, dim=1)
        sliced_l_embed_data = torch.chunk(l_embed_data, seq_len, dim=1)
        sliced_d_embed_data = torch.chunk(d_embed_data, seq_len, dim=1)
        sliced_q_behavior_embed_data = torch.chunk(q_behavior_embed_data, seq_len, dim=1)
        sliced_l_behavior_embed_data = torch.chunk(l_behavior_embed_data, seq_len, dim=1)
        sliced_d_data = torch.chunk(d_data, seq_len, dim=1)
        sliced_q_corr_weight = torch.chunk(q_corr_weight, seq_len, dim=1)
        sliced_l_corr_weight = torch.chunk(l_corr_weight, seq_len, dim=1)

        batch_pred, batch_pred_type = [], []
        for i in range(1, seq_len - 1):
            # embedding layer, get material embeddings and neighbors embeddings for each time step t
            q = sliced_q_embed_data[i].squeeze(1)  # (batch_size, emebeding_size_q)
            a = sliced_a_embed_data[i].squeeze(1)
            l = sliced_l_embed_data[i].squeeze(1)
            d = sliced_d_embed_data[i].squeeze(1)
            d_1 = sliced_d_embed_data[i - 1].squeeze(1)
            q_b = sliced_q_behavior_embed_data[i].squeeze(1)
            l_b = sliced_l_behavior_embed_data[i].squeeze(1)
            d_t = sliced_d_data[i]
            d_t_1 = sliced_d_data[i - 1]
            correlation_weight_q_curr = sliced_q_corr_weight[i].squeeze(1)
            correlation_weight_l_curr = sliced_l_corr_weight[i].squeeze(1)
            correlation_weight_q_next = sliced_q_corr_weight[i + 1].squeeze(1)

            #update knowledge state
            self.knowledge_MANN(q, a, l, d_t, d_t_1, s_embed_data, correlation_weight_q_curr, correlation_weight_l_curr, d, d_1)

            #update behavior prefrence
            self.behavior_LSTM(q_b, d, l_b, d_t, s_embed_data) 

            #predict type
            type_attn = self.attn_type_pred(
                s_embed_data.unsqueeze(1), 
                torch.cat([q_b.unsqueeze(1), a.unsqueeze(1), l_b.unsqueeze(1), self.h.unsqueeze(1)], dim = 1)
            ).mean(1)
            batch_sliced_pred_type = self.behavior_prefrence(type_attn)
            batch_pred_type.append(batch_sliced_pred_type)

            #predict response
            q_next = sliced_q_embed_data[i + 1].squeeze(1)  # (batch_size, key_dim)
            read_content_next = self.read(correlation_weight_q_next, d_t)
            batch_sliced_pred = self.linear_out(torch.cat([read_content_next, q_next, self.h], dim = 1))
            batch_pred.append(batch_sliced_pred)

        self.stateful_hidden_states[s_data] = self.h.detach().clone()
        self.stateful_cell_states[s_data] = self.m.detach().clone()
        self.stateful_value_matrix[s_data] = self.value_matrix.detach().clone()

        batch_pred = torch.cat(batch_pred, dim=-1)
        batch_pred_type = torch.cat(batch_pred_type, dim=-1)

        return batch_pred, batch_pred_type
    
    def knowledge_MANN(self, q, a, l, d_t, d_t_1, s_embed_data, correlation_weight_q, correlation_weight_l, d, d_1):
        qa = torch.cat([q, a], dim=1)
        correlation_weight = (1 - d_t) * correlation_weight_q + d_t * correlation_weight_l
        self.value_matrix = self.write(correlation_weight, qa, l, d_t, d_t_1, s_embed_data, d, d_1)

    def write(self, correlation_weight, qa_embed, l_embed, d_t, d_t_1, s_embed_data, d, d_1):
        """
                write function is to update memory based on the interaction
                value_matrix: (batch_size, memory_size, memory_state_dim)
                correlation_weight: (batch_size, memory_size)
                qa_embedded: (batch_size, memory_state_dim)
                """
        batch_size = self.value_matrix.size(0)

        erase_vector = (1-d_t) * self.erase_E_Q(qa_embed) + d_t*self.erase_E_L(l_embed) + self.erase_E_stu(s_embed_data) + self.erase_E_bh(self.h) # (batch_size, value_dim)
        erase_vector = self.sigmoid(erase_vector)

        add_vector = (1-d_t)*self.add_D_Q(qa_embed) + d_t*self.add_D_L(l_embed) + self.add_D_stu(s_embed_data) + self.add_D_bh(self.h)   # (batch_size, value_dim)
        add_vector = self.tanh(add_vector)

        erase_reshaped = erase_vector.reshape(batch_size, 1, self.value_dim)
        cw_reshaped = correlation_weight.reshape(batch_size, self.num_concepts, 1)  # the multiplication is to generate weighted erase vector for each memory cell, therefore, the size is (batch_size, num_concepts, value_dim)
        erase_mul = erase_reshaped * cw_reshaped

        transition = self.transition_proj_M(torch.concat([d_1, d], -1)).reshape(-1, 1, self.value_dim)
        memory_after_erase = self.tanh(transition) * self.value_matrix * (1 - erase_mul)

        add_reshaped = add_vector.reshape(batch_size, 1, self.value_dim)  # the multiplication is to generate weighted add vector for each memory cell therefore, the size is (batch_size, num_concepts, value_dim)
        add_memory = add_reshaped * cw_reshaped
        updated_memory = memory_after_erase + add_memory

        return updated_memory

    def behavior_LSTM(self, q, d, l, d_t, s_embed_data):
        qd = torch.cat([q, d], dim = 1)
        ld = torch.cat([l, d], dim = 1)
        maped_embedded = (1 - d_t)*self.behavior_mapQ(qd) + d_t*self.behavior_mapL(ld)
        sumed_knowledge = self.sum_knowledge2behavior(torch.transpose(self.value_matrix, 1,2)).squeeze(2)

        i = self.sigmoid(self.W_i(maped_embedded) + self.W_ih(self.h) + self.W_i_knowledge(sumed_knowledge) + self.W_i_stu(s_embed_data))
        g = self.tanh(self.W_g(maped_embedded) + self.W_gh(self.h) + self.W_g_knowledge(sumed_knowledge) + self.W_g_stu(s_embed_data))
        f = self.sigmoid(self.W_f(maped_embedded) + self.W_fh(self.h) + self.W_f_knowledge(sumed_knowledge) + self.W_f_stu(s_embed_data))
        o = self.sigmoid(self.W_o(maped_embedded) + self.W_oh(self.h) + self.W_o_knowledge(sumed_knowledge) + self.W_o_stu(s_embed_data))

        self.m = f * self.m + i * g
        self.h = o * self.tanh(self.m)

    def read(self, correlation_weight, d_t):
        """
        read function is to read a student's knowledge level on part of concepts covered by a
        target question.
        we could view value_matrix as the latent representation of a student's knowledge
        in terms of all possible concepts.

        value_matrix: (batch_size, num_concepts, concept_embedding_dim)
        correlation_weight: (batch_size, num_concepts)
        """
        batch_size = self.value_matrix.size(0)

        d_1 = self.d_embed_matrix(d_t)
        d = self.d_embed_matrix(torch.zeros_like(d_t))
        transition = self.transition_proj_M(torch.concat([d_1, d], -1)).reshape(-1, 1, self.value_dim)
        value_matrix_reshaped = self.tanh(transition) * self.value_matrix

        value_matrix_reshaped = value_matrix_reshaped.reshape(batch_size * self.num_concepts, self.value_dim)
        correlation_weight_reshaped = correlation_weight.reshape(batch_size * self.num_concepts, 1)
        # a (10,3) * b (10,1) = c (10, 3)is every row vector of a multiplies the row scalar of b
        # the multiplication below is to scale the memory embedding by the correlation weight
        rc = value_matrix_reshaped * correlation_weight_reshaped
        read_content = rc.reshape(batch_size, self.num_concepts, self.value_dim)
        read_content = torch.sum(read_content, dim=1)  # sum over all concepts

        return read_content
