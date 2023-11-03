#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator
import random
import pickle
import math
import collections
import itertools
import time
from tqdm import tqdm
import os

def Identity(x):
    return x

class BoxOffsetIntersection(nn.Module):
    
    def __init__(self, dim):
        super(BoxOffsetIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))
        layer1_mean = torch.mean(layer1_act, dim=0) 
        gate = torch.sigmoid(self.layer2(layer1_mean))
        offset, _ = torch.min(embeddings, dim=0)

        return offset * gate

class CenterIntersection(nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings)) # (num_conj, dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0) # (num_conj, dim)
        embedding = torch.sum(attention * embeddings, dim=0)

        return embedding

class BetaIntersection(nn.Module):

    def __init__(self, dim):
        super(BetaIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(2 * self.dim, 2 * self.dim)
        self.layer2 = nn.Linear(2 * self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, alpha_embeddings, beta_embeddings):
        all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)
        layer1_act = F.relu(self.layer1(all_embeddings)) # (num_conj, batch_size, 2 * dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0) # (num_conj, batch_size, dim)

        alpha_embedding = torch.sum(attention * alpha_embeddings, dim=0)
        beta_embedding = torch.sum(attention * beta_embeddings, dim=0)

        return alpha_embedding, beta_embedding

class BetaProjection(nn.Module):
    def __init__(self, entity_dim, relation_dim, hidden_dim, projection_regularizer, num_layers):
        super(BetaProjection, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim) # 1st layer
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim) # final layer
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)
        self.projection_regularizer = projection_regularizer

    def forward(self, e_embedding, r_embedding):
        x = torch.cat([e_embedding, r_embedding], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)
        x = self.projection_regularizer(x)

        return x

class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)

class KGReasoning(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma, 
                 geo, test_batch_size=1,
                 box_mode=None, use_cuda=False,
                 query_name_dict=None, beta_mode=None, use_tbox=None, arguments=None):
        super(KGReasoning, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.geo = geo
        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).cuda() if self.use_cuda else torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1) # used in test_step
        self.query_name_dict = query_name_dict
        self.use_tbox = use_tbox
        self.arguments=arguments

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim
        
        if self.geo == 'box':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim)) # centor for entities
            activation, cen = box_mode
            self.cen = cen # hyperparameter that balances the in-box distance and the out-box distance
            if activation == 'none':
                self.func = Identity
            elif activation == 'relu':
                self.func = F.relu
            elif activation == 'softplus':
                self.func = F.softplus
        elif self.geo == 'vec':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim)) # center for entities
        elif self.geo == 'beta':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim * 2)) # alpha and beta
            self.entity_regularizer = Regularizer(1, 0.05, 1e9) # make sure the parameters of beta embeddings are positive
            self.projection_regularizer = Regularizer(1, 0.05, 1e9) # make sure the parameters of beta embeddings after relation projection are positive
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        # self.relation_embedding = nn.Linear(self.relation_dim, self.relation_dim)
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )

        if self.geo == 'box':
            self.offset_embedding = nn.Parameter(torch.zeros(nrelation, self.entity_dim))
            nn.init.uniform_(
                tensor=self.offset_embedding, 
                a=0., 
                b=self.embedding_range.item()
            )
            self.center_net = CenterIntersection(self.entity_dim)
            self.offset_net = BoxOffsetIntersection(self.entity_dim)
        elif self.geo == 'vec':
            self.center_net = CenterIntersection(self.entity_dim)
        elif self.geo == 'beta':
            hidden_dim, num_layers = beta_mode
            self.center_net = BetaIntersection(self.entity_dim)
            self.projection_net = BetaProjection(self.entity_dim * 2, 
                                             self.relation_dim, 
                                             hidden_dim, 
                                             self.projection_regularizer, 
                                             num_layers)

    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict, batch_generalized_queries_dict,batch_generalized_idxs_dict,max_number_generalized_queries_per_train_query, args):
        if self.geo == 'box':
            if self.use_tbox :
                return self.forward_box_tbox(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict,batch_generalized_queries_dict,batch_generalized_idxs_dict,max_number_generalized_queries_per_train_query)
            return self.forward_box(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
        elif self.geo == 'vec':
            return self.forward_vec(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
        elif self.geo == 'beta':
            return self.forward_beta(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)

    def embed_query_box(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using Query2box
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                if self.use_cuda:
                    offset_embedding = torch.zeros_like(embedding).cuda()
                else:
                    offset_embedding = torch.zeros_like(embedding)
                idx += 1
            else:
                embedding, offset_embedding, idx = self.embed_query_box(queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert False, "box cannot handle queries with negation"
                else:
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    r_offset_embedding = torch.index_select(self.offset_embedding, dim=0, index=queries[:, idx])
                    embedding += r_embedding
                    # embedding += self.relation_embedding(r_embedding)
                    offset_embedding += self.func(r_offset_embedding)
                idx += 1
        else:
            embedding_list = []
            offset_embedding_list = []
            for i in range(len(query_structure)):
                embedding, offset_embedding, idx = self.embed_query_box(queries, query_structure[i], idx)
                embedding_list.append(embedding)
                offset_embedding_list.append(offset_embedding)
            embedding = self.center_net(torch.stack(embedding_list))
            offset_embedding = self.offset_net(torch.stack(offset_embedding_list))

        return embedding, offset_embedding, idx

    def embed_query_vec(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using GQE
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                idx += 1
            else:
                embedding, idx = self.embed_query_vec(queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert False, "vec cannot handle queries with negation"
                else:
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    embedding += r_embedding
                idx += 1
        else:
            embedding_list = []
            for i in range(len(query_structure)):
                embedding, idx = self.embed_query_vec(queries, query_structure[i], idx)
                embedding_list.append(embedding)
            embedding = self.center_net(torch.stack(embedding_list))

        return embedding, idx

    def embed_query_beta(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using BetaE
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx]))
                idx += 1
            else:
                alpha_embedding, beta_embedding, idx = self.embed_query_beta(queries, query_structure[0], idx)
                embedding = torch.cat([alpha_embedding, beta_embedding], dim=-1)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    embedding = 1./embedding
                else:
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    embedding = self.projection_net(embedding, r_embedding)
                idx += 1
            alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)
        else:
            alpha_embedding_list = []
            beta_embedding_list = []
            for i in range(len(query_structure)):
                alpha_embedding, beta_embedding, idx = self.embed_query_beta(queries, query_structure[i], idx)
                alpha_embedding_list.append(alpha_embedding)
                beta_embedding_list.append(beta_embedding)
            alpha_embedding, beta_embedding = self.center_net(torch.stack(alpha_embedding_list), torch.stack(beta_embedding_list))

        return alpha_embedding, beta_embedding, idx

    def cal_logit_beta(self, entity_embedding, query_dist):
        alpha_embedding, beta_embedding = torch.chunk(entity_embedding, 2, dim=-1)
        entity_dist = torch.distributions.beta.Beta(alpha_embedding, beta_embedding)
        logit = self.gamma - torch.norm(torch.distributions.kl.kl_divergence(entity_dist, query_dist), p=1, dim=-1)
        return logit

    def forward_beta(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_idxs, all_alpha_embeddings, all_beta_embeddings = [], [], []
        all_union_idxs, all_union_alpha_embeddings, all_union_beta_embeddings = [], [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                alpha_embedding, beta_embedding, _ = \
                    self.embed_query_beta(self.transform_union_query(batch_queries_dict[query_structure], 
                                                                     query_structure), 
                                          self.transform_union_structure(query_structure), 
                                          0)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_alpha_embeddings.append(alpha_embedding)
                all_union_beta_embeddings.append(beta_embedding)
            else:
                alpha_embedding, beta_embedding, _ = self.embed_query_beta(batch_queries_dict[query_structure], 
                                                                           query_structure, 
                                                                           0)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_alpha_embeddings.append(alpha_embedding)
                all_beta_embeddings.append(beta_embedding)

        if len(all_alpha_embeddings) > 0:
            all_alpha_embeddings = torch.cat(all_alpha_embeddings, dim=0).unsqueeze(1)
            all_beta_embeddings = torch.cat(all_beta_embeddings, dim=0).unsqueeze(1)
            all_dists = torch.distributions.beta.Beta(all_alpha_embeddings, all_beta_embeddings)
        if len(all_union_alpha_embeddings) > 0:
            all_union_alpha_embeddings = torch.cat(all_union_alpha_embeddings, dim=0).unsqueeze(1)
            all_union_beta_embeddings = torch.cat(all_union_beta_embeddings, dim=0).unsqueeze(1)
            all_union_alpha_embeddings = all_union_alpha_embeddings.view(all_union_alpha_embeddings.shape[0]//2, 2, 1, -1)
            all_union_beta_embeddings = all_union_beta_embeddings.view(all_union_beta_embeddings.shape[0]//2, 2, 1, -1)
            all_union_dists = torch.distributions.beta.Beta(all_union_alpha_embeddings, all_union_beta_embeddings)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs] # positive samples for non-union queries in this batch
                positive_embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1))
                positive_logit = self.cal_logit_beta(positive_embedding, all_dists)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_alpha_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs] # positive samples for union queries in this batch
                positive_embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1))
                positive_union_logit = self.cal_logit_beta(positive_embedding, all_union_dists)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1).long()).view(batch_size, negative_size, -1))
                negative_logit = self.cal_logit_beta(negative_embedding, all_dists)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_alpha_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1).long()).view(batch_size, 1, negative_size, -1))
                negative_union_logit = self.cal_logit_beta(negative_embedding, all_union_dists)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs

    def transform_union_query(self, queries, query_structure):
        '''
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        '''
        if self.query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1] # remove union -1
        elif self.query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1), torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        queries = torch.reshape(queries, [queries.shape[0]*2, -1])
        return queries

    def transform_union_structure(self, query_structure):
        if self.query_name_dict[query_structure] == '2u-DNF':
            return ('e', ('r',))
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return ('e', ('r', 'r'))

    def cal_logit_box(self, entity_embedding, query_center_embedding, query_offset_embedding):
        delta = (entity_embedding - query_center_embedding).abs()
        distance_out = F.relu(delta - query_offset_embedding)
        distance_in = torch.min(delta, query_offset_embedding)
        logit = self.gamma - torch.norm(distance_out, p=1, dim=-1) - self.cen * torch.norm(distance_in, p=1, dim=-1)
        return logit

    def cal_logit_box_generalized(self, entity_embedding, query_center_embedding, query_offset_embedding, mask_for_distances):
        """
        new code: modified function to compute the value: (gama- dist_box(v,q)) in formular 4 of the q2b paper
        """
        delta = (entity_embedding - query_center_embedding).abs()
        distance_out = F.relu(delta - query_offset_embedding)
        distance_in = torch.min(delta, query_offset_embedding)
        distance = torch.norm(distance_out, p=1, dim=-1) - self.cen * torch.norm(distance_in, p=1, dim=-1)
        distance =distance * mask_for_distances
        distance = distance / mask_for_distances.sum(dim=1, keepdim=True)
        distance = torch.sum(distance, dim=1, keepdim=True)
        logit = self.gamma - distance
        return logit

    def forward_box_tbox(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict, batch_generalized_queries_dict,batch_generalized_idxs_dict, max_number_generalized_queries_per_train_query):
        all_center_embeddings, all_offset_embeddings, all_idxs = [], [], []
        all_union_center_embeddings, all_union_offset_embeddings, all_union_idxs = [], [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, offset_embedding, _ = \
                    self.embed_query_box(self.transform_union_query(batch_queries_dict[query_structure],
                                                                    query_structure),
                                         self.transform_union_structure(query_structure),
                                         0)
                all_union_center_embeddings.append(center_embedding)
                all_union_offset_embeddings.append(offset_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                center_embedding, offset_embedding, _ = self.embed_query_box(batch_queries_dict[query_structure],
                                                                             query_structure,
                                                                             0)
                all_center_embeddings.append(center_embedding)
                all_offset_embeddings.append(offset_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_center_embeddings) > 0 and len(all_offset_embeddings) > 0:
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)
            all_offset_embeddings = torch.cat(all_offset_embeddings, dim=0).unsqueeze(1)
        if len(all_union_center_embeddings) > 0 and len(all_union_offset_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_offset_embeddings = torch.cat(all_union_offset_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0] // 2, 2,
                                                                           1, -1)
            all_union_offset_embeddings = all_union_offset_embeddings.view(all_union_offset_embeddings.shape[0] // 2, 2,
                                                                           1, -1)
        ''' process generalized queries '''
        if len(batch_generalized_queries_dict) > 0 :
            all_generalized_center_embeddings = []
            all_generalized_offset_embeddings= []
            all_generalized_idxs = [] #list of index of the train queries; the order corresponds to the order of processed generatlized queries
            for generalized_query_structure in batch_generalized_queries_dict:
                if 'u' in self.query_name_dict[generalized_query_structure]:
                    logging.warning("TODO: change code to process union")
                    #TODO:process union queries
                    # center_embedding, offset_embedding, _ = \
                    #     self.embed_query_box(self.transform_union_query(batch_generalized_queries_dict[query_structure],
                    #                                                     query_structure),
                    #                          self.transform_union_structure(query_structure),
                    #                          0)
                    # all_union_center_embeddings.append(center_embedding)
                    # all_union_offset_embeddings.append(offset_embedding)
                    # all_union_idxs.extend(batch_idxs_dict[query_structure])
                else:

                    generalized_queries_center_embedding, generalized_queries_offset_embedding, _ = self.embed_query_box(batch_generalized_queries_dict[generalized_query_structure],
                                                                                 generalized_query_structure,
                                                                                 0)
                    all_generalized_center_embeddings.append(generalized_queries_center_embedding)
                    all_generalized_offset_embeddings.append(generalized_queries_offset_embedding)
                    all_generalized_idxs.extend(batch_generalized_idxs_dict[generalized_query_structure])

            if len(all_generalized_center_embeddings) > 0 and len(all_generalized_offset_embeddings) > 0:
                all_generalized_center_embeddings = torch.cat(all_generalized_center_embeddings, dim=0)
                all_generalized_offset_embeddings = torch.cat(all_generalized_offset_embeddings, dim=0)

            #map from index of train_query to indices of their generalizations
            dict_index_of_train_query_to_generatlizations=collections.defaultdict(list)
            for i, index_train_query in enumerate(all_generalized_idxs):
                dict_index_of_train_query_to_generatlizations[index_train_query].append(i)

            for index in dict_index_of_train_query_to_generatlizations:
                if self.arguments.cuda:
                    dict_index_of_train_query_to_generatlizations[index] = torch.LongTensor(dict_index_of_train_query_to_generatlizations[index]).cuda()
                else:
                    dict_index_of_train_query_to_generatlizations[index] = torch.LongTensor(
                        dict_index_of_train_query_to_generatlizations[index])

            #create a new tensor for each generalizations of each train query
            list_of_center_generalizations=[]
            list_of_offset_generalizations = []

            list_of_center_for_intersection_of_generalization=[]
            list_of_offset_for_intersection_of_generalization = []

            if self.arguments.cuda:
                mask_for_distances=torch.zeros(all_center_embeddings.size()[0],max_number_generalized_queries_per_train_query+1).cuda() #plus 1 to count also the train query
            else :
                mask_for_distances = torch.zeros(all_center_embeddings.size()[0],
                                                 max_number_generalized_queries_per_train_query + 1)  # plus 1 to count also the train query
            current_row_of_mask=0
            for index in all_idxs:
                ''' compute center and offset for the generalizations of train query with this index'''
                center_embeddings_for_this_generalizations=torch.index_select(all_generalized_center_embeddings,dim=0,index=dict_index_of_train_query_to_generatlizations[index])
                offset_embeddings_for_this_generalizations = torch.index_select(all_generalized_offset_embeddings,
                                                                                dim=0,
                                                                                index= dict_index_of_train_query_to_generatlizations[index])
                number_of_gen_queries = len(dict_index_of_train_query_to_generatlizations[index])
                '''compute intersection'''
                if self.arguments.intersec_gen:
                    intersection_center_embedding = self.center_net(center_embeddings_for_this_generalizations.view(number_of_gen_queries,1,-1))
                    intersection_offset_embedding = self.offset_net(offset_embeddings_for_this_generalizations.view(number_of_gen_queries,1,-1))
                    list_of_center_for_intersection_of_generalization.append(intersection_center_embedding)
                    list_of_offset_for_intersection_of_generalization.append(intersection_offset_embedding)

                '''Pad fake generalized queries to make each train query having the same number of generalizations'''

                mask_for_distances[current_row_of_mask,0:number_of_gen_queries+1]=1.0
                current_row_of_mask+=1
                number_of_pad_tensor= max_number_generalized_queries_per_train_query- number_of_gen_queries
                dim_of_embeeding=center_embeddings_for_this_generalizations.size()[-1]

                if self.arguments.cuda:
                    random_pad_tensor=torch.zeros(number_of_pad_tensor,dim_of_embeeding).cuda()
                else:
                    random_pad_tensor = torch.zeros(number_of_pad_tensor, dim_of_embeeding)
                center_embeddings_for_this_generalizations=torch.cat([center_embeddings_for_this_generalizations,random_pad_tensor],dim=0)
                list_of_center_generalizations.append(center_embeddings_for_this_generalizations.view(1,max_number_generalized_queries_per_train_query,dim_of_embeeding))

                if self.arguments.cuda:
                    random_pad_offset_tensor = torch.zeros(number_of_pad_tensor, dim_of_embeeding).cuda()
                else:
                    random_pad_offset_tensor = torch.zeros(number_of_pad_tensor, dim_of_embeeding)
                offset_embeddings_for_this_generalizations = torch.cat(
                    [offset_embeddings_for_this_generalizations, random_pad_offset_tensor], dim=0)
                list_of_offset_generalizations.append(offset_embeddings_for_this_generalizations.view(1,max_number_generalized_queries_per_train_query,dim_of_embeeding))

            all_generalized_center_embeddings_in_order_of_processed_train_queries=torch.cat(list_of_center_generalizations,dim=0)
            all_generalized_offset_embeddings_in_order_of_processed_train_queries=torch.cat(list_of_offset_generalizations,dim=0)

            ''' Merge embedding of train_queries with embeddings of their generalizations'''
            merged_all_center_embeddings=torch.cat([all_center_embeddings,all_generalized_center_embeddings_in_order_of_processed_train_queries],dim=1)
            merged_all_offset_embeddings=torch.cat([all_offset_embeddings,all_generalized_offset_embeddings_in_order_of_processed_train_queries],dim=1)
            if self.arguments.intersec_gen:
                all_center_embeddings_for_intersection=torch.cat(list_of_center_for_intersection_of_generalization,dim=0).unsqueeze(1)
                all_offset_embeddings_for_intersection = torch.cat(list_of_offset_for_intersection_of_generalization,
                                                           dim=0).unsqueeze(1)
        else:
            merged_all_center_embeddings=all_center_embeddings
            merged_all_offset_embeddings=all_offset_embeddings
        ''' compute distances'''
        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:

                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=positive_sample_regular).unsqueeze(1)
                # if self.arguments.mean_gen_new:
                #     pass
                #     #positive_logit= self.cal_logit_box_generalized(positive_embedding, merged_all_center_embeddings, merged_all_offset_embeddings, mask_for_distances)
                #     # logging.info("using mean gen new!")
                # else:
                '''compute distance to train and its generalizations'''
                positive_logit_distance_to_train_query=self.cal_logit_box(positive_embedding,all_center_embeddings,all_offset_embeddings)
                positive_logit = self.cal_logit_box(positive_embedding, merged_all_center_embeddings, merged_all_offset_embeddings) #distaces to all generalizations
                ''' get mean of distance to train query and its generalizations'''
                positive_logit=positive_logit*mask_for_distances  #remove padded values
                # TODO: We have some options here, mean and sum, and possible weights
                if self.arguments.mean_gen_new:
                    positive_logit=torch.cat((positive_logit_distance_to_train_query,positive_logit),dim=1)
                if self.arguments.mean_gen:
                    # positive_logit = torch.mean(positive_logit,dim=1,keepdim=True)
                    positive_logit=positive_logit/mask_for_distances.sum(dim=1,keepdim=True)
                    positive_logit=torch.sum(positive_logit,dim=1,keepdim=True)
                    #logging.warning("Using mean")
                elif self.arguments.intersec_gen:
                    '''distance to the intersection'''
                    distance_to_intersection = self.cal_logit_box(positive_embedding,
                                                                  all_center_embeddings_for_intersection,
                                                                  all_offset_embeddings_for_intersection)
                    positive_logit=positive_logit_distance_to_train_query+distance_to_intersection
                # else:
                #     positive_logit = torch.sum(positive_logit, dim=1, keepdim=True)
                    #logging.warning("Using sum")
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_box(positive_embedding, all_union_center_embeddings,
                                                          all_union_offset_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=negative_sample_regular.view(-1).long()).view(batch_size,
                                                                                                            negative_size,
                                                                                                            -1)
                negative_logit = self.cal_logit_box(negative_embedding, all_center_embeddings, all_offset_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=negative_sample_union.view(-1).long()).view(batch_size, 1,
                                                                                                          negative_size,
                                                                                                          -1)
                negative_union_logit = self.cal_logit_box(negative_embedding, all_union_center_embeddings,
                                                          all_union_offset_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs


    def forward_box(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_center_embeddings, all_offset_embeddings, all_idxs = [], [], []
        all_union_center_embeddings, all_union_offset_embeddings, all_union_idxs = [], [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, offset_embedding, _ = \
                    self.embed_query_box(self.transform_union_query(batch_queries_dict[query_structure], 
                                                                    query_structure), 
                                         self.transform_union_structure(query_structure), 
                                         0)
                all_union_center_embeddings.append(center_embedding)
                all_union_offset_embeddings.append(offset_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                center_embedding, offset_embedding, _ = self.embed_query_box(batch_queries_dict[query_structure], 
                                                                             query_structure, 
                                                                             0)
                all_center_embeddings.append(center_embedding)
                all_offset_embeddings.append(offset_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_center_embeddings) > 0 and len(all_offset_embeddings) > 0:
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)
            all_offset_embeddings = torch.cat(all_offset_embeddings, dim=0).unsqueeze(1)
        if len(all_union_center_embeddings) > 0 and len(all_union_offset_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_offset_embeddings = torch.cat(all_union_offset_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0]//2, 2, 1, -1)
            all_union_offset_embeddings = all_union_offset_embeddings.view(all_union_offset_embeddings.shape[0]//2, 2, 1, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1)
                positive_logit = self.cal_logit_box(positive_embedding, all_center_embeddings, all_offset_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_box(positive_embedding, all_union_center_embeddings, all_union_offset_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1).long()).view(batch_size, negative_size, -1)
                negative_logit = self.cal_logit_box(negative_embedding, all_center_embeddings, all_offset_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1).long()).view(batch_size, 1, negative_size, -1)
                negative_union_logit = self.cal_logit_box(negative_embedding, all_union_center_embeddings, all_union_offset_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs
    
    def cal_logit_vec(self, entity_embedding, query_embedding):
        distance = entity_embedding - query_embedding
        logit = self.gamma - torch.norm(distance, p=1, dim=-1)
        return logit

    def forward_vec(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_center_embeddings, all_idxs = [], []
        all_union_center_embeddings, all_union_idxs = [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, _ = self.embed_query_vec(self.transform_union_query(batch_queries_dict[query_structure], 
                                                                    query_structure), 
                                                                self.transform_union_structure(query_structure), 0)
                all_union_center_embeddings.append(center_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                center_embedding, _ = self.embed_query_vec(batch_queries_dict[query_structure], query_structure, 0)
                all_center_embeddings.append(center_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_center_embeddings) > 0:
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)
        if len(all_union_center_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0]//2, 2, 1, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1)
                positive_logit = self.cal_logit_vec(positive_embedding, all_center_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_vec(positive_embedding, all_union_center_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1).long()).view(batch_size, negative_size, -1)
                negative_logit = self.cal_logit_vec(negative_embedding, all_center_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1).long()).view(batch_size, 1, negative_size, -1)
                negative_union_logit = self.cal_logit_vec(negative_embedding, all_union_center_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step):
        model.train()
        optimizer.zero_grad()
        generalizations=None
        if args.tbox:
            positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures, generalizations = next(train_iterator)
            """generalizations is a list of generalizations of each train query
               each element of this list is a list of tuples, where each tuple is a pair of (query,query_structure)
               Examples: generalizations[0]=[([26837, 27],  ('e', ('r',))), 
                                              ([33463, 27], ('e', ('r',))), 
                                              ([40626, 27], ('e', ('r',))), 
                                              ([26711, 27, 18], ('e', ('r', 'r'))), 
                                              ([13723, 18], ('e', ('r',)))
                                            ]
               
            """
            #logging.info('QUUERY BATCH')
            # logging.info(batch_queries)
            # print('QUERY STRUCTURES:')
            # print(query_structures)
            # print('GENERALIZATIONS:')
            # print(generalizations)
        else:
            positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(train_iterator)

        '''Reorganize train queries'''
        batch_queries_dict = collections.defaultdict(list) # dict: q_shape -> [flatten queries of this shape]
        batch_idxs_dict = collections.defaultdict(list) # dict: q_shape -> [idx1,....,idxn from batch queries]
        for i, query in enumerate(batch_queries): # group queries with same structure
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
            else:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])

        '''new code Reorganize generalized queries'''
        if generalizations is not None:
            batch_generalized_queries_dict = collections.defaultdict(
                list)  # dict: q_shape -> [flatten queries of this shape]
            batch_generalized_idxs_dict = collections.defaultdict(
                list)  # dict: q_shape -> [idx1,....,idxn from general queries]
            max_number_generalized_queries_per_train_query = 0;
            for i, generalized_queries in enumerate(generalizations):
                if len(generalized_queries) > max_number_generalized_queries_per_train_query:
                    max_number_generalized_queries_per_train_query=len(generalized_queries)
                for each_query in generalized_queries:
                    batch_generalized_queries_dict[each_query[-1]].append(each_query[0])
                    batch_generalized_idxs_dict[each_query[-1]].append(i)
            for generalized_query_structure in batch_generalized_queries_dict:
                if args.cuda:
                    batch_generalized_queries_dict[generalized_query_structure] = torch.LongTensor(batch_generalized_queries_dict[generalized_query_structure]).cuda()
                else:
                    batch_generalized_queries_dict[generalized_query_structure] = torch.LongTensor(batch_generalized_queries_dict[generalized_query_structure])

            '''end of new code'''


        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()




        #TODO: clean code; else branch is redundant?
        if args.tbox:
            positive_logit, negative_logit, subsampling_weight, _ = model(positive_sample, negative_sample,
                                                                          subsampling_weight, batch_queries_dict,batch_idxs_dict,
                                                                          batch_generalized_queries_dict,batch_generalized_idxs_dict, max_number_generalized_queries_per_train_query, args)
            # Kien code
            if args.mean_gen_new:
                positive_score = F.logsigmoid(positive_logit)
                generalization_weights=[1]
                number_of_generalizations=positive_score.shape[1]
                for i in range(1,number_of_generalizations):
                    generalization_weights.append(1/(number_of_generalizations-1))
                generalization_weights=torch.tensor(generalization_weights)
                if args.cuda:
                    generalization_weights=generalization_weights.cuda()

                positive_score = positive_score*generalization_weights
                positive_score =positive_score.sum(dim=1,keepdim=True)
                positive_score = positive_score.squeeze(dim=1)
            # end Kien code

        else:
            positive_logit, negative_logit, subsampling_weight, _ = model(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict, None, None, None, args)
            positive_score = F.logsigmoid(positive_logit)
            positive_score = positive_score.squeeze(dim=1)
        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)




        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        loss.backward()
        optimizer.step()
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }
        return log



    def evaluate_specializations(model, easy_answers, hard_answers, args, test_dataloader,
                                                                       negative_sample, queries, queries_unflatten, query_structures, specializations):
        '''new code Reorganize specialized queries'''
        if specializations is not None:
            batch_spe_queries_dict = collections.defaultdict(
                list)  # dict: q_shape -> [flatten queries of this shape]
            batch_spe_idxs_dict = collections.defaultdict(
                list)  # dict: q_shape -> [idx1,....,idxn from specialized queries]
            batch_query_idx_spe_idxes = collections.defaultdict(
                list)  # dict: q_shape -> [idx1,....,idxn from specialized queries]
            # max_number_spe_queries_per_train_query = 0;
            spe_negative_sample = []
            q_idx = 0
            for i, spe_queries in enumerate(specializations):
                # if len(spe_queries) > max_number_spe_queries_per_train_query:
                # max_number_spe_queries_per_train_query = len(spe_queries)
                for spe_idx, each_query in enumerate(spe_queries):
                    batch_spe_queries_dict[each_query[-1]].append(each_query[0])
                    # append id of original query
                    batch_spe_idxs_dict[each_query[-1]].append(q_idx)
                    # print('ppp %d'%(spe_idx))
                    batch_query_idx_spe_idxes[i].append(q_idx)
                    spe_negative_sample.append(negative_sample[i])
                    q_idx += 1
            for spe_query_structure in batch_spe_queries_dict:
                if args.cuda:
                    batch_spe_queries_dict[spe_query_structure] = torch.LongTensor(
                        batch_spe_queries_dict[spe_query_structure]).cuda()
                else:
                    batch_spe_queries_dict[spe_query_structure] = torch.LongTensor(
                        batch_spe_queries_dict[spe_query_structure])

            # print(batch_spe_queries_dict)
            # print(batch_spe_idxs_dict)

            '''end of new code'''
        batch_generalized_queries_dict = collections.defaultdict(
            list)  # dict: q_shape -> [flatten queries of this shape]
        batch_generalized_idxs_dict = collections.defaultdict(
            list)  # dict: q_shape -> [idx1,....,idxn from general queries]
        max_number_generalized_queries_per_train_query = 0;

        # print('Negative sample: ')
        spe_negative_sample = torch.stack(spe_negative_sample)
        # print(spe_negative_sample)
        # get the scores for all specializations in the batch
        _, negative_logit, _, idxs = model(None, spe_negative_sample, None, batch_spe_queries_dict, batch_spe_idxs_dict,
                                           batch_generalized_queries_dict, batch_generalized_idxs_dict,
                                           max_number_generalized_queries_per_train_query, args)

        # queries_unflatten = [queries_unflatten[i] for i in idxs]
        # query_structures = [query_structures[i] for i in idxs]
        argsort = torch.argsort(negative_logit, dim=1, descending=True)
        ranking = argsort.clone().to(torch.float)
        if len(
                argsort) == args.test_batch_size:  # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
            ranking = ranking.scatter_(1, argsort,
                                       model.batch_entity_range)  # achieve the ranking of all entities
        else:  # otherwise, create a new torch Tensor for batch_entity_range
            if args.cuda:
                ranking = ranking.scatter_(1,
                                           argsort,
                                           torch.arange(model.nentity).to(torch.float).repeat(
                                               argsort.shape[0],
                                               1).cuda()
                                           )  # achieve the ranking of all entities
            else:
                ranking = ranking.scatter_(1,
                                           argsort,
                                           torch.arange(model.nentity).to(torch.float).repeat(
                                               argsort.shape[0],
                                               1)
                                           )  # achieve the ranking of all entities

        # we enumerate all queries in the batch and perform the min over all specializations
        all_rankings = 0
        for i, (query, query_structure) in enumerate(zip(queries_unflatten, query_structures)):
            # identify all rows in ranking belonging to query i
            hard_answer = hard_answers[query]
            easy_answer = easy_answers[query]
            num_hard = len(hard_answer)
            num_easy = len(easy_answer)
            assert len(hard_answer.intersection(easy_answer)) == 0
            spe_ranking = []
            for idx in batch_query_idx_spe_idxes[i]:
                cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                cur_ranking, indices = torch.sort(cur_ranking)
                masks = indices >= num_easy
                if args.cuda:
                    answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                else:
                    answer_list = torch.arange(num_hard + num_easy).to(torch.float)
                cur_ranking = cur_ranking - answer_list + 1  # filtered setting
                cur_ranking = cur_ranking[masks]  # only take indices that belong to the hard answers

                spe_ranking.append(cur_ranking)

            # do torch.min over all the processed rankings

            spe_ranking = torch.stack(spe_ranking)
            # print(spe_ranking)
            cur_ranking, _ = torch.min(spe_ranking, dim=0, keepdim=True)
            mrr = torch.mean(1. / cur_ranking).item()
            h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
            h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
            h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()
            all_rankings = cur_ranking
        return all_rankings


    def evaluate_queries(model, easy_answers, hard_answers, args, test_dataloader,
                                                      negative_sample, queries, queries_unflatten, query_structures, specializations):
        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        for i, query in enumerate(queries):
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
            else:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
        if args.cuda:
            negative_sample = negative_sample.cuda()
        batch_generalized_queries_dict = collections.defaultdict(
            list)  # dict: q_shape -> [flatten queries of this shape]
        batch_generalized_idxs_dict = collections.defaultdict(
            list)  # dict: q_shape -> [idx1,....,idxn from general queries]
        max_number_generalized_queries_per_train_query = 0;
        _, negative_logit, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict,
                                           batch_generalized_queries_dict, batch_generalized_idxs_dict,
                                           max_number_generalized_queries_per_train_query, args)
        queries_unflatten = [queries_unflatten[i] for i in idxs]
        query_structures = [query_structures[i] for i in idxs]
        argsort = torch.argsort(negative_logit, dim=1, descending=True)
        ranking = argsort.clone().to(torch.float)
        if len(
                argsort) == args.test_batch_size:  # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
            ranking = ranking.scatter_(1, argsort, model.batch_entity_range)  # achieve the ranking of all entities
        else:  # otherwise, create a new torch Tensor for batch_entity_range
            if args.cuda:
                ranking = ranking.scatter_(1,
                                           argsort,
                                           torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                              1).cuda()
                                           )  # achieve the ranking of all entities
            else:
                ranking = ranking.scatter_(1,
                                           argsort,
                                           torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                              1)
                                           )  # achieve the ranking of all entities
        all_rank = 0
        for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
            # print(query)
            # print(hard_answers.keys())
            # print(query in hard_answers)

            hard_answer = hard_answers[query]
            easy_answer = easy_answers[query]
            num_hard = len(hard_answer)
            num_easy = len(easy_answer)
            assert len(hard_answer.intersection(easy_answer)) == 0
            cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
            cur_ranking, indices = torch.sort(cur_ranking)
            masks = indices >= num_easy
            if args.cuda:
                answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
            else:
                answer_list = torch.arange(num_hard + num_easy).to(torch.float)
            cur_ranking = cur_ranking - answer_list + 1  # filtered setting
            cur_ranking = cur_ranking[masks]  # only take indices that belong to the hard answers

            mrr = torch.mean(1. / cur_ranking).item()

            h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
            h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
            h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()
            all_rank = cur_ranking
        return all_rank

    @staticmethod
    def test_step(model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, save_result=False, save_str="", save_empty=False):
        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)

        with torch.no_grad():
            if args.query_rew:
                for negative_sample, queries, queries_unflatten, query_structures, specializations in tqdm(test_dataloader, disable=not args.print_on_screen):

                    '''new code Reorganize specialized queries'''
                    if specializations is not None:
                        batch_spe_queries_dict = collections.defaultdict(
                            list)  # dict: q_shape -> [flatten queries of this shape]
                        batch_spe_idxs_dict = collections.defaultdict(
                            list)  # dict: q_shape -> [idx1,....,idxn from specialized queries]
                        batch_query_idx_spe_idxes = collections.defaultdict(
                            list)  # dict: q_shape -> [idx1,....,idxn from specialized queries]
                        # max_number_spe_queries_per_train_query = 0;
                        spe_negative_sample = []
                        q_idx = 0
                        for i, spe_queries in enumerate(specializations):
                            # if len(spe_queries) > max_number_spe_queries_per_train_query:
                            # max_number_spe_queries_per_train_query = len(spe_queries)
                            for spe_idx, each_query in enumerate(spe_queries):
                                batch_spe_queries_dict[each_query[-1]].append(each_query[0])
                                # append id of original query
                                batch_spe_idxs_dict[each_query[-1]].append(q_idx)
                                # print('ppp %d'%(spe_idx))
                                batch_query_idx_spe_idxes[i].append(q_idx)
                                spe_negative_sample.append(negative_sample[i])
                                q_idx += 1
                        for spe_query_structure in batch_spe_queries_dict:
                            if args.cuda:
                                batch_spe_queries_dict[spe_query_structure] = torch.LongTensor(
                                    batch_spe_queries_dict[spe_query_structure]).cuda()
                            else:
                                batch_spe_queries_dict[spe_query_structure] = torch.LongTensor(
                                    batch_spe_queries_dict[spe_query_structure])

                        # print(batch_spe_queries_dict)
                        # print(batch_spe_idxs_dict)

                        '''end of new code'''
                    batch_generalized_queries_dict = collections.defaultdict(
                        list)  # dict: q_shape -> [flatten queries of this shape]
                    batch_generalized_idxs_dict = collections.defaultdict(
                        list)  # dict: q_shape -> [idx1,....,idxn from general queries]
                    max_number_generalized_queries_per_train_query = 0;

                    # print('Negative sample: ')
                    spe_negative_sample = torch.stack(spe_negative_sample)
                    if args.cuda:
                        spe_negative_sample = spe_negative_sample.cuda()
                    #print(spe_negative_sample.size())
                    # get the scores for all specializations in the batch
                    _, negative_logit, _, idxs = model(None, spe_negative_sample, None, batch_spe_queries_dict,
                                                       batch_spe_idxs_dict,
                                                       batch_generalized_queries_dict, batch_generalized_idxs_dict,
                                                       max_number_generalized_queries_per_train_query, args)

                    # queries_unflatten = [queries_unflatten[i] for i in idxs]
                    # query_structures = [query_structures[i] for i in idxs]
                    argsort = torch.argsort(negative_logit, dim=1, descending=True)
                    ranking = argsort.clone().to(torch.float)
                    if len(
                            argsort) == args.test_batch_size:  # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                        ranking = ranking.scatter_(1, argsort,
                                                   model.batch_entity_range)  # achieve the ranking of all entities
                    else:  # otherwise, create a new torch Tensor for batch_entity_range
                        if args.cuda:
                            ranking = ranking.scatter_(1,
                                                       argsort,
                                                       torch.arange(model.nentity).to(torch.float).repeat(
                                                           argsort.shape[0],
                                                           1).cuda()
                                                       )  # achieve the ranking of all entities
                        else:
                            ranking = ranking.scatter_(1,
                                                       argsort,
                                                       torch.arange(model.nentity).to(torch.float).repeat(
                                                           argsort.shape[0],
                                                           1)
                                                       )  # achieve the ranking of all entities

                    # we enumerate all queries in the batch and perform the min over all specializations
                    #all_rankings = 0
                    for i, (query, query_structure) in enumerate(zip(queries_unflatten, query_structures)):
                        # identify all rows in ranking belonging to query i
                        hard_answer = hard_answers[query]
                        easy_answer = easy_answers[query]
                        num_hard = len(hard_answer)
                        num_easy = len(easy_answer)
                        assert len(hard_answer.intersection(easy_answer)) == 0
                        spe_ranking = []
                        for idx in batch_query_idx_spe_idxes[i]:
                            cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                            cur_ranking, indices = torch.sort(cur_ranking)
                            masks = indices >= num_easy
                            if args.cuda:
                                answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                            else:
                                answer_list = torch.arange(num_hard + num_easy).to(torch.float)
                            cur_ranking = cur_ranking - answer_list + 1  # filtered setting
                            cur_ranking = cur_ranking[masks]  # only take indices that belong to the hard answers

                            spe_ranking.append(cur_ranking)

                        # do torch.min over all the processed rankings

                        spe_ranking = torch.stack(spe_ranking)
                        # print(spe_ranking)
                        cur_ranking, _ = torch.min(spe_ranking, dim=0, keepdim=True)
                        mrr = torch.mean(1. / cur_ranking).item()
                        h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                        h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                        h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()


                    #print(negative_sample)
                    # group specializations per query structure

                        #print('Min result:')
                        #print(cur_ranking)



                        logs[query_structure].append({
                            'MRR': mrr,
                            'HITS1': h1,
                            'HITS3': h3,
                            'HITS10': h10,
                            'num_hard_answer': num_hard,
                        })



                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

            else:

                for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader, disable=not args.print_on_screen):
                    #print(negative_sample)
                    batch_queries_dict = collections.defaultdict(list)
                    batch_idxs_dict = collections.defaultdict(list)
                    for i, query in enumerate(queries):
                        batch_queries_dict[query_structures[i]].append(query)
                        batch_idxs_dict[query_structures[i]].append(i)
                    for query_structure in batch_queries_dict:
                        if args.cuda:
                            batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
                        else:
                            batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                    if args.cuda:
                        negative_sample = negative_sample.cuda()
                    batch_generalized_queries_dict = collections.defaultdict(
                        list)  # dict: q_shape -> [flatten queries of this shape]
                    batch_generalized_idxs_dict = collections.defaultdict(
                        list)  # dict: q_shape -> [idx1,....,idxn from general queries]
                    max_number_generalized_queries_per_train_query = 0;
                    _, negative_logit, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict, batch_generalized_queries_dict, batch_generalized_idxs_dict, max_number_generalized_queries_per_train_query, args)
                    queries_unflatten = [queries_unflatten[i] for i in idxs]
                    query_structures = [query_structures[i] for i in idxs]
                    argsort = torch.argsort(negative_logit, dim=1, descending=True)
                    ranking = argsort.clone().to(torch.float)
                    if len(argsort) == args.test_batch_size: # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                        ranking = ranking.scatter_(1, argsort, model.batch_entity_range) # achieve the ranking of all entities
                    else: # otherwise, create a new torch Tensor for batch_entity_range
                        if args.cuda:
                            ranking = ranking.scatter_(1,
                                                       argsort,
                                                       torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                                          1).cuda()
                                                       ) # achieve the ranking of all entities
                        else:
                            ranking = ranking.scatter_(1,
                                                       argsort,
                                                       torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                                          1)
                                                       ) # achieve the ranking of all entities
                    for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
                        #print(query)
                        #print(hard_answers.keys())
                        #print(query in hard_answers)

                        hard_answer = hard_answers[query]
                        easy_answer = easy_answers[query]
                        num_hard = len(hard_answer)
                        num_easy = len(easy_answer)
                        assert len(hard_answer.intersection(easy_answer)) == 0
                        cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                        cur_ranking, indices = torch.sort(cur_ranking)
                        masks = indices >= num_easy
                        if args.cuda:
                            answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                        else:
                            answer_list = torch.arange(num_hard + num_easy).to(torch.float)
                        cur_ranking = cur_ranking - answer_list + 1 # filtered setting
                        cur_ranking = cur_ranking[masks] # only take indices that belong to the hard answers

                        mrr = torch.mean(1./cur_ranking).item()

                        h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                        h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                        h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                        'num_hard_answer': num_hard,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]])/len(logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics