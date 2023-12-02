'''
Created on July 1, 2020
@author: Tinglin Huang (tinglin.huang@zju.edu.cn)
'''
__author__ = "huangtinglin"

import os

import random
from collections import UserList, defaultdict

import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from time import time
from prettytable import PrettyTable
import pickle

from utils.parser import parse_args
from utils.data_loader import load_data
from modules.KGIN import Recommender
from utils.evaluate import test
from utils.custom_evaluate import custom_test
from utils.helper import early_stopping

n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0


def get_feed_dict(train_entity_pairs, start, end, train_user_set):
    # print(train_entity_pairs)

    def negative_sampling(user_item, train_user_set):
      # For each positive user-item interaction, we sample a random negative item interaction:

      neg_items = []
      # print(user_item)
      for user, _ in user_item.cpu().numpy():
          user = int(user)
          while True:
              neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
              if neg_item not in train_user_set[user] \
                  and neg_item not in test_user_neg_set[user]:
                  break
          neg_items.append(neg_item)
      return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    # print(entity_pairs)
    feed_dict['users'] = entity_pairs[:, 0]
    # print(feed_dict['users'])
    feed_dict['pos_items'] = entity_pairs[:, 1]
    # print(feed_dict['pos_items'])
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs,
                                                                train_user_set)).to(device)
    # print(feed_dict['neg_items'])

    return feed_dict


if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)

    adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    # print(train_cf_pairs)

    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))
    # print(test_cf_pairs)


    neg_items = []
    test_user_neg_set = defaultdict(list)
    for user, _ in test_cf_pairs.cpu().numpy():
        user = int(user)
        while True:
            neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
            if neg_item not in user_dict['test_user_set'][user]:
                break
        neg_items.append(neg_item)
        test_user_neg_set[user].append(neg_item)
    # print(neg_items)
    # print(test_user_neg_set)
    
    pos_DTItest = torch.stack(
      (test_cf_pairs[:, 0], test_cf_pairs[:, 1], torch.ones(test_cf_pairs.shape[0], dtype=torch.long)),
      1
    )
    # print(pos_DTItest)
    neg_DTItest = torch.stack(
      (test_cf_pairs[:, 0], torch.tensor(neg_items), torch.zeros(test_cf_pairs.shape[0], dtype=torch.long)),
      1
    )
    # print(neg_DTItest)
    DTItest = torch.cat((pos_DTItest, neg_DTItest))
    # print(DTItest, DTItest.shape)
  
    """define model"""
    # print(mean_mat_list[0])
    # print(device)
    model = Recommender(n_params, args, graph, mean_mat_list[0]).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False

    train_loss_list = []
    test_auc_list = []
    test_aupr_list = []

    print("start training ...")
    for epoch in range(args.epoch):
        """training CF"""
        # shuffle training data
        if args.splitting_mode == 'per-user':
          index = np.arange(len(train_cf))
          np.random.shuffle(index)
          train_cf_pairs = train_cf_pairs[index]

        """training"""
        loss, s, cor_loss = 0, 0, 0
        train_s_t = time()

        while s + args.batch_size <= len(train_cf):
          # print(s)
          batch = get_feed_dict(
            train_cf_pairs, 
            s, s + args.batch_size,
            user_dict['train_user_set']
          )
          # print(batch['users'])
          # print(batch['neg_items'])

          batch_loss, _, _, batch_cor = model(batch)
          # print(batch_cor)

          batch_loss = batch_loss
          optimizer.zero_grad()
          batch_loss.backward()
          optimizer.step()

          loss += batch_loss
          cor_loss += batch_cor
          s += args.batch_size

        train_e_t = time()

        train_loss_list.append(loss.item())

        if epoch % 10 == 9 or epoch == 1:
            """testing"""
            test_s_t = time()

            if args.evaluation_approach == 'KGIN':
              ret = test(model, user_dict, n_params)
            else:
              # deepDR model evaluation approach
              ret = custom_test(
                model, DTItest, test_cf_pairs, train_cf_pairs, n_params
              )

            test_auc_list.append(float(ret['auc']))
            test_aupr_list.append(float(ret['aupr']))

            test_e_t = time()

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "testing time", "Loss", "recall", "auc", "aupr"]
            train_res.add_row(
                [epoch, "{:.4f}".format(train_e_t - train_s_t), "{:.4f}".format(test_e_t - test_s_t), "{:.6f}".format(loss.item()),
                  ret['recall'], ret['auc'], ret['aupr']]
            )
            print(train_res)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][3], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=10)
            if should_stop:
                break

            """save weight"""
            if ret['recall'][3] == cur_best_pre_0 and args.save:
                os.chdir('/content/drive/MyDrive/EKGDR-main')
                # print(os.getcwd())
                torch.save(model, args.out_dir + 'model_' + args.dataset + '.pt')
                print('Saved.')

            if ret['recall'][3] == cur_best_pre_0:
              os.chdir('/content/drive/MyDrive/EKGDR-main')
              # print(os.getcwd())
              with open('results.pickle', 'wb') as handle:
                  pickle.dump(ret, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            # print(loss, cor_loss)
            
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
            print('using time %.4f, training loss at epoch %d: %.6f, cor: %.6f' % (
            train_e_t - train_s_t, epoch, loss.item(), cor_loss.item() if args.n_factors >= 2 else cor_loss))

    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))
