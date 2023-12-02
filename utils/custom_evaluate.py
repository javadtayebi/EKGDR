from .metrics import *
from .parser import parse_args

import torch
import pytrec_eval
from sklearn.metrics import roc_curve, precision_recall_curve
import numpy as np

import multiprocessing
import heapq
from time import time
from collections import defaultdict
from pprint import pprint


cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)
device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")


def get_auc_aupr(rating, DTItest):
  pred_list = []
  ground_truth = []
  for ele in DTItest:
    pred_list.append(rating[ele[0], ele[1]])
    ground_truth.append(ele[2])
  
  fpr, tpr, _ = roc_curve(ground_truth, pred_list)
  p, r, _ = precision_recall_curve(ground_truth, pred_list)

  return "{:.6f}".format(AUC(ground_truth, pred_list)), fpr, tpr, "{:.6f}".format(AUPR(ground_truth, pred_list)), p, r


def rec2dict(run):
    m, n = run.shape
    return {str(i): {str(int(run[i, j])): float(1.0 / (j + 1)) for j in range(n)} for i in range(m)}


def test2user_dict(test):
  test = test.numpy()

  dict_ = defaultdict(dict)
  for pair in test:
    dict_[str(pair[0])].update(
      {str(pair[1]): 1}
    )
  
  return dict_


def test2item_dict(test):
  test = test.numpy()

  dict_ = defaultdict(dict)
  for pair in test:
    dict_[str(pair[1])].update(
      {str(pair[0]): 1}
    )
  
  return dict_


class Evaluator:
    def __init__(self, metrics):
        self.result = {}
        self.metrics = metrics

    def evaluate(self, predict, test):
        evaluator = pytrec_eval.RelevanceEvaluator(test, self.metrics)
        self.result = evaluator.evaluate(predict)

    def show(self, metrics):
        result = {}
        for metric in metrics:
            res = pytrec_eval.compute_aggregated_measure(metric, [user[metric] for user in self.result.values()])
            result[metric] = res
            # print('{}={}'.format(metric, res))
        return result

    def show_all(self):
        key = next(iter(self.result.keys()))
        keys = self.result[key].keys()
        return self.show(keys)


def custom_test(model, DTItest, test_cf_pairs, train_cf_pairs, n_params, item_user_recall=False):
    """
    Here we are going to test model in deepDR way;
    Unlike KGIN (which works per-user), train/test splitting in deepDR is done over whole positive/negative pairs,
    therefore the AUC and AUPR metric will be calculated on whole test pairs,
    but @K metrics will be calculated per-item(drug).
    """
    result = {
      'precision': np.zeros(len(Ks)),
      'recall': [],
      'ndcg': np.zeros(len(Ks)),
      'hit_ratio': np.zeros(len(Ks)),
      'auc': 0.,
      'fpr': None,
      'tpr': None,
      'aupr': 0.,
      'p': None,
      'r': None,
    }

    global n_items , n_users
    n_items = n_params['n_items']
    n_users = n_params['n_users']


    # pool = multiprocessing.Pool(cores)


    entity_gcn_emb, user_gcn_emb = model.generate()

    u_g_embeddings = user_gcn_emb
    # print(u_g_embeddings.shape)
    i_g_embddings = entity_gcn_emb[:n_items]
    # print(i_g_embddings.shape)

    # whole user-item rating matrix:
    rating = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()
    # print(rating.shape)
    # print(torch.t(rating))

    # calculating AUC and AUPR on whole test pos/neg pairs:
    result['auc'], result['fpr'], result['tpr'], result['aupr'], result['p'], result['r'] = get_auc_aupr(rating, DTItest)
    # print(result['auc'], result['aupr'])


    # calculating @K metrics:
    rating = torch.sigmoid(rating)
    # print(rating.shape)

    rating[train_cf_pairs[:, 0], train_cf_pairs[:, 1]] = 0

    max_K = max(Ks)

    _, rec_items = torch.topk(
      torch.t(rating) if item_user_recall else rating, 
      max_K, 
      dim=1
    )
    # print(rec_items.shape)
    
    run = rec2dict(rec_items[:, 0:max_K])
    # print(run)
    test = test2item_dict(test_cf_pairs) if item_user_recall else test2user_dict(test_cf_pairs)
    # print(test)
    
    evaluator = Evaluator({'recall'})
    evaluator.evaluate(run, test)
    recall_result = evaluator.show(
      ['recall_5', 'recall_10', 'recall_15', 'recall_20', 'recall_30', 'recall_100', 'recall_200'])
    # print(recall_result)
    
    for re in recall_result:
      result['recall'].append(float("{:.6f}".format(recall_result[re])))


    # pool.close()
    return result