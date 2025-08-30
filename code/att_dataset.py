import pandas as pd
import numpy as np
import random
import math
import multiprocessing
import time
from collections import defaultdict
from loguru import logger

class Dataset(object):
    def __init__(self, file_prefix, is_valid=1):
        self.data = file_prefix
        logger.info(self.data)
        if self.data == '../data_process/Tmall/processed_data_att/Tmall':
            self.train = pd.read_csv(file_prefix + "_train.csv", header=None, 
                                     names=['user_id', 'item_id', 'cat_id', 'seller_id', 'brand_id', 'timestamp', 'action_type'], 
                                     dtype={'user_id': np.int32, 'item_id': np.int32, 'cat_id': np.int32, 'seller_id': np.int32, 'brand_id': np.int32, 'timestamp': np.int32, 'action_type': np.int32})
            self.valid = pd.read_csv(file_prefix + "_valid.csv", header=None, 
                                     names=['user_id', 'item_id', 'cat_id', 'seller_id', 'brand_id', 'timestamp', 'action_type'], 
                                     dtype={'user_id': np.int32, 'item_id': np.int32, 'cat_id': np.int32, 'seller_id': np.int32, 'brand_id': np.int32, 'timestamp': np.int32, 'action_type': np.int32})
        elif self.data == '../data_process/kaggle/processed_data_att/kaggle':
            self.train = pd.read_csv(file_prefix + "_train.csv", header=None, 
                                     names=['user_id', 'item_id', 'cat_id', 'brand_id', 'timestamp', 'action_type'], 
                                     dtype={'user_id': np.int32, 'item_id': np.int32, 'cat_id': np.int32, 'brand_id': np.int32, 'timestamp': np.int32, 'action_type': np.int32})
            self.valid = pd.read_csv(file_prefix + "_valid.csv", header=None, 
                                     names=['user_id', 'item_id', 'cat_id', 'brand_id', 'timestamp', 'action_type'], 
                                     dtype={'user_id': np.int32, 'item_id': np.int32, 'cat_id': np.int32, 'brand_id': np.int32, 'timestamp': np.int32, 'action_type': np.int32})

        
        if is_valid == 0:
            self.train = pd.concat([self.train, self.valid], axis='index')
            if self.data == '../data_process/Tmall/processed_data_att/Tmall':
                self.bet_val_test = pd.read_csv(file_prefix + "_between_val_test.csv", header=None,
                                                names=['user_id', 'item_id', 'cat_id', 'seller_id', 'brand_id', 'timestamp', 'action_type'],
                                                dtype={'user_id': np.int32, 'item_id': np.int32, 'cat_id': np.int32, 'seller_id': np.int32, 'brand_id': np.int32, 'timestamp': np.int32, 'action_type': np.int32})
                self.valid = pd.read_csv(file_prefix + "_test.csv", header=None,
                                        names=['user_id', 'item_id', 'cat_id', 'seller_id', 'brand_id', 'timestamp', 'action_type'],
                                        dtype={'user_id': np.int32, 'item_id': np.int32, 'cat_id': np.int32, 'seller_id': np.int32, 'brand_id': np.int32, 'timestamp': np.int32, 'action_type': np.int32})
            elif self.data == '../data_process/kaggle/processed_data_att/kaggle':
                self.bet_val_test = pd.read_csv(file_prefix + "_between_val_test.csv", header=None,
                                                names=['user_id', 'item_id', 'cat_id', 'brand_id', 'timestamp', 'action_type'],
                                                dtype={'user_id': np.int32, 'item_id': np.int32, 'cat_id': np.int32, 'brand_id': np.int32, 'timestamp': np.int32, 'action_type': np.int32})
                self.valid = pd.read_csv(file_prefix + "_test.csv", header=None,
                                        names=['user_id', 'item_id', 'cat_id', 'brand_id', 'timestamp', 'action_type'],
                                        dtype={'user_id': np.int32, 'item_id': np.int32, 'cat_id': np.int32, 'brand_id': np.int32, 'timestamp': np.int32, 'action_type': np.int32})
            self.train = pd.concat([self.train, self.bet_val_test], axis='index')
    
        self.candidate = pd.read_csv(file_prefix + "_negative.csv", header=None)

    def fix_length(self, maxlen=50):
        self.maxlen = maxlen
        self.train.sort_values(by=['user_id', 'timestamp'], axis='index', ascending=True, inplace=True, kind='mergesort')
        self.user_maxid = np.max(self.train.user_id.unique())
        self.item_maxid = np.max(self.train.item_id.unique())  # note that item_maxid > item_size, but it doesn't matter
        self.user_set = set(self.train.user_id.unique())
        self.item_set = set(self.train.item_id.unique())
        self.item_list = list(range(1, self.item_maxid + 1))  # index from 1 to maxid

        self.cat_maxid = np.max(self.train.cat_id.unique())
        self.brand_maxid = np.max(self.train.brand_id.unique())
        self.cat_set = set(self.train.cat_id.unique())
        self.brand_set = set(self.train.brand_id.unique())
        
        if self.data == '../data_process/Tmall/processed_data_att/Tmall':
            self.seller_maxid = np.max(self.train.seller_id.unique())
            self.seller_set = set(self.train.seller_id.unique())
        elif self.data == '../data_process/kaggle/processed_data_att/kaggle':
            self.seller_maxid = 0
            self.seller_set = {}
        
        logger.info('real item size: %d, user_maxid: %d, item_maxid: %d' % (len(self.item_set), self.user_maxid, self.item_maxid))
        logger.info('real cat size: %d, cat_maxid: %d' % (len(self.cat_set), self.cat_maxid))
        logger.info('real seller size: %d, seller_maxid: %d' % (len(self.seller_set), self.seller_maxid))
        logger.info('real brand size: %d, brand_maxid: %d' % (len(self.brand_set), self.brand_maxid))

        self.train = self.train.groupby(['user_id']).tail(maxlen + 1)
        
        self.train_seq = {}
        self.valid_seq = {}
        self.valid_neg_cand = {}
        self.train_behavior_seq = {}
        self.valid_behavior_seq = {}
        
        self.train_cat_seq = {}
        self.valid_cat_seq = {}
        self.train_seller_seq = {}
        self.valid_seller_seq = {}
        self.train_brand_seq = {}
        self.valid_brand_seq = {}
        
        for u in self.user_set:
            items = self.train[self.train['user_id'] == u].item_id.values
            seq = np.pad(items, (maxlen+1-len(items), 0), 'constant')  # padding from the left side
            
            # for attributes
            cats = self.train[self.train['user_id'] == u].cat_id.values
            brands = self.train[self.train['user_id'] == u].brand_id.values
            cats_seq = np.pad(cats, (maxlen+1-len(cats), 0), 'constant')  # padding from the left side
            brands_seq = np.pad(brands, (maxlen+1-len(brands), 0), 'constant')  # padding from the left side
            
            self.train_cat_seq[u] = list(cats_seq)
            self.train_brand_seq[u] = list(brands_seq)
            
            if self.data == '../data_process/Tmall/processed_data_att/Tmall':    
                sellers = self.train[self.train['user_id'] == u].seller_id.values
                sellers_seq = np.pad(sellers, (maxlen+1-len(sellers), 0), 'constant')  # padding from the left side
                self.train_seller_seq[u] = list(sellers_seq)
            elif self.data == '../data_process/kaggle/processed_data_att/kaggle':
                self.train_seller_seq[u] = np.zeros(len(cats_seq)).tolist()
                
            behaviors = self.train[self.train['user_id'] == u].action_type.values
            if self.data == '../data_process/Tmall/processed_data_att/Tmall':
                behavior_seq = np.pad(behaviors, (maxlen + 1 - len(behaviors), 0), 'constant', constant_values=4)
            elif self.data == '../data_process/kaggle/processed_data_att/kaggle':
                behavior_seq = np.pad(behaviors, (maxlen + 1 - len(behaviors), 0), 'constant', constant_values=4)
            else:
                behavior_seq = None
            
            self.train_seq[u] = list(seq)
            self.train_behavior_seq[u] = list(behavior_seq)
            
        # create mask that mark 1 when item is padding, purchases, purchased items
        self.train_mask_real_pos_items = defaultdict(list)
        for u in self.user_set:
            purchase_item = set()
            for i, behavior in enumerate(self.train_behavior_seq[u]):
                if behavior == 0:
                    purchase_item.add(self.train_seq[u][i])

            for i, item in enumerate(self.train_seq[u]):

                if item in purchase_item:
                    self.train_mask_real_pos_items[u].append(0.0)
                else:
                    self.train_mask_real_pos_items[u].append(1.0)
        valid_user_list = list(self.valid.user_id.unique())
        for u in valid_user_list:
            if u in self.train_seq.keys():
                seq = self.train_seq[u][1:]
                target_item = self.valid[self.valid['user_id'] == u].item_id.values[0]
                seq.append(target_item)
                self.valid_seq[u] = seq
                behavior_seq = self.train_behavior_seq[u][1:]
                target_behavior = self.valid[self.valid['user_id'] == u].action_type.values[0]
                behavior_seq.append(target_behavior)
                self.valid_behavior_seq[u] = behavior_seq
                cat_seq = self.train_cat_seq[u][1:]
                target_cat = self.valid[self.valid['user_id'] == u].cat_id.values[0]
                cat_seq.append(target_cat)
                self.valid_cat_seq[u] = cat_seq
                
                if self.data == '../data_process/Tmall/processed_data_att/Tmall':
                    seller_seq = self.train_seller_seq[u][1:]
                    target_seller = self.valid[self.valid['user_id'] == u].seller_id.values[0]
                    seller_seq.append(target_seller)
                    self.valid_seller_seq[u] = seller_seq
                elif self.data == '../data_process/kaggle/processed_data_att/kaggle':
                    self.valid_seller_seq[u] = np.zeros(len(cat_seq)).tolist()
                
                brand_seq = self.train_brand_seq[u][1:]
                target_brand = self.valid[self.valid['user_id'] == u].brand_id.values[0]
                brand_seq.append(target_brand)
                self.valid_brand_seq[u] = brand_seq
                self.valid_neg_cand[u] = list(self.candidate[self.candidate[0] == u].values[0][1:])
            else:
                continue

    def gen_batch_candidate(self,batch_target):
        candidates = []
        for i in range(len(batch_target)):
            tar = batch_target[i][0]
            # keep the real item at the first place
            cand = [tar]
            items = self.item_list.copy()
            items.remove(tar)
            cand.extend(items)
            candidates.append(cand)
        return np.array(candidates)
    
    def sample_batch(self, batch_size=128):
        '''
        batch_uid: users,
        batch_x:   inps,
        batch_xb:  behs,
        batch_yp:  poss,
        batch_yn:  negs,
        batch_yb:  target_feedback,
        batch_cm:  mask_pos,
        
        batch_xac: cates,
        batch_xas: sells,
        batch_xab: brans,
        batch_yac: target_cat,
        batch_yas: target_seller,
        batch_yab: target_brand
        '''
        batch_x = []
        batch_xb = []
        batch_yp = []
        batch_yb = []
        batch_yn = []
        batch_cm = []
        batch_xac = []
        batch_xab = []
        batch_yac = []
        batch_yab = []
        batch_xas = []
        batch_yas = []

        batch_uid = random.sample(self.user_set, batch_size)

        for u in batch_uid:
            x = self.train_seq[u][:-1]
            cm = self.train_mask_real_pos_items[u][:-1]
            xac = self.train_cat_seq[u][:-1]
            xab = self.train_brand_seq[u][:-1]
            yac = self.train_cat_seq[u][1:]
            yab = self.train_brand_seq[u][1:]
            xas = self.train_seller_seq[u][:-1]
            yas = self.train_seller_seq[u][1:]

            xb = self.train_behavior_seq[u][:-1]
            yp = self.train_seq[u][1:]
            yb = self.train_behavior_seq[u][1:]
            yn = random.sample(self.item_set.difference(set(self.train_seq[u])), self.maxlen)
            batch_x.append(x)
            batch_cm.append(cm)

            batch_xb.append(xb)
            batch_yp.append(yp)
            batch_yb.append(yb)
            batch_yn.append(yn)
            batch_xac.append(xac)
            batch_xab.append(xab)
            batch_yac.append(yac)
            batch_yab.append(yab)
            batch_xas.append(xas)
            batch_yas.append(yas)
        return batch_uid, batch_x, batch_xb, batch_yp, batch_yn, batch_yb, batch_cm, batch_xac, batch_xas, batch_xab, batch_yac, batch_yas, batch_yab




