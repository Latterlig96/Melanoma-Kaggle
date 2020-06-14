import tensorflow as tf 
import pandas as pd 
import numpy as np
from sklearn.model_selection import GroupKFold,StratifiedKFold,KFold
import os 
import random 
from collections import Counter,defaultdict 



class Fold_Creator:

    def __init__(self,
                train_df_path = None,
                test_df_path = None,
                tfrecord_path = None,
                fold_type = None,
                n_splits = None,
                shuffle = False,
                random_state = None,
                group_col = None):

        """
            Class responsible for creating different types of folds 
            because creating folds from tfrecords is not really possible 
            so we must prepare other way to create good quality folds.
            Args:
            - train_df_path - path to train data 
            - test_df_path - path to test data 
            - tfrecord_path - path to tfrecord files
            - fold_type - Type of fold creation:
                - StratifiedKfold
                - GroupKFold
                - KFold
                - TFDataset - Perform KFold on set of TFRecordDataset paths,
                    we dont need other types of folds creation since we dont really know what's in these files
                    (if we did not create them).
                - StratifiedGroupKFold
            - n_splits - number of folds to be created
            - shuffle - whether to shuffle the data before folding
            - random_state - random number generator
            - group_col - when using StratifiedKfold or GroupKfold, a grouping column must be 
            specified.
        """        

        self.train_df_path = train_df_path 
        self.test_df_path = test_df_path
        self.tfrecord_path = tfrecord_path
        self.fold_type = fold_type
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.group_col = group_col

        if self.fold_type != 'TFDataset':
            self.read_df()
        else:
            pass 

    def read_df(self):
        """
            Read pandas dataframes from given paths
        """
        self.train_df = pd.read_csv(self.train_df_path)
        self.test_df = pd.read_csv(self.test_df_path)
    
    def format_path_train(self,img_name):
        return './Dataset/JPEG/train/' + img_name + '.jpg'
    
    def format_path_valid(self,img_name):
        return './Dataset/JPEG/train/' + img_name + '.jpg'

    def post_process(self,train_fold,valid_fold):
        train_paths_fold = train_fold.image_name.apply(self.format_path_train).values
        train_labels_fold = train_fold.target.values
        valid_paths_fold = valid_fold.image_name.apply(self.format_path_valid).values
        valid_labels_fold = valid_fold.target.values
        return train_paths_fold,train_labels_fold,valid_paths_fold,valid_labels_fold
    

    def stratified_group_k_fold(self,n_splits):
        """
            Implementation of Stratified GroupKFold Cross Validation 
            made by Jakub Wąsikowski that can be found under this link: 
            https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
        """
        labels_num = np.max(self.train_df.target.values) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, g in zip(self.train_df.target.values,self.train_df[self.group_col].values):
            y_counts_per_group[g][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        def eval_y_counts_per_fold(y_counts, fold):
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(n_splits)])
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)
    
        groups_and_y_counts = list(y_counts_per_group.items())
        random.Random(self.random_state).shuffle(groups_and_y_counts)

        for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(n_splits):
                fold_eval = eval_y_counts_per_fold(y_counts, i)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(g)

        all_groups = set(self.train_df[self.group_col])
        for i in range(n_splits):
            train_groups = all_groups - groups_per_fold[i]
            test_groups = groups_per_fold[i]

            train_indices = [i for i, g in enumerate(self.train_df[self.group_col]) if g in train_groups]
            test_indices = [i for i, g in enumerate(self.train_df[self.group_col]) if g in test_groups]

            yield train_indices, test_indices


    def create_folds_generator(self):
        
        if self.fold_type == 'KFold':
            kf = KFold(n_splits=self.n_splits,shuffle=self.shuffle,random_state=self.random_state)
            for trn_idx,val_idx in kf.split(self.train_df,self.train_df.target.values):
                train_fold = self.train_df.iloc[trn_idx]
                val_fold = self.train_df.iloc[val_idx]
                train_path,train_label,valid_path,valid_label = self.post_process(train_fold,val_fold)
                yield trn_idx,val_idx,train_path,train_label,valid_path,valid_label

        elif self.fold_type == 'StratifiedKFold':
            skf = StratifiedKFold(n_splits=self.n_splits,shuffle=self.shuffle,random_state=self.random_state)
            for trn_idx,val_idx in skf.split(self.train_df,self.train_df.target.values,
                                            groups=np.array(self.train_df[self.group_col].values) if self.group_col != None else None):
                train_fold = self.train_df.iloc[trn_idx]
                val_fold = self.train_df.iloc[val_idx]
                train_path,train_label,valid_path,valid_label = self.post_process(train_fold,val_fold)
                yield trn_idx,val_idx,train_path,train_label,valid_path,valid_label

        elif self.fold_type == 'GroupKFold': 
            gkf = GroupKFold(n_splits=self.n_splits)
            for trn_idx,val_idx in gkf.split(self.train_df,self.train_df.target.values,
                                            groups=np.array(self.train_df[self.group_col].values) if self.group_col != None else None):
                train_fold = self.train_df.iloc[trn_idx]
                val_fold = self.train_df.iloc[val_idx]
                train_path,train_label,valid_path,valid_label = self.post_process(train_fold,val_fold)
                yield trn_idx,val_idx,train_path,train_label,valid_path,valid_label

        else:
            sgkf = self.stratified_group_k_fold(self.n_splits)
            for trn_idx,val_idx in sgkf:
                train_fold = self.train_df.iloc[trn_idx]
                val_fold = self.train_df.iloc[val_idx]
                train_path,train_label,valid_path,valid_label = self.post_process(train_fold,val_fold)
                yield trn_idx,val_idx,train_path,train_label,valid_path,valid_label

    def create_tfrecord_fold_generator(self):
        kf = KFold(n_splits=self.n_splits,shuffle=self.shuffle,random_state=self.random_state)
        for trn_idx,val_idx in kf.split(self.tfrecord_path):
            training_paths = [self.tfrecord_path[idx] for idx in trn_idx]
            validation_paths = [self.tfrecord_path[idx] for idx in val_idx]

            yield training_paths,validation_paths