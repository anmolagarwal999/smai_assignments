
import numpy as np
import math
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from random import choices


class RandomForestClass:

    def __init__(self,n_trees,impurity_measuring_criterion, max_depth=None,min_samples_leaf=2, max_features=None):
        self.trained_already=False

        self.n_trees=n_trees
        self.impurity_measuring_criterion=impurity_measuring_criterion
        self.max_depth=max_depth
        self.min_samples_leaf=min_samples_leaf
        self.max_features=max_features 
        self.num_classes=2

        ###################  
        self.trees_arr=[]
        self.fill_trees()
        ##################

    def fill_trees(self):
        for i in range(self.n_trees):
            curr_tree_obj=DecisionTreeClassifier(random_state=5,
                                            criterion=self.impurity_measuring_criterion,
                                            max_depth=self.max_depth,
                                            min_samples_leaf=self.min_samples_leaf,
                                            max_features=self.max_features)
            self.trees_arr.append(curr_tree_obj)
        return

    def fetch_sample_data(self, X_data,y_data, fraction=1):
        num_samples=X_data.shape[0]
        indices=list(range(num_samples))
        # print("Tot sampples is ", num_samples)
        assert(len(y_data)==num_samples)
        wanted_samples=int(num_samples*fraction)
        chosen_indices=choices(indices, k=wanted_samples)
        X_chosen=pd.DataFrame([X_data.iloc[i] for i in chosen_indices])
        y_chosen=[y_data.iloc[i] for i in chosen_indices]
        return X_chosen, y_chosen


    
    def fit_train_set(self, X_train, y_train):
        for curr_tree in self.trees_arr:
            X_chosen, y_chosen=self.fetch_sample_data(X_train, y_train)
            curr_tree.fit(X_chosen, y_chosen)
        self.trained_already=True
        return
    
 
    
    def fetch_results(self, X_test, return_type="class_label"):

        assert(self.trained_already)
        assert(return_type in ["class_label","class_freqs"])
        # assert(X_test.ndim==2)
        ans=[]
        num_samples_in_tc=X_test.shape[0]
        for i in range(num_samples_in_tc):
            ans.append([])
        for curr_tree in self.trees_arr:
            curr_ans=curr_tree.predict(X_test)
            assert(len(curr_ans)==num_samples_in_tc)
            for idx, sample_ans in enumerate(curr_ans):
                ans[idx].append(sample_ans)
        if return_type=="class_freqs":
            return ans
        ans_label=[]
        for curr_freqs_arr in ans:
            predicted_label=Counter(curr_freqs_arr).most_common(1)[0][0]
            ans_label.append(predicted_label)
        return ans_label

        





            


