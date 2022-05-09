
import numpy as np
import math
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import random
import datetime
from numpy import linalg as LA

#implement Kmeans from scratch
class Kmeans:

    def __init__(self, train_x, train_y, num_clusters, init_method="rand_part", dist_metric="euclidean"):

        assert(num_clusters>0)
        self.train_x=train_x
        # self.test_x=test_x
        self.train_y=train_y
        # self.test_y=test_y
        self.num_clusters=num_clusters
        

        ################################
        self.flatten_features()
        ############################

        self.num_train_examples=self.train_x.shape[0]
        self.num_features=self.train_x.shape[1]
        self.dist_metric=dist_metric
        self.init_method=init_method
        self.current_clusters=[-1]*self.num_train_examples

        ##########################################################################
        self.cluster_mids=np.zeros((self.num_clusters, self.num_features))
        self.reassignments_arr=[]
        self.optimizing_function_arr=[]
        self.final_sil_score=None
        ##########################################################################

    

        ##################################
        # print("Shape of train set is ", train_x.shape)
        # print("Shape of test set is ", test_x.shape)
        ###################################
     

        ########################
        self.assign_initial_clusters(self.init_method)
        
        ##########################

        self.perform_knn()

    def update_cluster_centres(self):
        num_points=[0]*self.num_clusters
        self.cluster_mids=np.zeros((self.num_clusters, self.num_features))
        # print("init cluster mapping is ", self.current_clusters)

        for idx, curr_elem in enumerate(self.train_x):
            # print("Shape of curr elem is ", curr_elem.shape)
            current_cluster=self.current_clusters[idx]
            num_points[current_cluster]+=1
            self.cluster_mids[current_cluster]+=curr_elem
        # print("without division mids are ", self.cluster_mids)

        for idx, curr_mid in enumerate(self.cluster_mids):
            # assert(num_points[idx]>0)
            # an empty cluster can also appear
            # see example here: https://user.ceng.metu.edu.tr/~tcan/ceng465_f1314/Schedule/KMeansEmpty.html
            if num_points[idx]==0:
                # randomly select a mean
                curr_mid=np.random.rand(self.num_features)
            else:
                curr_mid/=num_points[idx]
            
        # print("updated cluster mids are ", self.cluster_mids)
    
    def get_dist_btw_vectors(self,x1, x2):
        if self.dist_metric=="euclidean":
            dv=x1-x2
            assert(dv.shape[0]==self.num_features)
            ans=0

            for curr_dx in dv:
                ans+=curr_dx**2
            ans=math.sqrt(ans)
            return ans
        else:
            # assert(self.dist_metric=="manhattan")
            return np.abs(x1-x2).sum()
    
    def cal_sil_score(self):
        # no point in storing, calculate on the fly
        overall_sum=0
        tot_sil_score=0
        norms=[LA.norm(x)**2 for x in self.train_x]
        ##############################
        curr_ans=np.dot(self.train_x, np.transpose(self.train_x))
        # print("Train x is ", self.train_x)
        # print("curr ans is ", curr_ans)
        # print("Shape is ", curr_ans.shape)
        ###########################


        for idx, curr_elem in enumerate(self.train_x):
            current_cluster_id=self.current_clusters[idx]
            ########
            # start_node = datetime.datetime.now()
            ########
            dis_dict={}
            cnt_dict={}
            for idx_neigh, neigh_elem in enumerate(self.train_x):
                if idx==idx_neigh:
                    continue
                # predicted_dis=self.get_dist_btw_vectors(curr_elem, neigh_elem)
                # current_dis=LA.norm(neigh_elem-curr_elem)
                # current_dis=np.abs(neigh_elem-curr_elem).sum()
                if self.dist_metric=="euclidean":
                    current_dis=math.sqrt(norms[idx]+norms[idx_neigh]-2*curr_ans[idx][idx_neigh])
                else:
                    current_dis=self.get_dist_btw_vectors(curr_elem, neigh_elem)
                # if predicted_dis!=current_dis:
                #     print(idx)
                #     print(idx_neigh)
                #     print(curr_elem)
                #     print("norm 1 is ", LA.norm(curr_elem))
                #     print("norm 1 is ", np.dot(curr_elem, curr_elem))
                #     print("norm 2 is ", LA.norm(neigh_elem))
                #     print("Certain dot product is is ", np.dot(curr_elem, neigh_elem))
                #     print("Calculated doc dot product is is ", curr_ans[idx][idx_neigh])
                #     print("Surprise: ", predicted_dis, current_dis)
                #     print(self.get_dist_btw_vectors(curr_elem, neigh_elem))
                #     return
                neigh_cluster_id=self.current_clusters[idx_neigh]
                if neigh_cluster_id not in dis_dict:
                    dis_dict[neigh_cluster_id]=0
                    cnt_dict[neigh_cluster_id]=0
                dis_dict[neigh_cluster_id]+=current_dis
                cnt_dict[neigh_cluster_id]+=1
            ################################
            a_param=0
            if (current_cluster_id in cnt_dict):
                # assert( cnt_dict[current_cluster_id]!=0)
                a_param=dis_dict[current_cluster_id]/cnt_dict[current_cluster_id]
            ##################################
            b_param=-1
            for this_cluster in cnt_dict:
                if this_cluster==current_cluster_id:
                    continue
                avg_dist=dis_dict[this_cluster]/cnt_dict[this_cluster]
                if b_param==-1 or b_param>avg_dist:
                    b_param=avg_dist
            assert(b_param!=-1)
            node_sil_score=(b_param-a_param)/(max(a_param, b_param))
            tot_sil_score+=node_sil_score
            ####################
            # end_node = datetime.datetime.now()
            # print("time taken for this node is ", (end_node-start_node).total_seconds())
            #################

        avg_sil_score=tot_sil_score/self.num_train_examples
        self.final_sil_score=avg_sil_score
        return        
                
    def perform_one_iteration(self):

        num_reassigns=0
        curr_optimization_function_val=0

        # print("clusters before reinit are ", self.current_clusters)
        # first find nearest distances
        for data_point_idx, curr_data_point in enumerate(self.train_x):
            closest_mid_yet=-1
            closest_mid_dist=-1
            for idx, curr_cluster_mid in enumerate(self.cluster_mids):
                curr_dist=self.get_dist_btw_vectors(curr_data_point, curr_cluster_mid)
                if closest_mid_yet==-1 or curr_dist<closest_mid_dist:
                    closest_mid_dist=curr_dist
                    closest_mid_yet=idx
            
            old_cluster_idx=self.current_clusters[data_point_idx]
            curr_optimization_function_val+=closest_mid_dist
            
            if old_cluster_idx!=closest_mid_yet:
                num_reassigns+=1
                self.current_clusters[data_point_idx]=closest_mid_yet
         
        # print("clusters AFTER reinit are ", self.current_clusters)           
        
        # calculate number of changes
        # print("Number of reassignments is ", num_reassigns)
        self.reassignments_arr.append(num_reassigns)
        self.optimizing_function_arr.append(curr_optimization_function_val)
        
        # recalculate cluster 
        self.update_cluster_centres()
        return 

    def assign_random_clusters(self):
        np.random.seed(22)

        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html
        # draw from [low, high) ie high is exclusive
        # self.current_clusters=np.random.randint(low=1, high=self.num_clusters+1 ,size=self.num_train_examples)
        self.current_clusters=np.random.randint(low=0, high=self.num_clusters ,size=self.num_train_examples)
        # print("initial clusters assigned")
        # print("init clusters for k means is ", self.current_clusters)
        return

    # decide random labels
    def flatten_features(self):
        self.train_x=self.fetch_flattened_2d_version(self.train_x)
        # self.test_x=self.fetch_flattened_2d_version(self.test_x)

        # print("Shape of train array after flattening is ", self.train_x.shape)
        # print("Shape of test array after flattening is ", self.test_x.shape)
        return

    def assign_initial_clusters(self, init_method="rand_part"):
        # print("method chosen is ", init_method)
        assert(init_method=="rand_part" or init_method=='forgy')
        if init_method=="rand_part":
            # print("inside")
            self.assign_random_clusters()
            # print("now update")
            self.update_cluster_centres()
        else:
            assert(init_method=="forgy")
            lucky_point_idxs=random.sample(range(0, self.num_train_examples),self.num_clusters)
            # print(lucky_point_idxs)
            for idx, this_idx in enumerate(lucky_point_idxs):
                self.cluster_mids[idx]=np.copy(self.train_x[this_idx])

        return

    def fetch_flattened_2d_version(self, arr):
        n_dims=arr.ndim
        assert(n_dims==3)
        # print("Init shape is ", arr.shape)
        # print("Num features is ", arr.shape[0])
        ans=np.array(list(map(lambda x: x.flatten(),arr)))
        return ans
    
    def perform_knn(self):
        # we have some cluster labels and current cluster mids
        changes_in_this_it=-1
        it_num=-1
        while changes_in_this_it!=0:
            it_num+=1
            self.perform_one_iteration()
            changes_in_this_it=self.reassignments_arr[-1]
            print("current iteration ", it_num)
            print("Changes in current iteration is ", changes_in_this_it)
        print("Now cal sil scores")
        start_time = datetime.datetime.now()
        self.cal_sil_score()        
        end_time = datetime.datetime.now()

        time_diff = (end_time - start_time)
        execution_time = time_diff.total_seconds()

        print("Time taken for silhouette is ", execution_time)

        return


def plot_tsne_visual(input_arr, output_arr):
    # assume input has just 2 dimensions
    n_samples=input_arr.shape[0]
    MAX_POINTS_TO_CONSIDER=2000
    if n_samples>MAX_POINTS_TO_CONSIDER:
        input_arr=input_arr[0:MAX_POINTS_TO_CONSIDER,:]
        output_arr=output_arr[0:MAX_POINTS_TO_CONSIDER]
    
    tsne_model = TSNE(n_components=2, random_state=12)
    tsne_data = tsne_model.fit_transform(input_arr)
    ###############################
    # creating a new data frame which help us in ploting the result data
    tsne_data = np.vstack((tsne_data.T, output_arr)).T
    tsne_df = pd.DataFrame(data=tsne_data, columns=("dimension_1", "dimension_2", "category"))

    sns.FacetGrid(tsne_df, hue="category", height=6).map(plt.scatter, "dimension_1", "dimension_2").add_legend()
    plt.show()
    return

