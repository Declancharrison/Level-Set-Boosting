#!/usr/bin/python3

### IMPORTS ###

import numpy as np
import copy
import os.path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as MSE
from scipy.spatial import KDTree
import tqdm
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import warnings
import os
import sys
import gc
from multiprocessing import Pool
from multiprocessing import shared_memory
import time
import psutil
import threading
from itertools import repeat
import multiprocessing
warnings.filterwarnings("ignore", category=FutureWarning)

### Node Class ###

class node:
    '''
    Substructure of the LSBoost class. Stores instructions for predicting on an instance within a level set at a given point in time for the LS structure.
    '''
    def __init__(self, weak_learners, learning_rate = 1, means = None, right_child=None, node_name='node'):
        # level set weak learner bucket
        self.weak_learners = weak_learners
        
        # level set information (time dependent)
        self.learning_rate = learning_rate

        # level set replacement with mean information vice centers
        self.means = means

        # node relation
        self.right_child = right_child
        self.node_name = node_name

class snapshotNode:
    '''
    Substructure of the LSBoost class. Informs data structure when a round is beginning in order to create level sets by a 'snapshot' of the current predictions
    '''
    def __init__(self, learning_rate = 1, right_child = None):
        # node learning rate
        self.learning_rate = learning_rate

        # node location relation
        self.right_child = right_child

        # update indicator
        self.node_name = 'snapshotNode'

class headNode:
    '''
    Substructure of the LSBoost class. Contains the initial model and is the 'head' (or start) of the model
    '''
    def __init__(self, initial_model):
        self.initial_model = initial_model

class LSBoostingRegressor:
    '''
     
    '''
    def __init__(self, T = 100, num_bins = 10, min_group_size=1, global_gamma=.1, weak_learner = LinearRegression(), min_val=0, max_val=1, bin_type = 'default', initial_model = None, train_predictions = [], val_predictions = [], head_node_name ='head_node',  learning_rate = 1, center_mean = False, final_round = False,  n_jobs = 1):
        
        # multiprocessing for MacOS
        if sys.platform == "darwin":
            multiprocessing.set_start_method("fork", force=True)
        ### Hyperparameters ###

        # number of rounds
        self.T = T

        # number of level sets
        self.num_bins = num_bins

        # minimum number of samples needed in level set to train new predictor
        self.min_group_size = min_group_size

        # scaling factor for required error decrease per round / required decrease in MSE over level set for validation updates
        self.global_gamma = global_gamma

        # weak learner class
        self.weak_learner = weak_learner

        # minimum value in labels
        self.min_val = min_val

        # maximum value in labels
        self.max_val = max_val

        # type of level set creation. Either 'default' for equal width or 'distribution' for equal number of instances per level set 
        self.bin_type = bin_type

        # base model to boost. If None, begins from best weak learner over all data.
        self.initial_model = initial_model

        # allows warm start if hyperparameter tuning with a base model
        self.train_predictions = copy.deepcopy(train_predictions)

        # allows warm start if hyperparameter tuning with a base model with validation
        self.val_predictions = copy.deepcopy(val_predictions)

        # weighted average amount for new weak learner with previous level sets predictions
        self.learning_rate = learning_rate

        # set to true for rounding to training mean in each level set rather than mean for 0 calibration error.
        self.center_mean = center_mean

        # round final predictions, set to False to get output from final weak learners
        self.final_round = final_round

        # number of cores to use when training and predicting
        if n_jobs == -1:
            self.n_jobs = os.cpu_count()
        elif n_jobs > 0:
            self.n_jobs = n_jobs
        else:
            print('That is not a valid cpu count, defaulting to 1')
            self.n_jobs = 1

        ## Structural components
        self.head_node = None
        self.tail_node = None
        self.current_node = None
        self.node_list = []
        self.head_node_name = head_node_name
        
        ## Information tracking 
        self.updates = 1
        self.round_count = 0
        self.train_list = []
        self.val_list = []
        self.times = [0]
        self.preds_history = []
        
        ## Global values
        global GLOBAL_global_gamma
        global GLOBAL_min_group_size
        GLOBAL_global_gamma = copy.deepcopy(self.global_gamma)
        GLOBAL_min_group_size = copy.deepcopy(self.min_group_size)
        
    def append_node(self, new_node):
        '''
        Appends new node to end of data structure (tail node) and sets new tail node as new node
        '''
        self.tail_node.right_child = new_node
        self.tail_node = new_node
        self.current_node = self.tail_node
        self.node_list.append(new_node.node_name)

    def center_means(self, predictions, y, inference = False):
        '''
        Calibration function which occurs after each round. Sets predictions to mean of level set. In inference setting, uses historical means.
        '''
        # extend preds to 2-d on same y value
        preds_2d = np.stack((predictions, np.ones(len(predictions))), axis = 1)

        # use kd search algorithm to get closest centers
        closest_center_indices = self.kd_tree.query(preds_2d)[1]

        # if training
        if inference == False:
            # compute means or append center if no points in level set
            means_list = []
            for i in range(len(self.discretized_points)):
                if i in closest_center_indices:
                    means_list.append((y[closest_center_indices == i]).mean())
                else:
                    means_list.append(self.discretized_points[i])
            means_array = np.array(means_list)
        # if predicting
        else: 
            # replace with historical means
            means_array = y

        # return calibrated predictions. return level set includion to reduce computations
        return np.take(means_array, closest_center_indices), closest_center_indices, means_array


    def center_points(self, predictions, return_center = True):
        '''
        Calibration function which occurs after each round. Sets predictions to center of level set.
        '''
        # extend preds to 2-d on same y value
        preds_2d = np.stack((predictions, np.ones(len(predictions))), axis = 1)

        # use kd search algorithm to get closest centers
        closest_center_indices = self.kd_tree.query(preds_2d)[1]
        
        # return closest center value for each prediction
        if return_center == False:
            return np.take(self.discretized_points, closest_center_indices)
        else:
            # optional return level set includion to reduce computations
            return np.take(self.discretized_points, closest_center_indices), closest_center_indices

    def update(self):
        '''
        Fixes level set inclusion, fits weak learners over each level set, stores weak learners, updates predictions.
        '''

        # start time for research computations
        start_time = time.time()

        # create snapshot at start of round
        self.snapshot_train = copy.deepcopy(self.train_predictions)
        
        # save copy of centers and send to shared memory
        closest_centers_train = copy.deepcopy(self.closest_centers_train)
        create_shared_memory_nparray(closest_centers_train, f'closest_centers_train_{self.round_count}', True)

        # append snapshot node to data structure
        self.append_node(snapshotNode(learning_rate = self.learning_rate))

        # copy job count
        cores = copy.deepcopy(self.n_jobs)

        # find level sets which will be update
        occupied_level_sets = np.unique(closest_centers_train)   

        # instantiate pool process
        executor = Pool(cores)

        # fit weak learners in parallel over non-empty level sets
        result = executor.starmap(worker, zip(occupied_level_sets, repeat(self.round_count)))
        
        # wait for all computations to terminate and close executor
        executor.close()
        executor.join() 

        # delete entries which correspond to small level sets
        #TODO DELETE
        # result = [x for x in result if x is not None]
        result = filter(None, result)

        # list to store weak learner updates
        weak_learner_bucket = []
        
        # iterate through updates
        for update in result:
            
            # increase weak learner count
            self.updates += 1
            
            # level set update information
            index = update[0]
            hypothesis_preds_train = update[1]
            locals()['clf'+str(self.updates)] = copy.deepcopy(update[2])
            indices_train = update[3]

            # add weak learner to round bucket
            weak_learner_bucket.append((index, locals()['clf'+str(self.updates)]))

            # replace current predictions with weighted predictions from weak learner update
            np.put(self.train_predictions, np.where(indices_train == 1), ((1 - self.current_node.learning_rate) * self.train_predictions[indices_train]) + (self.current_node.learning_rate * hypothesis_preds_train))
            
        # check if termination round and if final update needs to not be rounded
        if self.flag == True:
            # add flag to structure indicating termination round with no rounding
            self.append_node(node(None, learning_rate = self.learning_rate, node_name = 'flag'))
            means_array = None
        else:
            # center predictions to mean of level set
            if self.center_mean == True:
                self.train_predictions, self.closest_centers_train, means_array = self.center_means(self.train_predictions, self.y_train, inference = False)
            # center predictions to center of level set
            else:
                means_array = None
                self.train_predictions, self.closest_centers_train = self.center_points(self.train_predictions)
        
        # collect garbage
        gc.collect()

        # append histrocial weak learners + means to data structure 
        self.append_node(node(weak_learner_bucket, learning_rate = self.learning_rate, means = means_array, right_child=None, node_name='node'))

        # store MSE for research purposes
        self.train_list.append(MSE(self.y_train, self.train_predictions))

        # release shared memory
        release_shared(f'closest_centers_train_{self.round_count}')

        # update round count
        self.round_count += 1
        
        # store time data for research purposes
        end_time = time.time()
        self.times.append(end_time - start_time)
        self.updates_history.append(self.updates)
        
        return True

    def update_validation(self):
        '''
        Fixes level set inclusion, fits weak learners over each level set, stores weak learners, updates predictions. Requires weak learners do factor of global_gamma better to create level set update
        '''

        # create snapshot at start of round
        self.snapshot_train = copy.deepcopy(self.train_predictions)
        self.snapshot_val = copy.deepcopy(self.val_predictions)
        
        # save copy of centers and send to shared memory
        closest_centers_train = copy.deepcopy(self.closest_centers_train)
        closest_centers_val = copy.deepcopy(self.closest_centers_val)
        create_shared_memory_nparray(closest_centers_train, f'closest_centers_train_{self.round_count}', True)
        create_shared_memory_nparray(closest_centers_val, f'closest_centers_val_{self.round_count}', True)
        
        # create snapshot for shared memory
        validation_predictions = copy.deepcopy(self.val_predictions)
        create_shared_memory_nparray(validation_predictions, f'validation_predictions_{self.round_count}', True)

        # append snapshot node to data structure
        self.append_node(snapshotNode(learning_rate = self.learning_rate))

    
        # copy job count        
        cores = copy.deepcopy(self.n_jobs)

    
        # find level sets which will be update
        occupied_level_sets = np.unique(closest_centers_train) 

        # instantiate pool process
        executor = Pool(cores)

        # fit weak learners in parallel over non-empty level sets
        result = executor.starmap(worker_validation, zip(occupied_level_sets, repeat(self.round_count)))
        
        # wait for all computations to terminate and close executor
        executor.close()
        executor.join()
       
        # delete entries which correspond to small level sets
        result = filter(None, result)

        # list to store weak learner updates
        weak_learner_bucket = []

        # iterate through update
        for update in result:
            # increase weak learner count
            self.updates += 1

            # level set update information
            index = update[0]
            hypothesis_preds_train = update[1]
            hypothesis_preds_val = update[2]
            locals()['clf'+str(self.updates)] = copy.deepcopy(update[3])
            indices_train = update[4]
            indices_val = update[5]

            # add weak learner to round bucket
            weak_learner_bucket.append((index, locals()['clf'+str(self.updates)]))
            
            # replace current predictions with weighted predictions from weak learner update
            np.put(self.train_predictions, np.where(indices_train == 1), ((1 - self.current_node.learning_rate) * self.train_predictions[indices_train]) + (self.current_node.learning_rate * hypothesis_preds_train))
            np.put(self.val_predictions, np.where(indices_val == 1), ((1 - self.current_node.learning_rate) * self.val_predictions[indices_val]) + (self.current_node.learning_rate * hypothesis_preds_val))
            
            continue

        # check if termination round and if final update needs to not be rounded
        if self.flag == True:
            # add flag to structure indicating termination round with no rounding
            self.append_node(node(None, learning_rate = self.learning_rate, node_name = 'flag'))
            means_array = None
        else:
            
            if self.center_mean == True:   
                # center predictions to mean of level set
                self.train_predictions, self.closest_centers_train, means_array = self.center_means(self.train_predictions, self.y_train, inference = False)
                self.val_predictions, self.closest_centers_val, _ = self.center_means(self.val_predictions, means_array, inference = True)
            else:
                # center predictions to center of level set
                means_array = None
                self.train_predictions, self.closest_centers_train = self.center_points(self.train_predictions)
                self.val_predictions, self.closest_centers_val = self.center_points(self.val_predictions)
            
        # collect garbage
        gc.collect()

        # append histrocial weak learners + means to data structure 
        self.append_node(node(weak_learner_bucket, learning_rate = self.learning_rate, means = means_array, right_child=None, node_name='node'))

        # store MSE for research purposes
        self.train_list.append(MSE(self.y_train, self.train_predictions))
        self.val_list.append(MSE(self.y_val, self.val_predictions))

        # release shared memory
        release_shared(f'closest_centers_train_{self.round_count}')
        release_shared(f'closest_centers_val_{self.round_count}')
        release_shared(f'validation_predictions_{self.round_count}')

        # update round count
        self.round_count += 1

        return True
    
    def predict(self, X):
        '''
        Inference pass of data structure on matrix X.
        '''

        # initialize counter
        counter = 0

        # set current node to start of data structure
        self.current_node = self.head_node

        # initialize predictions to initial model predictions
        predictions = self.head_node.initial_model(X)    

        # replace out-of-bounds predictions with min/max val
        np.put(predictions, np.where(predictions < self.min_val), self.min_val)
        np.put(predictions, np.where(predictions > self.max_val), self.max_val)
        

        # center points
        if self.center_mean == True:
            # center points with historical means
            predictions, closest_centers, _ = self.center_means(predictions, self.initial_means, inference = True)
        else:
            # center points with level set centers
            predictions, closest_centers = self.center_points(predictions)

        # if the head node contains a right child, walk to right child
        if self.current_node.right_child != None:
            self.current_node = self.current_node.right_child
        else:
            return predictions

        # set round flag
        flag = False

        # copy core counts
        cores = copy.deepcopy(self.n_jobs)

        # traverse data structure
        while True:
            # if at end of list, return predictions
            if self.current_node == None:
                return predictions

            # take a snapshot of predcition for level set inclusion
            if 'snapshotNode' == self.current_node.node_name:
                
                # walk to right child
                self.current_node = self.current_node.right_child

                # set flag condition
                if 'flag' == self.current_node.node_name:
                    flag = True
                    self.current_node = self.current_node.right_child

                # copy predictions for shared memory
                weak_learners_predictions = copy.deepcopy(predictions)

                # lock threads
                lock = threading.Lock()

                # parallel predict
                Parallel(n_jobs=cores, require="sharedmem")(delayed(update_predictions)(weak_learner[0], weak_learner[1].predict, X, closest_centers, weak_learners_predictions, lock) for weak_learner in self.current_node.weak_learners)
                
                # copy predictions and delete old instance to ensure parallel process closes
                new_predictions = copy.deepcopy(weak_learners_predictions)
                del weak_learners_predictions

                # update predictions with weighted average
                predictions = ((1 - self.current_node.learning_rate) * predictions) + (self.current_node.learning_rate * new_predictions)
                
                # flag implies no rounding and final update
                if flag == True:

                    # replace out-of-bounds predictions with min/max value
                    if (predictions < self.min_val).sum() != 0:
                        np.put(predictions, np.where(predictions < self.min_val), self.min_val)
                    if (predictions > self.max_val).sum() != 0:
                        np.put(predictions, np.where(predictions > self.max_val), self.max_val)
                    return predictions
                else:
                    counter += 1
                    
                    # center points
                    if self.center_mean == True:
                        # center points with historical mean
                        predictions, closest_centers, _ = self.center_means(predictions, self.current_node.means, inference = True)
                    else:
                        # center points with level set centers
                        predictions, closest_centers = self.center_points(predictions)
                    
                    # walk to right child
                    self.current_node = self.current_node.right_child
        
    def track(self, X, y, loss_fn):
        '''
        Compute loss_fn function on y and predictions at each round.
        '''

        # set current node to start of data structure
        self.current_node = self.head_node

        # initialize predictions to initial model predictions
        predictions = self.head_node.initial_model(X)   

        # record loss_fn
        error_list = [loss_fn(y, predictions)]

        # replace out-of-bounds predictions with min/max val
        np.put(predictions, np.where(predictions < self.min_val), self.min_val)
        np.put(predictions, np.where(predictions > self.max_val), self.max_val)

        # center points
        if self.center_mean == True:
            # center points with historical means
            predictions, closest_centers, _ = self.center_means(predictions,  self.initial_means, inference = True)
        else:
            #center points with level set centers
            predictions, closest_centers = self.center_points(predictions)

        # if the head node contains a right child, walk to right child
        if self.current_node.right_child != None:
            self.current_node = self.current_node.right_child
        else:
            return loss_fn(y, predictions)

        # copy core counts
        cores = copy.deepcopy(self.n_jobs)

        # set round flag
        flag = False

        # traverse data structure
        while True:
            # fill error list to contain T + 1 entries
            if self.current_node == None:
                if (self.T-len(error_list) + 1 > 0):
                    fill_list = np.full(self.T-len(error_list) + 1,error_list[-1]).tolist()
                    error_list += fill_list
                return error_list

            # snapshot implies start of round
            if 'snapshotNode' == self.current_node.node_name:
                
                # walk to right child
                self.current_node = self.current_node.right_child

                #set flag condition
                if 'flag' == self.current_node.node_name:
                    flag = True
                    self.current_node = self.current_node.right_child

                # copy predictions for shared memory
                weak_learners_predictions = copy.deepcopy(predictions)

                # lock threads
                lock = threading.Lock()

                # parallel predict
                Parallel(n_jobs=cores, require="sharedmem")(delayed(update_predictions)(weak_learner[0], weak_learner[1].predict, X, closest_centers, weak_learners_predictions, lock) for weak_learner in self.current_node.weak_learners)
                
                # copy predictions and delete old instance to ensure parallel process closes
                new_predictions = copy.deepcopy(weak_learners_predictions)
                del weak_learners_predictions

                # update predictions with weighted average
                predictions = ((1 - self.current_node.learning_rate) * predictions) + (self.current_node.learning_rate * new_predictions)
                
                # flag implies no rounding and final update
                if flag == True:
                    # replace out-of-bounds predictions with min/max value
                    if (predictions < self.min_val).sum() != 0:
                        np.put(predictions, np.where(predictions < self.min_val), self.min_val)
                    if (predictions > self.max_val).sum() != 0:
                        np.put(predictions, np.where(predictions > self.max_val), self.max_val)

                    # record error
                    error_list.append(loss_fn(y, predictions))
                    
                    # fill error list to contain T + 1 entries
                    if self.T-len(error_list) + 1 > 0:
                        fill_list = np.full(self.T-len(error_list) + 1,error_list[-1]).tolist()
                        error_list += fill_list
                    return error_list
                else:
                    # center points
                    if self.center_mean == True:
                        # center points with historical mean
                        predictions, closest_centers, _ = self.center_means(predictions, self.current_node.means, inference = True)
                    else:
                        # center points with level set centers
                        predictions, closest_centers = self.center_points(predictions)
                    
                    # record error
                    error_list.append(loss_fn(y, predictions))

                    # walk to right child
                    self.current_node = self.current_node.right_child

    def fit(self, x_train, y_train):
        '''
        Outer method for LSBoost to fit training data.
        '''
        try:
            # store data for other methods
            self.x_train = x_train
            self.y_train = y_train

            # set min max val by extremes of training data
            self.min_val = y_train.min()
            self.max_val = y_train.max()

            # initialize shared data list
            globals()[f'shms'] = []
            
            # share training data
            create_shared_memory_nparray(x_train,'x_train', True)
            create_shared_memory_nparray(y_train,'y_train', True)
           
            # initialize shared global values from hyperparameters
            global GLOBAL_wk_learner
            global GLOBALS_num_lvl_sets
            GLOBAL_wk_learner = copy.deepcopy(self.weak_learner)
            GLOBALS_num_lvl_sets = copy.deepcopy(self.num_bins)

            # set initializations
            if self.initial_model == None:
                initial_model = copy.deepcopy(self.weak_learner).fit(self.x_train,self.y_train).predict
            else:
                initial_model = self.initial_model.predict

            # optionality for defining initial predictions. Used when running experiments and initial model is ensemble which takes long time to make predictions
            if len(self.train_predictions) == 0:
                self.train_predictions = initial_model(self.x_train)

            # calculate error of your initial model
            self.train_list = [MSE(self.y_train, self.train_predictions)]

            # replace out-of-bounds predictions with min/max val
            np.put(self.train_predictions, np.where(self.train_predictions < self.min_val), self.min_val)
            np.put(self.train_predictions, np.where(self.train_predictions > self.max_val), self.max_val)
            
            # initialize structure
            self.head_node = headNode(initial_model)
            self.tail_node = self.head_node 
            self.current_node = self.head_node 
            self.node_list = [self.head_node_name]
        
            # create level set distributions based on style of level set separation
            if self.bin_type == 'distribution':
                # create level sets by equal instance counts
                self.discretized_endpoints = bin_by_dist(self.y_train, num_bins=self.num_bins)
                self.discretized_points = center_points(self.discretized_endpoints)       
            else:
                # create level sets by equal distance
                self.discretized_endpoints = np.linspace(self.min_val, self.max_val, self.num_bins+1)
                self.discretized_points = self.discretized_endpoints
            
            # initialize kd_tree for finding closest center
            self.kd_tree = KDTree(np.stack((self.discretized_points, np.ones(len(self.discretized_points))), axis = 1))
            
            #initialize round and update counters
            self.round_count = 0
            self.updates = 0

            #define iterator for progress bar
            iterator = tqdm(range(self.T), bar_format = '{l_bar}{bar} {n_fmt}/{total_fmt} {remaining}{postfix}', desc = 'LS', colour = 'green', leave = True, postfix={'Training Error':'{:.7f}'.format(self.train_list[-1])})
            
            #center points
            if self.center_mean == True:
                # center level set with mean
                self.train_predictions, self.closest_centers_train, means_array = self.center_means(self.train_predictions, self.y_train, inference = False)
                self.initial_means = means_array
            else:
                # center level set with center
                self.train_predictions, self.closest_centers_train = self.center_points(self.train_predictions)

            # set flag condition
            self.flag = False

            # store updates per round
            self.updates_history = [0]

            # run for T rounds or stopping conditions
            for i in iterator:
                
                # store history of predictions for research purposes
                self.preds_history.append(copy.deepcopy(self.train_predictions))
                
                # set iterator values
                iterator.set_postfix({'Training Error':'{:.7f}'.format(self.train_list[-1])})
                
                # final round updates
                if i == (self.T -1):
                    if self.final_round == False:
                        self.flag = True
                    self.update()   
                
                # rounds 1 and 2 updates
                elif ((i == 0) or (i == 1)):
                    self.update()
                    continue
                
                # stopping criteria
                elif (self.train_list[-2] - self.train_list[-1] <= self.global_gamma/(self.num_bins)):
                    if self.final_round == False:
                        self.flag = True
                        self.update()
                    print(f'Early Termination at round: {self.round_count}')
                    iterator.set_postfix({'Training Error':'{:.7f}'.format(self.train_list[-1])})
                    return

                # all other rounds
                else:
                    self.update()

                # update iterator
                iterator.set_postfix({'Training Error':'{:.7f}'.format(self.train_list[-1])})

        # release all stored memory if early termination from outside errors
        except KeyboardInterrupt:
            while len(globals()['shms']) != 0:
                release_shared(globals()["shms"][0])
        else:
            while len(globals()['shms']) != 0:
                release_shared(globals()["shms"][0])
        finally:
            while len(globals()['shms']) != 0:
                release_shared(globals()["shms"][0])

            if len(globals()['shms']) == 0:
                print('Memory released!')
            else:
                print('Memory could not be released in time, please restart kernel')

        return 

    def fit_validation(self, x_train, y_train, x_val, y_val):
        '''
        Outer method for LSBoost to fit training data with validating updates
        '''
        try:
            # store data for other methods
            self.x_train = x_train
            self.y_train = y_train
            self.x_val = x_val
            self.y_val = y_val

            # set min max val by extremes of training data
            self.min_val = y_train.min()
            self.max_val = y_train.max()

            # initialize shared data list
            globals()[f'shms'] = []
      
            # share training/validation data
            create_shared_memory_nparray(x_train,'x_train', True)
            create_shared_memory_nparray(y_train,'y_train', True)
            create_shared_memory_nparray(x_val,'x_val', True)
            create_shared_memory_nparray(y_val,'y_val', True)
           
          # initialize shared global values from hyperparameters
            global GLOBAL_wk_learner
            global GLOBALS_num_lvl_sets
            GLOBAL_wk_learner = copy.deepcopy(self.weak_learner)
            GLOBALS_num_lvl_sets = copy.deepcopy(self.num_bins)

            # store total_labels
            total_labels = np.append(y_train, y_val)
            
            # set initializations
            if self.initial_model == None:
                initial_model = copy.deepcopy(self.weak_learner).fit(self.x_train,self.y_train).predict
            else:
                initial_model = self.initial_model.predict

            # optionality for defining initial predictions. Used when running experiments and initial model is ensemble which takes long time to make predictions
            if len(self.train_predictions) == 0:
                self.train_predictions = initial_model(self.x_train)
                self.val_predictions = initial_model(self.x_val)

            # calculate error of your initial model
            self.train_list = [MSE(self.y_train, self.train_predictions)]
            self.val_list   = [MSE(self.y_val, self.val_predictions)]

            # replace out-of-bounds predictions with min/max val
            np.put(self.train_predictions, np.where(self.train_predictions < self.min_val), self.min_val)
            np.put(self.train_predictions, np.where(self.train_predictions > self.max_val), self.max_val)
            np.put(self.val_predictions, np.where(self.val_predictions < self.min_val), self.min_val)
            np.put(self.val_predictions, np.where(self.val_predictions > self.max_val), self.max_val)

            # initialize structure
            self.head_node = headNode(initial_model)
            self.tail_node = self.head_node 
            self.current_node = self.head_node 
            self.node_list = [self.head_node_name]
        
            # create bin distributions based on style of level set separation
            if self.bin_type == 'distribution':
                # create level sets by equal instance counts
                self.discretized_endpoints = bin_by_dist(total_labels, num_bins=self.num_bins)
                self.discretized_points = center_points(self.discretized_endpoints)       
            else:
                # create level sets by equal distance
                self.discretized_endpoints = np.linspace(self.min_val, self.max_val, self.num_bins+1)
                self.discretized_points = self.discretized_endpoints


            # initialize kd_tree for finding closest center
            self.kd_tree = KDTree(np.stack((self.discretized_points, np.ones(len(self.discretized_points))), axis = 1))

           
            # initialize round and update counters
            self.round_count = 0
            self.updates = 0

            # run for T rounds, iterator will keep track of number of rounds
            lowest_val_error = self.val_list[0]

            # define iterator for progress bar
            iterator = tqdm(range(self.T), bar_format = '{l_bar}{bar} {n_fmt}/{total_fmt} {remaining}{postfix}', desc = 'LS', colour = 'green', leave = True, postfix={'Training Error':'{:.7f}'.format(self.train_list[-1]), 'Val Error':'{:.7f}'.format(self.val_list[-1]), 'Lowest Val Error': '{:.7f}'.format(lowest_val_error)})
            
            # center points
            if self.center_mean == True:
                # center level set with mean
                self.train_predictions, self.closest_centers_train, means_array = self.center_means(self.train_predictions, self.y_train, inference = False)
                self.val_predictions, self.closest_centers_val, _ = self.center_means(self.val_predictions, means_array, inference = True)
                self.initial_means = means_array
            else:
                # center level set with center
                self.train_predictions, self.closest_centers_train = self.center_points(self.train_predictions)
                self.val_predictions, self.closest_centers_val = self.center_points(self.val_predictions)
            
            # set flag condition
            self.flag = False

            # run for T rounds or stopping conditions
            for i in iterator:

                # store history of predictions for research purposes
                self.preds_history.append(copy.deepcopy(self.train_predictions))

                # update lowest validation error
                if lowest_val_error > self.val_list[-1]:
                    lowest_val_error = self.val_list[-1]

                # set iterator values
                iterator.set_postfix({'Training Error':'{:.7f}'.format(self.train_list[-1]), 'Val Error':'{:.7f}'.format(self.val_list[-1]), 'Lowest Val Error': '{:.7f}'.format(lowest_val_error)})
                
                # final round updates
                if i == (self.T -1):
                    if self.final_round == False:
                        self.flag = True
                    self.update_validation()   

                # rounds 1 and 2 updates   
                elif ((i == 0) or (i == 1)):
                    self.update_validation()
                    continue

                # stopping criteria
                elif (self.train_list[-2] - self.train_list[-1] <= self.global_gamma/(self.num_bins)):
                    if self.final_round == False:
                        self.flag = True
                        self.update_validation()
                    print(f'Early Termination at round: {self.round_count}')
                    iterator.set_postfix({'Training Error':'{:.7f}'.format(self.train_list[-1]), 'Val Error':'{:.7f}'.format(self.val_list[-1]), 'Lowest Val Error': '{:.7f}'.format(lowest_val_error)})
                    return
                # all other rounds
                else:
                    self.update_validation()

                # update iterator
                iterator.set_postfix({'Training Error':'{:.7f}'.format(self.train_list[-1]), 'Val Error':'{:.7f}'.format(self.val_list[-1]), 'Lowest Val Error': '{:.7f}'.format(lowest_val_error)})
        
        # release all stored memory if early termination from outside errors
        except KeyboardInterrupt:
            while len(globals()['shms']) != 0:
                release_shared(globals()["shms"][0])
        else:
            while len(globals()['shms']) != 0:
                release_shared(globals()["shms"][0])
        finally:
            while len(globals()['shms']) != 0:
                release_shared(globals()["shms"][0])

            if len(globals()['shms']) == 0:
                print('Memory released!')
            else:
                print('Memory could not be released in time, please restart kernel')
        return 


### END LSBoost CLASS ###

### LSBoost PARALLEL FUNCTIONS ###

def create_shared_memory_nparray(data, name, create = False):
    '''
    Creates array in shared memory for parallel processing.
    '''
    d_size = np.dtype(np.float64).itemsize * np.prod(data.shape)
    try:
        shm = shared_memory.SharedMemory(create=create, size=d_size, name=f'{name}_buf')
    except:
        shm = shared_memory.SharedMemory(create=False, size=d_size, name=f'{name}_buf')
        return shm
    
    dst = np.ndarray(shape=data.shape, dtype=np.float64, buffer=shm.buf)
    dst[:] = data[:]
    globals()[f'global_{name}_shm'] = shm
    globals()[f'shms'].append(name)
    globals()[f'global_{name}_shape'] = data.shape
    return shm

def release_shared(name):
    '''
    Releases named array in shared memory.
    '''
    shm = shared_memory.SharedMemory(name=f'{name}_buf')
    shm.close()
    shm.unlink() 
    globals()[f'shms'].remove(name)
    del globals()[f'global_{name}_shm']
    del globals()[f'global_{name}_shape'] 

def worker(index, counter):
    '''
    Parallel process worker. Fits weak learner on level set data, returns level set information
    '''
    # load shared memory array
    x_train_shared = np.ndarray(globals()['global_x_train_shape'], dtype = np.float64, buffer=globals()['global_x_train_shm'].buf)
    y_train_shared = np.ndarray(globals()['global_y_train_shape'], dtype = np.float64, buffer=globals()['global_y_train_shm'].buf)
    
    # load closest level set indices
    closest_centers_train = np.ndarray(globals()['global_y_train_shape'], dtype = np.float64, buffer = globals()[f'global_closest_centers_train_{counter}_shm'].buf) 
    
    # get level set indices for workers assigned level set
    indices_train = get_level_set_indices(index, closest_centers_train)

    # return none if level set is too small
    if indices_train.sum() <= GLOBAL_min_group_size:
        return None

    # copy weak learner from class 
    locals()['clf_intermediate'+str(index)] = copy.deepcopy(GLOBAL_wk_learner)
    
    # fit weak learner over level set data
    locals()['clf_intermediate'+str(index)].fit(x_train_shared[indices_train], y_train_shared[indices_train])
    
    # compute weak learner predictions on data
    hypothesis_preds_train = locals()['clf_intermediate'+str(index)].predict(x_train_shared[indices_train])

    # return update information
    return [index, hypothesis_preds_train, locals()['clf_intermediate'+str(index)], indices_train]

def worker_validation(index, counter):
    '''
    Parallel process worker. Fits weak learner on level set data, returns level set information based on if validation update allowed
    '''
    # load shared memory array
    x_train_shared = np.ndarray(globals()['global_x_train_shape'], dtype = np.float64, buffer=globals()['global_x_train_shm'].buf)
    y_train_shared = np.ndarray(globals()['global_y_train_shape'], dtype = np.float64, buffer=globals()['global_y_train_shm'].buf)
    x_val_shared = np.ndarray(globals()['global_x_val_shape'], dtype = np.float64, buffer=globals()['global_x_val_shm'].buf)
    y_val_shared = np.ndarray(globals()['global_y_val_shape'], dtype = np.float64, buffer=globals()['global_y_val_shm'].buf)
    
    # load validation predictions of current model
    model_ls_predictions_val = np.ndarray(globals()[f'global_validation_predictions_{counter}_shape'], dtype = np.float64, buffer = globals()[f'global_validation_predictions_{counter}_shm'].buf)
    
    # load closest level set indices
    closest_centers_train = np.ndarray(globals()['global_y_train_shape'], dtype = np.float64, buffer = globals()[f'global_closest_centers_train_{counter}_shm'].buf) 
    closest_centers_val = np.ndarray(globals()['global_y_val_shape'], dtype = np.float64, buffer = globals()[f'global_closest_centers_val_{counter}_shm'].buf) #GLOBALS_val_predictions_shm.buf)
    
    # get level set indices for workers assigned level set
    indices_train = get_level_set_indices(index, closest_centers_train)
    indices_val = get_level_set_indices(index, closest_centers_val)

    # return none if level set is too small
    if indices_train.sum() <= GLOBAL_min_group_size:
        return None

    if indices_val.sum() <= GLOBAL_min_group_size:
        return None
    # copy weak learner from class 
    locals()['clf_intermediate'+str(index)] = copy.deepcopy(GLOBAL_wk_learner)
    
    # fit weak learner over level set data
    locals()['clf_intermediate'+str(index)].fit(x_train_shared[indices_train], y_train_shared[indices_train])

    # compute weak learner predictions on data
    hypothesis_preds_val = locals()['clf_intermediate'+str(index)].predict(x_val_shared[indices_val])
    hypothesis_preds_train = locals()['clf_intermediate'+str(index)].predict(x_train_shared[indices_train])
    
    # compute information to determine whether or not to accept update
    ls_hypothesis_error_val = MSE(y_val_shared[indices_val], hypothesis_preds_val)
    ls_model_error_val = MSE(y_val_shared[indices_val], model_ls_predictions_val[indices_val])
    update = (ls_hypothesis_error_val < (1-GLOBAL_global_gamma)*ls_model_error_val)

    # return update information
    if update == False:
        # update denied
        return None

    # update acecepted, return data
    return [index, hypothesis_preds_train, hypothesis_preds_val, locals()['clf_intermediate'+str(index)], indices_train, indices_val]

def update_predictions(index, predict, X,  closest_centers, new_predictions, lock):
    '''
    Parallel process prediction updates in the inference pass
    '''
    # lock threads
    with lock:

        # get boolean level set indices
        indices = get_level_set_indices(index, closest_centers)

        # return none if level set too small
        if indices.sum() == 0:
            return
        
        # compute weak learner predictions
        wk_learner_predictions = predict(X[indices])
        
        # replace level set predictions with weak learner predictions
        np.put(new_predictions, np.where(indices == True), wk_learner_predictions)


### LSBoost FUNCTIONS ### 

def bin_by_dist(y, num_bins):
    '''
    Produce level sets in the interval [0,1] with roughly equivalent number of training labels in each set.
    
    Inputs: 
        - y: training labels (array)
        - num_bins: number of level sets (int)
    Outputs:
        - discretized_endpoints: level set endpoints (array)
    '''
    #set lnumber of points
    nlen = len(y)

    #create equal bins
    discretized_endpoints = np.interp(np.linspace(y.min(), nlen, num_bins + 1), np.arange(nlen), np.sort(y))

    #set endpoints to be 0,1 to fashion [0,1] interval
    discretized_endpoints[-1] = y.max() + 1e-8
    discretized_endpoints[0] = y.min()

    #return level set endpoints
    return discretized_endpoints 

def get_closest_center(predictions, centers_list):
    '''
    returns closest level set (deprecated; memory suck with large level set count)
    '''
    return np.argmin(np.abs(predictions[:, None] - centers_list), axis = 1)

def center_points(discretized_endpoints):
    '''
    Create center points of level sets given the endpoints of each level set

    Inputs:
        - discretized_endpoints: endpoints of level sets (array)
    Outputs:
        - discretized_centers: center centers in each level set (array)
    '''
    #initialize centers set
    discretized_centers = []

    #iterate through endpoints and find halfway point between current endpoint and next endpoint
    for i in range(0,len(discretized_endpoints)-1):
        discretized_centers.append((discretized_endpoints[i] + discretized_endpoints[i + 1]) / 2)

    #return center points
    return discretized_centers

def get_level_set_indices(index, closest_centers):
    '''
    Returns boolean true false series for closest level set
    '''
    return (closest_centers == index).astype('bool')