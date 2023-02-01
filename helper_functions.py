#!/usr/bin/python3

### IMPORTS ###
import pandas as pd
import numpy as np
from multiprocessing import Pool
import seaborn as sns
import matplotlib.pyplot as plt
import folktables
from folktables import ACSDataSource
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
 
### Helper Functions ###

def get_data(task = 'ACSIncome', target = 0, states_list = ['CA'], year = '2018', cutoff = 1e9, scaler=None, distribution_shift_states = [], random_state = 42):
    '''
    Loads in folktables Census data.
    '''
    ACSIncome = folktables.BasicProblem(
    features=['AGEP','COW','SCHL','MAR','OCCP','POBP','RELP','WKHP','SEX','RAC1P','PINCP',
    ],
    target=['AGEP','COW','SCHL','MAR','OCCP','POBP','RELP','WKHP','SEX','RAC1P','PINCP',
    ][target-1],
    preprocess=folktables.adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1))

    ACSTravelTime = folktables.BasicProblem(
    features=['AGEP','SCHL','MAR','SEX','DIS','ESP','MIG','RELP','RAC1P','PUMA','ST','CIT','OCCP','JWTR','POWPUMA','POVPIP',"JWMNP",
    ],
    target=['AGEP','SCHL','MAR','SEX','DIS','ESP','MIG','RELP','RAC1P','PUMA','ST','CIT','OCCP','JWTR','POWPUMA','POVPIP',"JWMNP",
    ][target-1],
    preprocess=travel_time_filter,
    postprocess=lambda x: np.nan_to_num(x, -1))

    ACSIncomePovertyRatio = folktables.BasicProblem(
    features=['AGEP','SCHL','MAR','SEX','DIS','ESP','MIG','CIT','MIL','ANC','NATIVITY','RELP','DEAR','DEYE','DREM','RAC1P','GCL','ESR','OCCP','WKHP','POVPIP',
    ],
    target=['AGEP','SCHL','MAR','SEX','DIS','ESP','MIG','CIT','MIL','ANC','NATIVITY','RELP','DEAR','DEYE','DREM','RAC1P','GCL','ESR','OCCP','WKHP','POVPIP',
    ][target-1],
    preprocess=lambda x: x,
    postprocess=lambda x: np.nan_to_num(x, -1))

    # fetch data
    data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=states_list, download=True)
    data, labels, _ = locals()[task].df_to_numpy(acs_data)
    data = np.delete(data, target-1, axis = 1)
    
    # drop NaN
    indices_to_drop = np.isnan(labels)
    data = data[~indices_to_drop]
    labels = labels[~indices_to_drop]
    
    #cutoff for income
    if cutoff == None:
        cutoff = np.max(labels)
    indices_to_keep = labels <= cutoff
    labels = labels[indices_to_keep]
    data = data[indices_to_keep]

    #normalize labels 0,1
    labels = (labels - labels.min()) / (labels.max() - labels.min())
    return data, labels

def travel_time_filter(data):
        """
        Filters for the travel time prediction task
        """
        df = data
        df = df[df['AGEP'] > 16]
        df = df[df['PWGTP'] >= 1]
        df = df[df['ESR'] == 1]
        return df

def print_data_report(x_train, x_test, y_train, y_test):
    '''
    Prints information from the dataset
    '''
    min_val = min(y_train.min(),y_test.min())
    max_val = max(y_train.max(),y_test.max())
    print(f'Training data size: {len(x_train)}')
    print(f'Test data size: {len(x_test)}')
    print(f'Max y value: {max_val}')
    print(f'Min y value: {min_val}')
    print(f'Mean train label: {y_train.mean()}')
    print(f'Mean test label: {y_test.mean()}')

    print(f'Std train label: {y_train.std()}')
    print(f'Std test label: {y_test.std()}')
    plt.figure(figsize = (15,8))
    plt.hist(np.concatenate((y_train, y_test), axis = None), bins=300)
    plt.ylabel('# Instances')
    plt.xlabel('Label Value')
    plt.title('Label Distribution')
    

def test_train_graph(errors_dict, title='Blank', ref_line = None, two_tone = False, dotted = 'test', y_axis = 'MSE', path = ''): 
    '''
    Function which plots errors from dictionary with legend keys from dictionary
    '''
    random_set = list(errors_dict.values())[0] 
    x = np.linspace(0,len(random_set)-1,len(random_set))
    errors_dict['axis'] = x
    graph_df = pd.DataFrame.from_dict(errors_dict)
    if two_tone == True:
        ### SET GRAPHS LABELS HERE
        xaxis = 'Number of rounds'
        yaxis = y_axis

        plt.figure(figsize = (15,10))
        sns.set_style("whitegrid")
        num_colors = int((len(errors_dict) - 1)/2 + 1)
        color_palette = sns.color_palette("Set1",num_colors)

        index = 0
        two_loop = 0
        legend_list = []
        for column in graph_df.columns:
            if column == 'axis':
                continue
            sns_plt = sns.lineplot(data=graph_df, x='axis', y=column, color=color_palette[index])
            two_loop += .5
            if two_loop%1 == 0:
                index += 1
            legend_list.append(column)
        
        sns_plt.set_title(title, fontsize = 15)
        sns_plt.set_xlabel(xaxis, fontsize = 13)
        sns_plt.set_ylabel(yaxis, fontsize = 13)
        sns_plt.grid(False)
        sns_plt.patch.set_edgecolor('black')
        sns_plt.patch.set_linewidth('1')
        legend = sns_plt.legend(loc = 'best', labels = legend_list)
        legend_lines = legend.get_lines()
        for i in range(len(legend_list)):
            if dotted in legend_list[i]:
                legend_lines[i].set_linestyle('--')
                sns_plt.lines[i].set_linestyle('--')

        two_loop = 0
        index = 0
        sns_plt.plot(ref_line, graph_df['train error (LS)'][ref_line], marker = "*", color=color_palette[index])
        sns_plt.plot(ref_line, graph_df['test error (LS)'][ref_line], marker = "*", color=color_palette[index])
    else:    
        graph_df_melted = graph_df.melt('axis',var_name = 'Errors Set', value_name = 'Error')

        ### SET GRAPHS LABELS HERE
        xaxis = 'Number of rounds'
        yaxis = y_axis

        plt.figure(figsize = (15,10))
        sns.set_style("whitegrid")
        num_colors = len(errors_dict) - 1
        color_palette = sns.color_palette("dark",num_colors)
        sns_plt = sns.lineplot(data=graph_df_melted, x='axis', y='Error', hue='Errors Set', palette=color_palette)
        sns_plt.set_title(title, fontsize = 15)
        sns_plt.set_xlabel(xaxis, fontsize = 13)
        sns_plt.set_ylabel(yaxis, fontsize = 13)
        sns_plt.grid(False)
        sns_plt.patch.set_edgecolor('black')
        sns_plt.patch.set_linewidth('1')
        legend_list = []
        for column in graph_df.columns:
            if column == 'axis':
                continue
            legend_list.append(column)
        legend = sns_plt.legend(labels = legend_list)
        legend_lines = legend.get_lines()
        for i in range(len(legend_list)):
            if dotted in legend_list[i]:
                legend_lines[i].set_linestyle('--')
                sns_plt.lines[i].set_linestyle('--')
        sns_plt.plot(ref_line, graph_df['train error'][ref_line], marker = "*", color=color_palette[0])
        sns_plt.plot(ref_line, graph_df['test error'][ref_line], marker = "*", color=color_palette[1])

def worker_msce(count, index):
    '''
    Parallel process worker to compute MSCE
    '''
    return count * ((GLOBAL_difference_array[GLOBAL_idx_start[index]:GLOBAL_idx_start[index+1]].mean()) **2)

def MSCE(y, predictions):
    '''
    Function to compute K_2(f, D), also known as mean squared calibration error (MSCE)
    '''
    idx_sort = np.argsort(predictions, kind='mergesort')
    sorted_predictions = predictions[idx_sort]
    sorted_y = y[idx_sort]
    difference_array = sorted_y - sorted_predictions
    _, idx_start, count = np.unique(sorted_predictions, return_counts=True, return_index=True)
    global GLOBAL_difference_array
    global GLOBAL_idx_start
    GLOBAL_difference_array = difference_array 
    GLOBAL_idx_start = idx_start
 
    executor = Pool(8)

    result = executor.starmap(worker_msce, zip(count, range(len(idx_start)-1)))

    executor.close()
    executor.join()

    return 1/len(y) * np.array(result).sum()
