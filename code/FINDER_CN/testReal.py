#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from FINDER import FINDER
import tensorflow as tf
import numpy as np #矩阵处理
import time
import pandas as pd #数据分析
import os






def GetSolution(STEPRATIO, MODEL_FILE_CKPT, num): #得到解决方案
    ######################################################################################################################
    ##................................................Get Solution (model).....................................................
    # global prob_value
    tf.reset_default_graph()
    dqn = FINDER()
    data_test_path = '../results/result_contrast_100/graphs_400_500/'
    # data_test_path = '../results/result_contrast/'
    # data_test_path = '../data/real/'
#     data_test_name = ['Crime','HI-II-14','Digg','Enron','Gnutella31','Epinions','Facebook','Youtube','Flickr']
#     data_test_name = ['Krebs']
    data_test_name = ['g_test_%d'%num]
    # data_test_name = ['g_test_30_50']

    model_file_path = './models/Model_erdos_renyi'
    model_file_ckpt = MODEL_FILE_CKPT
    model_file = model_file_path + '/'+model_file_ckpt
    ## save_dir
    # save_dir = '../results/FINDER_CN/real'
    save_dir = '../results/result_contrast_100/solutions_finder_ga_modify_400_500_4'
    # save_dir = '../results/result_contrast'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    ## begin computing...
    print ('The best model is :%s'%(model_file))
    dqn.LoadModel(model_file)
    df = pd.DataFrame(np.arange(1*len(data_test_name)).reshape((1,len(data_test_name))),index=['time'], columns=data_test_name)
    # df_prob = pd.DataFrame(np.arange(1*len(data_test_name)*int(num_edge)).reshape(len(data_test_name),-1),index=data_test_name)
    #################################### modify to choose which stepRatio to get the solution
    stepRatio = STEPRATIO
    num_edge = 0
    for j in range(len(data_test_name)):
        print ('\nTesting dataset %s'%data_test_name[j])
        data_test = data_test_path + data_test_name[j] + '.txt'
        # solution, time , num_edge_tem, prob_value = dqn.EvaluateRealData(model_file, data_test, save_dir, stepRatio)
        solution, time, num_edge_tem= dqn.EvaluateRealData(model_file, data_test, save_dir, stepRatio)
        num_edge = num_edge_tem
        df.iloc[0,j] = time
        print('Data:%s, time:%.2f'%(data_test_name[j], time))
    # df_prob = pd.DataFrame(np.arange(1 * len(data_test_name) * int(num_edge)).reshape(len(data_test_name), -1),index=data_test_name)
    # for j in range(num_edge):
    #     df_prob.iloc[0,j] = prob_value[j]

    save_dir_local = '../results/result_contrast_100/sol_times_finder_ga_modify_400_500_4'
    # save_dir_local = '../results/result_contrast'
    # save_dir_local = save_dir + '/StepRatio_%.4f' % stepRatio
    if not os.path.exists(save_dir_local):
        os.mkdir(save_dir_local)
    df.to_csv(save_dir_local + '/sol_time_%d.csv'%num, encoding='utf-8', index=False)
    # df.to_csv(save_dir_local + '/sol_time_end_30_50.csv', encoding='utf-8', index=False)
    # df_prob.to_csv(save_dir_local + '/sol_prob.csv', encoding='utf-8', index=False)


def GetSolution_contrast(STEPRATIO, MODEL_FILE_CKPT): #得到解决方案
    ######################################################################################################################
    ##................................................Get Solution (model).....................................................
    global prob_value
    dqn = FINDER()
    data_test_path = '../data/real/'
#     data_test_name = ['Crime','HI-II-14','Digg','Enron','Gnutella31','Epinions','Facebook','Youtube','Flickr']
    data_test_name = ['Krebs']


    model_file_path = './models/Model_barabasi_albert'
    model_file_ckpt = MODEL_FILE_CKPT
    model_file = model_file_path + '/'+model_file_ckpt
    ## save_dir
    save_dir = '../results/FINDER_CN/real'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    stepRatio = STEPRATIO
    save_dir_local = save_dir + '/StepRatio_%.4f/' % stepRatio
    data_prob_name = save_dir_local + 'sol_prob.csv'

    ## begin computing...
    print ('The best model is :%s'%(model_file))
    dqn.LoadModel(model_file)
    df = pd.DataFrame(np.arange(1*len(data_test_name)).reshape((1,len(data_test_name))),index=['time'], columns=data_test_name)
    # df_prob = pd.DataFrame(np.arange(1*len(data_test_name)*int(num_edge)).reshape(len(data_test_name),-1),index=data_test_name)
    #################################### modify to choose which stepRatio to get the solution
    # stepRatio = STEPRATIO
    num_edge = 0
    for j in range(len(data_test_name)):
        print ('\nTesting dataset %s'%data_test_name[j])
        data_test = data_test_path + data_test_name[j] + '.txt'
        solution, time , num_edge_tem= dqn.EvaluateRealData_inputprob(model_file, data_test, data_prob_name, save_dir, stepRatio)
        num_edge = num_edge_tem
        df.iloc[0,j] = time
        print('Data:%s, time:%.2f'%(data_test_name[j], time))




    if not os.path.exists(save_dir_local):
        os.mkdir(save_dir_local)
    df.to_csv(save_dir_local + '/sol_contrast_time.csv', encoding='utf-8', index=False)
    # df_prob.to_csv(save_dir_local + '/sol_prob.csv', encoding='utf-8', index=False)

def GetSolution_contrast_hda(STEPRATIO, MODEL_FILE_CKPT): #得到解决方案
    ######################################################################################################################
    ##................................................Get Solution (model).....................................................
    global prob_value
    dqn = FINDER()
    data_test_path = '../data/real/'
#     data_test_name = ['Crime','HI-II-14','Digg','Enron','Gnutella31','Epinions','Facebook','Youtube','Flickr']
    data_test_name = ['Krebs']


    model_file_path = './models/Model_barabasi_albert'
    model_file_ckpt = MODEL_FILE_CKPT
    model_file = model_file_path + '/'+model_file_ckpt
    ## save_dir
    save_dir = '../results/FINDER_CN/real'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    stepRatio = STEPRATIO
    save_dir_local = save_dir + '/StepRatio_%.4f/' % stepRatio
    data_prob_name = save_dir_local + 'sol_prob.csv'

    ## begin computing...
    print ('The best model is :%s'%(model_file))
    dqn.LoadModel(model_file)
    df = pd.DataFrame(np.arange(1*len(data_test_name)).reshape((1,len(data_test_name))),index=['time'], columns=data_test_name)
    # df_prob = pd.DataFrame(np.arange(1*len(data_test_name)*int(num_edge)).reshape(len(data_test_name),-1),index=data_test_name)
    #################################### modify to choose which stepRatio to get the solution
    # stepRatio = STEPRATIO
    num_edge = 0
    for j in range(len(data_test_name)):
        print ('\nTesting dataset %s'%data_test_name[j])
        data_test = data_test_path + data_test_name[j] + '.txt'
        solution, time , num_edge_tem= dqn.EvaluateRealData_inputprob_hda(model_file, data_test, data_prob_name, save_dir, stepRatio)
        num_edge = num_edge_tem
        df.iloc[0,j] = time
        print('Data:%s, time:%.2f'%(data_test_name[j], time))




    if not os.path.exists(save_dir_local):
        os.mkdir(save_dir_local)
    df.to_csv(save_dir_local + '/sol_contrast_hda_time.csv', encoding='utf-8', index=False)
    # df_prob.to_csv(save_dir_local + '/sol_prob.csv', encoding='utf-8', index=False)

def EvaluateSolution(STEPRATIO, MODEL_FILE_CKPT, STRTEGYID): #评估解决方案
    #######################################################################################################################
    ##................................................Evaluate Solution.....................................................
    dqn = FINDER()
    data_test_path = '../data/real/'
#     data_test_name = ['Crime', 'HI-II-14', 'Digg', 'Enron', 'Gnutella31', 'Epinions', 'Facebook', 'Youtube', 'Flickr']
    data_test_name = ['Crime']
    save_dir = '../results/FINDER_CN/real/StepRatio_%.4f/'%STEPRATIO
    ## begin computing...
    df = pd.DataFrame(np.arange(2 * len(data_test_name)).reshape((2, len(data_test_name))), index=['solution', 'time'], columns=data_test_name)
    for i in range(len(data_test_name)):
        print('\nEvaluating dataset %s' % data_test_name[i])
        data_test = data_test_path + data_test_name[i] + '.txt'
        solution = save_dir + data_test_name[i] + '.txt'
        t1 = time.time()
        # strategyID: 0:no insert; 1:count; 2:rank; 3:multiply
        ################################## modify to choose which strategy to evaluate
        strategyID = STRTEGYID
        score, MaxCCList = dqn.EvaluateSol(data_test, solution, strategyID, reInsertStep=0.001) #生成了两个txt文件 data test 和solution？？
        t2 = time.time()
        df.iloc[0, i] = score
        df.iloc[1, i] = t2 - t1
        result_file = save_dir + '/MaxCCList_Strategy_' + data_test_name[i] + '.txt'
        with open(result_file, 'w') as f_out:
            for j in range(len(MaxCCList)):
                f_out.write('%.8f\n' % MaxCCList[j])
        print('Data: %s, score:%.6f' % (data_test_name[i], score))
    df.to_csv(save_dir + '/solution_score.csv', encoding='utf-8', index=False)


def main():
    model_file_ckpt = 'nrange_5_10_iter_171900.ckpt'
    for i in range(16):
        GetSolution(0.01, model_file_ckpt, i)

    # GetSolution(0.01, model_file_ckpt)
    # GetSolution_contrast(0.01, model_file_ckpt) #相同的图结构 相同的激活概率 不同的激活判断 多次试验进行对比
    # GetSolution_contrast_hda(0.01, model_file_ckpt)
    # EvaluateSolution(0.01, model_file_ckpt, 0)



if __name__=="__main__":
    main()
