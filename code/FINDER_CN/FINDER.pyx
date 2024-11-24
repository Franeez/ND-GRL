#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 00:33:33 2017

@author: fanchangjun
"""

from __future__ import print_function, division #输出（变为一个函数） 和 精确除法（/），若还想用截断除法，则使用//   __future__即摒弃py2的用法用py3的。


import tensorflow as tf
# import tensorflow.compat.v1 as tf #无tf 1.x 虚拟环境下转换库的版本
import pandas as pd #加入为了读取文件数据 csv
import numpy as np
import networkx as nx
import random
import math
import time
import pickle as cp #序列化和反序列化 与数据存储有关？
import sys
from tqdm import tqdm
import PrepareBatchGraph  #等待pyd文件产生后 import
import graph
import nstep_replay_mem
import nstep_replay_mem_prioritized
import mvc_env
import utils
import scipy.linalg as linalg #线性代数库
import os

# Hyper Parameters:
cdef double GAMMA = 0.99  # decay rate of past observations 参数γ 强化学习迭代公式中？
cdef int UPDATE_TIME = 50  #更新时间 次数？？ 每几次更新?
cdef int EMBEDDING_SIZE = 64 #嵌入的维数？
cdef int MAX_ITERATION = 500000  #最大迭代次数 原来是500000
cdef double LEARNING_RATE = 0.05   #学习率  图神经网络？
cdef int MEMORY_SIZE = 250000 #内存块的大小 对应最大迭代数目
cdef double Alpha = 0.0001 ## weight of reconstruction loss 重建损失部分的参数
########################### hyperparameters for priority(start)#########################################
cdef double epsilon = 0.0000001  # small amount to avoid zero priority
cdef double alpha = 0.6  # [0~1] convert the importance of TD error to priority
cdef double beta = 0.4  # importance-sampling, from initial value increasing to 1
cdef double beta_increment_per_sampling = 0.001
cdef double TD_err_upper = 1.  # clipped abs error
########################## hyperparameters for priority(end)#########################################
# cdef int K_num = 5
cdef int N_STEP = 3 #5改为3 再改为5
cdef int NUM_MIN = 5
cdef int NUM_MAX = 10
cdef int REG_HIDDEN = 32 #？？
cdef int BATCH_SIZE = 64
cdef double initialization_stddev = 0.1  # 默认初始化标准差设置为0.01
cdef int n_valid = 200
cdef int aux_dim = 2 #修改后 原来是4
cdef int num_env = 1 #环境数，即一个DQN环境
cdef double inf = 2147483647/2
#########################  embedding method ##########################################################
cdef int max_bp_iter = 3 #？？ 最大迭代次数？？
cdef int aggregatorID = 0 #0:sum; 1:mean; 2:GCN  聚集函数选择
cdef int embeddingMethod = 1   #0:structure2vec; 1:graphsage  算法选择 下面条件判断

cdef double L2_normalization = 0.001


class FINDER:

    def __init__(self): #魔术方法，创建时自动增加成员变量（里面的属性为实例属性，对应（不同于） 类属性）
        # init some parameters
        self.weight_decay = L2_normalization
        self.embedding_size = EMBEDDING_SIZE
        self.learning_rate = LEARNING_RATE

        self.g_type = 'barabasi_albert' #erdos_renyi, powerlaw, small-world， barabasi_albert  图的类型名称
        self.TrainSet = graph.py_GSet()
        self.TestSet = graph.py_GSet()
        self.inputs = dict() #空字典
        self.inputs_ga = dict()

        self.reg_hidden = REG_HIDDEN #？？？？？？？？？？？？？？？降维后的维数？
        self.utils = utils.py_Utils() #一些关于图的操作

        ############----------------------------- variants of DQN(start) ------------------- ################################### 不同DQN的变体
        self.IsHuberloss = False #一种函数huberloss 损失函数选择判断
        self.IsDoubleDQN = False #DQN变种方法选择判断
        self.IsPrioritizedSampling = False #优先取样 这个对应prioritized memory  是否带权选择，cpp还没看懂 sumtree等...
        self.IsMultiStepDQN = True     ##(if IsNStepDQN=False, N_STEP==1) 多步DQN方法

        ############----------------------------- variants of DQN(end) ------------------- ###################################

        self.pop_size = 100
        # self.features = 50
        self.selection = 0.2

        self.mutation1 = 1. / self.embedding_size
        self.mutation2 = 1. / (self.embedding_size * self.embedding_size)
        self.mutation3 = 1. / (2 * self.embedding_size * self.embedding_size)
        self.mutation4 = 1. / (self.embedding_size * self.reg_hidden)
        self.mutation5 = 1. / (self.reg_hidden + aux_dim)


        self.generations = 200
        self.num_parents = int(self.pop_size * self.selection)
        self.num_children = self.pop_size - self.num_parents

        # self.truth_ph1 = tf.placeholder(tf.float32, [1, 2, self.embedding_size])
        # self.truth_ph2 = tf.placeholder(tf.float32, [1, self.embedding_size, self.embedding_size])
        # self.truth_ph3 = tf.placeholder(tf.float32, [1, 2*self.embedding_size, self.embedding_size])
        # self.truth_ph4 = tf.placeholder(tf.float32, [1, self.embedding_size, self.reg_hidden])
        # self.truth_ph5 = tf.placeholder(tf.float32, [1, self.reg_hidden + aux_dim, 1])
        # self.truth_ph6 = tf.placeholder(tf.float32, [1, self.embedding_size, 1])

        self.crossover_mat_ph1 = tf.placeholder(tf.float32, [self.num_children, 2, self.embedding_size], name="crossover_mat_ph1")
        self.crossover_mat_ph2 = tf.placeholder(tf.float32, [self.num_children, self.embedding_size, self.embedding_size], name="crossover_mat_ph2")
        self.crossover_mat_ph3 = tf.placeholder(tf.float32, [self.num_children, self.embedding_size, self.embedding_size],name="crossover_mat_ph3")
        self.crossover_mat_ph4 = tf.placeholder(tf.float32, [self.num_children, 2*self.embedding_size, self.embedding_size], name="crossover_mat_ph4")
        self.crossover_mat_ph5 = tf.placeholder(tf.float32, [self.num_children, self.embedding_size, self.reg_hidden], name="crossover_mat_ph5")
        self.crossover_mat_ph6 = tf.placeholder(tf.float32, [self.num_children, self.reg_hidden + aux_dim, 1], name="crossover_mat_ph6")
        self.crossover_mat_ph7 = tf.placeholder(tf.float32, [self.num_children, self.embedding_size, 1], name="crossover_mat_ph7")

        self.mutation_val_ph1 = tf.placeholder(tf.float32, [self.num_children, 2, self.embedding_size], name="mutation_val_ph1")
        self.mutation_val_ph2 = tf.placeholder(tf.float32, [self.num_children, self.embedding_size, self.embedding_size], name="mutation_val_ph2")
        self.mutation_val_ph3 = tf.placeholder(tf.float32, [self.num_children, self.embedding_size, self.embedding_size], name="mutation_val_ph3")
        self.mutation_val_ph4 = tf.placeholder(tf.float32, [self.num_children, 2*self.embedding_size, self.embedding_size], name="mutation_val_ph4")
        self.mutation_val_ph5 = tf.placeholder(tf.float32, [self.num_children, self.embedding_size, self.reg_hidden], name="mutation_val_ph5")
        self.mutation_val_ph6 = tf.placeholder(tf.float32, [self.num_children, self.reg_hidden + aux_dim, 1], name="mutation_val_ph6")
        self.mutation_val_ph7 = tf.placeholder(tf.float32, [self.num_children, self.embedding_size, 1], name="mutation_val_ph7")






        #Simulator
        self.ngraph_train = 0
        self.ngraph_test = 0
        self.env_list=[] #环境列表
        self.g_list=[] #图列表
        self.pred=[] # pred Q值

        if self.IsPrioritizedSampling:
            self.nStepReplayMem = nstep_replay_mem_prioritized.py_Memory(epsilon,alpha,beta,beta_increment_per_sampling,TD_err_upper,MEMORY_SIZE)
        else:
            self.nStepReplayMem = nstep_replay_mem.py_NStepReplayMem(MEMORY_SIZE)

        for i in range(num_env): #num_env==1，i==0 则有一个DQN环境
            self.env_list.append(mvc_env.py_MvcEnv(NUM_MAX)) #norm == NUM_MAX
            self.g_list.append(graph.py_Graph())

        self.test_env = mvc_env.py_MvcEnv(NUM_MAX) #测试环境

        self.test_env2 = mvc_env.py_MvcEnv(NUM_MAX) #计算n步后reward使用，确定maxQ

        # [batch_size, node_cnt]
        self.action_select = tf.sparse_placeholder(tf.float32, name="action_select")#占位符，参数：数据类型，规模（省略），名称 删除节点标记
        # [node_cnt, batch_size]
        self.rep_global = tf.sparse_placeholder(tf.float32, name="rep_global") #节点所属batch记录
        # [node_cnt, node_cnt]
        self.n2nsum_param = tf.sparse_placeholder(tf.float32, name="n2nsum_param") #节点与节点相连记录


        # self.n2nsum_param_A = tf.sparse_placeholder(tf.float32, name="n2nsum_param_A")

        # [node_cnt, node_cnt]
        self.laplacian_param = tf.sparse_placeholder(tf.float32, name="laplacian_param") #laplacian矩阵
        # [batch_size, node_cnt]
        self.subgsum_param = tf.sparse_placeholder(tf.float32, name="subgsum_param") #子图所含节点记录

        #self.subgsum_param_A = tf.sparse_placeholder(tf.float32, name="subgsum_param_A")  # 子图所含节点记录


        # [batch_size,1]
        self.target = tf.placeholder(tf.float32, [BATCH_SIZE,1], name="target") #大小是 [批数据量，1]    目标Q值 与之相对的是pred Q
        # [batch_size, aux_dim]
        self.aux_input = tf.placeholder(tf.float32, name="aux_input") #????附加信息？？？ aux_dim维信息？？？

        #[batch_size, 1]
        if self.IsPrioritizedSampling:
            self.ISWeights = tf.placeholder(tf.float32, [BATCH_SIZE, 1], name='IS_weights') #权重 计算loss时使用
                                                                                                                            #self.init_new_vars_op_alter
        self.loss_tem, self.trainStep_tem, self.q_pred_tem, self.q_on_all_tem, self.Q_param_list_tem, self.nodes_size_tem, self.q_print_tem = self.BuildNet_alter()

        # init GA                                                                                                                                                                                                                     #self.init_new_vars_op_alter_GA
        self.loss_result, self.loss_result_f, self.loss_result_selection, self.trainStep_GA, self.step, self.best_val, self.individual1, self.individual2, self.individual3, self.individual4, self.individual5, self.individual6, self.individual7 = self.BuildNet_alter_GA()
        self.loss, self.trainStep, self.q_pred, self.q_on_all, self.Q_param_list, self.nodes_size, self.q_print = self.BuildNet_alter_GA_DQN()
        self.lossT, self.trainStepT, self.q_predT, self.q_on_allT, self.Q_param_listT, self.nodes_size, self.q_print = self.BuildNet_alter_GA_DQN()

        # # init Q network    pred
        # self.loss,self.trainStep,self.q_pred, self.q_on_all,self.Q_param_list ,self.nodes_size, self.q_print= self.BuildNet_alter() #[loss,trainStep,q_pred, q_on_all, ...]
        # #init Target Q Network
        # self.lossT,self.trainStepT,self.q_predT, self.q_on_allT,self.Q_param_listT, self.nodes_size, self.q_print = self.BuildNet_alter() #两个DQN训练，其中一个表示target Q  另一个表示pred Q
        #takesnapsnot
        self.copyTargetQNetworkOperation = [a.assign(b) for a,b in zip(self.Q_param_listT,self.Q_param_list)] #variable的赋值过程 b的值给a  返回的是赋值操作


        self.UpdateTargetQNetwork = tf.group(*self.copyTargetQNetworkOperation) #*list是取list中的所有内容(值)  返回的是一个操作 run后将所有赋值执行
        # saving and loading networks
        self.saver = tf.train.Saver(max_to_keep=None) #max_to_keep 就是保存最近的几个模型 默认是5  None或0则是保存全部模型
        #self.session = tf.InteractiveSession()
        config = tf.ConfigProto(device_count={"GPU": 2, "CPU": 8},  # limit to num_cpu_core CPU usage  最大cpu数值
                                inter_op_parallelism_threads=100,
                                intra_op_parallelism_threads=100,
                                log_device_placement= False,
                                allow_soft_placement = True) #intra：设置多个操作的并行运算数，这里是100个操作一起算  inter：一个操作中并行运算线程数，这里是100个一次，通常用在大型矩阵等   log：是否打印使用哪个设备执行哪个操作，打印到终端上
        config.gpu_options.allow_growth = True #动态申请显存  也可以使用限制占有率的方法 二选一 这里是第一种动态分配
        self.session = tf.Session(config = config) #创建一个tensorflow对话框

        # self.session = tf_debug.LocalCLIDebugWrapperSession(self.session)
        # self.session.run(tf.global_variables_initializer(), feed_dict={self.action_select:np.random.rand(1), self.rep_global:np.random.rand(1), self.n2nsum_param:np.random.rand(1), self.laplacian_param:np.random.rand(1), self.subgsum_param:np.random.rand(1), self.target:np.random.rand(BATCH_SIZE, 1), self.aux_input:np.random.rand(1)})#初始化
        print(self.session.run(tf.report_uninitialized_variables()))
        # self.session.run(self.init_new_vars_op_alter)
        # self.session.run(self.init_new_vars_op_alter_GA)
        lac_var = []
        [lac_var.append(i) for i in tf.global_variables()]
        print(lac_var)
        self.session.run(tf.variables_initializer(lac_var[0:46]))
        print(self.session.run(tf.report_uninitialized_variables()))
        # lac_var = []
        # [lac_var.append(i) for i in tf.global_variables()]
        # print(lac_var)
        # lac_var1 = []
        # [lac_var1.append(i) for i in tf.local_variables()]
        # print(lac_var1)
        # self.session.run(tf.variables_initializer(lac_var[0:]))


#################################################New code for FINDER#####################################
    def BuildNet(self):
        # [2, embed_dim 64] 随机矩阵，即初始化节点的嵌入表示 W1
        w_n2l = tf.Variable(tf.truncated_normal([2, self.embedding_size], stddev=initialization_stddev), tf.float32)# trunncated_normal：截断产生正态分布随机数，随机数-均值>2*stddev（标准差） 则重新生成随机数，即范围【mean默认0 - 2 * stddev默认1, mean + 2 * stddev】
        # [embed_dim, embed_dim] 权重X2 节点相连邻居的聚合处理相关权重       W3
        p_node_conv = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32)#卷积层1 即初始卷积层
        if embeddingMethod == 1:    #'graphsage'方法
            # [embed_dim, embed_dim] 该算法下 节点自身表示的权重处理       W2
            p_node_conv2 = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32)
            # [2*embed_dim, embed_dim] 权重W^(l)
            p_node_conv3 = tf.Variable(tf.truncated_normal([2*self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32)

        #[reg_hidden+aux_dim, 1]
        if self.reg_hidden > 0: #降维 32
            #[2*embed_dim, reg_hidden]
           # h1_weight = tf.Variable(tf.truncated_normal([2 * self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            # [embed_dim 64, reg_hidden 32]
            h1_weight = tf.Variable(tf.truncated_normal([self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            #[reg_hidden1, reg_hidden2]
            # h2_weight = tf.Variable(tf.truncated_normal([self.reg_hidden1, self.reg_hidden2], stddev=initialization_stddev), tf.float32)
            #[reg_hidden+aux_dim 32+2, 1]
            h2_weight = tf.Variable(tf.truncated_normal([self.reg_hidden + aux_dim, 1], stddev=initialization_stddev), tf.float32) #32+4维？？
            #[reg_hidden2 + aux_dim 32+2, 1]
            last_w = h2_weight
        else: #reg_hidden<=0
            #[2*embed_dim, reg_hidden]
            h1_weight = tf.Variable(tf.truncated_normal([2 * self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            # [embed_dim, reg_hidden]
            # h1_weight = tf.Variable(tf.truncated_normal([self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            #[2*embed_dim, reg_hidden]
            last_w = h1_weight

        ## [embed_dim 64, 1]       W4
        cross_product = tf.Variable(tf.truncated_normal([self.embedding_size, 1], stddev=initialization_stddev), tf.float32) #内积？ 外积？


        #[node_cnt, 2]
        nodes_size = tf.shape(self.n2nsum_param)[0] #输出矩阵的维数的第一维 即节点数
        node_input = tf.ones((nodes_size,2)) #全1矩阵 Xv

        y_nodes_size = tf.shape(self.subgsum_param)[0] #即子图数目，亦即虚拟节点的表达实现过程
        # [batch_size, 2]
        y_node_input = tf.ones((y_nodes_size,2))


        #[node_cnt, 2] * [2, embed_dim] = [node_cnt, embed_dim]
        input_message = tf.matmul(tf.cast(node_input,tf.float32), w_n2l) #矩阵相乘 cast是转换数据类型，为float32 最后为【节点数，嵌入向量维数】 初始每个节点的嵌入表示，随机化

        #[node_cnt, embed_dim]  # no sparse  初始化节点嵌入表示
        input_potential_layer = tf.nn.relu(input_message) #将输入进行ReLu函数处理

        # # no sparse
        # [batch_size, embed_dim]
        y_input_message = tf.matmul(tf.cast(y_node_input,tf.float32), w_n2l)
        #[batch_size, embed_dim]  # no sparse 初始化虚拟节点表示 按子图batch来
        y_input_potential_layer = tf.nn.relu(y_input_message) #tf.nn 提供神经网络相关支持 比如卷积 池化



        #input_potential_layer = input_message 循环次数，即图嵌入表示算法循环的次数K
        cdef int lv = 0

        #[node_cnt, embed_dim], no sparse 标准化处理
        cur_message_layer = input_potential_layer
        cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1) #一种标准化处理，即每个元素除以 所属列的全部元素的平方和开跟  axis表示在哪一维度计算norm 这里按行计算

        #[batch_size, embed_dim], no sparse
        y_cur_message_layer = y_input_potential_layer
        # [batch_size, embed_dim] 标准化
        y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)

        while lv < max_bp_iter: #最大迭代 ：此处是3
            lv = lv + 1
            #[node_cnt, node_cnt] * [node_cnt, embed_dim] = [node_cnt, embed_dim], dense 按照边的记录矩阵n2nsum_param来聚合邻居节点信息
            n2npool = tf.sparse_tensor_dense_matmul(tf.cast(self.n2nsum_param,tf.float32), cur_message_layer) #稀疏矩阵与dense矩阵相乘，因为表达不同 一个是sparseplaceholder 一个是variable   这是聚集邻居节点表示 求和

            #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
            node_linear = tf.matmul(n2npool, p_node_conv) #卷积 相当于乘以权重W2 或者 AGG的权重部分 前者是structure2vec 算法  后者是graphSAGE算法

            # [batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim]
            y_n2npool = tf.sparse_tensor_dense_matmul(tf.cast(self.subgsum_param,tf.float32), cur_message_layer)
            #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
            y_node_linear = tf.matmul(y_n2npool, p_node_conv)


            if embeddingMethod == 0: # 'structure2vec'算法
                #[node_cnt, embed_dim] + [node_cnt, embed_dim] = [node_cnt, embed_dim], return tensed matrix
                merged_linear = tf.add(node_linear,input_message) #元素相加 与节点本身表示相加  input_message相当于输入Xv
                #[node_cnt, embed_dim]
                cur_message_layer = tf.nn.relu(merged_linear)

                #[batch_size, embed_dim] + [batch_size, embed_dim] = [batch_size, embed_dim], return tensed matrix
                y_merged_linear = tf.add(y_node_linear, y_input_message)
                #[batch_size, embed_dim]
                y_cur_message_layer = tf.nn.relu(y_merged_linear)
            else:   # 'graphsage'算法
                #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
                cur_message_layer_linear = tf.matmul(tf.cast(cur_message_layer, tf.float32), p_node_conv2) # AGG函数结果，即代表自身节点表达 hv(l-1)

                #[[node_cnt, embed_dim] [node_cnt, embed_dim]] = [node_cnt, 2*embed_dim], return tensed matrix
                merged_linear = tf.concat([node_linear, cur_message_layer_linear], 1) #按照第二个维度拼接 即按列拼 节点和邻居拼接 hv(l-1)拼接hN(v)
                #[node_cnt, 2*embed_dim]*[2*embed_dim, embed_dim] = [node_cnt, embed_dim]
                cur_message_layer = tf.nn.relu(tf.matmul(merged_linear, p_node_conv3)) #COM结果 乘权重Wl

                #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
                y_cur_message_layer_linear = tf.matmul(tf.cast(y_cur_message_layer, tf.float32), p_node_conv2)
                #[[batch_size, embed_dim] [batch_size, embed_dim]] = [batch_size, 2*embed_dim], return tensed matrix
                y_merged_linear = tf.concat([y_node_linear, y_cur_message_layer_linear], 1)
                #[batch_size, 2*embed_dim]*[2*embed_dim, embed_dim] = [batch_size, embed_dim]
                y_cur_message_layer = tf.nn.relu(tf.matmul(y_merged_linear, p_node_conv3))

            cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1) #标准化节点表示[node_cnt, embed_dim]
            y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1) #标准化虚拟节点 [batch_size, embed_dim]




        # self.node_embedding = cur_message_layer
        #[batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim], dense
      #  y_potential = tf.sparse_tensor_dense_matmul(tf.cast(self.subgsum_param,tf.float32), cur_message_layer)
        y_potential = y_cur_message_layer#？？？？？？？？？？？
        #[batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim] 是否限制每个batch至多选1个节点？？？
        action_embed = tf.sparse_tensor_dense_matmul(tf.cast(self.action_select, tf.float32), cur_message_layer) #根据action_select这个数据输入，选择动作，即删除节点的嵌入表示

     #   embed_s_a = tf.concat([action_embed,y_potential],1)

         # # [batch_size, embed_dim, embed_dim] za*zs
        temp = tf.matmul(tf.expand_dims(action_embed, axis=2),tf.expand_dims(y_potential, axis=1)) #前一项将行向量 变为列向量？ 虽然多了一维度？ 后一个变为每一个batch都是一个一行embed列向量；多维矩阵相乘，除了最后两维，其他都相等才能算
        # # [batch_size, embed_dim]
        Shape = tf.shape(action_embed) #输出维度
        # # [batch_size, embed_dim], first transform 相当于每个batch对应的action嵌入向量乘一个与所处batch相关的一个常数（该数通过此batch对应的虚拟节点表示与一个随机向量的内积值）
        embed_s_a = tf.reshape(tf.matmul(temp, tf.reshape(tf.tile(cross_product,[Shape[0],1]),[Shape[0],Shape[1],1])),Shape)#即每个temp对应的batch乘cross_product然后再转置 最后全部结果组合成一个shape规格的数据，其实就是每个batch emberd与crossproduct内积乘以actionselect的embed并用行向量表示，组合成一个矩阵

        #[batch_size, embed_dim]
        last_output = embed_s_a

        if self.reg_hidden > 0: #降维   之后就是ReLu    然后*W5 权重 得到Q
            #[batch_size, embed_dim] * [embed_dim, reg_hidden] = [batch_size, reg_hidden], dense
            hidden = tf.matmul(embed_s_a, h1_weight) #batch_size，reg_hidden
            #[batch_size, reg_hidden]
            last_output = tf.nn.relu(hidden)

        # if reg_hidden == 0: ,[[batch_size, 2*embed_dim], [batch_size, aux_dim]] = [batch_size, 2*embed_dim+aux_dim]
        # if reg_hidden > 0: ,[[batch_size, reg_hidden], [batch_size, aux_dim]] = [batch_size, reg_hidden+aux_dim]
        last_output = tf.concat([last_output, self.aux_input], 1) #aux_input代表什么？？？
        #if reg_hidden == 0: ,[batch_size, 2*embed_dim+aux_dim] * [2*embed_dim+aux_dim, 1] = [batch_size, 1]
        #if reg_hidden > 0: ,[batch_size, reg_hidden+aux_dim] * [reg_hidden+aux_dim, 1] = [batch_size, 1]
        q_pred = tf.matmul(last_output, last_w)



        ## first order reconstruction loss 重构损失，乘2是为了和下面edge数目多算一倍抵消，该loss的形式是yi（yi-求和yj...）   j是i的邻居  这项对每一维度求和&每一对相连节点求和
        loss_recons = 2 * tf.trace(tf.matmul(tf.transpose(cur_message_layer), tf.sparse_tensor_dense_matmul(tf.cast(self.laplacian_param,tf.float32), cur_message_layer))) #trace 矩阵的迹
        edge_num = tf.sparse_reduce_sum(tf.cast(self.n2nsum_param, tf.float32)) #不指定维度，计算所有元素总和
        loss_recons = tf.divide(loss_recons, edge_num) #除以边数 计算平均的重构损失


        if self.IsPrioritizedSampling:
            self.TD_errors = tf.reduce_sum(tf.abs(self.target - q_pred), axis=1)    # for updating Sumtree    reduce后从二维变为一维数据 值不变
            if self.IsHuberloss:
                loss_rl = tf.losses.huber_loss(self.ISWeights * self.target, self.ISWeights * q_pred)
            else:
                loss_rl = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.target, q_pred))
        else:
            if self.IsHuberloss:
                loss_rl = tf.losses.huber_loss(self.target, q_pred) #根据target（目标，真实）-预测pred的每一项结果按huberloss的规则分段函数计算最后得到loss
            else:
                loss_rl = tf.losses.mean_squared_error(self.target, q_pred) #平方和取平均 即batch张图的结果取均值E

        loss = loss_rl + Alpha * loss_recons #Q-learning损失 和 重构损失

        trainStep = tf.train.AdamOptimizer(self.learning_rate).minimize(loss) #训练DQN 返回的是优化更新后的var_list(变量列表)

        #[node_cnt, batch_size] * [batch_size, embed_dim] = [node_cnt, embed_dim]
        rep_y = tf.sparse_tensor_dense_matmul(tf.cast(self.rep_global, tf.float32), y_potential) #每个节点对应的batch的虚拟节点表示

      #  embed_s_a_all = tf.concat([cur_message_layer,rep_y],1)

        # # # [node_cnt, embed_dim, embed_dim]
        temp1 = tf.matmul(tf.expand_dims(cur_message_layer, axis=2),tf.expand_dims(rep_y, axis=1))#计算每个s a对应的Q，每个节点对应的嵌入表示a，相对应的batch表示s
        # # [node_cnt embed_dim]
        Shape1 = tf.shape(cur_message_layer)
        # # [node_cnt, embed_dim], first transform
        embed_s_a_all = tf.reshape(tf.matmul(temp1, tf.reshape(tf.tile(cross_product,[Shape1[0],1]),[Shape1[0],Shape1[1],1])),Shape1)

        #[node_cnt, embed_dim]
        last_output = embed_s_a_all
        if self.reg_hidden > 0:
            #[node_cnt, embed_dim] * [embed_dim, reg_hidden] = [node_cnt, reg_hidden1]
            hidden = tf.matmul(embed_s_a_all, h1_weight)
            #Relu, [node_cnt, reg_hidden1]
            last_output = tf.nn.relu(hidden)
            #[node_cnt, reg_hidden1] * [reg_hidden1, reg_hidden2] = [node_cnt, reg_hidden2]

        #[node_cnt, batch_size] * [batch_size, aux_dim] = [node_cnt, aux_dim]
        rep_aux = tf.sparse_tensor_dense_matmul(tf.cast(self.rep_global, tf.float32), self.aux_input)

        #if reg_hidden == 0: , [[node_cnt, 2 * embed_dim], [node_cnt, aux_dim]] = [node_cnt, 2*embed_dim + aux_dim]
        #if reg_hidden > 0: , [[node_cnt, reg_hidden], [node_cnt, aux_dim]] = [node_cnt, reg_hidden + aux_dim]
        last_output = tf.concat([last_output,rep_aux],1)

        #if reg_hidden == 0: , [node_cnt, 2 * embed_dim + aux_dim] * [2 * embed_dim + aux_dim, 1] = [node_cnt，1]
        #f reg_hidden > 0: , [node_cnt, reg_hidden + aux_dim] * [reg_hidden + aux_dim, 1] = [node_cnt，1]
        q_on_all = tf.matmul(last_output, last_w) #每个节点在当前被删的Q值记录

        return loss, trainStep, q_pred, q_on_all, tf.trainable_variables() , nodes_size#损失，神经网络训练trainstep表示，q预测（根据action选择计算Q），q表（计算所有节点在当前被删后的Q），神经网络训练相关参数



    def BuildNet_alter(self):
        # [2, embed_dim 64] 随机矩阵，即初始化节点的嵌入表示 W1
        w_n2l = tf.Variable(tf.truncated_normal([2, self.embedding_size], stddev=initialization_stddev), tf.float32)# trunncated_normal：截断产生正态分布随机数，随机数-均值>2*stddev（标准差） 则重新生成随机数，即范围【mean默认0 - 2 * stddev默认1, mean + 2 * stddev】
        # [embed_dim, embed_dim] 权重X2 节点相连邻居的聚合处理相关权重       W3
        p_node_conv = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32)#卷积层1 即初始卷积层
        if embeddingMethod == 1:    #'graphsage'方法
            # [embed_dim, embed_dim] 该算法下 节点自身表示的权重处理       W2
            p_node_conv2 = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32)
            # [2*embed_dim, embed_dim] 权重W^(l)
            p_node_conv3 = tf.Variable(tf.truncated_normal([2*self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32)

        #[reg_hidden+aux_dim, 1]
        if self.reg_hidden > 0: #降维 32
            #[2*embed_dim, reg_hidden]
           # h1_weight = tf.Variable(tf.truncated_normal([2 * self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            # [embed_dim 64, reg_hidden 32]
            h1_weight = tf.Variable(tf.truncated_normal([self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            #[reg_hidden1, reg_hidden2]
            # h2_weight = tf.Variable(tf.truncated_normal([self.reg_hidden1, self.reg_hidden2], stddev=initialization_stddev), tf.float32)
            #[reg_hidden+aux_dim 32+2, 1]
            h2_weight = tf.Variable(tf.truncated_normal([self.reg_hidden + aux_dim, 1], stddev=initialization_stddev), tf.float32) #32+4维？？
            #[reg_hidden2 + aux_dim 32+2, 1]
            last_w = h2_weight
        else: #reg_hidden<=0
            #[2*embed_dim, reg_hidden]
            h1_weight = tf.Variable(tf.truncated_normal([2 * self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            # [embed_dim, reg_hidden]
            # h1_weight = tf.Variable(tf.truncated_normal([self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            #[2*embed_dim, reg_hidden]
            last_w = h1_weight

        ## [embed_dim 64, 1]       W4
        cross_product = tf.Variable(tf.truncated_normal([self.embedding_size, 1], stddev=initialization_stddev), tf.float32) #内积？ 外积？

        # init_new_vars_op_alter = tf.initialize_variables([w_n2l, p_node_conv, p_node_conv2, p_node_conv3, h1_weight, last_w, cross_product])

        #[node_cnt, 2]
        nodes_size = tf.shape(self.n2nsum_param)[0] #输出矩阵的维数的第一维 即节点数
        node_input = tf.ones((nodes_size,2)) #全1矩阵 Xv

        y_nodes_size = tf.shape(self.subgsum_param)[0] #即子图数目，亦即虚拟节点的表达实现过程
        # [batch_size, 2]
        y_node_input = tf.ones((y_nodes_size,2))


        #[node_cnt, 2] * [2, embed_dim] = [node_cnt, embed_dim]
        input_message = tf.matmul(tf.cast(node_input,tf.float32), w_n2l) #矩阵相乘 cast是转换数据类型，为float32 最后为【节点数，嵌入向量维数】 初始每个节点的嵌入表示，随机化

        #[node_cnt, embed_dim]  # no sparse  初始化节点嵌入表示
        input_potential_layer = tf.nn.relu(input_message) #将输入进行ReLu函数处理

        # # no sparse
        # [batch_size, embed_dim]
        y_input_message = tf.matmul(tf.cast(y_node_input,tf.float32), w_n2l)
        #[batch_size, embed_dim]  # no sparse 初始化虚拟节点表示 按子图batch来
        y_input_potential_layer = tf.nn.relu(y_input_message) #tf.nn 提供神经网络相关支持 比如卷积 池化



        #input_potential_layer = input_message 循环次数，即图嵌入表示算法循环的次数K
        cdef int lv = 0

        #[node_cnt, embed_dim], no sparse 标准化处理
        cur_message_layer = input_potential_layer
        cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1) #一种标准化处理，即每个行向量标准化

        #[batch_size, embed_dim], no sparse
        y_cur_message_layer = y_input_potential_layer
        # [batch_size, embed_dim] 标准化
        y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)

        while lv < max_bp_iter: #最大迭代 ：此处是3
            lv = lv + 1
            #[node_cnt, node_cnt] * [node_cnt, embed_dim] = [node_cnt, embed_dim], dense 按照边的记录矩阵n2nsum_param来聚合邻居节点信息
            n2npool = tf.sparse_tensor_dense_matmul(tf.cast(self.n2nsum_param,tf.float32), cur_message_layer) #稀疏矩阵与dense矩阵相乘，因为表达不同 一个是sparseplaceholder 一个是variable   这是聚集邻居节点表示 求和

            #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
            node_linear = tf.matmul(n2npool, p_node_conv) #卷积 相当于乘以权重W2 或者 AGG的权重部分 前者是structure2vec 算法  后者是graphSAGE算法

            # [batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim]
            y_n2npool = tf.sparse_tensor_dense_matmul(tf.cast(self.subgsum_param,tf.float32), cur_message_layer)
            #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
            y_node_linear = tf.matmul(y_n2npool, p_node_conv)


            if embeddingMethod == 0: # 'structure2vec'算法
                #[node_cnt, embed_dim] + [node_cnt, embed_dim] = [node_cnt, embed_dim], return tensed matrix
                merged_linear = tf.add(node_linear,input_message) #元素相加 与节点本身表示相加  input_message相当于输入Xv
                #[node_cnt, embed_dim]
                cur_message_layer = tf.nn.relu(merged_linear)

                #[batch_size, embed_dim] + [batch_size, embed_dim] = [batch_size, embed_dim], return tensed matrix
                y_merged_linear = tf.add(y_node_linear, y_input_message)
                #[batch_size, embed_dim]
                y_cur_message_layer = tf.nn.relu(y_merged_linear)
            else:   # 'graphsage'算法
                #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
                cur_message_layer_linear = tf.matmul(tf.cast(cur_message_layer, tf.float32), p_node_conv2) # AGG函数结果，即代表自身节点表达 hv(l-1)

                #[[node_cnt, embed_dim] [node_cnt, embed_dim]] = [node_cnt, 2*embed_dim], return tensed matrix
                merged_linear = tf.concat([node_linear, cur_message_layer_linear], 1) #按照第二个维度拼接 即按列拼 节点和邻居拼接 hv(l-1)拼接hN(v)
                #[node_cnt, 2*embed_dim]*[2*embed_dim, embed_dim] = [node_cnt, embed_dim]
                cur_message_layer = tf.nn.relu(tf.matmul(merged_linear, p_node_conv3)) #COM结果 乘权重Wl

                #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
                y_cur_message_layer_linear = tf.matmul(tf.cast(y_cur_message_layer, tf.float32), p_node_conv2)
                #[[batch_size, embed_dim] [batch_size, embed_dim]] = [batch_size, 2*embed_dim], return tensed matrix
                y_merged_linear = tf.concat([y_node_linear, y_cur_message_layer_linear], 1)
                #[batch_size, 2*embed_dim]*[2*embed_dim, embed_dim] = [batch_size, embed_dim]
                y_cur_message_layer = tf.nn.relu(tf.matmul(y_merged_linear, p_node_conv3))

            cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1) #标准化节点表示[node_cnt, embed_dim]
            y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1) #标准化虚拟节点 [batch_size, embed_dim]




        # self.node_embedding = cur_message_layer
        #[batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim], dense
      #  y_potential = tf.sparse_tensor_dense_matmul(tf.cast(self.subgsum_param,tf.float32), cur_message_layer)
        y_potential = y_cur_message_layer#？？？？？？？？？？？
        #[batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim] 是否限制每个batch至多选1个节点？？？
        action_embed = tf.sparse_tensor_dense_matmul(tf.cast(self.action_select, tf.float32), cur_message_layer) #根据action_select这个数据输入，选择动作，即删除节点的嵌入表示

     #   embed_s_a = tf.concat([action_embed,y_potential],1)

         # # [batch_size, embed_dim, embed_dim] za*zs
        temp = tf.matmul(tf.expand_dims(action_embed, axis=2),tf.expand_dims(y_potential, axis=1)) #前一项将行向量 变为列向量？ 虽然多了一维度？ 后一个变为每一个batch都是一个一行embed列向量；多维矩阵相乘，除了最后两维，其他都相等才能算
        # # [batch_size, embed_dim]
        Shape = tf.shape(action_embed) #输出维度
        # # [batch_size, embed_dim], first transform 相当于每个batch对应的action嵌入向量乘一个与所处batch相关的一个常数（该数通过此batch对应的虚拟节点表示与一个随机向量的内积值）
        embed_s_a = tf.reshape(tf.matmul(temp, tf.reshape(tf.tile(cross_product,[Shape[0],1]),[Shape[0],Shape[1],1])),Shape)#即每个temp对应的batch乘cross_product然后再转置 最后全部结果组合成一个shape规格的数据，其实就是每个batch emberd与crossproduct内积乘以actionselect的embed并用行向量表示，组合成一个矩阵

        #[batch_size, embed_dim]
        last_output = embed_s_a

        if self.reg_hidden > 0: #降维   之后就是ReLu    然后*W5 权重 得到Q
            #[batch_size, embed_dim] * [embed_dim, reg_hidden] = [batch_size, reg_hidden], dense
            hidden = tf.matmul(embed_s_a, h1_weight) #batch_size，reg_hidden
            #[batch_size, reg_hidden]
            last_output = tf.nn.relu(hidden)

        # if reg_hidden == 0: ,[[batch_size, 2*embed_dim], [batch_size, aux_dim]] = [batch_size, 2*embed_dim+aux_dim]
        # if reg_hidden > 0: ,[[batch_size, reg_hidden], [batch_size, aux_dim]] = [batch_size, reg_hidden+aux_dim]
        last_output = tf.concat([last_output, self.aux_input], 1) #aux_input代表什么？？？
        #if reg_hidden == 0: ,[batch_size, 2*embed_dim+aux_dim] * [2*embed_dim+aux_dim, 1] = [batch_size, 1]
        #if reg_hidden > 0: ,[batch_size, reg_hidden+aux_dim] * [reg_hidden+aux_dim, 1] = [batch_size, 1]
        q_pred = tf.matmul(last_output, last_w) #每个batch中选择一个节点的对应Q


        # loss1 = tf.reduce_sum(q_pred)
        ## first order reconstruction loss 重构损失，乘2是为了和下面edge数目多算一倍抵消，该loss的形式是yi（yi-求和yj...）   j是i的邻居  这项对每一维度求和&每一对相连节点求和
        loss_recons = tf.trace(tf.matmul(tf.transpose(cur_message_layer), tf.sparse_tensor_dense_matmul(tf.cast(self.laplacian_param,tf.float32), cur_message_layer))) #trace 矩阵的迹
        # loss2 =
        edge_num = tf.sparse_reduce_sum(tf.cast(self.n2nsum_param, tf.float32)) #不指定维度，计算所有元素总和

        loss_recons = tf.divide(loss_recons, edge_num) #除以边数 计算平均的重构损失
        # loss2 = loss_recons


        if self.IsPrioritizedSampling:
            self.TD_errors = tf.reduce_sum(tf.abs(self.target - q_pred), axis=1)    # for updating Sumtree    reduce后从二维变为一维数据 值不变
            if self.IsHuberloss:
                loss_rl = tf.losses.huber_loss(self.ISWeights * self.target, self.ISWeights * q_pred)
            else:
                loss_rl = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.target, q_pred))
        else:
            if self.IsHuberloss:
                loss_rl = tf.losses.huber_loss(self.target, q_pred) #根据target（目标，真实）-预测pred的每一项结果按huberloss的规则分段函数计算最后得到loss
            else:
                loss_rl = tf.losses.mean_squared_error(self.target, q_pred) #平方和取平均 即batch张图的结果取均值E
        # loss1 = loss_rl
        # loss2 = loss_recons

        l2_loss = self.weight_decay * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
        loss = loss_rl + Alpha * loss_recons #Q-learning损失 和 重构损失
        loss = loss + l2_loss

        trainStep = tf.train.AdamOptimizer(self.learning_rate).minimize(loss) #训练DQN

        #[node_cnt, batch_size] * [batch_size, embed_dim] = [node_cnt, embed_dim]
        rep_y = tf.sparse_tensor_dense_matmul(tf.cast(self.rep_global, tf.float32), y_potential) #每个节点对应的batch的虚拟节点表示

      #  embed_s_a_all = tf.concat([cur_message_layer,rep_y],1)

        # # # [node_cnt, embed_dim, embed_dim]
        temp1 = tf.matmul(tf.expand_dims(cur_message_layer, axis=2),tf.expand_dims(rep_y, axis=1))#计算每个s a对应的Q，每个节点对应的嵌入表示a，相对应的batch表示s
        # # [node_cnt embed_dim]
        Shape1 = tf.shape(cur_message_layer)
        # # [node_cnt, embed_dim], first transform
        embed_s_a_all = tf.reshape(tf.matmul(temp1, tf.reshape(tf.tile(cross_product,[Shape1[0],1]),[Shape1[0],Shape1[1],1])),Shape1)

        #[node_cnt, embed_dim]
        last_output = embed_s_a_all
        if self.reg_hidden > 0:
            #[node_cnt, embed_dim] * [embed_dim, reg_hidden] = [node_cnt, reg_hidden1]
            hidden = tf.matmul(embed_s_a_all, h1_weight)
            #Relu, [node_cnt, reg_hidden1]
            last_output = tf.nn.relu(hidden)
            #[node_cnt, reg_hidden1] * [reg_hidden1, reg_hidden2] = [node_cnt, reg_hidden2]

        #[node_cnt, batch_size] * [batch_size, aux_dim] = [node_cnt, aux_dim]
        rep_aux = tf.sparse_tensor_dense_matmul(tf.cast(self.rep_global, tf.float32), self.aux_input)

        #if reg_hidden == 0: , [[node_cnt, 2 * embed_dim], [node_cnt, aux_dim]] = [node_cnt, 2*embed_dim + aux_dim]
        #if reg_hidden > 0: , [[node_cnt, reg_hidden], [node_cnt, aux_dim]] = [node_cnt, reg_hidden + aux_dim]
        last_output = tf.concat([last_output,rep_aux],1)

        #if reg_hidden == 0: , [node_cnt, 2 * embed_dim + aux_dim] * [2 * embed_dim + aux_dim, 1] = [node_cnt，1]
        #f reg_hidden > 0: , [node_cnt, reg_hidden + aux_dim] * [reg_hidden + aux_dim, 1] = [node_cnt，1]
        q_on_all = tf.matmul(last_output, last_w) #每个节点在当前被删的Q值记录
        # loss2 = tf.reduce_sum(q_on_all)
        q_print = tf.print(q_on_all,[q_on_all],"q_on_all: ",summarize=50)

        return loss, trainStep, q_pred, q_on_all, tf.trainable_variables() , nodes_size, q_print#损失，神经网络训练trainstep表示，q预测（根据action选择计算Q），q表（计算所有节点在当前被删后的Q），神经网络训练相关参数







    def BuildNet_alter_GA(self):
        # [2, embed_dim 64] 随机矩阵，即初始化节点的嵌入表示 W1
        w_n2l = tf.Variable(tf.truncated_normal([self.pop_size, 2, self.embedding_size], stddev=initialization_stddev),tf.float32)  # trunncated_normal：截断产生正态分布随机数，随机数-均值>2*stddev（标准差） 则重新生成随机数，即范围【mean默认0 - 2 * stddev默认1, mean + 2 * stddev】
        # [embed_dim, embed_dim] 权重X2 节点相连邻居的聚合处理相关权重       W3
        p_node_conv = tf.Variable(tf.truncated_normal([self.pop_size, self.embedding_size, self.embedding_size], stddev=initialization_stddev),tf.float32)  # 卷积层1 即初始卷积层
        if embeddingMethod == 1:  # 'graphsage'方法
            # [embed_dim, embed_dim] 该算法下 节点自身表示的权重处理       W2
            p_node_conv2 = tf.Variable(tf.truncated_normal([self.pop_size, self.embedding_size, self.embedding_size], stddev=initialization_stddev),tf.float32)
            # [2*embed_dim, embed_dim] 权重W^(l)
            p_node_conv3 = tf.Variable(tf.truncated_normal([self.pop_size, 2 * self.embedding_size, self.embedding_size], stddev=initialization_stddev),tf.float32)

        # [reg_hidden+aux_dim, 1]
        if self.reg_hidden > 0:  # 降维 32
            # [2*embed_dim, reg_hidden]
            # h1_weight = tf.Variable(tf.truncated_normal([2 * self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            # [embed_dim 64, reg_hidden 32]
            h1_weight = tf.Variable(tf.truncated_normal([self.pop_size, self.embedding_size, self.reg_hidden], stddev=initialization_stddev),tf.float32)
            # [reg_hidden1, reg_hidden2]
            # h2_weight = tf.Variable(tf.truncated_normal([self.reg_hidden1, self.reg_hidden2], stddev=initialization_stddev), tf.float32)
            # [reg_hidden+aux_dim 32+2, 1]
            h2_weight = tf.Variable(tf.truncated_normal([self.pop_size, self.reg_hidden + aux_dim, 1], stddev=initialization_stddev),tf.float32)  # 32+4维？？
            # [reg_hidden2 + aux_dim 32+2, 1]
            last_w = h2_weight
        else:  # reg_hidden<=0
            # [2*embed_dim, reg_hidden]
            h1_weight = tf.Variable(tf.truncated_normal([self.pop_size, 2 * self.embedding_size, self.reg_hidden], stddev=initialization_stddev),tf.float32)
            # [embed_dim, reg_hidden]
            # h1_weight = tf.Variable(tf.truncated_normal([self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            # [2*embed_dim, reg_hidden]
            last_w = h1_weight

        ## [embed_dim 64, 1]       W4
        cross_product = tf.Variable(tf.truncated_normal([self.pop_size, self.embedding_size, 1], stddev=initialization_stddev),tf.float32)  # 内积？ 外积？

        # init_new_vars_op_alter_GA = tf.initialize_variables([w_n2l, p_node_conv, p_node_conv2, p_node_conv3, h1_weight, last_w, cross_product])

        def loopcaculate(x):
            w_n2l_tem, p_node_conv_tem, p_node_conv2_tem, p_node_conv3_tem, h1_weight_tem, last_w_tem, cross_product_tem = x

            # [node_cnt, 2]
            nodes_size = tf.shape(self.n2nsum_param)[0]  # 输出矩阵的维数的第一维 即节点数
            node_input = tf.ones((nodes_size, 2))  # 全1矩阵 Xv

            y_nodes_size = tf.shape(self.subgsum_param)[0]  # 即子图数目，亦即虚拟节点的表达实现过程
            # [batch_size, 2]
            y_node_input = tf.ones((y_nodes_size, 2))

            # [node_cnt, 2] * [2, embed_dim] = [node_cnt, embed_dim]
            input_message = tf.matmul(tf.cast(node_input, tf.float32),w_n2l_tem)  # 矩阵相乘 cast是转换数据类型，为float32 最后为【节点数，嵌入向量维数】 初始每个节点的嵌入表示，随机化

            # [node_cnt, embed_dim]  # no sparse  初始化节点嵌入表示
            input_potential_layer = tf.nn.relu(input_message)  # 将输入进行ReLu函数处理

            # # no sparse
            # [batch_size, embed_dim]
            y_input_message = tf.matmul(tf.cast(y_node_input, tf.float32), w_n2l_tem)
            # [batch_size, embed_dim]  # no sparse 初始化虚拟节点表示 按子图batch来
            y_input_potential_layer = tf.nn.relu(y_input_message)  # tf.nn 提供神经网络相关支持 比如卷积 池化

            # input_potential_layer = input_message 循环次数，即图嵌入表示算法循环的次数K
            cdef int lv = 0

            # [node_cnt, embed_dim], no sparse 标准化处理
            cur_message_layer = input_potential_layer
            cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)  # 一种标准化处理，即每个行向量标准化

            # [batch_size, embed_dim], no sparse
            y_cur_message_layer = y_input_potential_layer
            # [batch_size, embed_dim] 标准化
            y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)

            while lv < max_bp_iter:  # 最大迭代 ：此处是3
                lv = lv + 1
                # [node_cnt, node_cnt] * [node_cnt, embed_dim] = [node_cnt, embed_dim], dense 按照边的记录矩阵n2nsum_param来聚合邻居节点信息
                n2npool = tf.sparse_tensor_dense_matmul(tf.cast(self.n2nsum_param, tf.float32),cur_message_layer)  # 稀疏矩阵与dense矩阵相乘，因为表达不同 一个是sparseplaceholder 一个是variable   这是聚集邻居节点表示 求和

                # [node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
                node_linear = tf.matmul(n2npool,p_node_conv_tem)  # 卷积 相当于乘以权重W2 或者 AGG的权重部分 前者是structure2vec 算法  后者是graphSAGE算法

                # [batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim]
                y_n2npool = tf.sparse_tensor_dense_matmul(tf.cast(self.subgsum_param, tf.float32), cur_message_layer)
                # [batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
                y_node_linear = tf.matmul(y_n2npool, p_node_conv_tem)

                if embeddingMethod == 0:  # 'structure2vec'算法
                    # [node_cnt, embed_dim] + [node_cnt, embed_dim] = [node_cnt, embed_dim], return tensed matrix
                    merged_linear = tf.add(node_linear, input_message)  # 元素相加 与节点本身表示相加  input_message相当于输入Xv
                    # [node_cnt, embed_dim]
                    cur_message_layer = tf.nn.relu(merged_linear)

                    # [batch_size, embed_dim] + [batch_size, embed_dim] = [batch_size, embed_dim], return tensed matrix
                    y_merged_linear = tf.add(y_node_linear, y_input_message)
                    # [batch_size, embed_dim]
                    y_cur_message_layer = tf.nn.relu(y_merged_linear)
                else:  # 'graphsage'算法
                    # [node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
                    cur_message_layer_linear = tf.matmul(tf.cast(cur_message_layer, tf.float32),p_node_conv2_tem)  # AGG函数结果，即代表自身节点表达 hv(l-1)

                    # [[node_cnt, embed_dim] [node_cnt, embed_dim]] = [node_cnt, 2*embed_dim], return tensed matrix
                    merged_linear = tf.concat([node_linear, cur_message_layer_linear],1)  # 按照第二个维度拼接 即按列拼 节点和邻居拼接 hv(l-1)拼接hN(v)
                    # [node_cnt, 2*embed_dim]*[2*embed_dim, embed_dim] = [node_cnt, embed_dim]
                    cur_message_layer = tf.nn.relu(tf.matmul(merged_linear, p_node_conv3_tem))  # COM结果 乘权重Wl

                    # [batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
                    y_cur_message_layer_linear = tf.matmul(tf.cast(y_cur_message_layer, tf.float32), p_node_conv2_tem)
                    # [[batch_size, embed_dim] [batch_size, embed_dim]] = [batch_size, 2*embed_dim], return tensed matrix
                    y_merged_linear = tf.concat([y_node_linear, y_cur_message_layer_linear], 1)
                    # [batch_size, 2*embed_dim]*[2*embed_dim, embed_dim] = [batch_size, embed_dim]
                    y_cur_message_layer = tf.nn.relu(tf.matmul(y_merged_linear, p_node_conv3_tem))

                cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)  # 标准化节点表示[node_cnt, embed_dim]
                y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)  # 标准化虚拟节点 [batch_size, embed_dim]

            # self.node_embedding = cur_message_layer
            # [batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim], dense
            #  y_potential = tf.sparse_tensor_dense_matmul(tf.cast(self.subgsum_param,tf.float32), cur_message_layer)
            y_potential = y_cur_message_layer  # ？？？？？？？？？？？
            # [batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim] 是否限制每个batch至多选1个节点？？？
            action_embed = tf.sparse_tensor_dense_matmul(tf.cast(self.action_select, tf.float32),cur_message_layer)  # 根据action_select这个数据输入，选择动作，即删除节点的嵌入表示

            #   embed_s_a = tf.concat([action_embed,y_potential],1)

            # # [batch_size, embed_dim, embed_dim] za*zs
            temp = tf.matmul(tf.expand_dims(action_embed, axis=2), tf.expand_dims(y_potential,axis=1))  # 前一项将行向量 变为列向量？ 虽然多了一维度？ 后一个变为每一个batch都是一个一行embed列向量；多维矩阵相乘，除了最后两维，其他都相等才能算
            # # [batch_size, embed_dim]
            Shape = tf.shape(action_embed)  # 输出维度
            # # [batch_size, embed_dim], first transform 相当于每个batch对应的action嵌入向量乘一个与所处batch相关的一个常数（该数通过此batch对应的虚拟节点表示与一个随机向量的内积值）
            embed_s_a = tf.reshape(tf.matmul(temp, tf.reshape(tf.tile(cross_product_tem, [Shape[0], 1]), [Shape[0], Shape[1], 1])),Shape)  # 即每个temp对应的batch乘cross_product然后再转置 最后全部结果组合成一个shape规格的数据，其实就是每个batch emberd与crossproduct内积乘以actionselect的embed并用行向量表示，组合成一个矩阵

            # [batch_size, embed_dim]
            last_output = embed_s_a

            if self.reg_hidden > 0:  # 降维   之后就是ReLu    然后*W5 权重 得到Q
                # [batch_size, embed_dim] * [embed_dim, reg_hidden] = [batch_size, reg_hidden], dense
                hidden = tf.matmul(embed_s_a, h1_weight_tem)  # batch_size，reg_hidden
                # [batch_size, reg_hidden]
                last_output = tf.nn.relu(hidden)

            # if reg_hidden == 0: ,[[batch_size, 2*embed_dim], [batch_size, aux_dim]] = [batch_size, 2*embed_dim+aux_dim]
            # if reg_hidden > 0: ,[[batch_size, reg_hidden], [batch_size, aux_dim]] = [batch_size, reg_hidden+aux_dim]
            last_output = tf.concat([last_output, self.aux_input], 1)  # aux_input代表什么？？？
            # if reg_hidden == 0: ,[batch_size, 2*embed_dim+aux_dim] * [2*embed_dim+aux_dim, 1] = [batch_size, 1]
            # if reg_hidden > 0: ,[batch_size, reg_hidden+aux_dim] * [reg_hidden+aux_dim, 1] = [batch_size, 1]
            q_pred = tf.matmul(last_output, last_w_tem)  # 每个batch中选择一个节点的对应Q

            # loss1 = tf.reduce_sum(q_pred)
            ## first order reconstruction loss 重构损失，乘2是为了和下面edge数目多算一倍抵消，该loss的形式是yi（yi-求和yj...）   j是i的邻居  这项对每一维度求和&每一对相连节点求和
            loss_recons = tf.trace(tf.matmul(tf.transpose(cur_message_layer),tf.sparse_tensor_dense_matmul(tf.cast(self.laplacian_param, tf.float32),cur_message_layer)))  # trace 矩阵的迹
            # loss2 =
            edge_num = tf.sparse_reduce_sum(tf.cast(self.n2nsum_param, tf.float32))  # 不指定维度，计算所有元素总和

            loss_recons = tf.divide(loss_recons, edge_num)  # 除以边数 计算平均的重构损失
            # loss2 = loss_recons

            if self.IsPrioritizedSampling:
                self.TD_errors = tf.reduce_sum(tf.abs(self.target - q_pred),axis=1)  # for updating Sumtree    reduce后从二维变为一维数据 值不变
                if self.IsHuberloss:
                    loss_rl = tf.losses.huber_loss(self.ISWeights * self.target, self.ISWeights * q_pred)
                else:
                    loss_rl = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.target, q_pred))
            else:
                if self.IsHuberloss:
                    loss_rl = tf.losses.huber_loss(self.target,q_pred)  # 根据target（目标，真实）-预测pred的每一项结果按huberloss的规则分段函数计算最后得到loss
                else:
                    loss_rl = tf.losses.mean_squared_error(self.target, q_pred)  # 平方和取平均 即batch张图的结果取均值E
            # loss1 = loss_rl
            # loss2 = loss_recons

            # l2_loss = self.weight_decay * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
            loss = loss_rl + Alpha * loss_recons  # Q-learning损失 和 重构损失
            # loss = loss + l2_loss
            return loss

        # out_dtype = [tf.float32]
        loss_result = tf.map_fn(loopcaculate, elems=(w_n2l, p_node_conv, p_node_conv2, p_node_conv3, h1_weight, last_w, cross_product), dtype=tf.float32)






        # loss_tem = tf.constant([-1],tf.float32)
        # fu_1 = tf.constant([-1],tf.float32)


        # cdef int lv = 0

        # for gen_i in range(self.pop_size):
        #
        #     # [node_cnt, 2]
        #     nodes_size = tf.shape(self.n2nsum_param)[0]  # 输出矩阵的维数的第一维 即节点数
        #     node_input = tf.ones((nodes_size, 2))  # 全1矩阵 Xv
        #
        #     y_nodes_size = tf.shape(self.subgsum_param)[0]  # 即子图数目，亦即虚拟节点的表达实现过程
        #     # [batch_size, 2]
        #     y_node_input = tf.ones((y_nodes_size, 2))
        #
        #     # [node_cnt, 2] * [2, embed_dim] = [node_cnt, embed_dim]
        #     input_message = tf.matmul(tf.cast(node_input, tf.float32),w_n2l[gen_i])  # 矩阵相乘 cast是转换数据类型，为float32 最后为【节点数，嵌入向量维数】 初始每个节点的嵌入表示，随机化
        #
        #     # [node_cnt, embed_dim]  # no sparse  初始化节点嵌入表示
        #     input_potential_layer = tf.nn.relu(input_message)  # 将输入进行ReLu函数处理
        #
        #     # # no sparse
        #     # [batch_size, embed_dim]
        #     y_input_message = tf.matmul(tf.cast(y_node_input, tf.float32), w_n2l[gen_i])
        #     # [batch_size, embed_dim]  # no sparse 初始化虚拟节点表示 按子图batch来
        #     y_input_potential_layer = tf.nn.relu(y_input_message)  # tf.nn 提供神经网络相关支持 比如卷积 池化
        #
        #     # input_potential_layer = input_message 循环次数，即图嵌入表示算法循环的次数K
        #     lv = 0
        #
        #     # [node_cnt, embed_dim], no sparse 标准化处理
        #     cur_message_layer = input_potential_layer
        #     cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)  # 一种标准化处理，即每个行向量标准化
        #
        #     # [batch_size, embed_dim], no sparse
        #     y_cur_message_layer = y_input_potential_layer
        #     # [batch_size, embed_dim] 标准化
        #     y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)
        #
        #     while lv < max_bp_iter:  # 最大迭代 ：此处是3
        #         lv = lv + 1
        #         # [node_cnt, node_cnt] * [node_cnt, embed_dim] = [node_cnt, embed_dim], dense 按照边的记录矩阵n2nsum_param来聚合邻居节点信息
        #         n2npool = tf.sparse_tensor_dense_matmul(tf.cast(self.n2nsum_param, tf.float32),cur_message_layer)  # 稀疏矩阵与dense矩阵相乘，因为表达不同 一个是sparseplaceholder 一个是variable   这是聚集邻居节点表示 求和
        #
        #         # [node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
        #         node_linear = tf.matmul(n2npool,p_node_conv[gen_i])  # 卷积 相当于乘以权重W2 或者 AGG的权重部分 前者是structure2vec 算法  后者是graphSAGE算法
        #
        #         # [batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim]
        #         y_n2npool = tf.sparse_tensor_dense_matmul(tf.cast(self.subgsum_param, tf.float32), cur_message_layer)
        #         # [batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
        #         y_node_linear = tf.matmul(y_n2npool, p_node_conv[gen_i])
        #
        #         if embeddingMethod == 0:  # 'structure2vec'算法
        #             # [node_cnt, embed_dim] + [node_cnt, embed_dim] = [node_cnt, embed_dim], return tensed matrix
        #             merged_linear = tf.add(node_linear, input_message)  # 元素相加 与节点本身表示相加  input_message相当于输入Xv
        #             # [node_cnt, embed_dim]
        #             cur_message_layer = tf.nn.relu(merged_linear)
        #
        #             # [batch_size, embed_dim] + [batch_size, embed_dim] = [batch_size, embed_dim], return tensed matrix
        #             y_merged_linear = tf.add(y_node_linear, y_input_message)
        #             # [batch_size, embed_dim]
        #             y_cur_message_layer = tf.nn.relu(y_merged_linear)
        #         else:  # 'graphsage'算法
        #             # [node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
        #             cur_message_layer_linear = tf.matmul(tf.cast(cur_message_layer, tf.float32),p_node_conv2[gen_i])  # AGG函数结果，即代表自身节点表达 hv(l-1)
        #
        #             # [[node_cnt, embed_dim] [node_cnt, embed_dim]] = [node_cnt, 2*embed_dim], return tensed matrix
        #             merged_linear = tf.concat([node_linear, cur_message_layer_linear],1)  # 按照第二个维度拼接 即按列拼 节点和邻居拼接 hv(l-1)拼接hN(v)
        #             # [node_cnt, 2*embed_dim]*[2*embed_dim, embed_dim] = [node_cnt, embed_dim]
        #             cur_message_layer = tf.nn.relu(tf.matmul(merged_linear, p_node_conv3[gen_i]))  # COM结果 乘权重Wl
        #
        #             # [batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
        #             y_cur_message_layer_linear = tf.matmul(tf.cast(y_cur_message_layer, tf.float32), p_node_conv2[gen_i])
        #             # [[batch_size, embed_dim] [batch_size, embed_dim]] = [batch_size, 2*embed_dim], return tensed matrix
        #             y_merged_linear = tf.concat([y_node_linear, y_cur_message_layer_linear], 1)
        #             # [batch_size, 2*embed_dim]*[2*embed_dim, embed_dim] = [batch_size, embed_dim]
        #             y_cur_message_layer = tf.nn.relu(tf.matmul(y_merged_linear, p_node_conv3[gen_i]))
        #
        #         cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)  # 标准化节点表示[node_cnt, embed_dim]
        #         y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)  # 标准化虚拟节点 [batch_size, embed_dim]
        #
        #     # self.node_embedding = cur_message_layer
        #     # [batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim], dense
        #     #  y_potential = tf.sparse_tensor_dense_matmul(tf.cast(self.subgsum_param,tf.float32), cur_message_layer)
        #     y_potential = y_cur_message_layer  # ？？？？？？？？？？？
        #     # [batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim] 是否限制每个batch至多选1个节点？？？
        #     action_embed = tf.sparse_tensor_dense_matmul(tf.cast(self.action_select, tf.float32),cur_message_layer)  # 根据action_select这个数据输入，选择动作，即删除节点的嵌入表示
        #
        #     #   embed_s_a = tf.concat([action_embed,y_potential],1)
        #
        #     # # [batch_size, embed_dim, embed_dim] za*zs
        #     temp = tf.matmul(tf.expand_dims(action_embed, axis=2), tf.expand_dims(y_potential,axis=1))  # 前一项将行向量 变为列向量？ 虽然多了一维度？ 后一个变为每一个batch都是一个一行embed列向量；多维矩阵相乘，除了最后两维，其他都相等才能算
        #     # # [batch_size, embed_dim]
        #     Shape = tf.shape(action_embed)  # 输出维度
        #     # # [batch_size, embed_dim], first transform 相当于每个batch对应的action嵌入向量乘一个与所处batch相关的一个常数（该数通过此batch对应的虚拟节点表示与一个随机向量的内积值）
        #     embed_s_a = tf.reshape(tf.matmul(temp, tf.reshape(tf.tile(cross_product[gen_i], [Shape[0], 1]), [Shape[0], Shape[1], 1])),Shape)  # 即每个temp对应的batch乘cross_product然后再转置 最后全部结果组合成一个shape规格的数据，其实就是每个batch emberd与crossproduct内积乘以actionselect的embed并用行向量表示，组合成一个矩阵
        #
        #     # [batch_size, embed_dim]
        #     last_output = embed_s_a
        #
        #     if self.reg_hidden > 0:  # 降维   之后就是ReLu    然后*W5 权重 得到Q
        #         # [batch_size, embed_dim] * [embed_dim, reg_hidden] = [batch_size, reg_hidden], dense
        #         hidden = tf.matmul(embed_s_a, h1_weight[gen_i])  # batch_size，reg_hidden
        #         # [batch_size, reg_hidden]
        #         last_output = tf.nn.relu(hidden)
        #
        #     # if reg_hidden == 0: ,[[batch_size, 2*embed_dim], [batch_size, aux_dim]] = [batch_size, 2*embed_dim+aux_dim]
        #     # if reg_hidden > 0: ,[[batch_size, reg_hidden], [batch_size, aux_dim]] = [batch_size, reg_hidden+aux_dim]
        #     last_output = tf.concat([last_output, self.aux_input], 1)  # aux_input代表什么？？？
        #     # if reg_hidden == 0: ,[batch_size, 2*embed_dim+aux_dim] * [2*embed_dim+aux_dim, 1] = [batch_size, 1]
        #     # if reg_hidden > 0: ,[batch_size, reg_hidden+aux_dim] * [reg_hidden+aux_dim, 1] = [batch_size, 1]
        #     q_pred = tf.matmul(last_output, last_w[gen_i])  # 每个batch中选择一个节点的对应Q
        #
        #     # loss1 = tf.reduce_sum(q_pred)
        #     ## first order reconstruction loss 重构损失，乘2是为了和下面edge数目多算一倍抵消，该loss的形式是yi（yi-求和yj...）   j是i的邻居  这项对每一维度求和&每一对相连节点求和
        #     loss_recons = tf.trace(tf.matmul(tf.transpose(cur_message_layer),tf.sparse_tensor_dense_matmul(tf.cast(self.laplacian_param, tf.float32),cur_message_layer)))  # trace 矩阵的迹
        #     # loss2 =
        #     edge_num = tf.sparse_reduce_sum(tf.cast(self.n2nsum_param_A, tf.float32))  # 不指定维度，计算所有元素总和
        #
        #     loss_recons = tf.divide(loss_recons, edge_num)  # 除以边数 计算平均的重构损失
        #     # loss2 = loss_recons
        #
        #     if self.IsPrioritizedSampling:
        #         self.TD_errors = tf.reduce_sum(tf.abs(self.target - q_pred),axis=1)  # for updating Sumtree    reduce后从二维变为一维数据 值不变
        #         if self.IsHuberloss:
        #             loss_rl = tf.losses.huber_loss(self.ISWeights * self.target, self.ISWeights * q_pred)
        #         else:
        #             loss_rl = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.target, q_pred))
        #     else:
        #         if self.IsHuberloss:
        #             loss_rl = tf.losses.huber_loss(self.target,q_pred)  # 根据target（目标，真实）-预测pred的每一项结果按huberloss的规则分段函数计算最后得到loss
        #         else:
        #             loss_rl = tf.losses.mean_squared_error(self.target, q_pred)  # 平方和取平均 即batch张图的结果取均值E
        #     # loss1 = loss_rl
        #     # loss2 = loss_recons
        #
        #     # l2_loss = self.weight_decay * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
        #     loss = loss_rl + Alpha * loss_recons  # Q-learning损失 和 重构损失
        #     # loss = loss + l2_loss
        #     # loss_tem2 = tf.constant([loss, 1], tf.float32)
        #     # loss_tem = tf.concat([loss_tem, tf.multiply(fu_1,loss_tem2)],0)
        #
        #     # loss_tem = tf.concat([loss_tem, fu_1 * loss], 0)
        #     loss_tem = tf.concat([loss_tem, fu_1*loss],1)

            # gen_fitness.extend(-1*loss)
        # top_vals, top_ind = tf.nn.top_k(tf.gather(loss_tem, indices=[map(lambda x:x**1),range(1,self.pop_size)]), k=self.pop_size)



        # top_vals, top_ind = tf.nn.top_k(tf.gather(loss_result,axis=0, indices=[0]),k=self.pop_size)
        top_vals, top_ind = tf.nn.top_k(tf.reshape(loss_result, [1,-1]), k=self.pop_size)
        # best_val = tf.reduce_max(top_vals)
        # best_ind = tf.argmax(top_vals, 0)
        best_val = top_vals[0][0]
        best_ind = top_ind[0][0]
        best_individual1 = tf.gather(w_n2l, best_ind)
        best_individual2 = tf.gather(p_node_conv, best_ind)
        best_individual3 = tf.gather(p_node_conv2, best_ind)
        best_individual4 = tf.gather(p_node_conv3, best_ind)
        best_individual5 = tf.gather(h1_weight, best_ind)
        best_individual6 = tf.gather(last_w, best_ind)
        best_individual7 = tf.gather(cross_product, best_ind)

        population_sorted1 = tf.gather(w_n2l, top_ind)
        population_sorted2 = tf.gather(p_node_conv, top_ind)
        population_sorted3 = tf.gather(p_node_conv2, top_ind)
        population_sorted4 = tf.gather(p_node_conv3, top_ind)
        population_sorted5 = tf.gather(h1_weight, top_ind)
        population_sorted6 = tf.gather(last_w, top_ind)
        population_sorted7 = tf.gather(cross_product, top_ind)

        parents1 = tf.slice(population_sorted1, [0, 0, 0, 0], [1, self.num_parents, 2, self.embedding_size])
        parents2 = tf.slice(population_sorted2, [0, 0, 0, 0], [1, self.num_parents, self.embedding_size, self.embedding_size])
        parents3 = tf.slice(population_sorted3, [0, 0, 0, 0], [1, self.num_parents, self.embedding_size, self.embedding_size])
        parents4 = tf.slice(population_sorted4, [0, 0, 0, 0], [1, self.num_parents, 2 * self.embedding_size, self.embedding_size])
        parents5 = tf.slice(population_sorted5, [0, 0, 0, 0], [1, self.num_parents, self.embedding_size, self.reg_hidden])
        parents6 = tf.slice(population_sorted6, [0, 0, 0, 0], [1, self.num_parents, self.reg_hidden + aux_dim, 1])
        parents7 = tf.slice(population_sorted7, [0, 0, 0, 0], [1, self.num_parents, self.embedding_size, 1])


        # 排序的序号
        rand_parent1_ix = np.random.choice(self.num_parents, self.num_children)
        rand_parent2_ix = np.random.choice(self.num_parents, self.num_children)

        rand1_parent1 = tf.gather(parents1[0], rand_parent1_ix)
        rand1_parent2 = tf.gather(parents1[0], rand_parent2_ix)
        rand2_parent1 = tf.gather(parents2[0], rand_parent1_ix)
        rand2_parent2 = tf.gather(parents2[0], rand_parent2_ix)
        rand3_parent1 = tf.gather(parents3[0], rand_parent1_ix)
        rand3_parent2 = tf.gather(parents3[0], rand_parent2_ix)
        rand4_parent1 = tf.gather(parents4[0], rand_parent1_ix)
        rand4_parent2 = tf.gather(parents4[0], rand_parent2_ix)
        rand5_parent1 = tf.gather(parents5[0], rand_parent1_ix)
        rand5_parent2 = tf.gather(parents5[0], rand_parent2_ix)
        rand6_parent1 = tf.gather(parents6[0], rand_parent1_ix)
        rand6_parent2 = tf.gather(parents6[0], rand_parent2_ix)
        rand7_parent1 = tf.gather(parents7[0], rand_parent1_ix)
        rand7_parent2 = tf.gather(parents7[0], rand_parent2_ix)

        rand1_parent1_sel = tf.multiply(rand1_parent1, self.crossover_mat_ph1)
        rand1_parent2_sel = tf.multiply(rand1_parent2, tf.subtract(1., self.crossover_mat_ph1))
        rand2_parent1_sel = tf.multiply(rand2_parent1, self.crossover_mat_ph2)
        rand2_parent2_sel = tf.multiply(rand2_parent2, tf.subtract(1., self.crossover_mat_ph2))
        rand3_parent1_sel = tf.multiply(rand3_parent1, self.crossover_mat_ph3)
        rand3_parent2_sel = tf.multiply(rand3_parent2, tf.subtract(1., self.crossover_mat_ph3))
        rand4_parent1_sel = tf.multiply(rand4_parent1, self.crossover_mat_ph4)
        rand4_parent2_sel = tf.multiply(rand4_parent2, tf.subtract(1., self.crossover_mat_ph4))
        rand5_parent1_sel = tf.multiply(rand5_parent1, self.crossover_mat_ph5)
        rand5_parent2_sel = tf.multiply(rand5_parent2, tf.subtract(1., self.crossover_mat_ph5))
        rand6_parent1_sel = tf.multiply(rand6_parent1, self.crossover_mat_ph6)
        rand6_parent2_sel = tf.multiply(rand6_parent2, tf.subtract(1., self.crossover_mat_ph6))
        rand7_parent1_sel = tf.multiply(rand7_parent1, self.crossover_mat_ph7)
        rand7_parent2_sel = tf.multiply(rand7_parent2, tf.subtract(1., self.crossover_mat_ph7))

        children1_after_sel = tf.add(rand1_parent1_sel, rand1_parent2_sel)
        children2_after_sel = tf.add(rand2_parent1_sel, rand2_parent2_sel)
        children3_after_sel = tf.add(rand3_parent1_sel, rand3_parent2_sel)
        children4_after_sel = tf.add(rand4_parent1_sel, rand4_parent2_sel)
        children5_after_sel = tf.add(rand5_parent1_sel, rand5_parent2_sel)
        children6_after_sel = tf.add(rand6_parent1_sel, rand6_parent2_sel)
        children7_after_sel = tf.add(rand7_parent1_sel, rand7_parent2_sel)

        mutated_children1 = tf.add(children1_after_sel, self.mutation_val_ph1)
        mutated_children2 = tf.add(children2_after_sel, self.mutation_val_ph2)
        mutated_children3 = tf.add(children3_after_sel, self.mutation_val_ph3)
        mutated_children4 = tf.add(children4_after_sel, self.mutation_val_ph4)
        mutated_children5 = tf.add(children5_after_sel, self.mutation_val_ph5)
        mutated_children6 = tf.add(children6_after_sel, self.mutation_val_ph6)
        mutated_children7 = tf.add(children7_after_sel, self.mutation_val_ph7)

        new_population1 = tf.concat([parents1[0], mutated_children1], 0)
        new_population2 = tf.concat([parents2[0], mutated_children2], 0)
        new_population3 = tf.concat([parents3[0], mutated_children3], 0)
        new_population4 = tf.concat([parents4[0], mutated_children4], 0)
        new_population5 = tf.concat([parents5[0], mutated_children5], 0)
        new_population6 = tf.concat([parents6[0], mutated_children6], 0)
        new_population7 = tf.concat([parents7[0], mutated_children7], 0)


        loss_result_f = tf.map_fn(loopcaculate, elems=(new_population1, new_population2, new_population3, new_population4, new_population5, new_population6, new_population7), dtype=tf.float32)


        # top_vals_f, top_ind_f = tf.nn.top_k(tf.gather(loss_result_f,axis=0, indices=[0]),k=self.pop_size)
        top_vals_f, top_ind_f = tf.nn.top_k(tf.reshape(loss_result_f, [1,-1]), k=self.pop_size)
        # best_val = tf.reduce_max(top_vals)
        # best_ind = tf.argmax(top_vals, 0)
        best_val_f = top_vals_f[0][0]
        best_ind_f = top_ind_f[0][0]

        trainStep = tf.train.AdamOptimizer(self.learning_rate).minimize(tf.gather(loss_result_f,axis=0, indices=[best_ind_f]))

        best_individual1_f = tf.gather(new_population1, best_ind_f)
        best_individual2_f = tf.gather(new_population2, best_ind_f)
        best_individual3_f = tf.gather(new_population3, best_ind_f)
        best_individual4_f = tf.gather(new_population4, best_ind_f)
        best_individual5_f = tf.gather(new_population5, best_ind_f)
        best_individual6_f = tf.gather(new_population6, best_ind_f)
        best_individual7_f = tf.gather(new_population7, best_ind_f)

        w_n2l_selection = tf.concat([w_n2l, new_population1, tf.expand_dims(best_individual1_f, axis=0)], 0)
        p_node_conv_selection = tf.concat([p_node_conv, new_population2, tf.expand_dims(best_individual2_f, axis=0)], 0)
        p_node_conv2_selection = tf.concat([p_node_conv2, new_population3, tf.expand_dims(best_individual3_f, axis=0)], 0)
        p_node_conv3_selection = tf.concat([p_node_conv3, new_population4, tf.expand_dims(best_individual4_f, axis=0)], 0)
        h1_weight_selection = tf.concat([h1_weight, new_population5, tf.expand_dims(best_individual5_f, axis=0)], 0)
        last_w_selection = tf.concat([last_w, new_population6, tf.expand_dims(best_individual6_f, axis=0)], 0)
        cross_product_selection = tf.concat([cross_product, new_population7, tf.expand_dims(best_individual7_f,axis=0)], 0)


        loss_result_selection = tf.map_fn(loopcaculate, elems=(w_n2l_selection, p_node_conv_selection, p_node_conv2_selection, p_node_conv3_selection, h1_weight_selection, last_w_selection, cross_product_selection), dtype=tf.float32)

        # top_vals_end, top_ind_end = tf.nn.top_k(tf.gather(loss_result_selection,axis=0, indices=[0]),k=self.pop_size)
        top_vals_end, top_ind_end = tf.nn.top_k(tf.reshape(loss_result_selection, [1,-1]), k=self.pop_size)
        # best_val = tf.reduce_max(top_vals)
        # best_ind = tf.argmax(top_vals, 0)
        best_val_end = top_vals_end[0][0]
        best_ind_end = top_ind_end[0][0]
        best_individual1_end = tf.gather(w_n2l_selection, best_ind_end)
        best_individual2_end = tf.gather(p_node_conv_selection, best_ind_end)
        best_individual3_end = tf.gather(p_node_conv2_selection, best_ind_end)
        best_individual4_end = tf.gather(p_node_conv3_selection, best_ind_end)
        best_individual5_end = tf.gather(h1_weight_selection, best_ind_end)
        best_individual6_end = tf.gather(last_w_selection, best_ind_end)
        best_individual7_end = tf.gather(cross_product_selection, best_ind_end)

        population_sorted1_front = tf.gather(w_n2l_selection, top_ind_end[0:int(self.pop_size/2)])
        population_sorted2_front = tf.gather(p_node_conv_selection, top_ind_end[0:int(self.pop_size/2)])
        population_sorted3_front = tf.gather(p_node_conv2_selection, top_ind_end[0:int(self.pop_size/2)])
        population_sorted4_front = tf.gather(p_node_conv3_selection, top_ind_end[0:int(self.pop_size/2)])
        population_sorted5_front = tf.gather(h1_weight_selection, top_ind_end[0:int(self.pop_size/2)])
        population_sorted6_front = tf.gather(last_w_selection, top_ind_end[0:int(self.pop_size/2)])
        population_sorted7_front = tf.gather(cross_product_selection, top_ind_end[0:int(self.pop_size/2)])

        back_ind_sorted = tf.random_shuffle(top_ind_end[int(self.pop_size/2):])

        population_sorted1_back = tf.gather(w_n2l_selection, back_ind_sorted[0:int(self.pop_size/2)])
        population_sorted2_back = tf.gather(p_node_conv_selection, back_ind_sorted[0:int(self.pop_size/2)])
        population_sorted3_back = tf.gather(p_node_conv2_selection, back_ind_sorted[0:int(self.pop_size/2)])
        population_sorted4_back = tf.gather(p_node_conv3_selection, back_ind_sorted[0:int(self.pop_size/2)])
        population_sorted5_back = tf.gather(h1_weight_selection, back_ind_sorted[0:int(self.pop_size/2)])
        population_sorted6_back = tf.gather(last_w_selection, back_ind_sorted[0:int(self.pop_size/2)])
        population_sorted7_back = tf.gather(cross_product_selection, back_ind_sorted[0:int(self.pop_size/2)])

        new_population1_end = tf.concat([population_sorted1_front, population_sorted1_back], 0)
        new_population2_end = tf.concat([population_sorted2_front, population_sorted2_back], 0)
        new_population3_end = tf.concat([population_sorted3_front, population_sorted3_back], 0)
        new_population4_end = tf.concat([population_sorted4_front, population_sorted4_back], 0)
        new_population5_end = tf.concat([population_sorted5_front, population_sorted5_back], 0)
        new_population6_end = tf.concat([population_sorted6_front, population_sorted6_back], 0)
        new_population7_end = tf.concat([population_sorted7_front, population_sorted7_back], 0)





        step = tf.group(w_n2l.assign(new_population1_end[0]), p_node_conv.assign(new_population2_end[0]), p_node_conv2.assign(new_population3_end[0]), p_node_conv3.assign(new_population4_end[0]),
                        h1_weight.assign(new_population5_end[0]), last_w.assign(new_population6_end[0]), cross_product.assign(new_population7_end[0]))







        # trainStep = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)  # 训练DQN
        #
        # # [node_cnt, batch_size] * [batch_size, embed_dim] = [node_cnt, embed_dim]
        # rep_y = tf.sparse_tensor_dense_matmul(tf.cast(self.rep_global, tf.float32),y_potential)  # 每个节点对应的batch的虚拟节点表示
        #
        # #  embed_s_a_all = tf.concat([cur_message_layer,rep_y],1)
        #
        # # # # [node_cnt, embed_dim, embed_dim]
        # temp1 = tf.matmul(tf.expand_dims(cur_message_layer, axis=2),tf.expand_dims(rep_y, axis=1))  # 计算每个s a对应的Q，每个节点对应的嵌入表示a，相对应的batch表示s
        # # # [node_cnt embed_dim]
        # Shape1 = tf.shape(cur_message_layer)
        # # # [node_cnt, embed_dim], first transform
        # embed_s_a_all = tf.reshape(tf.matmul(temp1, tf.reshape(tf.tile(cross_product, [Shape1[0], 1]), [Shape1[0], Shape1[1], 1])), Shape1)
        #
        # # [node_cnt, embed_dim]
        # last_output = embed_s_a_all
        # if self.reg_hidden > 0:
        #     # [node_cnt, embed_dim] * [embed_dim, reg_hidden] = [node_cnt, reg_hidden1]
        #     hidden = tf.matmul(embed_s_a_all, h1_weight)
        #     # Relu, [node_cnt, reg_hidden1]
        #     last_output = tf.nn.relu(hidden)
        #     # [node_cnt, reg_hidden1] * [reg_hidden1, reg_hidden2] = [node_cnt, reg_hidden2]
        #
        # # [node_cnt, batch_size] * [batch_size, aux_dim] = [node_cnt, aux_dim]
        # rep_aux = tf.sparse_tensor_dense_matmul(tf.cast(self.rep_global, tf.float32), self.aux_input)
        #
        # # if reg_hidden == 0: , [[node_cnt, 2 * embed_dim], [node_cnt, aux_dim]] = [node_cnt, 2*embed_dim + aux_dim]
        # # if reg_hidden > 0: , [[node_cnt, reg_hidden], [node_cnt, aux_dim]] = [node_cnt, reg_hidden + aux_dim]
        # last_output = tf.concat([last_output, rep_aux], 1)
        #
        # # if reg_hidden == 0: , [node_cnt, 2 * embed_dim + aux_dim] * [2 * embed_dim + aux_dim, 1] = [node_cnt，1]
        # # f reg_hidden > 0: , [node_cnt, reg_hidden + aux_dim] * [reg_hidden + aux_dim, 1] = [node_cnt，1]
        # q_on_all = tf.matmul(last_output, last_w)  # 每个节点在当前被删的Q值记录
        # # loss2 = tf.reduce_sum(q_on_all)
        # q_print = tf.print(q_on_all, [q_on_all], "q_on_all: ", summarize=50)

        # return loss, trainStep, q_pred, q_on_all, tf.trainable_variables(), nodes_size, q_print  # 损失，神经网络训练trainstep表示，q预测（根据action选择计算Q），q表（计算所有节点在当前被删后的Q），神经网络训练相关参数
        return loss_result, loss_result_f, loss_result_selection, trainStep, step, best_val_end, best_individual1_end, best_individual2_end, best_individual3_end, best_individual4_end, best_individual5_end, best_individual6_end, best_individual7_end


                                    #  individual1, individual2, individual3, individual4, individual5, individual6, individual7
    def BuildNet_alter_GA_DQN(self):
        # [2, embed_dim 64] 随机矩阵，即初始化节点的嵌入表示 W1
        w_n2l = tf.Variable(tf.reshape(self.individual1, shape= [2, self.embedding_size]), tf.float32)# trunncated_normal：截断产生正态分布随机数，随机数-均值>2*stddev（标准差） 则重新生成随机数，即范围【mean默认0 - 2 * stddev默认1, mean + 2 * stddev】
        # [embed_dim, embed_dim] 权重X2 节点相连邻居的聚合处理相关权重       W3
        p_node_conv = tf.Variable(tf.reshape(self.individual2, shape= [self.embedding_size, self.embedding_size]), tf.float32)#卷积层1 即初始卷积层
        if embeddingMethod == 1:    #'graphsage'方法
            # [embed_dim, embed_dim] 该算法下 节点自身表示的权重处理       W2
            p_node_conv2 = tf.Variable(tf.reshape(self.individual3, shape = [self.embedding_size, self.embedding_size]), tf.float32)
            # [2*embed_dim, embed_dim] 权重W^(l)
            p_node_conv3 = tf.Variable(tf.reshape(self.individual4, shape = [2*self.embedding_size, self.embedding_size]), tf.float32)

        #[reg_hidden+aux_dim, 1]
        if self.reg_hidden > 0: #降维 32
            #[2*embed_dim, reg_hidden]
           # h1_weight = tf.Variable(tf.truncated_normal([2 * self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            # [embed_dim 64, reg_hidden 32]
            h1_weight = tf.Variable(tf.reshape(self.individual5,shape = [self.embedding_size, self.reg_hidden]), tf.float32)
            #[reg_hidden1, reg_hidden2]
            # h2_weight = tf.Variable(tf.truncated_normal([self.reg_hidden1, self.reg_hidden2], stddev=initialization_stddev), tf.float32)
            #[reg_hidden+aux_dim 32+2, 1]
            h2_weight = tf.Variable(tf.reshape(self.individual6, shape = [self.reg_hidden + aux_dim, 1]), tf.float32) #32+4维？？
            #[reg_hidden2 + aux_dim 32+2, 1]
            last_w = h2_weight
        else: #reg_hidden<=0
            #[2*embed_dim, reg_hidden]
            h1_weight = tf.Variable(tf.truncated_normal([2 * self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            # [embed_dim, reg_hidden]
            # h1_weight = tf.Variable(tf.truncated_normal([self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            #[2*embed_dim, reg_hidden]
            last_w = h1_weight

        ## [embed_dim 64, 1]       W4
        cross_product = tf.Variable(tf.reshape(self.individual7, shape = [self.embedding_size, 1]), tf.float32) #内积？ 外积？


        #[node_cnt, 2]
        nodes_size = tf.shape(self.n2nsum_param)[0] #输出矩阵的维数的第一维 即节点数
        node_input = tf.ones((nodes_size,2)) #全1矩阵 Xv

        y_nodes_size = tf.shape(self.subgsum_param)[0] #即子图数目，亦即虚拟节点的表达实现过程
        # [batch_size, 2]
        y_node_input = tf.ones((y_nodes_size,2))


        #[node_cnt, 2] * [2, embed_dim] = [node_cnt, embed_dim]
        input_message = tf.matmul(tf.cast(node_input,tf.float32), w_n2l) #矩阵相乘 cast是转换数据类型，为float32 最后为【节点数，嵌入向量维数】 初始每个节点的嵌入表示，随机化

        #[node_cnt, embed_dim]  # no sparse  初始化节点嵌入表示
        input_potential_layer = tf.nn.relu(input_message) #将输入进行ReLu函数处理

        # # no sparse
        # [batch_size, embed_dim]
        y_input_message = tf.matmul(tf.cast(y_node_input,tf.float32), w_n2l)
        #[batch_size, embed_dim]  # no sparse 初始化虚拟节点表示 按子图batch来
        y_input_potential_layer = tf.nn.relu(y_input_message) #tf.nn 提供神经网络相关支持 比如卷积 池化



        #input_potential_layer = input_message 循环次数，即图嵌入表示算法循环的次数K
        cdef int lv = 0

        #[node_cnt, embed_dim], no sparse 标准化处理
        cur_message_layer = input_potential_layer
        cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1) #一种标准化处理，即每个行向量标准化

        #[batch_size, embed_dim], no sparse
        y_cur_message_layer = y_input_potential_layer
        # [batch_size, embed_dim] 标准化
        y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)

        while lv < max_bp_iter: #最大迭代 ：此处是3
            lv = lv + 1
            #[node_cnt, node_cnt] * [node_cnt, embed_dim] = [node_cnt, embed_dim], dense 按照边的记录矩阵n2nsum_param来聚合邻居节点信息
            n2npool = tf.sparse_tensor_dense_matmul(tf.cast(self.n2nsum_param,tf.float32), cur_message_layer) #稀疏矩阵与dense矩阵相乘，因为表达不同 一个是sparseplaceholder 一个是variable   这是聚集邻居节点表示 求和

            #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
            node_linear = tf.matmul(n2npool, p_node_conv) #卷积 相当于乘以权重W2 或者 AGG的权重部分 前者是structure2vec 算法  后者是graphSAGE算法

            # [batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim]
            y_n2npool = tf.sparse_tensor_dense_matmul(tf.cast(self.subgsum_param,tf.float32), cur_message_layer)
            #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
            y_node_linear = tf.matmul(y_n2npool, p_node_conv)


            if embeddingMethod == 0: # 'structure2vec'算法
                #[node_cnt, embed_dim] + [node_cnt, embed_dim] = [node_cnt, embed_dim], return tensed matrix
                merged_linear = tf.add(node_linear,input_message) #元素相加 与节点本身表示相加  input_message相当于输入Xv
                #[node_cnt, embed_dim]
                cur_message_layer = tf.nn.relu(merged_linear)

                #[batch_size, embed_dim] + [batch_size, embed_dim] = [batch_size, embed_dim], return tensed matrix
                y_merged_linear = tf.add(y_node_linear, y_input_message)
                #[batch_size, embed_dim]
                y_cur_message_layer = tf.nn.relu(y_merged_linear)
            else:   # 'graphsage'算法
                #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
                cur_message_layer_linear = tf.matmul(tf.cast(cur_message_layer, tf.float32), p_node_conv2) # AGG函数结果，即代表自身节点表达 hv(l-1)

                #[[node_cnt, embed_dim] [node_cnt, embed_dim]] = [node_cnt, 2*embed_dim], return tensed matrix
                merged_linear = tf.concat([node_linear, cur_message_layer_linear], 1) #按照第二个维度拼接 即按列拼 节点和邻居拼接 hv(l-1)拼接hN(v)
                #[node_cnt, 2*embed_dim]*[2*embed_dim, embed_dim] = [node_cnt, embed_dim]
                cur_message_layer = tf.nn.relu(tf.matmul(merged_linear, p_node_conv3)) #COM结果 乘权重Wl

                #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
                y_cur_message_layer_linear = tf.matmul(tf.cast(y_cur_message_layer, tf.float32), p_node_conv2)
                #[[batch_size, embed_dim] [batch_size, embed_dim]] = [batch_size, 2*embed_dim], return tensed matrix
                y_merged_linear = tf.concat([y_node_linear, y_cur_message_layer_linear], 1)
                #[batch_size, 2*embed_dim]*[2*embed_dim, embed_dim] = [batch_size, embed_dim]
                y_cur_message_layer = tf.nn.relu(tf.matmul(y_merged_linear, p_node_conv3))

            cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1) #标准化节点表示[node_cnt, embed_dim]
            y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1) #标准化虚拟节点 [batch_size, embed_dim]




        # self.node_embedding = cur_message_layer
        #[batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim], dense
      #  y_potential = tf.sparse_tensor_dense_matmul(tf.cast(self.subgsum_param,tf.float32), cur_message_layer)
        y_potential = y_cur_message_layer#？？？？？？？？？？？
        #[batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim] 是否限制每个batch至多选1个节点？？？
        action_embed = tf.sparse_tensor_dense_matmul(tf.cast(self.action_select, tf.float32), cur_message_layer) #根据action_select这个数据输入，选择动作，即删除节点的嵌入表示

     #   embed_s_a = tf.concat([action_embed,y_potential],1)

         # # [batch_size, embed_dim, embed_dim] za*zs
        temp = tf.matmul(tf.expand_dims(action_embed, axis=2),tf.expand_dims(y_potential, axis=1)) #前一项将行向量 变为列向量？ 虽然多了一维度？ 后一个变为每一个batch都是一个一行embed列向量；多维矩阵相乘，除了最后两维，其他都相等才能算
        # # [batch_size, embed_dim]
        Shape = tf.shape(action_embed) #输出维度
        # # [batch_size, embed_dim], first transform 相当于每个batch对应的action嵌入向量乘一个与所处batch相关的一个常数（该数通过此batch对应的虚拟节点表示与一个随机向量的内积值）
        embed_s_a = tf.reshape(tf.matmul(temp, tf.reshape(tf.tile(cross_product,[Shape[0],1]),[Shape[0],Shape[1],1])),Shape)#即每个temp对应的batch乘cross_product然后再转置 最后全部结果组合成一个shape规格的数据，其实就是每个batch emberd与crossproduct内积乘以actionselect的embed并用行向量表示，组合成一个矩阵

        #[batch_size, embed_dim]
        last_output = embed_s_a

        if self.reg_hidden > 0: #降维   之后就是ReLu    然后*W5 权重 得到Q
            #[batch_size, embed_dim] * [embed_dim, reg_hidden] = [batch_size, reg_hidden], dense
            hidden = tf.matmul(embed_s_a, h1_weight) #batch_size，reg_hidden
            #[batch_size, reg_hidden]
            last_output = tf.nn.relu(hidden)

        # if reg_hidden == 0: ,[[batch_size, 2*embed_dim], [batch_size, aux_dim]] = [batch_size, 2*embed_dim+aux_dim]
        # if reg_hidden > 0: ,[[batch_size, reg_hidden], [batch_size, aux_dim]] = [batch_size, reg_hidden+aux_dim]
        last_output = tf.concat([last_output, self.aux_input], 1) #aux_input代表什么？？？
        #if reg_hidden == 0: ,[batch_size, 2*embed_dim+aux_dim] * [2*embed_dim+aux_dim, 1] = [batch_size, 1]
        #if reg_hidden > 0: ,[batch_size, reg_hidden+aux_dim] * [reg_hidden+aux_dim, 1] = [batch_size, 1]
        q_pred = tf.matmul(last_output, last_w) #每个batch中选择一个节点的对应Q


        # loss1 = tf.reduce_sum(q_pred)
        ## first order reconstruction loss 重构损失，乘2是为了和下面edge数目多算一倍抵消，该loss的形式是yi（yi-求和yj...）   j是i的邻居  这项对每一维度求和&每一对相连节点求和
        loss_recons = tf.trace(tf.matmul(tf.transpose(cur_message_layer), tf.sparse_tensor_dense_matmul(tf.cast(self.laplacian_param,tf.float32), cur_message_layer))) #trace 矩阵的迹
        # loss2 =
        edge_num = tf.sparse_reduce_sum(tf.cast(self.n2nsum_param, tf.float32)) #不指定维度，计算所有元素总和

        loss_recons = tf.divide(loss_recons, edge_num) #除以边数 计算平均的重构损失
        # loss2 = loss_recons


        if self.IsPrioritizedSampling:
            self.TD_errors = tf.reduce_sum(tf.abs(self.target - q_pred), axis=1)    # for updating Sumtree    reduce后从二维变为一维数据 值不变
            if self.IsHuberloss:
                loss_rl = tf.losses.huber_loss(self.ISWeights * self.target, self.ISWeights * q_pred)
            else:
                loss_rl = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.target, q_pred))
        else:
            if self.IsHuberloss:
                loss_rl = tf.losses.huber_loss(self.target, q_pred) #根据target（目标，真实）-预测pred的每一项结果按huberloss的规则分段函数计算最后得到loss
            else:
                loss_rl = tf.losses.mean_squared_error(self.target, q_pred) #平方和取平均 即batch张图的结果取均值E
        # loss1 = loss_rl
        # loss2 = loss_recons

        l2_loss = self.weight_decay * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
        loss = loss_rl + Alpha * loss_recons #Q-learning损失 和 重构损失
        loss = loss + l2_loss

        trainStep = tf.train.AdamOptimizer(self.learning_rate).minimize(loss) #训练DQN

        #[node_cnt, batch_size] * [batch_size, embed_dim] = [node_cnt, embed_dim]
        rep_y = tf.sparse_tensor_dense_matmul(tf.cast(self.rep_global, tf.float32), y_potential) #每个节点对应的batch的虚拟节点表示

      #  embed_s_a_all = tf.concat([cur_message_layer,rep_y],1)

        # # # [node_cnt, embed_dim, embed_dim]
        temp1 = tf.matmul(tf.expand_dims(cur_message_layer, axis=2),tf.expand_dims(rep_y, axis=1))#计算每个s a对应的Q，每个节点对应的嵌入表示a，相对应的batch表示s
        # # [node_cnt embed_dim]
        Shape1 = tf.shape(cur_message_layer)
        # # [node_cnt, embed_dim], first transform
        embed_s_a_all = tf.reshape(tf.matmul(temp1, tf.reshape(tf.tile(cross_product,[Shape1[0],1]),[Shape1[0],Shape1[1],1])),Shape1)

        #[node_cnt, embed_dim]
        last_output = embed_s_a_all
        if self.reg_hidden > 0:
            #[node_cnt, embed_dim] * [embed_dim, reg_hidden] = [node_cnt, reg_hidden1]
            hidden = tf.matmul(embed_s_a_all, h1_weight)
            #Relu, [node_cnt, reg_hidden1]
            last_output = tf.nn.relu(hidden)
            #[node_cnt, reg_hidden1] * [reg_hidden1, reg_hidden2] = [node_cnt, reg_hidden2]

        #[node_cnt, batch_size] * [batch_size, aux_dim] = [node_cnt, aux_dim]
        rep_aux = tf.sparse_tensor_dense_matmul(tf.cast(self.rep_global, tf.float32), self.aux_input)

        #if reg_hidden == 0: , [[node_cnt, 2 * embed_dim], [node_cnt, aux_dim]] = [node_cnt, 2*embed_dim + aux_dim]
        #if reg_hidden > 0: , [[node_cnt, reg_hidden], [node_cnt, aux_dim]] = [node_cnt, reg_hidden + aux_dim]
        last_output = tf.concat([last_output,rep_aux],1)

        #if reg_hidden == 0: , [node_cnt, 2 * embed_dim + aux_dim] * [2 * embed_dim + aux_dim, 1] = [node_cnt，1]
        #f reg_hidden > 0: , [node_cnt, reg_hidden + aux_dim] * [reg_hidden + aux_dim, 1] = [node_cnt，1]
        q_on_all = tf.matmul(last_output, last_w) #每个节点在当前被删的Q值记录
        # loss2 = tf.reduce_sum(q_on_all)
        q_print = tf.print(q_on_all,[q_on_all],"q_on_all: ",summarize=50)

        return loss, trainStep, q_pred, q_on_all, tf.trainable_variables() , nodes_size, q_print#损失，神经网络训练trainstep表示，q预测（根据action选择计算Q），q表（计算所有节点在当前被删后的Q），神经网络训练相关参数






    def species_origin(self,population_size,chromosome_length):
        population = [[]]
        # 二维列表，包含染色体和基因
        for i in range(population_size):
            temporary = []
            # 染色体暂存器
            for j in range(chromosome_length):
                temporary.append(random.randint(0, 1))
                # 随机产生一个染色体,由二进制数组成
            population.append(temporary)
            # 将染色体添加到种群中
        return population[1:]
        # 将种群返回，种群是个二维数组，个体和染色体两维

    # 从二进制到十进制
    # input:种群,染色体长度
    def translation(self,population, chromosome_length):
        temporary = []
        for i in range(len(population)):
            total = 0
            for j in range(chromosome_length):
                total += population[i][j] * (math.pow(2, j))
                # 从第一个基因开始，每位对2求幂，再求和
                # 如：0101 转成十进制为：1 * 2^0 + 0 * 2^1 + 1 * 2^2 + 0 * 2^3 = 1 + 0 + 4 + 0 = 5
            temporary.append(total)
            # 一个染色体编码完成，由一个二进制数编码为一个十进制数
        return temporary

    # 返回种群中所有个体编码完成后的十进制数

    # 目标函数相当于环境 对染色体进行筛选，这里是2*sin(x)+cos(x)
    def function(self,population, chromosome_length, max_value):

        function1 = []
        temporary = self.translation(population, chromosome_length)
        # 暂存种群中的所有的染色体(十进制)
        for i in range(len(temporary)):
            x = temporary[i] * max_value / (math.pow(2, chromosome_length) - 1)
            # 一个基因代表一个决策变量，其算法是先转化成十进制，然后再除以2的基因个数次方减1(固定值)。
            function1.append(2 * math.sin(x) + math.cos(x))
            # 这里将2*sin(x)+cos(x)作为目标函数，也是适应度函数







    def gen_graph(self, num_min, num_max):  #gen生成图，无向图
        cdef int max_n = num_max #节点数最大限制
        cdef int min_n = num_min #节点数最小限制
        cdef int cur_n = np.random.randint(max_n - min_n + 1) + min_n #一个随机整数 范围[min,max+1)

        if self.g_type == 'erdos_renyi':
            g = nx.erdos_renyi_graph(n=cur_n, p=0.3) #n节点 概率p连接的ER网络
            # for u,v in g.edges:
            #     g.add_edge(u,v,weight=random.uniform(0,1))
        elif self.g_type == 'powerlaw':  #生成的图符合幂律分布？
            g = nx.powerlaw_cluster_graph(n=cur_n, m=4, p=0.05)#节点数n 每个节点随即连边m条，加入新边后增加一个三角形结构的概率为p
            # for u, v in g.edges:
            #     g.add_edge(u, v, weight=random.uniform(0, 1))
        elif self.g_type == 'small_world':
            g = nx.connected_watts_strogatz_graph(n=cur_n, k=4, p=0.5) #n节点 每个节点k邻居 随机化重连边概率p ws小世界网络
            # for u, v in g.edges:
            #     g.add_edge(u, v, weight=random.uniform(0, 1))
        elif self.g_type == 'barabasi_albert':
            g = nx.barabasi_albert_graph(n=cur_n, m=4) #n个节点 每次加入m边的 BA无标度网络
            # for u, v in g.edges:
            #     g.add_edge(u, v, weight=random.uniform(0, 1))
        # print('num of nodes:',g.number_of_nodes())
        # print('num of edges:',g.number_of_edges())
        # # a=list(g.nodes)
        # # b=[e for e in g.edges]
        # # print(a,' ;',b)
        # self.GenNetwork(g)


        return g #networkx的图 带权

    def gen_new_graphs(self, num_min, num_max):
        print('\ngenerating new training graphs...')
        sys.stdout.flush() #windows上一样 （针对print） 不影响，Linux上加入则会及时输出，不加则需要缓存满输出，比如一段程序运行结束后整体输出..

        self.ClearTrainGraphs()
        cdef int i
        for i in tqdm(range(500)): #插入1000张图
            g = self.gen_graph(num_min, num_max)

            self.InsertGraph(g, is_test=False)

    def ClearTrainGraphs(self):
        self.ngraph_train = 0
        self.TrainSet.Clear()

    def ClearTestGraphs(self):
        self.ngraph_test = 0
        self.TestSet.Clear()

    def InsertGraph(self,g,is_test):
        cdef int t
        if is_test:
            t = self.ngraph_test
            self.ngraph_test += 1
            self.TestSet.InsertGraph(t, self.GenNetwork(g)) #编号t，和py_graph GenNetwork返回根据networkx图g构造的 节点为g.node，边（from、 to）py_graph类型
        else:
            t = self.ngraph_train
            self.ngraph_train += 1
            self.TrainSet.InsertGraph(t, self.GenNetwork(g))

    def InsertGraph_prob(self,g,prob,is_test):
        cdef int t
        if is_test:
            t = self.ngraph_test
            self.ngraph_test += 1
            self.TestSet.InsertGraph(t, self.GenNetwork_prob(g,prob))  # 编号t，和py_graph GenNetwork返回根据networkx图g构造的 节点为g.node，边（from、 to）py_graph类型
        else:
            t = self.ngraph_train
            self.ngraph_train += 1
            self.TrainSet.InsertGraph(t, self.GenNetwork_prob(g,prob))

    def PrepareValidData(self): #预评估数据 R值
        print('\ngenerating validation graphs...') #验证网络 测试
        sys.stdout.flush()

        cdef double result_degree = 0.0 #度
        cdef double result_betweeness = 0.0 #介数
        for i in tqdm(range(n_valid)): #200次循环   按规模平均
            g = self.gen_graph(NUM_MIN, NUM_MAX) #30-50规模的网络
            self.InsertGraph(g,is_test=True)
            # self.TestSet.Get(i)

            # g_test = [self.TestSet.Get_value(i)]

            # g_test=graph.py_GSet()
            # test_test = mvc_env.py_MvcEnv(NUM_MAX)
            # test_test.s0(self.TestSet.Get(i))
            # g_test.append(test_test.graph)

            # print('111111111\n')
            g_degree = g.copy()
            g_betweenness = g.copy()
            val_degree, sol = self.HXA(g_degree, 'HDA')
            result_degree += val_degree #R值求和 200次
            # print('222222222\n')
            val_betweenness, sol = self.HXA(g_betweenness, 'HBA')
            result_betweeness += val_betweenness
            # print('333333333\n')
            # self.InsertGraph(g, is_test=True)         #与上面的激活概率设置不同 GenNetwork了两次/三次
        print ('Validation of HDA: %.6f'%(result_degree / n_valid)) #平均R值 平均了200次
        print ('Validation of HBA: %.6f'%(result_betweeness / n_valid))

    def Run_simulator(self, int num_seq, double eps, TrainSet, int n_step): #运行 模拟装置
        cdef int num_env = len(self.env_list) #环境数目
        cdef int n = 0
        cdef int i
        # pred_pred=[]
        while n < num_seq: #num_seq是之后一次取内存块中多少环境数     每一轮每一个环境都进行操作判断，符合要求的记录加入memory中，直到能够有num_seq个记录可取
            # print('*\n')
            for i in range(num_env): #一个环境里只有一个graph
                if self.env_list[i].graph.num_nodes == 0 or self.env_list[i].isTerminal(): #空图则重新sample一张图，非空图如果达到终态，即操作完毕，则保存到memory中同时重新sample一张图，非空图非终态时，则省去if里的操作
                    # print('=\n')
                    if self.env_list[i].graph.num_nodes > 0 and self.env_list[i].isTerminal():
                        if(self.env_list[i].k_num != 10):
                            n = n + 1 #记录内存中已有环境数目
                            # print('testtest\n')
                            self.nStepReplayMem.Add(self.env_list[i], n_step) #存入内存块中  n步DQN
                            #print ('add experience transition!')
                    g_sample= TrainSet.Sample() #随机取出trainset中的一张图graph
                    self.env_list[i].s0(g_sample) #加入到环境中  numCoveredEdges重新置0
                    self.g_list[i] = self.env_list[i].graph #同时graph也加入到g_list中
            if n >= num_seq:
                break
            pred_pred = [] #每一轮只选一个动作，只需要一轮扩散的节点判断结果
            Random = False
            if random.uniform(0,1) >= eps: #随意生成一个[0,1]之间的实数 e-贪婪策略
                pred = self.PredictWithCurrentQNet(self.g_list, [env.action_list for env in self.env_list]) #对应的真实Q值表
                # print('test1\n')
            else:
                Random = True

            for i in range(num_env): #此时环境中的图已经是新取样的  每一步都取随机数判断random
                if (Random):
                    a_t = self.env_list[i].randomAction() #一定概率随机选择动作，即随机选择被删节点
                    # print('test2\n')
                else:
                    # a_t = self.argMax(pred[i])
                    # print('test3\n')

                    pred_pred.append(self.env_list[i].decycling_dfs_action_list())
                    print("pred_pred: ",len(pred_pred[i]),"pred: ",len(pred[i]))
                    for j in range(len(pred_pred[i])):
                        if pred_pred[i][j] == 0:
                            pred[i][j] = -inf
                    a_t = self.argMax(pred[i])  # 子图i中最大Q值对应的节点编号
                    # print('test3\n')
                    # if len(self.env_list[i].action_list)==0:
                    #     a_t = self.argMax(pred[i])
                    # else:
                    #     pred_pred.append(self.env_list[i].influenceAction())
                    #     for j in range(len(pred_pred[i])):
                    #         if pred_pred[i][j] == 0:
                    #             pred[i][j] = -inf
                    #     a_t = self.argMax(pred[i]) #子图i中最大Q值对应的节点编号
                self.env_list[i].step(a_t)#环境中执行删除节点操作，修改相应的变量记录
                # print('test4\n')
                # print(n)
                # print('\n')
    #pass


#目前与原来的没有区别
    def Run_simulator_test(self, int num_seq, double eps, TrainSet, int n_step): #运行 模拟装置
        cdef int num_env = len(self.env_list) #环境数目
        cdef int n = 0
        cdef int i
        # pred_pred=[]
        while n < num_seq: #num_seq是之后一次取内存块中多少环境数     每一轮每一个环境都进行操作判断，符合要求的记录加入memory中，直到能够有num_seq个记录可取
            for i in range(num_env): #一个环境里只有一个graph
                if self.env_list[i].graph.num_nodes == 0 or self.env_list[i].isTerminal(): #空图则重新sample一张图，非空图如果达到终态，即操作完毕，则保存到memory中同时重新sample一张图，非空图非终态时，则省去if里的操作
                    if self.env_list[i].graph.num_nodes > 0 and self.env_list[i].isTerminal():
                        n = n + 1 #记录内存中已有环境数目
                        self.nStepReplayMem.Add(self.env_list[i], n_step) #存入内存块中  n步DQN
                        #print ('add experience transition!')
                    g_sample= TrainSet.Sample() #随机取出trainset中的一张图graph
                    self.env_list[i].s0(g_sample) #加入到环境中  numCoveredEdges重新置0
                    self.g_list[i] = self.env_list[i].graph #同时graph也加入到g_list中
            if n >= num_seq:
                break
            pred_pred = [] #每一轮只选一个动作，只需要一轮扩散的节点判断结果
            Random = False
            if random.uniform(0,1) >= eps: #随意生成一个[0,1]之间的实数 e-贪婪策略
                pred = self.PredictWithCurrentQNet(self.g_list, [env.action_list for env in self.env_list]) #对应的真实Q值表

            else:
                Random = True

            for i in range(num_env): #此时环境中的图已经是新取样的  每一步都取随机数判断random
                if (Random):
                    a_t = self.env_list[i].randomAction() #一定概率随机选择动作，即随机选择被删节点
                else:
                    if len(self.env_list[i].action_list)==0:
                        a_t = self.argMax(pred[i])
                    else:
                        pred_pred.append(self.env_list[i].decycling_dfs_action_list())
                        for j in range(len(pred_pred[i])):
                            if pred_pred[i][j] == 0:
                                pred[i][j] = -inf
                        a_t = self.argMax(pred[i]) #子图i中最大Q值对应的节点编号
                self.env_list[i].step(a_t)#环境中执行删除节点操作，修改相应的变量记录

            # result = self.PredictWithCurrentQNet(self.nodes_size, [env.action_list for env in self.env_list])
            # print('node[]==%d,'%result)

    #pass

    def PlayGame_test(self, int n_traj, double eps):  # 启动模拟 eps决定了动作随机选取的概率  n_traj是人工设置
        self.Run_simulator_test(n_traj, eps, self.TrainSet, N_STEP)







    def PlayGame(self,int n_traj, double eps): #启动模拟 eps决定了动作随机选取的概率  n_traj是人工设置
        self.Run_simulator(n_traj, eps, self.TrainSet, N_STEP)


    def SetupGA(self):
        sample = self.nStepReplayMem.Sampling(BATCH_SIZE)
        ness = False
        cdef int i, j, bsize
        cdef int n_graphs = len(sample.g_list)

        for i in range(BATCH_SIZE):
            if (not sample.list_term[i]):
                ness = True
                break
        if ness:  # 存在非终态记录时
            # double_pred = self.infulencespread_node(sample)
            list_pred = self.PredictWithSnapshot_GA(sample.g_list, sample.list_s_primes)
            # for i in range(len(list_pred)):
            #     for j in range(len(double_pred[i])):
            #         if (double_pred[i][j] == 0):
            #             list_pred[i][j] = -inf
            # list_pred_isnotdouble.append(np.multiply(np.array(double_pred[i]), np.array(list_pred_tem[i])).tolist())

        list_target = np.zeros([BATCH_SIZE, 1])

        for i in range(BATCH_SIZE):
            q_rhs = 0
            if (not sample.list_term[i]):
                q_rhs = GAMMA * self.Max(list_pred[i])
            q_rhs += sample.list_rt[i]  # 每张图的r+gamma*maxQ
            list_target[i] = q_rhs
            # list_target.append(q_rhs)
        # lac_var = []
        # def f(x):
        #     lac_var.append(x)
        #     return x



        for i in range(0,n_graphs,BATCH_SIZE):
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            # batch_idxes = []
            for j in range(i, i + bsize):
                batch_idxes[j-i] = j
                # batch_idxes.append(j)
            batch_idxes = np.int32(batch_idxes)

            self.SetupTrain(batch_idxes, sample.g_list, sample.list_st, sample.list_at,list_target)
            my_dict = {}
            my_dict[self.action_select] = self.inputs['action_select']
            my_dict[self.rep_global] = self.inputs['rep_global']
            my_dict[self.n2nsum_param] = self.inputs['n2nsum_param']
            # my_dict[self.n2nsum_param_A] = self.inputs['n2nsum_param_A']

            my_dict[self.laplacian_param] = self.inputs['laplacian_param']
            my_dict[self.subgsum_param] = self.inputs['subgsum_param']
        #    my_dict[self.subgsum_param_A] = self.inputs['subgsum_param_A']
            my_dict[self.aux_input] = np.array(self.inputs['aux_input'])
            my_dict[self.ISWeights] = np.mat(sample.ISWeights).T
            my_dict[self.target] = self.inputs['target']

            # result = self.session.run([self.loss,self.trainStep,self.q_pred],feed_dict=my_dict) #得到具体的loss和训练DQN网络trainstep
            # self.session.run(self.q_print, feed_dict=my_dict)
            # lac_var = []
            # [lac_var.append(k) for k in tf.global_variables()]
            # # tf.report_uninitialized_variables()
            # # need_init = tf.map_fn(f,elems=tf.expand_dims(tf.report_uninitialized_variables(), axis=0))
            # print(lac_var)
            # self.session.run(tf.variables_initializer(lac_var[46:]), feed_dict = my_dict) #115? 116?
            # print(self.session.run(tf.report_uninitialized_variables()))

        # crossover_mat1 = np.ones(shape=[self.generations, self.num_children, 2, self.embedding_size])
        # crossover_mat2 = np.ones(shape=[self.generations, self.num_children, self.embedding_size, self.embedding_size])
        # crossover_mat3 = np.ones(shape=[self.generations, self.num_children, self.embedding_size, self.embedding_size])
        # crossover_mat4 = np.ones(shape=[self.generations, self.num_children, 2 * self.embedding_size, self.embedding_size])
        # crossover_mat5 = np.ones(shape=[self.generations, self.num_children, self.embedding_size, self.reg_hidden])
        # crossover_mat6 = np.ones(shape=[self.generations, self.num_children, self.reg_hidden + aux_dim, 1])
        # crossover_mat7 = np.ones(shape=[self.generations, self.num_children, self.embedding_size, 1])
        #
        # mutation_values1 = np.random.normal(size=[self.generations, self.num_children, 2, self.embedding_size])
        # mutation_values2 = np.random.normal(size=[self.generations, self.num_children, self.embedding_size, self.embedding_size])
        # mutation_values3 = np.random.normal(size=[self.generations, self.num_children, self.embedding_size, self.embedding_size])
        # mutation_values4 = np.random.normal(size=[self.generations, self.num_children, 2 * self.embedding_size, self.embedding_size])
        # mutation_values5 = np.random.normal(size=[self.generations, self.num_children, self.embedding_size, self.reg_hidden])
        # mutation_values6 = np.random.normal(size=[self.generations, self.num_children, self.reg_hidden + aux_dim, 1])
        # mutation_values7 = np.random.normal(size=[self.generations, self.num_children, self.embedding_size, 1])




        # def generations_opration(x):
        for i in range(self.generations):
            # crossover
            # crossover_mat1_tem, crossover_mat2_tem, crossover_mat3_tem, crossover_mat4_tem, crossover_mat5_tem, crossover_mat6_tem, crossover_mat7_tem, mutation_values1_tem, mutation_values2_tem, mutation_values3_tem, mutation_values4_tem, mutation_values5_tem, mutation_values6_tem, mutation_values7_tem = x


            crossover_mat1 = np.ones(shape=[self.num_children, 2, self.embedding_size])
            crossover_mat2 = np.ones(shape=[self.num_children, self.embedding_size, self.embedding_size])
            crossover_mat3 = np.ones(shape=[self.num_children, self.embedding_size, self.embedding_size])
            crossover_mat4 = np.ones(shape=[self.num_children, 2 * self.embedding_size, self.embedding_size])
            crossover_mat5 = np.ones(shape=[self.num_children, self.embedding_size, self.reg_hidden])
            crossover_mat6 = np.ones(shape=[self.num_children, self.reg_hidden + aux_dim, 1])
            crossover_mat7 = np.ones(shape=[self.num_children, self.embedding_size, 1])

            crossover_point1 = np.random.choice(np.arange(1, self.embedding_size-1, step=1), self.num_children)
            for pop_ix in range(self.num_children):
                crossover_mat1[pop_ix, 0, 0:crossover_point1[pop_ix]] = 0
                crossover_mat1[pop_ix, 1, 0:crossover_point1[pop_ix]] = 0

            for num in range(self.num_children):
                crossover_point2 = np.random.choice(np.arange(1, self.embedding_size-1, step=1), self.embedding_size)
                for pop_ix in range(self.embedding_size):
                    crossover_mat2[num, pop_ix, 0:crossover_point2[pop_ix]] = 0

            for num in range(self.num_children):
                crossover_point3 = np.random.choice(np.arange(1, self.embedding_size-1, step=1), self.embedding_size)
                for pop_ix in range(self.embedding_size):
                    crossover_mat3[num, pop_ix, 0:crossover_point3[pop_ix]] = 0

            for num in range(self.num_children):
                crossover_point4 = np.random.choice(np.arange(1, self.embedding_size - 1, step=1), 2 * self.embedding_size)
                for pop_ix in range(2 * self.embedding_size):
                    crossover_mat4[num, pop_ix, 0:crossover_point4[pop_ix]] = 0

            for num in range(self.num_children):
                crossover_point5 = np.random.choice(np.arange(1, self.reg_hidden - 1, step=1), self.embedding_size)
                for pop_ix in range(self.embedding_size):
                    crossover_mat5[num, pop_ix, 0:crossover_point5[pop_ix]] = 0

            crossover_point6 = np.random.choice(np.arange(1, self.reg_hidden + aux_dim-1, step=1), self.num_children)
            for pop_ix in range(self.num_children):
                crossover_mat6[pop_ix, 0:crossover_point6[pop_ix], 0] = 0

            crossover_point7 = np.random.choice(np.arange(1, self.embedding_size-1, step=1), self.num_children)
            for pop_ix in range(self.num_children):
                crossover_mat7[pop_ix, 0:crossover_point7[pop_ix], 0] = 0

            # mutation
            mutation_prob_mat1 = np.random.uniform(size=[self.num_children, 2, self.embedding_size])
            mutation_prob_mat2 = np.random.uniform(size=[self.num_children, self.embedding_size, self.embedding_size])
            mutation_prob_mat3 = np.random.uniform(size=[self.num_children, self.embedding_size, self.embedding_size])
            mutation_prob_mat4 = np.random.uniform(size=[self.num_children, 2*self.embedding_size, self.embedding_size])
            mutation_prob_mat5 = np.random.uniform(size=[self.num_children, self.embedding_size, self.reg_hidden])
            mutation_prob_mat6 = np.random.uniform(size=[self.num_children, self.reg_hidden + aux_dim, 1])
            mutation_prob_mat7 = np.random.uniform(size=[self.num_children, self.embedding_size, 1])

            mutation_values1 = np.random.normal(size=[self.num_children, 2, self.embedding_size])
            mutation_values2 = np.random.normal(size=[self.num_children, self.embedding_size, self.embedding_size])
            mutation_values3 = np.random.normal(size=[self.num_children, self.embedding_size, self.embedding_size])
            mutation_values4 = np.random.normal(size=[self.num_children, 2 * self.embedding_size, self.embedding_size])
            mutation_values5 = np.random.normal(size=[self.num_children, self.embedding_size, self.reg_hidden])
            mutation_values6 = np.random.normal(size=[self.num_children, self.reg_hidden + aux_dim, 1])
            mutation_values7 = np.random.normal(size=[self.num_children, self.embedding_size, 1])



            mutation_values1[mutation_prob_mat1 <= self.mutation1] = 0
            mutation_values2[mutation_prob_mat2 <= self.mutation2] = 0
            mutation_values3[mutation_prob_mat3 <= self.mutation2] = 0
            mutation_values4[mutation_prob_mat4 <= self.mutation3] = 0
            mutation_values5[mutation_prob_mat5 <= self.mutation4] = 0
            mutation_values6[mutation_prob_mat6 <= self.mutation5] = 0
            mutation_values7[mutation_prob_mat7 <= self.mutation1] = 0

            feed_dict_v = {self.crossover_mat_ph1: crossover_mat1,
                         self.crossover_mat_ph2: crossover_mat2,
                         self.crossover_mat_ph3: crossover_mat3,
                         self.crossover_mat_ph4: crossover_mat4,
                         self.crossover_mat_ph5: crossover_mat5,
                         self.crossover_mat_ph6: crossover_mat6,
                         self.crossover_mat_ph7: crossover_mat7,
                         self.mutation_val_ph1: mutation_values1,
                         self.mutation_val_ph2: mutation_values2,
                         self.mutation_val_ph3: mutation_values3,
                         self.mutation_val_ph4: mutation_values4,
                         self.mutation_val_ph5: mutation_values5,
                         self.mutation_val_ph6: mutation_values6,
                         self.mutation_val_ph7: mutation_values7,
                         self.action_select: my_dict[self.action_select],
                         self.rep_global: my_dict[self.rep_global],
                         self.n2nsum_param: my_dict[self.n2nsum_param],
                         self.laplacian_param: my_dict[self.laplacian_param],
                         self.subgsum_param: my_dict[self.subgsum_param],
                         self.aux_input: my_dict[self.aux_input],
                         self.ISWeights: my_dict[self.ISWeights],
                         self.target: my_dict[self.target]}

            if i == 0:
                lac_var = []
                [lac_var.append(k) for k in tf.global_variables()]
                # tf.report_uninitialized_variables()
                # need_init = tf.map_fn(f,elems=tf.expand_dims(tf.report_uninitialized_variables(), axis=0))
                print(lac_var)
                self.session.run(tf.variables_initializer(lac_var[46:]), feed_dict=feed_dict_v)  # 115? 116?
                print(self.session.run(tf.report_uninitialized_variables()))


            self.session.run([self.loss_result, self.loss_result_f], feed_dict = feed_dict_v)
            # self.loss_result.run(feed_dict, session = self.session)
            # self.loss_result_f.run(feed_dict, session=self.session)
            self.trainStep_GA.run(feed_dict_v, session=self.session)
            self.session.run([self.loss_result_selection], feed_dict=feed_dict_v)
            # self.loss_result_selection.run(feed_dict, session=self.session)
            self.step.run(feed_dict_v, session=self.session)
            individual_all_result = self.session.run([self.individual1, self.individual2, self.individual3, self.individual4, self.individual5, self.individual6, self.individual7], feed_dict=feed_dict_v)
            # return individual_all_result

            # out_dtype = [np.tile([tf.float32] * self.embedding_size,(2,1)), np.tile([tf.float32] * self.embedding_size,(self.embedding_size,1)), np.tile([tf.float32] * self.embedding_size,(self.embedding_size,1)), np.tile([tf.float32] * self.embedding_size,(2*self.embedding_size,1)), np.tile([tf.float32] * self.reg_hidden,(2*self.embedding_size,1)),[tf.float32]*(self.reg_hidden + aux_dim).T, [tf.float32]*self.embedding_size.T]
            # individual_result = tf.map_fn(generations_opration, elems=(crossover_mat1, crossover_mat2, crossover_mat3, crossover_mat4, crossover_mat5, crossover_mat6, crossover_mat7, mutation_values1, mutation_values2, mutation_values3, mutation_values4, mutation_values5, mutation_values6, mutation_values7), dtype=out_dtype)

            # if i == self.generations-1:
            #     uninit_vars = []
            #     # 用 try & except 语句块捕获：
            #     for var in tf.global_variables():
            #         try:
            #             self.session.run(var)
            #         except tf.errors.FailedPreconditionError:
            #             uninit_vars.append(var)
            #     init_new_vars_op = tf.initialize_variables(uninit_vars,feed_dict={self.individual1:individual_all_result[0], self.individual2:individual_all_result[1], self.individual3:individual_all_result[2], self.individual4:individual_all_result[3], self.individual5:individual_all_result[4], self.individual6:individual_all_result[5], self.individual7:individual_all_result[6]})
            #     self.session.run(init_new_vars_op)






        ##



    def SetupTrain(self, idxes, g_list, covered, actions, target): #批量处理g_list
        self.m_y = target #新定义的变量m_y 目标Q值=r+gamma*maxQ
        self.inputs['target'] = self.m_y #target是什么 loss中的target?
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph(aggregatorID) #ID只影响N2N的元素计算，即邻接矩阵是否带权
        prepareBatchGraph.SetupTrain(idxes, g_list, covered, actions)

        # env_list_s = []
        # sol = []
        #
        # for i in range(0,len(g_list)): #num_env==1，i==0 则有一个DQN环境
        #     env_list_s.append(mvc_env.py_MvcEnv(NUM_MAX)) #norm == NUM_MAX
        #     # self.g_list.append(graph.py_Graph())
        #
        # # envgraph = mvc_env.py_MvcEnv(NUM_MAX)
        # for i in range(0,len(g_list)):
        #     env_list_s[i].s0(g_list[idxes[i]])
        #     for new_action in covered[idxes[i]]:
        #         if not env_list_s[i].isTerminal():
        #             env_list_s[i].stepWithoutReward(new_action)
        #     sol.append(env_list_s[i].node_act_flag)
        #
        # prepareBatchGraph.SetupTrain(idxes, g_list, covered, sol, actions)





        self.inputs['action_select'] = prepareBatchGraph.act_select #剩余子图num，剩余总节点num
        self.inputs['rep_global'] = prepareBatchGraph.rep_global #剩余总节点，剩余子图
        self.inputs['n2nsum_param'] = prepareBatchGraph.n2nsum_param #剩余图的A
        # self.inputs['n2nsum_param_A'] = prepareBatchGraph.n2nsum_param_A
        self.inputs['laplacian_param'] = prepareBatchGraph.laplacian_param #剩余图的D-A
        self.inputs['subgsum_param'] = prepareBatchGraph.subgsum_param #剩余图的 子图，节点 对应关系
      #  self.inputs['subgsum_param_A'] = prepareBatchGraph.subgsum_param_A
        self.inputs['aux_input'] = prepareBatchGraph.aux_feat #剩余图的一些测度（每个剩余子图一条记录）：【已删节点占总节点的比例 以删边占总边比例 twohop比例 1】


    def SetupPredAll(self, idxes, g_list, covered):
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph(aggregatorID)
        prepareBatchGraph.SetupPredAll(idxes, g_list, covered) #即actions==nullptr时

        # env_list_s = []
        # sol = []
        #
        # for i in range(0, len(g_list)):  # num_env==1，i==0 则有一个DQN环境
        #     env_list_s.append(mvc_env.py_MvcEnv(NUM_MAX))  # norm == NUM_MAX
        #     # self.g_list.append(graph.py_Graph())
        #
        # # envgraph = mvc_env.py_MvcEnv(NUM_MAX)
        # for i in range(0, len(g_list)):
        #     env_list_s[i].s0(g_list[idxes[i]])
        #     for new_action in covered[idxes[i]]:
        #         if not env_list_s[i].isTerminal():
        #             env_list_s[i].stepWithoutReward(new_action)
        #     sol.append(env_list_s[i].node_act_flag)
        # prepareBatchGraph.SetupPredAll(idxes, g_list, covered, sol)  # 即actions==nullptr时



        self.inputs['rep_global'] = prepareBatchGraph.rep_global #此时返回的是tf.sparsetensorvalue()
        self.inputs['n2nsum_param'] = prepareBatchGraph.n2nsum_param
        # self.inputs['n2nsum_param_A'] = prepareBatchGraph.n2nsum_param_A
        # self.inputs['laplacian_param'] = prepareBatchGraph.laplacian_param
        self.inputs['subgsum_param'] = prepareBatchGraph.subgsum_param
      #  self.inputs['subgsum_param_A'] = prepareBatchGraph.subgsum_param_A
        self.inputs['aux_input'] = prepareBatchGraph.aux_feat
        return prepareBatchGraph.idx_map_list #标记剩余图的节点还有哪些，按子图分开



#covered集合是已选 影响力节点 下一个选边际影响力最大的节点
    def Predict(self,g_list,covered,isSnapSnot): #issnapsnot是控制计算Q target 还是 Q pred 的标志 返回当前图的全部节点的Q表，以供动作选择
        cdef int n_graphs = len(g_list)
        cdef int i, j, k, bsize
        for i in range(0, n_graphs, BATCH_SIZE): #0到n_graph step==batch_size==64
            bsize = BATCH_SIZE #一组的数目，前几个都满足 都为batch_size个
            if (i + BATCH_SIZE) > n_graphs: #最后一组不足batch_size数目
                bsize = n_graphs - i #最后一组数目
            batch_idxes = np.zeros(bsize)#每组重新置零
            for j in range(i, i + bsize):
                batch_idxes[j - i] = j #编号
            batch_idxes = np.int32(batch_idxes) #根据g_list设立的图的编号

            idx_map_list = self.SetupPredAll(batch_idxes, g_list, covered)
            my_dict = {}
            my_dict[self.rep_global] = self.inputs['rep_global'] #将tf.sparsetensorvalue()与sparse_placeholder()通过dic字典对应起来
            my_dict[self.n2nsum_param] = self.inputs['n2nsum_param']
            # my_dict[self.n2nsum_param_A] = self.inputs['n2nsum_param_A']

            my_dict[self.subgsum_param] = self.inputs['subgsum_param']
        #    my_dict[self.subgsum_param_A] = self.inputs['subgsum_param_A']

            my_dict[self.aux_input] = np.array(self.inputs['aux_input']) #二维数组（矩阵）  与placeholder对应起来

            if isSnapSnot: #快照
                result = self.session.run([self.q_on_allT,self.nodes_size], feed_dict = my_dict) #init已经设置好session并且run初始化了
                self.session.run(self.q_print,feed_dict = my_dict)
            else:
                result = self.session.run([self.q_on_all,self.nodes_size], feed_dict = my_dict) #返回的result是[q_on_all] 这一list
                self.session.run(self.q_print, feed_dict=my_dict)
            raw_output = result[0] #结果 只含当前循环中的batch_size个图的结果 node_cnt==当前的数目  真实的每一个剩余节点的Q值
            # dist=0;
            # print('node=%d'%result[1])

            # for j in range(i, i + bsize):
            #     dist += g_list[j].num_nodes - len(covered[j])
            # print('raw_output==%d,dist=%d' % (len(raw_output), dist))
            # assert (dist == len(raw_output))

            pos = 0
            pred = []
            # dis=0; #测试
            for j in range(i, i + bsize): #遍历每个batch中的图
                idx_map = idx_map_list[j-i] #节点标记
                cur_pred = np.zeros(len(idx_map))
                for k in range(len(idx_map)): #遍历图的每个节点
                    if idx_map[k] < 0:
                        cur_pred[k] = -inf #没用的节点Q值取最小，因为目标是Q累计最大
                    else:
                        cur_pred[k] = raw_output[pos] #每一个剩余节点Q值
                        pos += 1
                for k in covered[j]: #遍历当前图的当前covered节点集，此处可以省略，因为上面的遍历过程包括了
                    cur_pred[k] = -inf
                pred.append(cur_pred) #[[],[],[]]的形式
                # print('cur_pred==%d,idx_map==%d,covered[]==%d,node==%d,pos_cur=%d'%(len(cur_pred),len(idx_map),len(covered[j]),g_list[j].num_nodes,pos))
                # dis+=g_list[j].num_nodes - len(covered[j])

            # print('pos==%d,raw_output==%d'%(pos,len(raw_output)))
            assert (pos == len(raw_output))
        return pred #每个子图的对应节点Q值结果记录，是不是对g_list数目有一定限制？ 否则会覆盖pred



    def Predict_GA(self,g_list,covered): #issnapsnot是控制计算Q target 还是 Q pred 的标志 返回当前图的全部节点的Q表，以供动作选择
        cdef int n_graphs = len(g_list)
        cdef int i, j, k, bsize
        for i in range(0, n_graphs, BATCH_SIZE): #0到n_graph step==batch_size==64
            bsize = BATCH_SIZE #一组的数目，前几个都满足 都为batch_size个
            if (i + BATCH_SIZE) > n_graphs: #最后一组不足batch_size数目
                bsize = n_graphs - i #最后一组数目
            batch_idxes = np.zeros(bsize)#每组重新置零
            for j in range(i, i + bsize):
                batch_idxes[j - i] = j #编号
            batch_idxes = np.int32(batch_idxes) #根据g_list设立的图的编号

            idx_map_list = self.SetupPredAll(batch_idxes, g_list, covered)
            my_dict = {}
            my_dict[self.rep_global] = self.inputs['rep_global'] #将tf.sparsetensorvalue()与sparse_placeholder()通过dic字典对应起来
            my_dict[self.n2nsum_param] = self.inputs['n2nsum_param']
            # my_dict[self.n2nsum_param_A] = self.inputs['n2nsum_param_A']

            my_dict[self.subgsum_param] = self.inputs['subgsum_param']
        #    my_dict[self.subgsum_param_A] = self.inputs['subgsum_param_A']

            my_dict[self.aux_input] = np.array(self.inputs['aux_input']) #二维数组（矩阵）  与placeholder对应起来


            result = self.session.run([self.q_on_all_tem,self.nodes_size_tem], feed_dict = my_dict) #返回的result是[q_on_all] 这一list
            self.session.run(self.q_print_tem, feed_dict=my_dict)
            raw_output = result[0] #结果 只含当前循环中的batch_size个图的结果 node_cnt==当前的数目  真实的每一个剩余节点的Q值
            # dist=0;
            # print('node=%d'%result[1])

            # for j in range(i, i + bsize):
            #     dist += g_list[j].num_nodes - len(covered[j])
            # print('raw_output==%d,dist=%d' % (len(raw_output), dist))
            # assert (dist == len(raw_output))

            pos = 0
            pred = []
            # dis=0; #测试
            for j in range(i, i + bsize): #遍历每个batch中的图
                idx_map = idx_map_list[j-i] #节点标记
                cur_pred = np.zeros(len(idx_map))
                for k in range(len(idx_map)): #遍历图的每个节点
                    if idx_map[k] < 0:
                        cur_pred[k] = -inf #没用的节点Q值取最小，因为目标是Q累计最大
                    else:
                        cur_pred[k] = raw_output[pos] #每一个剩余节点Q值
                        pos += 1
                for k in covered[j]: #遍历当前图的当前covered节点集，此处可以省略，因为上面的遍历过程包括了
                    cur_pred[k] = -inf
                pred.append(cur_pred) #[[],[],[]]的形式
                # print('cur_pred==%d,idx_map==%d,covered[]==%d,node==%d,pos_cur=%d'%(len(cur_pred),len(idx_map),len(covered[j]),g_list[j].num_nodes,pos))
                # dis+=g_list[j].num_nodes - len(covered[j])

            # print('pos==%d,raw_output==%d'%(pos,len(raw_output)))
            assert (pos == len(raw_output))
        return pred #每个子图的对应节点Q值结果记录，是不是对g_list数目有一定限制？ 否则会覆盖pred


    def PredictWithCurrentQNet(self,g_list,covered): #当前Q  pred Q表进行计算 每个g_list中的图对当前非covered集合节点进行Q_s_a计算
        result = self.Predict(g_list,covered,False)
        return result

    def PredictWithSnapshot(self,g_list,covered): #对target Q表进行预测 计算
        result = self.Predict(g_list,covered,True)
        return result

    def PredictWithSnapshot_GA(self,g_list,covered): #对target Q表进行预测 计算
        result = self.Predict_GA(g_list,covered)
        return result

    #pass
    def TakeSnapShot(self):
       self.session.run(self.UpdateTargetQNetwork) #执行赋值操作，即将Pred Q的相关参数的当前值赋给 Target Q中

    def infulencespread_node(self,sample):

        pred_pred = []
        for i in range(len(sample.g_list)):
            self.test_env2.s0(sample.g_list[i])
            # g_list.append(self.test_env2.graph)


            # if len(sample.list_st[i]) == 0:
            #     # a_t = self.argMax(pred[i])
            #     pred_pred.append([1 for k in range(self.test_env2.graph.num_nodes)])
            # else:
            #     for j in range(len(sample.list_s_primes[i])):
            #
            #         self.test_env2.stepWithoutReward(sample.list_s_primes[i][j])
            #
            #
            #     pred_pred.append(self.test_env2.influenceAction()) #此时的

            for j in range(len(sample.list_s_primes[i])):

                self.test_env2.stepWithoutReward(sample.list_s_primes[i][j])


            pred_pred.append(self.test_env2.influenceAction()) #此时的




        return pred_pred





    def Fit(self):  #计算loss值
        sample = self.nStepReplayMem.Sampling(BATCH_SIZE)
        ness = False
        cdef int i, j
        # double_list_pred_infl = []
        # double_list_predT_infl = []
        # list_pred_isnotdouble = []

        for i in range(BATCH_SIZE):
            if (not sample.list_term[i]):
                ness = True
                break
        if ness: #存在非终态记录时
            if self.IsDoubleDQN:
                double_list_pred = self.PredictWithCurrentQNet(sample.g_list, sample.list_s_primes)
                double_list_predT = self.PredictWithSnapshot(sample.g_list, sample.list_s_primes)
                # double_pred = self.infulencespread_node(sample)
                # for i in range(len(double_list_pred)):
                #     for j in range(len(double_pred[i])):
                #         if (double_pred[i][j]==0):
                #             double_list_pred[i][j] = -inf
                #             double_list_predT[i][j] = -inf
                   # double_list_pred_infl.append(np.multiply(np.array(double_pred[i]), np.array(double_list_pred[i])).tolist())
                   # double_list_predT_infl.append(np.multiply(np.array(double_pred[i]), np.array(double_list_predT[i])).tolist())

                list_pred = [a[self.argMax(b)] for a, b in zip(double_list_predT, double_list_pred)] #每个pred Q中最大的节点所对应的Target 中一样位置的Q值
            else:
                # double_pred = self.infulencespread_node(sample)
                list_pred = self.PredictWithSnapshot(sample.g_list, sample.list_s_primes)
                # for i in range(len(list_pred)):
                #     for j in range(len(double_pred[i])):
                #         if (double_pred[i][j] == 0):
                #             list_pred[i][j] = -inf
                    # list_pred_isnotdouble.append(np.multiply(np.array(double_pred[i]), np.array(list_pred_tem[i])).tolist())

        list_target = np.zeros([BATCH_SIZE, 1])


        for i in range(BATCH_SIZE):
            q_rhs = 0
            if (not sample.list_term[i]):
                if self.IsDoubleDQN:
                    q_rhs=GAMMA * list_pred[i]
                else:
                    q_rhs=GAMMA * self.Max(list_pred[i])
            q_rhs += sample.list_rt[i] #每张图的r+gamma*maxQ
            list_target[i] = q_rhs
            # list_target.append(q_rhs)
        if self.IsPrioritizedSampling:
            return self.fit_with_prioritized(sample.b_idx,sample.ISWeights,sample.g_list, sample.list_st, sample.list_at,list_target)
        else:
            return self.fit(sample.g_list, sample.list_st, sample.list_at,list_target)

    def fit_with_prioritized(self,tree_idx,ISWeights,g_list,covered,actions,list_target): #带权重（优先级）
        cdef double loss = 0.0
        cdef int n_graphs = len(g_list)
        cdef int i, j, bsize
        for i in range(0,n_graphs,BATCH_SIZE):
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            # batch_idxes = []
            for j in range(i, i + bsize):
                batch_idxes[j-i] = j
                # batch_idxes.append(j)
            batch_idxes = np.int32(batch_idxes)

            self.SetupTrain(batch_idxes, g_list, covered, actions,list_target)
            my_dict = {}
            my_dict[self.action_select] = self.inputs['action_select']
            my_dict[self.rep_global] = self.inputs['rep_global']
            my_dict[self.n2nsum_param] = self.inputs['n2nsum_param']
            # my_dict[self.n2nsum_param_A] = self.inputs['n2nsum_param_A']

            my_dict[self.laplacian_param] = self.inputs['laplacian_param']
            my_dict[self.subgsum_param] = self.inputs['subgsum_param']
        #    my_dict[self.subgsum_param_A] = self.inputs['subgsum_param_A']
            my_dict[self.aux_input] = np.array(self.inputs['aux_input'])
            my_dict[self.ISWeights] = np.mat(ISWeights).T
            my_dict[self.target] = self.inputs['target']

            result = self.session.run([self.trainStep,self.TD_errors,self.loss],feed_dict=my_dict)
            # self.session.run(self.q_print, feed_dict=my_dict)
            self.nStepReplayMem.batch_update(tree_idx, result[1])
            loss += result[2]*bsize
        return loss / len(g_list)


    def fit(self,g_list,covered,actions,list_target):
        cdef double loss = 0.0
        cdef int n_graphs = len(g_list)
        cdef int i, j, bsize
        for i in range(0,n_graphs,BATCH_SIZE):
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            # batch_idxes = []
            for j in range(i, i + bsize):
                batch_idxes[j-i] = j
                # batch_idxes.append(j)
            batch_idxes = np.int32(batch_idxes)

            self.SetupTrain(batch_idxes, g_list, covered, actions,list_target)
            my_dict = {}
            my_dict[self.action_select] = self.inputs['action_select']
            my_dict[self.rep_global] = self.inputs['rep_global']
            my_dict[self.n2nsum_param] = self.inputs['n2nsum_param']
            # my_dict[self.n2nsum_param_A] = self.inputs['n2nsum_param_A']

            my_dict[self.laplacian_param] = self.inputs['laplacian_param']
            my_dict[self.subgsum_param] = self.inputs['subgsum_param']
        #    my_dict[self.subgsum_param_A] = self.inputs['subgsum_param_A']
            my_dict[self.aux_input] = np.array(self.inputs['aux_input'])
            my_dict[self.target] = self.inputs['target']

            result = self.session.run([self.loss,self.trainStep,self.q_pred],feed_dict=my_dict) #得到具体的loss和训练DQN网络trainstep
            # self.session.run(self.q_print, feed_dict=my_dict)
            loss += result[0]*bsize
        # print("loss1 = %.6f loss2 = %.6f\n"%(result[2], result[3]))
        # print(self.inputs['target'])
        # print('\n')
        # print(result[2])
        # print('\n')

        # print(self.inputs['target']-result[2])
        # print('\n')
        # print("loss_1 = %.6f\n"%loss)
        # print("loss_2 = %.6f\n"%len(g_list))
        print("loss = %.6f\n"%(loss / len(g_list) ))
        return loss / len(g_list)   #损失的具体值


    def Train(self):
        try:
            self.PrepareValidData()
        except:
            print("prepare error")
        else:
            print("prepare success")
        # self.PrepareValidData() #测试200张测试图，可以输出平均的R值  测试图集200张
        self.gen_new_graphs(NUM_MIN, NUM_MAX) #每次训练构造一批新图 与测试的图规模一致   训练图集1000张图
        # print('one\n')
        cdef int i, iter, idx
        for i in range(10):
            try:
                self.PlayGame(50, 1)
            except:
                print("playgame error")
            else:
                print("placygame success")
            # self.PlayGame(10, 1) #训练图 并将记录加入memory中，重复了10次，每次50张，eps==1 最终有500张图存入memory 从TrainSet中抽取
            # print('two\n')
        # self.TakeSnapShot() #更新目标target Q
        # print('three\n')
        cdef double eps_start = 1.0
        cdef double eps_end = 0.05
        cdef double eps_step = 50000.0
        cdef int loss = 0
        cdef double frac, start, end, N_start, N_end

        #save_dir = './models/%s'%self.g_type
        save_dir = './models/Model_barabasi_albert'  #目前所在目录中  powerlaw类型网络机制模型

        if not os.path.exists(save_dir):
            os.mkdir(save_dir) #创建时上一级目录必须存在
        VCFile = '%s/ModelVC_%d_%d.csv'%(save_dir, NUM_MIN, NUM_MAX)
        f_out = open(VCFile, 'w')

        test_name = 'loss.txt'
        result_file = '%s/%s' % (save_dir, 'contrast' + test_name)  # 记录loss
        f_outloss = open(result_file,'w')


        self.SetupGA()
        for iter in range(MAX_ITERATION): #500000次迭代
            start = time.clock()  #windows上返回每次循环的消耗时间
            ###########-----------------------normal training data setup(start) -----------------##############################
            if iter and iter % 500 == 0: #每5000次重构图
                self.gen_new_graphs(NUM_MIN, NUM_MAX) #重建1000张新图
            eps = eps_end + max(0., (eps_start - eps_end) * (eps_step - iter) / eps_step)  #eps概率会变化，前10000次从1开始线性减小到0.05，之后不变都是0.05

            if iter % 10 == 0:
                self.PlayGame(10, eps) #每10次迭代后加10张图的结果到memory
            if iter % 300 == 0 and iter >= 200000:
                if(iter == 0):
                    N_start = start
                else:
                    N_start = N_end
                frac = 0.0
                test_start = time.time()
                for idx in range(n_valid):
                    frac += self.Test(idx) #每个测试图的R值Robustness
                test_end = time.time() #间隔时间记录
                f_out.write('%.16f\n'%(frac/n_valid))   #write vc into the file
                f_out.flush() # 强行把上面的write内容写入文件
                print('iter %d, eps %.4f, average size of vc:%.6f'%(iter, eps, frac/n_valid))
                print ('testing 200 graphs time: %.2fs'%(test_end-test_start))
                N_end = time.clock()
                print ('300 iterations total time: %.2fs\n'%(N_end-N_start)) #iter==0时也包括进来了？
                sys.stdout.flush() #间隔输出，不是一下全输出完毕 先将该句上面的执行过程中显示
                model_path = '%s/nrange_%d_%d_iter_%d.ckpt' % (save_dir, NUM_MIN, NUM_MAX, iter)
                self.SaveModel(model_path)
            if iter % UPDATE_TIME == 0: #每1000次更新target Q 参数
                self.TakeSnapShot()
            loss_store = self.Fit() #计算loss 同时会有更新参数的过程
            f_outloss.write('%.16f\n'%(loss_store))
            f_outloss.flush()

        f_out.close()
        f_outloss.close()

    def Train_load(self):
        self.LoadModel("./models/Model_barabasi_albert/nrange_5_10_iter_39000.ckpt")
        self.PrepareValidData() #测试200张测试图，可以输出平均的R值  测试图集200张
        self.gen_new_graphs(NUM_MIN, NUM_MAX) #每次训练构造一批新图 与测试的图规模一致   训练图集1000张图

        cdef int i, iter, idx
        for i in range(10):
            self.PlayGame(50, 1) #训练图 并将记录加入memory中，重复了10次，每次100张，eps==1 最终有1000张图存入memory 从TrainSet中抽取
        self.TakeSnapShot() #更新目标target Q
        cdef double eps_start = 1.0
        cdef double eps_end = 0.05
        cdef double eps_step = 50000.0
        cdef int loss = 0
        cdef double frac, start, end, N_start, N_end

        #save_dir = './models/%s'%self.g_type
        save_dir = './models/Model_barabasi_albert'  #目前所在目录中  powerlaw类型网络机制模型
        if not os.path.exists(save_dir):
            os.mkdir(save_dir) #创建时上一级目录必须存在
        VCFile = '%s/ModelVC_%d_%d.csv'%(save_dir, NUM_MIN, NUM_MAX)
        f_out = open(VCFile, 'w')

        test_name = 'loss.txt'
        result_file = '%s/%s' % (save_dir, 'contrast' + test_name)  # 记录loss
        f_outloss = open(result_file,'w')


        for iter in range(39001,MAX_ITERATION): #500000次迭代
            start = time.clock()  #windows上返回每次循环的消耗时间
            ###########-----------------------normal training data setup(start) -----------------##############################
            if iter and iter % 500 == 0: #每5000次重构图
                self.gen_new_graphs(NUM_MIN, NUM_MAX) #重建1000张新图
            eps = eps_end + max(0., (eps_start - eps_end) * (eps_step - iter) / eps_step)  #eps概率会变化，前10000次从1开始线性减小到0.05，之后不变都是0.05
            # eps = eps_end
            if iter % 10 == 0:
                self.PlayGame(10, eps) #每10次迭代后加10张图的结果到memory
            if iter % 300 == 0:
                if(iter == 39300):
                    N_start = start
                else:
                    N_start = N_end
                frac = 0.0
                test_start = time.time()
                for idx in range(n_valid):
                    frac += self.Test(idx) #每个测试图的R值Robustness
                test_end = time.time() #间隔时间记录
                f_out.write('%.16f\n'%(frac/n_valid))   #write vc into the file
                f_out.flush() # 强行把上面的write内容写入文件
                print('iter %d, eps %.4f, average size of vc:%.6f'%(iter, eps, frac/n_valid))
                print ('testing 200 graphs time: %.2fs'%(test_end-test_start))
                N_end = time.clock()
                print ('300 iterations total time: %.2fs\n'%(N_end-N_start)) #iter==0时也包括进来了？
                sys.stdout.flush() #间隔输出，不是一下全输出完毕 先将该句上面的执行过程中显示
                model_path = '%s/nrange_%d_%d_iter_%d.ckpt' % (save_dir, NUM_MIN, NUM_MAX, iter)
                self.SaveModel(model_path)
            if iter % UPDATE_TIME == 0: #每1000次更新target Q 参数
                self.TakeSnapShot()
            loss_store = self.Fit() #计算loss 同时会有更新参数的过程
            f_outloss.write('%.16f\n'%(loss_store))
            f_outloss.flush()
        f_out.close()
        f_outloss.close()

    def findModel(self):
        VCFile = './models/%s/ModelVC_%d_%d.csv'%(self.g_type, NUM_MIN, NUM_MAX)
        vc_list = []
        for line in open(VCFile):
            vc_list.append(float(line))

        start_loc = 33 #第34个元素开始 是csv文件的一些内置内容？
        max_vc = start_loc + np.argmax(vc_list[start_loc:])
        best_model_iter = 300 * max_vc #因为train中每300次写入一次当前结果
        best_model = './models/%s/nrange_%d_%d_iter_%d.ckpt' % (self.g_type, NUM_MIN, NUM_MAX, best_model_iter)
        return best_model

#一次测评结果
    def Evaluate1(self, g, save_dir, model_file=None): #评估1
        if model_file == None:  #if user do not specify the model_file
            model_file = self.findModel()
        print ('The best model is :%s'%(model_file))
        sys.stdout.flush()
        self.LoadModel(model_file)
        cdef double frac = 0.0
        cdef double frac_time = 0.0
        cdef double frac_cnr = 0.0
        cdef double frac_diff = 0.0
        result_file = '%s/test.csv' % (save_dir)
        with open(result_file, 'w') as f_out:
            print ('testing')
            sys.stdout.flush()
            self.InsertGraph(g, is_test=True)
            t1 = time.time()
            val, sol, CNR_r, diff, cnr_list= self.GetSol(0)
            t2 = time.time()
            for i in range(len(sol)):
                f_out.write(' %d\n' % sol[i])
            frac += val
            frac_cnr += CNR_r
            frac_diff += diff
            frac_time += (t2 - t1)
        print ('average size of vc: ', frac)
        print('average time: ', frac_time)
        print('average cnr: ', frac_cnr)
        print('average diff: ', frac_diff)
        # print('cnr_list: ',cnr_list)

#多次测评结果
    def Evaluate(self, data_test, model_file=None): #data_test是图数据集 的路径？   gml文件格式
        if model_file == None:  #if user do not specify the model_file
            model_file = self.findModel()
        print ('The best model is :%s'%(model_file))
        sys.stdout.flush()
        self.LoadModel(model_file)
        cdef int n_test = 100
        cdef int i
        result_list_score = []
        result_list_time = []
        result_list_CNR = []
        result_list_diff = []
        result_list_cnr_list = []
        sol_all = []
        sys.stdout.flush()
        for i in tqdm(range(n_test)):
            g_path = '%s/'%data_test + 'g_%d'%i
            g = nx.read_gml(g_path)
            self.InsertGraph(g, is_test=True) #在这之前test集没用进行clear处理？
            t1 = time.time()
            val, sol, CNR_r, diff, cnr_list = self.GetSol(i) #默认step==1 即取每次选取一个节点
            t2 = time.time()
            result_list_score.append(val)
            result_list_time.append(t2-t1)
            result_list_CNR.append(CNR_r)
            result_list_diff.append(diff)
            result_list_cnr_list.append(cnr_list)
            sol_all.append(sol)
        self.ClearTestGraphs()
        score_mean = np.mean(result_list_score)
        score_std = np.std(result_list_score)
        CNR_mean = np.mean(result_list_CNR)
        CNR_std = np.std(result_list_CNR)
        diff_mean = np.mean(result_list_diff)
        diff_std = np.std(result_list_diff)
        time_mean = np.mean(result_list_time)
        time_std = np.std(result_list_time)
        return  score_mean, score_std, time_mean, time_std, CNR_mean, CNR_std, diff_mean, diff_std, sol_all, result_list_cnr_list





    def EvaluateRealData_inputprob(self, model_file, data_test, data_prob, save_dir, stepRatio=0.0025):  #测试真实数据 data_test是什么
        cdef double solution_time = 0.0
        test_name = data_test.split('/')[-1] #以指定分隔符/分割data 再取最后的元素  '30-50' ？ 'g_0' ??

        test_prob_name = data_prob.split('/')[-1] #.csv

        save_dir_local = save_dir+'/StepRatio_%.4f'%stepRatio #路径下的一个新文件夹
        if not os.path.exists(save_dir_local):#make dir
            os.mkdir(save_dir_local)
        # prob_filename = 'prob'+ test_name #激活概率
        result_file = '%s/%s' %(save_dir_local, 'contrast'+test_name) #选择结果 对比
        prob_result_file = '%s/%s' %(save_dir_local, test_prob_name) #概率 已有

        g = nx.read_edgelist(data_test) #读取图信息 根据连边
        # g_test = []

        # prob = []

        # with open(prob_result_file, 'r') as f1_out:
        #     print('reading prob')
        #     sys.stdout.flush()
        #     print('number of edges:%d'%(nx.number_of_edges(g)))
        #
        #     probdatas = f1_out.readlines()
        #     for probdata in probdatas:
        #         prob.append(float(probdata));
        df = pd.read_csv(prob_result_file)
        df_test = np.array(df)
        df_test_data = df_test.tolist() #使用df_test_data[0]
        print('\n')
        print(df_test_data[0])
        print('\n')



        with open(result_file, 'w') as f_out:
            print ('testing')
            sys.stdout.flush()
            print ('number of nodes:%d'%(nx.number_of_nodes(g)))
            print ('number of edges:%d'%(nx.number_of_edges(g)))
            num_edge = nx.number_of_edges(g)
            if stepRatio > 0:
                step = np.max([int(stepRatio*nx.number_of_nodes(g)),1]) #step size
            else:
                step = 1
            self.InsertGraph_prob(g,df_test_data[0],is_test=True) #内含激活概率 后续改为区间..
            # g_test.append(self.TestSet.Get_value(0))
            t1 = time.time()
            solution = self.GetSolution(0,step) #内含每次激活判断概率
            t2 = time.time()
            solution_time = (t2 - t1)
            for i in range(len(solution)):
                f_out.write('%d\n' % solution[i])

        # with open(prob_result_file, 'w') as f1_out:
        #     print('writing prob')
        #     sys.stdout.flush()
        #     print('number of edges:%d'%(nx.number_of_edges(g)))
        #     # for i in range(len(g_test[0])):
        #     #     f1_out.write('%f\n' % g_test[0][i])
        #     g_test = [self.TestSet.Get_value(0)]
        #     for i in range(len(g_test[0]) - num_edge):
        #         f1_out.write('%f\n' % g_test[0][i+num_edge])



        self.ClearTestGraphs()
        return solution, solution_time, num_edge


    def EvaluateRealData(self, model_file, data_test, save_dir, stepRatio=0.0025):  #测试真实数据 data_test是什么
        cdef double solution_time = 0.0
        test_name = data_test.split('/')[-1] #以指定分隔符/分割data 再取最后的元素  '30-50' ？ 'g_0' ??

        save_dir_local = save_dir
        # save_dir_local = save_dir+'/StepRatio_%.4f'%stepRatio #路径下的一个新文件夹
        if not os.path.exists(save_dir_local):#make dir
            os.mkdir(save_dir_local)
        # prob_filename = 'prob'+ test_name #激活概率
        result_file = '%s/FINDER_GA_%s' %(save_dir_local, test_name) #选择结果
        # prob_result_file = '%s/%s' %(save_dir_local, prob_filename) #概率

        g = nx.read_edgelist(data_test) #读取图信息 根据连边
        # g_test = []
        with open(result_file, 'w') as f_out:
            print ('testing')
            sys.stdout.flush()
            print ('number of nodes:%d'%(nx.number_of_nodes(g)))
            print ('number of edges:%d'%(nx.number_of_edges(g)))
            num_edge = nx.number_of_edges(g)
            if stepRatio > 0:
                step = np.max([int(stepRatio*nx.number_of_nodes(g)),1]) #step size
            else:
                step = 1
            self.InsertGraph(g, is_test=True) #内含激活概率 后续改为区间..
            # g_test.append(self.TestSet.Get_value(0))
            t1 = time.time()
            solution, cnr = self.GetSolution(0,step) #内含每次激活判断概率
            t2 = time.time()
            if len(cnr) == 0:
                f_out.write('0.00\n')
            else:

                f_out.write('%.2f\n' % cnr[0])
            solution_time = (t2 - t1)
            for i in range(len(solution)):
                f_out.write('%d, %.2f\n' % (solution[i], cnr[i+1]))

        # with open(prob_result_file, 'w') as f1_out:
        #     print('writing prob')
        #     sys.stdout.flush()
        #     print('number of edges:%d'%(nx.number_of_edges(g)))
        #     # for i in range(len(g_test[0])):
        #     #     f1_out.write('%f\n' % g_test[0][i])
        #     g_test = [self.TestSet.Get_value(0)]
        #     print('\n')
        #     for i in range(len(g_test[0]) ):
        #         f1_out.write('%f\n' % g_test[0][i])
        #         print('%f '% g_test[0][i])
        #
        #     # for i in range(len(g_test[0]) - num_edge):
        #     #     f1_out.write('%f\n' % g_test[0][i+num_edge])
        #     #     print('%f '% g_test[0][i+num_edge])
        #     print('\n')


        self.ClearTestGraphs()
        return solution, solution_time, num_edge #, g_test[0]  # g_test[0][num_edge:]

    def GetSolution(self, int gid, int step = 1):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))  #Get一次重新Py_Graph了一张图，此时权重改变，这里需要修改
        g_list.append(self.test_env.graph)
        sol = []
        cnr = []
        start = time.time()
        cdef int iter = 0
        cdef int new_action
        sum_sort_time = 0
        flag = 0
        while (not self.test_env.isTerminal()):
            print ('Iteration:%d'%iter)
            iter += 1
            # pred_pred = []
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list]) #一张图
            start_time = time.time()
            batchSol = np.argsort(-list_pred[0])[:step] #从小到大的索引 取最大的Q值step个节点

            # if len(self.test_env.action_list) == 0:
            #     # new_action = self.argMax(list_pred[0])
            #     batchSol = np.argsort(-list_pred[0])[:step]
            # else:
            #     pred_pred.append(self.test_env.decyclingaction_list())
            #     for j in range(len(pred_pred[0])):
            #         if pred_pred[0][j] == 0:
            #             list_pred[0][j] = -inf
            #     # new_action = self.argMax(list_pred[0])  # 子图i中最大Q值对应的节点编号
            #     batchSol = np.argsort(-list_pred[0])[:step] #索引 最大step个节点

            end_time = time.time()
            sum_sort_time += (end_time-start_time)

            # rnum = 0
            for new_action in batchSol:
                if not self.test_env.isTerminal():
                    # self.test_env.stepWithoutReward(new_action)
                    if flag == 0:
                        # for tem in self.test_env.record_cycle:
                        #     if tem == 1:
                        #         rnum += 1
                        # cnr.append(rnum)
                        cnr.append(self.test_env.CNR)
                        # rnum = 0
                        flag = 1
                    self.test_env.step(new_action)
                    # for tem in self.test_env.record_cycle:
                    #     if tem == 1:
                    #         rnum += 1
                    # cnr.append(rnum)
                    cnr.append(self.test_env.CNR)
                    # rnum = 0
                    sol.append(new_action)
                else:
                    continue
        return sol, cnr





#真实数据集中使用
    def EvaluateSol(self, data_test, sol_file, strategyID, reInsertStep):
        sys.stdout.flush()
        g = nx.read_edgelist(data_test) #networkx类型    之后需要改 将prob加入networkx的图g 而不是py_Graph
        g_inner = self.GenNetwork(g) #py_graph类型
        print ('number of nodes:%d'%nx.number_of_nodes(g))
        print ('number of edges:%d'%nx.number_of_edges(g))
        nodes = list(range(nx.number_of_nodes(g)))
        sol = []
        for line in open(sol_file): #删除节点顺序 step内 从文件到sol中
            sol.append(int(line))
        print ('number of sol nodes:%d'%len(sol))
        sol_left = list(set(nodes)^set(sol)) #求集合的补集

        if strategyID > 0:  # 1，2，3 default
            start = time.time()
            if reInsertStep > 0 and reInsertStep < 1:
                step = np.max([int(reInsertStep*nx.number_of_nodes(g)),1]) #step size
            else:
                step = reInsertStep  #>=1 不取整？？？
            sol_reinsert = self.utils.reInsert(g_inner, sol, sol_left, strategyID, step)
            end = time.time()
            print ('reInsert time:%.6f'%(end-start))
        else:
            sol_reinsert = sol

        solution = sol_reinsert + sol_left
        print ('number of solution nodes:%d'%len(solution)) #全节点数？

        env_list_s = []

        env_list_s.append(mvc_env.py_MvcEnv(NUM_MAX))  # norm == NUM_MAX
        Robustness = 0.0
        env_list_s[0].s0(g_inner)
        for new_action in solution:
            if not env_list_s[0].isTerminal():
                env_list_s[0].step(new_action)
                # Robustness += -(env_list_s[0].CNR/(env_list_s[0].CNR_all*env_list_s[0].graph.num_nodes))
                # Robustness += -(env_list_s[0].CNR / env_list_s[0].CNR_all)
                Robustness += -env_list_s[0].CNR
            else:
                break

        # Robustness = env_list_s[0].decyclingratio() / env_list_s[0].cycle_node_all



        # Robustness = self.utils.getRobustnessInflu(g_inner, solution)
        # MaxCCList = self.utils.MaxInfluSzList
        MaxCCList = env_list_s[0].reward_seq
        return Robustness, MaxCCList


    def Test(self,int gid):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        cdef double cost = 0.0
        cdef int i
        sol = []
        pred_pred = []
        while (not self.test_env.isTerminal()):
            # print('--1\n')

            # cost += 1
            # pred_pred = []
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            # new_action = self.argMax(list_pred[0])

            # if len(self.test_env.action_list) == 0:



            # new_action = self.argMax(list_pred[0])



            # else:
            #     pred_pred.append(self.test_env.influenceAction())
            #     for j in range(len(pred_pred[0])):
            #         if pred_pred[0][j] == 0:
            #             list_pred[0][j] = -inf
            #     new_action = self.argMax(list_pred[0])  # 子图i中最大Q值对应的节点编号
            # new_action = self.argMax(list_pred[0])
            # self.test_env.stepWithoutReward(new_action)
            pred_pred.append(self.test_env.decycling_dfs_action_list())
            for j in range(len(pred_pred[0])):
                if pred_pred[0][j] == 0:
                    list_pred[0][j] = -inf
            new_action = self.argMax(list_pred[0])  # 子图i中最大Q值对应的节点编号




            self.test_env.step(new_action)
            sol.append(new_action)
        nodes = list(range(g_list[0].num_nodes))
        solution = sol + list(set(nodes)^set(sol))
        env_list_s = []

        env_list_s.append(mvc_env.py_MvcEnv(NUM_MAX))  # norm == NUM_MAX
        Robustness = 0.0
        env_list_s[0].s0(g_list[0])
        for new_action in solution:
            if not env_list_s[0].isTerminal():
                # env_list_s[0].stepWithoutReward(new_action)
                env_list_s[0].step(new_action)
                # Robustness += -(env_list_s[0].CNR/(env_list_s[0].CNR_all*env_list_s[0].graph.num_nodes))
                # Robustness += -(env_list_s[0].CNR / env_list_s[0].CNR_all)
                Robustness += -env_list_s[0].CNR
            else:
                break
        # Robustness = self.utils.getRobustnessInflu(g_list[0], solution)
        # Robustness = env_list_s[0].decyclingratio() / env_list_s[0].cycle_node_all




        return Robustness


    def GetSol(self, int gid, int step=1):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        cdef double cost = 0.0
        sol = []
        cdef int new_action
        while (not self.test_env.isTerminal()):
            pred_pred = []
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])

            # if len(self.test_env.action_list) == 0:
            #     # new_action = self.argMax(list_pred[0])
            #     batchSol = np.argsort(-list_pred[0])[:step]
            # else:
            #     pred_pred.append(self.test_env.decyclingaction_list())
            #     for j in range(len(pred_pred[0])):
            #         if pred_pred[0][j] == 0:
            #             list_pred[0][j] = -inf
            #     # new_action = self.argMax(list_pred[0])  # 子图i中最大Q值对应的节点编号
            #     batchSol = np.argsort(-list_pred[0])[:step]

            batchSol = np.argsort(-list_pred[0])[:step]
            for new_action in batchSol:
                if not self.test_env.isTerminal():
                    # self.test_env.stepWithoutReward(new_action)
                    self.test_env.step(new_action)
                    sol.append(new_action)
                else:
                    break
        nodes = list(range(g_list[0].num_nodes))
        solution = sol + list(set(nodes)^set(sol))

        env_list_s = []
        cnr_list = []

        env_list_s.append(mvc_env.py_MvcEnv(NUM_MAX))  # norm == NUM_MAX
        Robustness = 0.0
        env_list_s[0].s0(g_list[0])
        flag = 0
        diff = 0.0
        for new_action in solution:
            if not env_list_s[0].isTerminal():
                # env_list_s[0].stepWithoutReward(new_action)

                if flag == 0:
                    diff = env_list_s[0].CNR
                    cnr_list.append(env_list_s[0].CNR)
                    flag = 1

                env_list_s[0].step(new_action)
                # Robustness += -(env_list_s[0].CNR/(env_list_s[0].CNR_all*env_list_s[0].graph.num_nodes))
                # Robustness += -(env_list_s[0].CNR / env_list_s[0].CNR_all)
                Robustness += -env_list_s[0].CNR
            else:
                break
        cnr_list.append(env_list_s[0].CNR)
        CNR_result = env_list_s[0].CNR
        diff = CNR_result - diff
        # Robustness = self.utils.getRobustnessInflu(g_list[0], solution)
        # Robustness = env_list_s[0].decyclingratio() / env_list_s[0].cycle_node_all

        return Robustness, sol, CNR_result, diff, cnr_list


    def SaveModel(self,model_path): #存储模型model
        self.saver.save(self.session, model_path)
        print('model has been saved success!')

    def LoadModel(self,model_path):
        self.saver.restore(self.session, model_path)
        print('restore model from file successfully')

    def GenNetwork(self, g):    #networkx2four
        edges = g.edges()
        if len(edges) > 0:
            a, b = zip(*edges) #a是边的开始节点 b是边的结束节点
            A = np.array(a)
            B = np.array(b)
        else:
            A = np.array([0])
            B = np.array([0])
        # #测试
        # print(A)
        # print('\n')
        # print(B)
        return graph.py_Graph(len(g.nodes()), len(edges), A, B) # 没有给定权重，自动随机产生

    def GenNetwork_prob(self,g,prob):
        edges = g.edges()
        if len(edges) > 0:
            a, b = zip(*edges) #a是边的开始节点 b是边的结束节点
            A = np.array(a)
            B = np.array(b)
            prob_value = np.array(prob)
        else:
            A = np.array([0])
            B = np.array([0])
            prob_value = np.array([0])
        return graph.py_Graph(len(g.nodes()), len(edges), A, B, prob_value)


    def argMax(self, scores):
        cdef int n = len(scores)
        cdef int pos = -1
        cdef double best = -10000000
        cdef int i
        for i in range(n):
            if pos == -1 or scores[i] > best:
                pos = i
                best = scores[i]
        return pos


    def Max(self, scores):
        cdef int n = len(scores)
        cdef int pos = -1
        cdef double best = -10000000
        cdef int i
        for i in range(n):
            if pos == -1 or scores[i] > best:
                pos = i
                best = scores[i]
        return best


    def HXA(self, g, method):
        # 'HDA', 'HBA', 'HPRA', '' PR：PageRank
        sol = []
        # sol_k = []
        G = g.copy()
        while (nx.number_of_edges(G)>0):
            if method == 'HDA':
                dc = nx.degree_centrality(G) #返回dictionary
            elif method == 'HBA':
                dc = nx.betweenness_centrality(G)
            elif method == 'HCA':
                dc = nx.closeness_centrality(G)
            elif method == 'HPRA':
                dc = nx.pagerank(G)
            keys = list(dc.keys())
            values = list(dc.values())
            maxTag = np.argmax(values) #最大数的索引
            node = keys[maxTag]
            sol.append(int(node))
            G.remove_node(node) #对g有影响吗 无影响 移除后边对应删除 while判断时相关
        solution = sol + list(set(g.nodes())^set(sol)) #合并list  ^操作是补集， 此处得到的是除去最大度（介数）节点的剩余节点编号去重后的list与最大度节点list合并，即将最大度节点放到list的第一位
        solutions = [int(i) for i in solution]

        # #下面对K_num个最有影响力节点选出，评估影响力延展度
        # for i in range(K_num):
        #     sol_k.append(solutions[i])

        env_list_s = []
        env_list_s.append(mvc_env.py_MvcEnv(NUM_MAX)) #norm == NUM_MAX
        Robustness = 0.0

        env_list_s[0].s0(self.GenNetwork(g))

        for new_action in solutions:
            # print('```')
            if not env_list_s[0].isTerminal():
                # print('***')
                # env_list_s[0].stepWithoutReward(new_action)
                print(new_action)
                env_list_s[0].step(new_action)
                print('--------')
                # Robustness += -(env_list_s[0].CNR/(env_list_s[0].CNR_all*env_list_s[0].graph.num_nodes)) #取负数 为了最大化r  也就是最小化每次CNR
                # Robustness += -(env_list_s[0].CNR / env_list_s[0].CNR_all)
                Robustness += -env_list_s[0].CNR
                print(Robustness)
                # print(Robustness,'\n')
                # print('===')
            else:
                break
        # Robustness = env_list_s[0].getReward()

        # Robustness = env_list_s[0].decyclingratio()/env_list_s[0].cycle_node_all

        # Robustness = self.utils.getRobustnessInflu(self.GenNetwork_prob(g,g_test), solutions) #即python中的参数是 py_Graph list（vector转换来的） 得到R值
        return Robustness, sol #R值，sol是最大度节点排序


class contrast_tarjan:
    def __init__(self):
        self.env_list = []
        self.g_list = []
        for i in range(num_env): #num_env==1，i==0 则有一个DQN环境
            self.env_list.append(mvc_env.py_MvcEnv(NUM_MAX)) #norm == NUM_MAX
            self.g_list.append(graph.py_Graph())


    def GenNetwork(self, g):    #networkx2four
        edges = g.edges()
        if len(edges) > 0:
            a, b = zip(*edges) #a是边的开始节点 b是边的结束节点
            A = np.array(a)
            B = np.array(b)
        else:
            A = np.array([0])
            B = np.array([0])
        # #测试
        # print(A)
        # print('\n')
        # print(B)
        return graph.py_Graph(len(g.nodes()), len(edges), A, B) # 没有给定权重，自动随机产生

    # def mvc_test(self, g, id):
    #     print('test dl')
    #     self.env_list[0].s0(self.GenNetwork(g))
    #     self.g_list[0] = self.env_list[0].graph
    #     sol = []
    #     cnr = []
    #     save_dir = '../results/result_contrast_100/solutions_dl'
    #     if not os.path.exists(save_dir):
    #         os.mkdir(save_dir) #创建时上一级目录必须存在
    #     result_file = save_dir + '/dl_g_test_%d.txt' %id
    #     data_test_name = ['g_test_%d' % id]
    #     print("\n")
    #     t1 = time.time()
    #     while (self.env_list[0].isTerminal() != 1):
    #         print("\n")
    #         # print("terminal while [num] is :")
    #
    #         a_t = self.env_list[0].decycling_dfs_action_contrast()
    #         if (a_t == -1):
    #             break
    #         sol.append(a_t)
    #         print("reward(CNR) is:  ")
    #         print(self.env_list[0].CNR)
    #         cnr.append(self.env_list[0].CNR)
    #         print("\n")
    #         self.env_list[0].step(a_t)
    #         print("delete node is: ", a_t)
    #
    #         print("------------------")

    def test(self, g, id):
        print("start tarjan test model: ")
        print("g's edges: ")
        print(g.edges())
        self.env_list[0].s0(self.GenNetwork(g))
        self.g_list[0] = self.env_list[0].graph
        sol = []
        cnr = []
        save_dir = '../results/result_contrast_100/solutions_tarjan_400_500'
        # save_dir = '../results/result_contrast'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir) #创建时上一级目录必须存在
        result_file = save_dir + '/tarjan_g_test_%d.txt' %id
        data_test_name = ['g_test_%d' % id]
        # data_test_name = ['g_test_30_50']

        df = pd.DataFrame(np.arange(1 * len(data_test_name)).reshape((1, len(data_test_name))), index=['time'],columns=data_test_name)

        print("\n")
        t1 = time.time()
        while (self.env_list[0].isTerminal() != 1) :
            print("\n")
            # print("terminal while [num] is :")

            a_t = self.env_list[0].decycling_dfs_action_contrast()
            if(a_t == -1):
                break
            sol.append(a_t)
            print("reward(CNR) is:  ")
            print(self.env_list[0].CNR)
            cnr.append(self.env_list[0].CNR)
            print("\n")
            self.env_list[0].step(a_t)
            print("delete node is: ",a_t)

            print("------------------")


        print(self.env_list[0].CNR)
        cnr.append(self.env_list[0].CNR)
        t2 = time.time()
        print('\n')
        print("the sol is (selected node list):")
        for i in range(len(sol)):
            print(sol[i])
        solution_time = t2-t1
        # print("the time is %.4f"%(solution_time))
        print('Data:%s, time:%.2f' % (data_test_name[0], solution_time))
        df.iloc[0, 0] = solution_time
        save_dir_local = '../results/result_contrast_100/sol_times_tarjan_400_500'
        # save_dir_local = '../results/result_contrast'
        if not os.path.exists(save_dir_local):
            os.mkdir(save_dir_local)  # 创建时上一级目录必须存在
        df.to_csv(save_dir_local + '/sol_time_%d.csv' % id, encoding='utf-8', index=False)




        with open(result_file, 'w') as f_out:


            f_out.write('%.2f\n' % cnr[0])

            for i in range(len(sol)):
                f_out.write('%d, %.2f\n' % (sol[i], cnr[i+1]))


    def absolute_value(self, g, id): # need to ensure g has loop
        print("start absolute test model: ")
        print("g's edges: ")
        print(g.edges())
        self.env_list[0].s0(self.GenNetwork(g))
        self.g_list[0] = self.env_list[0].graph
        G = g.copy()
        sol = []
        cnr = []
        print("\n")
        save_dir = '../results/result_contrast_100/solutions_absolute_400_500'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)  # 创建时上一级目录必须存在
        result_file = save_dir + '/absolute_g_test_%d.txt' %id
        # result_file = '../results/result_contrast' + '/absolute_g_test_%d.txt' % id
        data_test_name = ['g_test_%d' % id]
        # data_test_name = ['g_test_30_50']
        df = pd.DataFrame(np.arange(1 * len(data_test_name)).reshape((1, len(data_test_name))), index=['time'],columns=data_test_name)
        t1 = time.time()
        while (self.env_list[0].isTerminal() != 1):
            print("\n")
            # print("terminal while [num] is :")
            sol_tem = []
            solution = list(set(g.nodes()) ^ set(sol))  # 合并list  ^操作是补集， 此处得到的是除去最大度（介数）节点的剩余节点编号去重后的list与最大度节点list合并，即将最大度节点放到list的第一位
            solutions = [int(i) for i in solution]
            for t in solutions:
                r_tem = self.env_list[0].step_fake(t)
                sol_tem.append(r_tem)
                self.env_list[0].step_fake_reverse(t)
            # a_t = solutions[sol_tem.index(max(sol_tem))]
            max_tem = max(sol_tem)
            max_node = []
            for i in range(len(sol_tem)):
                if sol_tem[i] == max_tem:
                    max_node.append(solutions[i])
            max_node_degree = dict(G.degree(max_node))
            degrees = [i for i in list(max_node_degree.values())]
            deg_to_node = [i for i in list(max_node_degree.keys())]
            a_t = deg_to_node[degrees.index(max(degrees))]







           # a_t = self.env_list[0].decycling_dfs_action_contrast()
            if (a_t == -1):
                break
            sol.append(a_t)
            print("reward(CNR) is:  ")
            print(self.env_list[0].CNR)
            cnr.append(self.env_list[0].CNR)
            print("\n")
            self.env_list[0].step(a_t)
            print("delete node is: ", a_t)

            print("------------------")

        print(self.env_list[0].CNR)
        cnr.append(self.env_list[0].CNR)
        t2 = time.time()
        print('\n')
        print("the sol is (selected node list):")

        for i in range(len(sol)):
            print(sol[i])
        solution_time = t2 - t1
        # print("the time is %.4f" % (solution_time))

        print('Data:%s, time:%.2f' % (data_test_name[0], solution_time))
        df.iloc[0, 0] = solution_time
        save_dir_local = '../results/result_contrast_100/sol_times_absolute_400_500'
        # save_dir_local = '../results/result_contrast'
        if not os.path.exists(save_dir_local):
            os.mkdir(save_dir_local)  # 创建时上一级目录必须存在
        df.to_csv(save_dir_local + '/sol_time_%d.csv' % id, encoding='utf-8', index=False)

        with open(result_file, 'w') as f_out:

            f_out.write('%.2f\n' % cnr[0])

            for i in range(len(sol)):
                f_out.write('%d, %.2f\n' % (sol[i], cnr[i+1]))

    def absolute_value_2_node(self, g, id): # need to ensure g has loop
        print("start absolute test model: ")
        print("g's edges: ")
        print(g.edges())
        self.env_list[0].s0(self.GenNetwork(g))
        self.g_list[0] = self.env_list[0].graph
        G = g.copy()
        sol = []
        cnr = []
        print("\n")
        save_dir = '../results/result_contrast_100/solutions_absolute_400_500'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)  # 创建时上一级目录必须存在
        result_file = save_dir + '/absolute_g_test_%d.txt' %id
        # result_file = '../results/result_contrast' + '/absolute_g_test_%d.txt' % id
        data_test_name = ['g_test_%d' % id]
        # data_test_name = ['g_test_30_50']
        df = pd.DataFrame(np.arange(1 * len(data_test_name)).reshape((1, len(data_test_name))), index=['time'],columns=data_test_name)
        t1 = time.time()
        while (self.env_list[0].isTerminal() != 1):
            print("\n")
            # print("terminal while [num] is :")
            sol_tem = []
            solution = list(set(g.nodes()) ^ set(sol))  # 合并list  ^操作是补集， 此处得到的是除去最大度（介数）节点的剩余节点编号去重后的list与最大度节点list合并，即将最大度节点放到list的第一位
            solutions = [int(i) for i in solution]
            for t in solutions:
                r_tem = self.env_list[0].step_fake(t)
                sol_tem.append(r_tem)
                self.env_list[0].step_fake_reverse(t)
            # a_t = solutions[sol_tem.index(max(sol_tem))]

            # max_tem = max(sol_tem)
            max_tem = sorted(sol_tem, reverse = True)
            max_node = []
            for i in range(len(sol_tem)):
                if len(max_tem)>2:
                    if sol_tem[i] == max_tem[0] or sol_tem[i] == max_tem[1]:
                        max_node.append(solutions[i])
            max_node_degree = dict(G.degree(max_node))

            max_item = sorted(max_node_degree.items(), key=lambda x: -x[1])
            sol_key = []
            for key,value in max_item:
                sol_key.append(int(key))
            if len(sol_key)>2:
                sol.append(sol_key[0])
                sol.append(sol_key[1])

            # degrees = [i for i in list(max_node_degree.values())]
            # deg_to_node = [i for i in list(max_node_degree.keys())]
            # a_t = deg_to_node[degrees.index(max(degrees))]




           # a_t = self.env_list[0].decycling_dfs_action_contrast()


            # if (a_t == -1):
            #     break
            # sol.append(a_t)
            print("reward(CNR) is:  ")
            print(self.env_list[0].CNR)
            cnr.append(self.env_list[0].CNR)
            print("\n")
            self.env_list[0].step(sol_key[0])
            print("delete node is: ", sol_key[0])
            self.env_list[0].step(sol_key[1])
            print("delete node is: ", sol_key[1])

            print("------------------")

        print(self.env_list[0].CNR)
        cnr.append(self.env_list[0].CNR)
        t2 = time.time()
        print('\n')
        print("the sol is (selected node list):")

        for i in range(len(sol)):
            print(sol[i])
        solution_time = t2 - t1
        # print("the time is %.4f" % (solution_time))

        print('Data:%s, time:%.2f' % (data_test_name[0], solution_time))
        df.iloc[0, 0] = solution_time
        save_dir_local = '../results/result_contrast_100/sol_times_absolute_400_500'
        # save_dir_local = '../results/result_contrast'
        if not os.path.exists(save_dir_local):
            os.mkdir(save_dir_local)  # 创建时上一级目录必须存在
        df.to_csv(save_dir_local + '/sol_time_%d.csv' % id, encoding='utf-8', index=False)

        with open(result_file, 'w') as f_out:

            f_out.write('%.2f\n' % cnr[0])

            for i in range(len(sol)):
                f_out.write('%d, %.2f\n' % (sol[i], cnr[i+1]))






    def Random_value(self, g, id):
        print("start random test model: ")
        print("g's edges: ")
        print(g.edges())
        self.env_list[0].s0(self.GenNetwork(g))
        self.g_list[0] = self.env_list[0].graph
        sol = []
        cnr = []
        data_test_name = ['g_test_%d' % id]
        # data_test_name = ['g_test_30_50']
        df = pd.DataFrame(np.arange(1 * len(data_test_name)).reshape((1, len(data_test_name))), index=['time'],columns=data_test_name)
        t1 = time.time()
        for i in range(len(g.nodes())):
            sol.append(i)
        random.shuffle(sol)

        print("\n")
        save_dir = '../results/result_contrast_100/solutions_random_400_500'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)  # 创建时上一级目录必须存在
        result_file = save_dir + '/random_g_test_%d.txt'%id
        # result_file = '../results/result_contrast' + '/random_g_test_%d.txt' % id
        ind = 0
        while (self.env_list[0].isTerminal() != 1):
            print("\n")

            a_t = sol[ind]
            ind = ind + 1


            print("reward(CNR) is:  ")
            print(self.env_list[0].CNR)
            cnr.append(self.env_list[0].CNR)
            print("\n")
            self.env_list[0].step(a_t)
            print("delete node is: ", a_t)

            print("------------------")

        print(self.env_list[0].CNR)
        cnr.append(self.env_list[0].CNR)
        t2 = time.time()
        print('\n')
        print("the sol is (selected node list):")

        for i in range(ind):
            print(sol[i])

        solution_time = t2 - t1
        # print("the time is %.4f" % (solution_time))
        print('Data:%s, time:%.2f' % (data_test_name[0], solution_time))
        df.iloc[0, 0] = solution_time
        save_dir_local = '../results/result_contrast_100/sol_times_random_400_500'
        # save_dir_local = '../results/result_contrast'
        if not os.path.exists(save_dir_local):
            os.mkdir(save_dir_local)  # 创建时上一级目录必须存在
        df.to_csv(save_dir_local + '/sol_time_%d.csv' % id, encoding='utf-8', index=False)

        with open(result_file, 'w') as f_out:

            f_out.write('%.2f\n' % cnr[0])

            for i in range(ind):
                f_out.write('%d, %.2f\n' % (sol[i], cnr[i+1]))


    def HXA_value(self, g, method, id):
        # 'HDA', 'HBA', 'HPRA', '' PR：PageRank
        sol = []
        # sol_k = []
        G = g.copy()
        data_test_name = ['g_test_%d' % id]
        # data_test_name = ['g_test_30_50']
        df = pd.DataFrame(np.arange(1 * len(data_test_name)).reshape((1, len(data_test_name))), index=['time'],columns=data_test_name)
        t1 = time.time()
        while (nx.number_of_edges(G) > 0):
            if method == 'HDA':
                dc = nx.degree_centrality(G)  # 返回dictionary
            elif method == 'HBA':
                dc = nx.betweenness_centrality(G)
            elif method == 'HCA':
                dc = nx.closeness_centrality(G)
            elif method == 'HPRA':
                dc = nx.pagerank(G)
            keys = list(dc.keys())
            values = list(dc.values())
            maxTag = np.argmax(values)  # 最大数的索引
            node = keys[maxTag]
            sol.append(int(node))
            G.remove_node(node)  # 对g有影响吗 无影响 移除后边对应删除 while判断时相关
        # solution = sol + list(set(g.nodes()) ^ set(sol))  # 合并list  ^操作是补集， 此处得到的是除去最大度（介数）节点的剩余节点编号去重后的list与最大度节点list合并，即将最大度节点放到list的第一位
        # solutions = [int(i) for i in solution]


        print("start HXA test model: ")
        print("g's edges: ")
        print(g.edges())
        self.env_list[0].s0(self.GenNetwork(g))
        self.g_list[0] = self.env_list[0].graph
        cnr = []

        print("\n")
        if method == 'HDA':
            save_dir = '../results/result_contrast_100/solutions_HDA_400_500'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)  # 创建时上一级目录必须存在
            result_file = save_dir + '/HDA_g_test_%d.txt'%id
            save_dir_local = '../results/result_contrast_100/sol_times_HDA_400_500'
            if not os.path.exists(save_dir_local):
                os.mkdir(save_dir_local)  # 创建时上一级目录必须存在
            # result_file = '../results/result_contrast' + '/HDA_g_test_%d.txt'%id
            # save_dir_local = '../results/result_contrast'
        elif method == 'HBA':
            save_dir = '../results/result_contrast_100/solutions_HBA_400_500'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)  # 创建时上一级目录必须存在
            result_file = save_dir + '/HBA_g_test_%d.txt'%id
            save_dir_local = '../results/result_contrast_100/sol_times_HBA_400_500'
            if not os.path.exists(save_dir_local):
                os.mkdir(save_dir_local)  # 创建时上一级目录必须存在
            # result_file = '../results/result_contrast' + '/HBA_g_test_%d.txt' % id
            # save_dir_local = '../results/result_contrast'
        elif method == 'HCA':
            save_dir = '../results/result_contrast_100/solutions_HCA_400_500'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)  # 创建时上一级目录必须存在
            result_file = save_dir + '/HCA_g_test_%d.txt'%id
            save_dir_local = '../results/result_contrast_100/sol_times_HCA_400_500'
            if not os.path.exists(save_dir_local):
                os.mkdir(save_dir_local)  # 创建时上一级目录必须存在
            # result_file = '../results/result_contrast' + '/HCA_g_test_%d.txt' % id
            # save_dir_local = '../results/result_contrast'
        elif method == 'HPRA':
            save_dir = '../results/result_contrast_100/solutions_HPRA_400_500'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)  # 创建时上一级目录必须存在
            result_file = save_dir + '/HPRA_g_test_%d.txt'%id
            save_dir_local = '../results/result_contrast_100/sol_times_HPRA_400_500'
            if not os.path.exists(save_dir_local):
                os.mkdir(save_dir_local)  # 创建时上一级目录必须存在
            # result_file = '../results/result_contrast' + '/HPRA_g_test_%d.txt' % id
            # save_dir_local = '../results/result_contrast'


        ind = 0
        while (self.env_list[0].isTerminal() != 1):
            print("\n")

            a_t = sol[ind]
            ind = ind + 1

            print("reward(CNR) is:  ")
            print(self.env_list[0].CNR)
            cnr.append(self.env_list[0].CNR)
            print("\n")
            self.env_list[0].step(a_t)
            print("delete node is: ", a_t)

            print("------------------")

        print(self.env_list[0].CNR)
        cnr.append(self.env_list[0].CNR)
        t2 = time.time()
        print('\n')
        print("the sol is (selected node list):")

        for i in range(ind):
            print(sol[i])

        solution_time = t2 - t1
        # print("the time is %.4f" % (solution_time))
        print('Data:%s, time:%.2f' % (data_test_name[0], solution_time))
        df.iloc[0, 0] = solution_time


        df.to_csv(save_dir_local + '/sol_time_%d.csv' %id, encoding='utf-8', index=False)


        with open(result_file, 'w') as f_out:

            f_out.write('%.2f\n' % cnr[0])

            for i in range(ind):
                f_out.write('%d, %.2f\n' % (sol[i], cnr[i + 1]))

    def solution_to_cnr(self, g, sol, id):
        print("start FINDER original test model: ")
        print("g's edges: ")
        print(g.edges())
        self.env_list[0].s0(self.GenNetwork(g))
        self.g_list[0] = self.env_list[0].graph
        cnr = []
        save_dir = '../results/result_contrast_100/solutions_FINDER_O_400_500'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)  # 创建时上一级目录必须存在
        result_file = save_dir + '/FINDER_original_g_test_%d.txt'%id
        data_test_name = ['g_test_%d' % id]
        df = pd.DataFrame(np.arange(1 * len(data_test_name)).reshape((1, len(data_test_name))), index=['time'],columns=data_test_name)

        ind = 0
        len_sol = len(sol)
        t1 = time.time()
        while (self.env_list[0].isTerminal() != 1):
            print("\n")
            if len_sol==ind:
                break

            a_t = sol[ind]
            ind = ind + 1

            print("reward(CNR) is:  ")
            print(self.env_list[0].CNR)
            cnr.append(self.env_list[0].CNR)
            print("\n")
            self.env_list[0].step(a_t)
            print("delete node is: ", a_t)

            print("------------------")

        print(self.env_list[0].CNR)
        cnr.append(self.env_list[0].CNR)
        t2 = time.time()
        print('\n')
        print("the sol is (selected node list):")

        for i in range(ind):
            print(sol[i])

        solution_time = t2 - t1
        # print("the time is %.4f" % (solution_time))
        print('Data:%s, time:%.2f' % (data_test_name[0], solution_time))
        df.iloc[0, 0] = solution_time
        save_dir_local = '../results/result_contrast_100/sol_times_FINDER_O_400_500'
        if not os.path.exists(save_dir_local):
            os.mkdir(save_dir_local)  # 创建时上一级目录必须存在
        df.to_csv(save_dir_local + '/sol_time_%d.csv' % id, encoding='utf-8', index=False)

        with open(result_file, 'w') as f_out:

            f_out.write('%.2f\n' % cnr[0])

            for i in range(ind):
                f_out.write('%d, %.2f\n' % (sol[i], cnr[i + 1]))




    def cal_core(self,g, sol):
        G = nx.Graph()
        G.add_nodes_from([int(i) for i in g.nodes()])
        G.add_edges_from([(int(u), int(v)) for u, v in g.edges()])
        slen = len(sol)
        if slen==0:
            flag = 1
            while(flag == 1):
                deg = dict(nx.degree(G))
                a = sorted(deg.items(), key=lambda x: x[1]) # 升序
                for x in a:
                    if x[1] == 1 or x[1] ==0:
                        G.remove_node(x[0])

                degrees = [i for i in list(dict(nx.degree(G)).values())]
                if min(degrees) >=2 :
                    flag = 0
                else:
                    flag = 1
        else:
            for i in slen:
                G.remove_node(sol[i])
            flag = 1
            while (flag == 1):
                deg = dict(nx.degree(G))
                a = sorted(deg.items(), key=lambda x: x[1])  # 升序
                for x in a:
                    if x[1] == 1 or x[1] == 0:
                        G.remove_node(x[0])

                degrees = [i for i in list(dict(nx.degree(G)).values())]
                if min(degrees) >= 2:
                    flag = 0
                else:
                    flag = 1

        if G.number_of_nodes() == 0:
            result = []
        else:
            deg = dict(nx.degree(G))
            a = sorted(deg.items(), key=lambda x: x[1], reverse = True)
            result = [x[0] for x in a]

        return result

    def CoreHD_value(self, g, id):
        sol = []
        G = g.copy()

        t1 = time.time()
        dc = self.cal_core(G, sol)
        while (dc != []):
            sol.append(dc[0])
            dc = self.cal_core(G, sol)


        self.env_list[0].s0(self.GenNetwork(g))
        self.g_list[0] = self.env_list[0].graph
        cnr = []
        save_dir = '../results/result_contrast_100/solutions_CoreHD_400_500'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)  # 创建时上一级目录必须存在
        result_file = save_dir + '/CoreHD_g_test_%d.txt'%id
        data_test_name = ['g_test_%d' % id]
        df = pd.DataFrame(np.arange(1 * len(data_test_name)).reshape((1, len(data_test_name))), index=['time'],columns=data_test_name)

        ind = 0
        len_sol = len(sol)

        while (self.env_list[0].isTerminal() != 1):
            print("\n")
            if len_sol==ind:
                break

            a_t = sol[ind]
            ind = ind + 1

            print("reward(CNR) is:  ")
            print(self.env_list[0].CNR)
            cnr.append(self.env_list[0].CNR)
            print("\n")
            self.env_list[0].step(a_t)
            print("delete node is: ", a_t)

            print("------------------")

        print(self.env_list[0].CNR)
        cnr.append(self.env_list[0].CNR)
        t2 = time.time()
        print('\n')
        print("the sol is (selected node list):")

        for i in range(ind):
            print(sol[i])

        solution_time = t2 - t1
        # print("the time is %.4f" % (solution_time))
        print('Data:%s, time:%.2f' % (data_test_name[0], solution_time))
        df.iloc[0, 0] = solution_time
        save_dir_local = '../results/result_contrast_100/sol_times_CoreHD_400_500'
        if not os.path.exists(save_dir_local):
            os.mkdir(save_dir_local)  # 创建时上一级目录必须存在
        df.to_csv(save_dir_local + '/sol_time_%d.csv' % id, encoding='utf-8', index=False)

        with open(result_file, 'w') as f_out:

            f_out.write('%.2f\n' % cnr[0])

            for i in range(ind):
                f_out.write('%d, %.2f\n' % (sol[i], cnr[i + 1]))
