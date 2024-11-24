# -*- coding: utf-8 -*-

from FINDER import FINDER  #使用的是pyd文件，即编译后的生成文件 或者说优先使用pyd文件中的内容连接 pyd是python的动态链接库，windows下，是pyd，Linux下是so
from FINDER import contrast_tarjan
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import csv

def testtest():
    # max_n = 500  # 节点数最大限制
    # min_n = 400  # 节点数最小限制
    #
    # cur_n = np.random.randint(max_n - min_n + 1) + min_n  # 一个随机整数 范围[min,max+1)
    #
    # g = nx.erdos_renyi_graph(n=cur_n, p=0.15)  # n节点 概率p连接的ER网络
    # sol = []
    # sol.append(0)
    # sol.append(1)
    # print(g.nodes())
    # print(set(g.nodes()))
    # print(set(g.nodes()) ^ set(sol))
    # print(list(set(g.nodes()) ^ set(sol)))
    x = [5, 6, 3, 8]
    b = sorted(zip(x, range(len(x))))
    print(b)
    b.reverse()
    print(b)
    print([x[1] for x in b])
    print([x[1] for x in b if x[0] > 3])
    b.sort(key=lambda x: x[0])
    print(b)
    c = [x[1] for x in b]
    print(c)


def special_data_generate():
    data_store_path = '../results/result_contrast/'
    g = nx.Graph()
    max_n = 50  # 节点数最大限制
    min_n = 30  # 节点数最小限制
    start = 0
    core_id = []
    node_all_num = [] #当前总节点数目
    for i in range(10):
        cur_n = np.random.randint(max_n - min_n + 1) + min_n  # 一个随机整数 范围[min,max+1)
        for j in range(start+1, cur_n+start):
            g.add_edge(start, j)

        core_id.append(start)
        start = start + cur_n
        if i == 9:
            node_all_num.append(start)

    g.add_node(node_all_num[0])
    node_all_num[0] = node_all_num[0] + 1
    g.add_node(node_all_num[0])

    a = [np.random.uniform(0, 5) for i in range(5 - 1)] #10是10个星形团的数目 5是5条链 链接两端2个节点
    a = [int(a[i]) for i in range(5-1)]
    a.append(0)
    a.append(5)
    a.sort()
    b = [a[i+1]-a[i] for i in range(5)]
    for i in range(len(b)):
        b[i] = int(b[i])+1
    print(b)
    print(core_id)
    t = 0
    for i in range(len(b)):
        if b[i] == 1:
            g.add_edge(core_id[t], node_all_num[0]-1)
            g.add_edge(core_id[t], node_all_num[0])
        else:
            for j in range(b[i]):
                if j == 0:
                    g.add_edge(core_id[t + j], node_all_num[0] - 1)
                elif j == b[i]-1:
                    g.add_edge(core_id[t + j], node_all_num[0])
                    g.add_edge(core_id[t + j - 1], core_id[t + j])
                else:
                    g.add_edge(core_id[t + j - 1], core_id[t + j])
                    g.add_edge(core_id[t + j], core_id[t + j + 1])
        t = t+b[i]




    # branch_n = np.random.randint(10 + 1) + 20
    # degree_low = branch_n + 1
    #
    # branch_num = []
    # for i in range(branch_n):
    #     branch_num.append(np.random.randint())

    return g
    # data_store_name = ['g_test_special']
    # data_store_test = data_store_path + data_store_name[0] + '.txt'
    # nx.write_edgelist(g, data_store_test)  # 对应read_edgelist()





def special_data_random_generate():
    # data_store_path = '../results/result_contrast/'
    g = nx.Graph()
    max_n = 30  # 节点数最大限制
    min_n = 10  # 节点数最小限制
    start = 0
    core_id = []
    node_all_num = [] #当前总节点数目
    for i in range(10):
        cur_n = np.random.randint(max_n - min_n + 1) + min_n  # 一个随机整数 范围[min,max+1)
        for j in range(start+1, cur_n+start):
            g.add_edge(start, j)

        core_id.append(start)
        start = start + cur_n
        if i == 9:
            node_all_num.append(start)

    g.add_node(node_all_num[0])
    node_all_num[0] = node_all_num[0] + 1
    g.add_node(node_all_num[0])

    a = [np.random.uniform(0, 5) for i in range(5 - 1)] #10是10个星形团的数目 5是5条链 链接两端2个节点
    a = [int(a[i]) for i in range(5-1)]
    a.append(0)
    a.append(5)
    a.sort()
    b = [a[i+1]-a[i] for i in range(5)]
    for i in range(len(b)):
        b[i] = int(b[i])+1
    print(b)
    print(core_id)
    t = 0
    for i in range(len(b)):
        if b[i] == 1:
            g.add_edge(core_id[t], node_all_num[0]-1)
            g.add_edge(core_id[t], node_all_num[0])
        else:
            for j in range(b[i]):
                if j == 0:
                    g.add_edge(core_id[t + j], node_all_num[0] - 1)
                elif j == b[i]-1:
                    g.add_edge(core_id[t + j], node_all_num[0])
                    g.add_edge(core_id[t + j - 1], core_id[t + j])
                else:
                    g.add_edge(core_id[t + j - 1], core_id[t + j])
                    g.add_edge(core_id[t + j], core_id[t + j + 1])
        t = t+b[i]

# 随机产生少量节点 连边
    for k in range(5):
        random_n = np.random.randint(10 - 5 + 1) + 5  # [5,11)
        for i in range(random_n):
            g.add_node(node_all_num[0]+i+1)
            if i == 0 or i == random_n - 1:
                for j in range(3):
                    from_node_id = np.random.randint(100 + 1)  # [0,100]
                    from_node_id = from_node_id % len(core_id)
                    g.add_edge(core_id[from_node_id], node_all_num[0]+i+1)
            else:
                g.add_edge(node_all_num[0]+i, node_all_num[0]+i+1)
                g.add_edge(node_all_num[0] + i + 1, node_all_num[0] + i + 2)
        node_all_num[0] = node_all_num[0] + random_n

    print(node_all_num[0])

    return g

def gen_special_100():
    data_store_path = '../results/result_contrast_100/graphs_400_500/'
    for i in range(100):
        g = special_data_random_generate()
        data_store_name = ['g_test_%d'%i]
        data_store_test = data_store_path + data_store_name[0] + '.txt'
        nx.write_edgelist(g, data_store_test)  # 对应read_edgelist()




# def read_gml_graph(data_test):
#     i = 0
#     g_path = '%s/' % data_test + 'g_%d' % i
#     g = nx.read_gml(g_path)
#     N = list(g.nodes())
#     M = list(g.edges())
#     for i in range(len(N)):
#         print(N[i])
#     for i in range(len(M)):
#         print(M[i][0],M[i][1])
#     file_path = '../data/synthetic/uniform_cost/30_50_edges'
#     if not os.path.exists('../data/synthetic/uniform_cost/30_50_edges'):
#         os.mkdir('../data/synthetic/uniform_cost/30_50_edges')
#     with open('%s/g_0.txt' % file_path, 'w') as fout:
#         for j in range(len(M)):
#             # fout.write('\n')
#             for k in range(len(M[j])):
#                 fout.write('%d  ' % M[j][k])
#             fout.write('\n')
#             # for k in range(len(result_list_cnr_list[j])):
#             #     fout.write('cnr=%.2f, ' % result_list_cnr_list[j][k])
#         # fout.write('\n')
#         fout.flush()
def gen_graph_one():
    data_store_path = '../results/result_contrast/'
    max_n = 50  # 节点数最大限制
    min_n = 30  # 节点数最小限制

    cur_n = np.random.randint(max_n - min_n + 1) + min_n  # 一个随机整数 范围[min,max+1)

    # g = nx.erdos_renyi_graph(n=cur_n, p=0.15)  # n节点 概率p连接的ER网络
    # g = nx.powerlaw_cluster_graph(n=cur_n, m=4, p=0.05)
    # g = nx.connected_watts_strogatz_graph(n=cur_n, k=4, p=0.5)
    g = nx.barabasi_albert_graph(n=cur_n, m=4)

    data_store_name = ['g_test_30_50']
    data_store_test = data_store_path + data_store_name[0] + '.txt'
    nx.write_edgelist(g, data_store_test)  # 对应read_edgelist()

def gen_graph():

    data_store_path = '../results/result_contrast_100/graphs_400_500/'
    max_n = 500  # 节点数最大限制
    min_n = 400  # 节点数最小限制
    for i in range(100):
        cur_n = np.random.randint(max_n - min_n + 1) + min_n  # 一个随机整数 范围[min,max+1)


    # g = nx.erdos_renyi_graph(n=cur_n, p=0.15)  # n节点 概率p连接的ER网络
    # g = nx.powerlaw_cluster_graph(n=cur_n, m=4, p=0.05)
    # g = nx.connected_watts_strogatz_graph(n=cur_n, k=4, p=0.5)
        g = nx.barabasi_albert_graph(n=cur_n, m=4)



        data_store_name = ['g_test_%d'%i]
        data_store_test = data_store_path + data_store_name[0] + '.txt'
        nx.write_edgelist(g, data_store_test) #对应read_edgelist()

def read_txt_graph():
    data_test_path = '../data/real/'
    data_test_name = ['netsience', '50-100']
    data_test = data_test_path + data_test_name[0]
    # read_gml_graph(data_test)
    G = nx.Graph()

    g_path = '%s' % data_test + '.txt'
    with open('%s'%g_path, 'r') as fout:
        lines = fout.readlines()
        for line in lines:
            line = line.strip("\n")
            line = line.split()
            # print(line)
            # print(line[0], line[1])
            G.add_edges_from([(int(line[0]), int(line[1]))])
    print(G.nodes())
    # print(G.edges())
    data_store_path = '../data/real/'
    data_store_name = ['g_test_100']
    data_store_test = data_store_path + data_store_name[0] + '.txt'
    nx.write_edgelist(G, data_store_test) #对应read_edgelist()





def test_1():
    data_test_path = '../results/result_contrast/'
    # solutions_dir = 'D:/Beihang_graduate_stage/git_warehouse/FINDER_original/code/results/solutions'

    data_test_name = ['g_test_30_50']
    data_test = data_test_path + data_test_name[0]
    g_path = '%s' % data_test + '.txt'
    g = nx.read_edgelist(g_path)

    # sol_node = []
    # with open('%s/%s.txt' % (solutions_dir, data_test_name[0]), 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         sol_node.append(int(line))

    print(g.nodes())
    print(g.edges())
    G = nx.Graph()
    G.add_nodes_from([int(i) for i in g.nodes()])
    G.add_edges_from([(int(u), int(v)) for u, v in g.edges()])
    print(G.nodes)
    print(G.edges)
    dqn = contrast_tarjan()

    # dqn.solution_to_cnr(g, sol_node, i)
    i = 3050
    # dqn.test(g, i)
    # print("=================")
    dqn.absolute_value(G, i)
    print("=================")
    # dqn.Random_value(g, i)
    # print("=================")
    # dqn.HXA_value(g, 'HDA', i)
    # dqn.HXA_value(g, 'HBA', i)
    # dqn.HXA_value(g, 'HCA', i)
    # dqn.HXA_value(g, 'HPRA', i)

    print('\n')
    print("success !")


def test_100():
    data_test_path = '../results/result_contrast_100/graphs_400_500/'
    # solutions_dir = 'D:/Beihang_graduate_stage/git_warehouse/FINDER_original/code/results/solutions2'
    for i in range(16, 100, 1):
        data_test_name = ['g_test_%d'%i]
        data_test = data_test_path + data_test_name[0]
        g_path = '%s' % data_test + '.txt'
        g = nx.read_edgelist(g_path)


        # sol_node = []
        # with open('%s/%s.txt' % (solutions_dir, data_test_name[0]), 'r') as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         sol_node.append(int(line))


        print(g.nodes())
        print(g.edges())
        G = nx.Graph()
        G.add_nodes_from([int(i) for i in g.nodes()])
        G.add_edges_from([(int(u), int(v)) for u, v in g.edges()])
        print(G.nodes)
        print(G.edges)
        dqn = contrast_tarjan()

        # dqn.solution_to_cnr(g, sol_node, i)


        dqn.test(g, i)
        print("=================")
        dqn.absolute_value(G, i)
        print("=================")
        dqn.Random_value(g, i)
        print("=================")
        dqn.HXA_value(g, 'HDA', i)
        dqn.HXA_value(g, 'HBA', i)
        dqn.HXA_value(g, 'HCA', i)
        dqn.HXA_value(g, 'HPRA', i)

        print('\n')
        print("success !")

def write_graph():


    max_n = 10  # 节点数最大限制
    min_n = 8  # 节点数最小限制
    cur_n = np.random.randint(max_n - min_n + 1) + min_n  # 一个随机整数 范围[min,max+1)


    # g = nx.erdos_renyi_graph(n=cur_n, p=0.15)  # n节点 概率p连接的ER网络
    # g = nx.powerlaw_cluster_graph(n=cur_n, m=4, p=0.05)
    # g = nx.connected_watts_strogatz_graph(n=cur_n, k=4, p=0.5)
    g = nx.barabasi_albert_graph(n=cur_n, m=4)


    data_store_path = '../results/result_contrast/'
    data_store_name = ['g_test']
    data_store_test = data_store_path + data_store_name[0] + '.txt'
    nx.write_edgelist(g, data_store_test) #对应read_edgelist()

def read_gml_graph():
    data_test_path = '../data/synthetic/uniform_cost/'
    data_test_name = ['30-50', '50-100']
    data_test = data_test_path + data_test_name[0]
    # read_gml_graph(data_test)

    i = 0
    g_path = '%s/' % data_test + 'g_%d' % i
    g = nx.read_gml(g_path)
    return g

def open_and_load(method1, method2):

    # open files and load data
    file_path = '../results/result_contrast_100/solutions_%s'%method1
    sol_all = []
    cnr_all = []
    # sol_len_min = len(g.nodes())
    # file_names = ['FINDER_GA_g_test.txt', 'tarjan_result.txt', 'random_result.txt', 'HDA_result.txt', 'HBA_result.txt', 'HCA_result.txt', 'HPRA_result.txt']

    for i in range(16):
        sol = []
        cnr = []
        file_name = ['%s_g_test_%d' % (method2, i)]
        with open('%s/%s.txt'%(file_path, file_name[0]), 'r') as f:
            lines = f.readlines()
            cnr.append(float(lines[0]))
            if len(lines) > 1:
                for line in lines[1:]:
                    node, cnrs = line.split(',', 1)

                    sol.append(node)
                    cnr.append(float(cnrs))


        # if len(sol) < sol_len_min:
        #     sol_len_min = len(sol)
        sol_all.append(sol)
        cnr_all.append(cnr)

    # nodes_num = int(sol_len_min*0.6)
    print(sol_all)
    return sol_all, cnr_all

# no ga method
def open_and_load1():
    # open files and load data
    file_path = 'D:/Beihang_graduate_stage/git_warehouse/FINDER_dfs_success_multi_versions/code/results/result_contrast/solutions'


    sol_all = []
    cnr_all = []
    # sol_len_min = len(g.nodes())
    # file_names = ['FINDER_GA_g_test.txt', 'tarjan_result.txt', 'random_result.txt', 'HDA_result.txt', 'HBA_result.txt', 'HCA_result.txt', 'HPRA_result.txt']

    for i in range(100):
        sol = []
        cnr = []
        file_name = ['FINDER_g_test_%d' %i]
        with open('%s/%s.txt' % (file_path, file_name[0]), 'r') as f:
            lines = f.readlines()
            cnr.append(float(lines[0]))
            if len(lines) > 1:
                for line in lines[1:]:
                    node, cnrs = line.split(',', 1)

                    sol.append(node)
                    cnr.append(float(cnrs))

        # if len(sol) < sol_len_min:
        #     sol_len_min = len(sol)
        sol_all.append(sol)
        cnr_all.append(cnr)

    # nodes_num = int(sol_len_min*0.6)
    print(sol_all)
    return sol_all, cnr_all

def sol_size():
    method1 = ['absolute', 'finder', 'finder_ga', 'FINDER_O', 'HDA', 'HBA', 'HCA', 'HPRA', 'random', 'tarjan']
    method2 = ['absolute', 'FINDER', 'FINDER_GA', 'FINDER_original', 'HDA', 'HBA', 'HCA', 'HPRA', 'random', 'tarjan']
    sol_len_ave = []
    for i in range(len(method1)):
        if i == 1:
            sol_all, cnr_all = open_and_load1()
        else:
            sol_all, cnr_all = open_and_load(method1[i],method2[i])
        len_ave = 0.0
        for j in range(len(sol_all)):
            len_ave = len_ave + len(sol_all[j])
        len_ave = len_ave / len(sol_all)
        sol_len_ave.append(len_ave)
    return sol_len_ave

def time_csv_add():
    file_path1 = 'D:/Beihang_graduate_stage/git_warehouse/FINDER_original/code/results/sol_times2'
    # file_path2 = '../results/result_contrast_100/sol_times_FINDER_O'
    file_path2 = '../results/result_contrast_100/sol_times_FINDER_O_400_500'
    time_result = []
    for i in range(16):
        file_name = ['sol_time_%d'%i]
        time1 = []
        time2 = []
        with open('%s/%s.csv'%(file_path1, file_name[0])) as f1:
            f1_csv = csv.reader(f1)
            header1 = next(f1_csv)
            for row in f1_csv:
                time1.append(float(row[0]))
        with open('%s/%s.csv'%(file_path2, file_name[0])) as f2:
            f2_csv = csv.reader(f2)
            header2 = next(f2_csv)
            for row in f2_csv:
                time2.append(float(row[0]))
        time_add = time1[0]+time2[0]
        time_result.append(time_add)

    return time_result


def time_csv_all(method):
    file_path = '../results/result_contrast_100/sol_times_%s'%method
    time_result = []
    for i in range(16):
        file_name = ['sol_time_%d'%i]
        time = []
        with open('%s/%s.csv'%(file_path, file_name[0])) as f:
            f_csv = csv.reader(f)
            header = next(f_csv)
            for row in f_csv:
                time.append(float(row[0]))
        time_result.append(time[0])
    return time_result

def time_csv_no_ga():
    file_path = 'D:/Beihang_graduate_stage/git_warehouse/FINDER_dfs_success_multi_versions/code/results/result_contrast/sol_times'
    time_result = []
    for i in range(100):
        file_name = ['sol_time_%d'%i]
        time = []
        with open('%s/%s.csv'%(file_path, file_name[0])) as f:
            f_csv = csv.reader(f)
            header = next(f_csv)
            for row in f_csv:
                time.append(float(row[0]))
        time_result.append(time[0])
    return time_result


def times_ave():
    # method = ['absolute', 'finder_ga_modify', 'FINDER_O', 'HDA', 'HBA', 'HCA', 'HPRA', 'random', 'tarjan']
    method = ['absolute_400_500', 'finder_ga_modify_400_500', 'FINDER_O_400_500', 'HDA_400_500', 'HBA_400_500', 'HCA_400_500', 'HPRA_400_500', 'random_400_500', 'tarjan_400_500']
    result_all = []
    result_ave = []
    # result = time_csv_all(method[0])
    # result_all.append(result)
    for i in range(len(method)):
        if i == 2:
            result = time_csv_add()
            result_all.append(result)
            continue
        # if i == 1:
        #     result = time_csv_no_ga()
        #     result_all.append(result)
        #     continue
        result = time_csv_all(method[i])
        result_all.append(result)
    for i in range(len(method)):
        result_ave.append(sum(result_all[i])/len(result_all[i]))
    return result_ave

def cnr_ave_pos():
    result_all = []

    # method1 = ['absolute', 'finder', 'finder_ga', 'FINDER_O', 'HDA', 'HBA', 'HCA', 'HPRA', 'random', 'tarjan']
    # method2 = ['absolute', 'FINDER', 'FINDER_GA', 'FINDER_original', 'HDA', 'HBA', 'HCA', 'HPRA', 'random', 'tarjan']
    method1 = ['absolute_400_500', 'finder_ga_modify_400_500_4', 'FINDER_O_400_500', 'HDA_400_500', 'HBA_400_500', 'HCA_400_500', 'HPRA_400_500', 'random_400_500', 'tarjan_400_500']
    method1 = ['absolute_400_500', 'finder_ga_modify_400_500', 'finder_ga_modify_400_500_4', 'FINDER_O_400_500']
    # method1 = ['absolute', 'finder_ga_modify', 'FINDER_O', 'HDA', 'HBA', 'HCA', 'HPRA', 'random', 'tarjan']
    # method2 = ['absolute', 'FINDER_GA', 'FINDER_original', 'HDA', 'HBA', 'HCA', 'HPRA', 'random', 'tarjan']
    method2 = ['absolute', 'FINDER_GA', 'FINDER_GA', 'FINDER_original']

    for i in range(len(method1)):
        # if i == 1:
        #     sol_all, cnr_all = open_and_load1()
        # else:
        #     sol_all, cnr_all = open_and_load(method1[i], method2[i])
        sol_all, cnr_all = open_and_load(method1[i], method2[i])
        max_len = 0
        result_ave = []
        for j in range(len(cnr_all)):
            if len(cnr_all[j])>max_len:
                max_len = len(cnr_all[j])
        for j in range(len(cnr_all)):
            if len(cnr_all[j])<max_len:
                for k in range(max_len-len(cnr_all[j])):
                    cnr_all[j].append(0.00)
        result_ave.extend([0.00]*max_len)
        for j in range(len(cnr_all)):
            for k in range(max_len):
                result_ave[k] = cnr_all[j][k] + result_ave[k]
        for j in range(max_len):
            result_ave[j] /= len(cnr_all)
        result_all.append(result_ave)

    return result_all


def calculate_ANC():
    method1 = ['absolute_400_500', 'finder_ga_modify_400_500', 'FINDER_O_400_500', 'HDA_400_500', 'HBA_400_500', 'HCA_400_500', 'HPRA_400_500', 'random_400_500', 'tarjan_400_500']
    method2 = ['absolute', 'FINDER_GA', 'FINDER_original', 'HDA', 'HBA', 'HCA', 'HPRA', 'random', 'tarjan']
    ANC_value = []
    for i in range(len(method1)):
        # if i == 1:
        #     sol_all, cnr_all = open_and_load1()
        # else:
        #     sol_all, cnr_all = open_and_load(method1[i], method2[i])
        sol_all, cnr_all = open_and_load(method1[i], method2[i])

        # result_ave = 0.0
        ANC_tem = []
        for j in range(len(cnr_all)):
            result_ave = 0.0
            if len(cnr_all[j]) < 2:
                result_ave = result_ave + 0.0
                ANC_tem.append(result_ave)
                continue
            for k in range(len(cnr_all[j])-1):
                result_ave = result_ave + cnr_all[j][k+1]/cnr_all[j][0]
            result_ave = result_ave/(len(cnr_all[j])-1)
            ANC_tem.append(result_ave)
        ANC_ave = 0.0
        for j in range(len(ANC_tem)):
            ANC_ave = ANC_ave + ANC_tem[j]
        ANC_ave = ANC_ave/len(ANC_tem)
        ANC_value.append(ANC_ave)

    # result_file = '../results/result_contrast_100' + '/ANC_400_500.txt'
    # with open(result_file, 'w') as f_out:
    #
    #     for i in range(len(ANC_value)):
    #         f_out.write('%.2f\n' % ANC_value[i])
        ANC_STD = np.array(ANC_tem)
        ANC_std = np.std(ANC_STD, ddof= 1)



    return ANC_value



def draw_ANC():
    method = ['absolute', 'finder_ga', 'FINDER_O', 'HDA', 'HBA', 'HCA', 'HPRA', 'random', 'tarjan']
    # method = ['finder_ga', 'FINDER_O', 'HDA', 'HBA', 'HCA', 'HPRA', 'random', 'tarjan']
    time_ave = calculate_ANC()
    time_ave_4 = []
    for i in range(len(time_ave)):
        time_ave_4.append(round(time_ave[i], 5))
    # draw bar
    fig = plt.figure(figsize=(50, 50))
    ax = fig.add_subplot(111)
    # x = ['FINDER_GA', 'Tarjan', 'Random', 'HDA', 'HBA', 'HCA', 'HPRA']
    # y = [i[nodes_num] for i in cnr_all]
    ax.bar(method, time_ave_4, label='accumulated normalized CNR', width=0.6)
    for i in range(len(method)):
        ax.text(i, time_ave_4[i]+0.01, time_ave_4[i], ha='center', fontsize=8)
    ax.legend()
    plt.title('accumulated normalized CNR value(400-500)', fontsize='larger', fontweight='heavy')
    plt.xlabel('methods')
    plt.ylabel('accumulated normalized CNR value')
    plt.ylim(0.0,1.0)
    # plt.savefig(r'D:\Beihang_graduate_stage\git_warehouse\FINDER\code\results\result_contrast_100\draw_pic\time_average_result.svg')
    plt.show()







def draw_time():
    # method = ['absolute', 'finder_ga', 'FINDER_O', 'HDA', 'HBA', 'HCA', 'HPRA', 'random', 'tarjan']
    method = ['finder_ga', 'FINDER_O', 'HDA', 'HBA', 'HCA', 'HPRA', 'random', 'tarjan']
    time_ave = times_ave()
    time_ave_4 = []
    for i in range(len(time_ave)):
        time_ave_4.append(round(time_ave[i], 5))
    # draw bar
    fig = plt.figure(figsize=(50, 50))
    ax = fig.add_subplot(111)
    # x = ['FINDER_GA', 'Tarjan', 'Random', 'HDA', 'HBA', 'HCA', 'HPRA']
    # y = [i[nodes_num] for i in cnr_all]
    ax.bar(method, time_ave_4[1:], label='time average', width=0.6)
    for i in range(len(method)):
        ax.text(i, time_ave_4[i+1]+0.01, time_ave_4[i+1], ha='center', fontsize=8)
    ax.legend()
    plt.title('average_time_result(400-500)', fontsize='larger', fontweight='heavy')
    plt.xlabel('methods')
    plt.ylabel('average time')
    plt.ylim(0.0,100.0)
    # plt.savefig(r'D:\Beihang_graduate_stage\git_warehouse\FINDER\code\results\result_contrast_100\draw_pic\time_average_result.svg')
    plt.show()

def draw_cnr_line():
    method = ['absolute', 'FINDER', 'FINDER_GA', 'FINDER_original', 'HDA', 'HBA', 'HCA', 'HPRA', 'random', 'tarjan']
    cnr_pos = cnr_ave_pos()
    # nodes_num = len(cnr_pos[0]) - 1
    nodes_num = len(cnr_pos[0])
    for i in range(len(cnr_pos)):
        if nodes_num > len(cnr_pos[i]):
            nodes_num = len(cnr_pos[i])
    nodes_num = nodes_num - 1
    # draw line
    fig = plt.figure(figsize=(50, 50))
    x_major_locator = MultipleLocator(10)
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_locator(x_major_locator)
    x = [i for i in range(nodes_num + 1)]
    ax.plot(x, cnr_pos[0][:(nodes_num + 1)], 'ro-', linewidth=2, label='absolute')
    ax.plot(x, cnr_pos[1][:(nodes_num + 1)], 'bo-', linewidth=2, label='FINDER_GA_modify')
    ax.plot(x, cnr_pos[2][:(nodes_num + 1)], 'go-', linewidth=2, label='FINDER_GA_modify_relative')
    ax.plot(x, cnr_pos[3][:(nodes_num + 1)], 'co-', linewidth=2, label='FINDER_original')



    # ax.plot(x, cnr_pos[0][:(nodes_num + 1)], 'ro-', linewidth=2, label='absolute')
    # ax.plot(x, cnr_pos[1][:(nodes_num + 1)], 'bo-', linewidth=2, label='FINDER_GA_modify')
    # ax.plot(x, cnr_pos[2][:(nodes_num + 1)], 'go-', linewidth=2, label='FINDER_original')
    # ax.plot(x, cnr_pos[3][:(nodes_num + 1)], 'co-', linewidth=2, label='HDA')
    # ax.plot(x, cnr_pos[4][:(nodes_num + 1)], 'mo-', linewidth=2, label='HBA')
    # ax.plot(x, cnr_pos[5][:(nodes_num + 1)], 'yo-', linewidth=2, label='HCA')
    # # ax.plot(x, cnr_pos[6][:(nodes_num + 1)], color='#412f1f', marker='o', line='-', linewidth=2, label='HCA')
    # ax.plot(x, cnr_pos[6][:(nodes_num + 1)], 'o-', linewidth=2, label='HPRA')
    # ax.plot(x, cnr_pos[7][:(nodes_num + 1)], 'o-', linewidth=2, label='random')
    # ax.plot(x, cnr_pos[8][:(nodes_num + 1)], 'o-', linewidth=2, label='tarjan')
    # # ax.plot(x, cnr_pos[9][:(nodes_num + 1)], 'o-', linewidth=2, label='tarjan')


    ax.legend()
    plt.title('line4(400-500)', fontsize='larger', fontweight='heavy')
    plt.xlabel('node seq ave')
    plt.ylabel('CNR')
    # plt.savefig(r'D:\Beihang_graduate_stage\git_warehouse\FINDER\code\results\result_contrast_100\draw_pic\line9_2.svg')
    plt.show()







def draw_line(nodes_num, cnr_all):
    # draw line
    fig = plt.figure(figsize=(50, 50))
    ax = fig.add_subplot(111)
    x = [i for i in range(nodes_num+1)]
    # ax.plot(x, cnr_all[0][:(nodes_num + 1)], 'ro-', linewidth=2, label='FINDER_GA')
    # ax.plot(x, cnr_all[1][:(nodes_num + 1)], 'bo-', linewidth=2, label='Tarjan')
    # ax.plot(x, cnr_all[2][:(nodes_num + 1)], 'go-', linewidth=2, label='Random')
    # ax.plot(x, cnr_all[3][:(nodes_num + 1)], 'co-', linewidth=2, label='HDA')
    # ax.plot(x, cnr_all[4][:(nodes_num + 1)], 'mo-', linewidth=2, label='HBA')
    # ax.plot(x, cnr_all[5][:(nodes_num + 1)], 'yo-', linewidth=2, label='HCA')
    ax.plot(x, cnr_all[6][:(nodes_num + 1)], 'ko-', linewidth=2, label='HPRA')
    ax.legend()
    plt.title('line7', fontsize='xx-large', fontweight='heavy')
    plt.xlabel('node seq')
    plt.ylabel('CNR')
    plt.savefig(r'D:\Beihang_graduate_stage\git_warehouse\FINDER\code\results\result_contrast\line7.svg')
    plt.show()

def draw_bar(nodes_num, cnr_all):
    # draw bar
    fig = plt.figure(figsize=(50, 50))
    ax = fig.add_subplot(111)
    x = ['FINDER_GA', 'Tarjan', 'Random', 'HDA', 'HBA', 'HCA', 'HPRA']
    y = [i[nodes_num] for i in cnr_all]
    ax.bar(x, y, label='CNR(delete 6 nodes)')
    for i in range(len(x)):
        ax.text(i, y[i]+0.01, y[i], ha='center', fontsize=8)
    ax.legend()
    plt.title('bar_result', fontsize='xx-large', fontweight='heavy')
    plt.xlabel('methods')
    plt.ylabel('CNR(delete 6 nodes)')
    plt.ylim(0.0,1.0)
    plt.savefig(r'D:\Beihang_graduate_stage\git_warehouse\FINDER\code\results\result_contrast\bar_result.svg')
    plt.show()


def draw_one_graph(cur_n):
    # g = nx.barabasi_albert_graph(n=cur_n, m=4)
    g = special_data_random_generate()
    # g = special_data_generate()
    plt.figure(figsize=(50, 50))
    # fig, a = plt.subplots(2, 4)
    plt.suptitle('a graph')
    pos = nx.spring_layout(g)
    plt.subplot(1, 1, 1)
    nx.draw(g, with_labels=True, node_size=100, font_size=8, pos=pos, width=2)
    plt.title('graph', fontsize='larger', fontweight='heavy')
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])
    plt.show()







def draw_graphs(g, sol_all, nodes_num, cnr_all):

    # draw graph
    F_GA = g.copy()
    tar_r = g.copy()
    ran_r = g.copy()
    HDA_r = g.copy()
    HBA_r = g.copy()
    HCA_r = g.copy()
    HPRA_r = g.copy()
    abso_r = g.copy()
    print(F_GA.nodes())
    print(g.nodes())
    print(g.edges())


    e = list(sol_all[0][:nodes_num])
    F_GA.remove_nodes_from(e)
    e = list(sol_all[1][:nodes_num])
    tar_r.remove_nodes_from(e)
    e = list(sol_all[2][:nodes_num])
    ran_r.remove_nodes_from(e)
    e = list(sol_all[3][:nodes_num])
    HDA_r.remove_nodes_from(e)
    e = list(sol_all[4][:nodes_num])
    HBA_r.remove_nodes_from(e)
    e = list(sol_all[5][:nodes_num])
    HCA_r.remove_nodes_from(e)
    e = list(sol_all[6][:nodes_num])
    HPRA_r.remove_nodes_from(e)
    e = list(sol_all[7][:nodes_num])
    abso_r.remove_nodes_from(e)


    # for i in range(nodes_num):
    #     F_GA.remove_node(sol_all[0][i])
    #     tar_r.remove_node(sol_all[1][i])
    #     ran_r.remove_node(sol_all[2][i])
    #     HDA_r.remove_node(sol_all[3][i])
    #     HBA_r.remove_node(sol_all[4][i])
    #     HCA_r.remove_node(sol_all[5][i])
    #     HPRA_r.remove_node(sol_all[6][i])


    plt.figure(figsize=(50, 50))
    # fig, a = plt.subplots(2, 4)
    plt.suptitle('some graph results')
    pos = nx.spring_layout(g)
    plt.subplot(3, 3, 1)
    nx.draw(g, with_labels=True, node_size=100, font_size=8, pos=pos, width=2)
    plt.title('graph', fontsize='larger', fontweight='heavy')
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])




    plt.subplot(3, 3, 2)
    nx.draw(F_GA, with_labels=True, node_size=100, font_size=8, pos=pos, width=2)
    plt.title('FINDER_GA(delete %d nodes), CNR is %.2f'%(nodes_num, cnr_all[0][nodes_num-1]), fontsize='larger', fontweight='heavy')
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 3, 3)
    nx.draw(tar_r, with_labels=True, node_size=100, font_size=8, pos=pos, width=2)
    plt.title('tarjan(delete %d nodes), CNR is %.2f'%(nodes_num, cnr_all[1][nodes_num-1]), fontsize='larger', fontweight='heavy')
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 3, 4)
    nx.draw(ran_r, with_labels=True, node_size=100, font_size=8, pos=pos, width=2)
    plt.title('random(delete %d nodes), CNR is %.2f'%(nodes_num, cnr_all[2][nodes_num-1]), fontsize='larger', fontweight='heavy')
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 3, 5)
    nx.draw(HDA_r, with_labels=True, node_size=100, font_size=8, pos=pos, width=2)
    plt.title('HDA(delete %d nodes), CNR is %.2f'%(nodes_num, cnr_all[3][nodes_num-1]), fontsize='larger', fontweight='heavy')
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 3, 6)
    nx.draw(HBA_r, with_labels=True, node_size=100, font_size=8, pos=pos, width=2)
    plt.title('HBA(delete %d nodes), CNR is %.2f'%(nodes_num, cnr_all[4][nodes_num-1]), fontsize='larger', fontweight='heavy')
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 3, 7)
    nx.draw(HCA_r, with_labels=True, node_size=100, font_size=8, pos=pos, width=2)
    plt.title('HCA(delete %d nodes), CNR is %.2f'%(nodes_num, cnr_all[5][nodes_num-1]), fontsize='larger', fontweight='heavy')
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 3, 8)
    nx.draw(HPRA_r, with_labels=True, node_size=100, font_size=8, pos=pos, width=2)
    plt.title('HPRA(delete %d nodes), CNR is %.2f'%(nodes_num, cnr_all[6][nodes_num-1]), fontsize='larger', fontweight='heavy')
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 3, 9)
    nx.draw(abso_r, with_labels=True, node_size=100, font_size=8, pos=pos, width=2)
    plt.title('Absolute(delete %d nodes), CNR is %.2f'%(nodes_num, cnr_all[7][nodes_num-1]), fontsize='larger', fontweight='heavy')
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])

    # plt.savefig(r'D:\Beihang_graduate_stage\git_warehouse\FINDER\code\results\result_contrast\some_results.svg')
    plt.show()



def main():
    # cur_n = 8
    # draw_one_graph(cur_n)
    # testtest()
    read_txt_graph()

    # tett = os.path.dirname(__file__) + os.sep + '../'
    # print(tett)

    # dqn = FINDER()
    # try:
    #     dqn.Train()
    # except Exception as e:
    #     print(e)
    #     print("error")



    # dqn.Train()


    # dqn.Train_load()



    # dqn = FINDER()
    # try:
    #     dqn.Train_load()
    # except Exception as e:
    #     print(e)
    #     print("error")



    # draw_time()
    # draw_cnr_line()
    # gen_graph()
    # draw_cnr_line()
    # test_100()
    # draw_ANC()



    # open files and load data
    # gen_graph_one()
    # test_1()

    # data_test_path = '../results/result_contrast/'
    # # solutions_dir = 'D:/Beihang_graduate_stage/git_warehouse/FINDER_original/code/results/solutions'
    #
    # data_test_name = ['g_test_30_50']
    # data_test = data_test_path + data_test_name[0]
    # g_path = '%s' % data_test + '.txt'
    # g = nx.read_edgelist(g_path)

    # print(g.number_of_nodes())
    # file_path = '../results/result_contrast'
    # sol_all = []
    # cnr_all = []
    # sol_len_min = len(g.nodes())
    # file_names = ['FINDER_GA_g_test_30_50.txt', 'tarjan_g_test_3050.txt', 'random_g_test_3050.txt', 'HDA_g_test_3050.txt', 'HBA_g_test_3050.txt', 'HCA_g_test_3050.txt', 'HPRA_g_test_3050.txt', 'absolute_g_test_3050.txt']
    #
    # for i in range(len(file_names)):
    #     sol = []
    #     cnr = []
    #     # file_name = ['%s_g_test_%d' % (file_names[i], i)]
    #     with open('%s/%s' % (file_path, file_names[i]), 'r') as f:
    #         lines = f.readlines()
    #         cnr.append(float(lines[0]))
    #         if len(lines) > 1:
    #             for line in lines[1:]:
    #                 node, cnrs = line.split(',', 1)
    #
    #                 sol.append(node)
    #                 cnr.append(float(cnrs))
    #
    #     if len(sol) < sol_len_min:
    #         sol_len_min = len(sol)
    #     sol_all.append(sol)
    #     cnr_all.append(cnr)
    #
    # nodes_num = int(sol_len_min*0.6)
    # # print(sol_all)
    # draw_graphs(g, sol_all, nodes_num, cnr_all)







if __name__=="__main__":
    main()
