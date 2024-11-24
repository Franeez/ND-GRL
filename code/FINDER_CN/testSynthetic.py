#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')  # __file__表示当前文件目录，dirname表示去掉文件后的目录， 好像有问题，__file__的分隔符是/还是\，另外../应该在最前面吧，代表相对路径
from FINDER import FINDER
from tqdm import tqdm

def main():
    dqn = FINDER()
    data_test_path = '../data/synthetic/uniform_cost/'  # python在windows下的路径是/还是\？？  该文件是Linux运行 或者都没问题？
#     data_test_name = ['30-50', '50-100', '100-200', '200-300', '300-400', '400-500']
    data_test_name =['30-50', '50-100']
    model_file = './models/Model_erdos_renyi/nrange_5_10_iter_270.ckpt'
    
    file_path = '../results/FINDER_CN/synthetic'
#     if not os.path.exists(file_path):
    if not os.path.exists('../results/FINDER_CN'):
        os.mkdir('../results/FINDER_CN')
    if not os.path.exists('../results/FINDER_CN/synthetic'):
        os.mkdir('../results/FINDER_CN/synthetic')
        
    with open('%s/result.txt'%file_path, 'w') as fout:
        for i in tqdm(range(len(data_test_name))):
            data_test = data_test_path + data_test_name[i]
            score_mean, score_std, time_mean, time_std, CNR_mean, CNR_std, diff_mean, diff_std, sol_all, result_list_cnr_list = dqn.Evaluate(data_test, model_file)
            fout.write('%.2f±%.2f, CNR: %.2f±%.2f, diff = %.2f±%.2f' % (score_mean * 100, score_std * 100, CNR_mean * 100, CNR_std * 100, diff_mean * 100, diff_std * 100)) #100次测试的结果记录
            for j in range(len(sol_all)):
                fout.write('\n')
                for k in range(len(sol_all[j])):
                    fout.write('%d, ' % sol_all[j][k])
                # for k in range(len(result_list_cnr_list[j])):
                #     fout.write('cnr=%.2f, ' % result_list_cnr_list[j][k])
            fout.write('\n')

            for j in range(len(result_list_cnr_list)):
                fout.write('\n')
                for k in range(len(result_list_cnr_list[j])):
                    fout.write('cnr=%.2f, ' % result_list_cnr_list[j][k])
            fout.write('\n')

            fout.flush() #写出大文件时，清空缓冲区

            print('data_test_%s has been tested!' % data_test_name[i])


if __name__=="__main__":
    main()
