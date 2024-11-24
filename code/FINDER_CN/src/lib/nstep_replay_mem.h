#ifndef NSTEP_REPLAY_MEM_H
#define NSTEP_REPLAY_MEM_H

#include <vector>
#include <random>
#include "graph.h"
#include "mvc_env.h"


class ReplaySample //#重放例子
{
public:
    ReplaySample(int batch_size);//#获得batch_size大小的图样例

    std::vector< std::shared_ptr<Graph> > g_list; //#多张图
    std::vector< std::vector<int>> list_st, list_s_primes; //#状态使用当前删除节点顺序存储记录，这样可以还原图的结构
    std::vector<int> list_at;//#删除节点记录
    std::vector<double> list_rt; //#reward
    std::vector<bool> list_term;
};

class NStepReplayMem //#n步重置内存
{
public:
      NStepReplayMem(int memory_size);

     void Add(std::shared_ptr<Graph> g,
                    std::vector<int> s_t,
                    int a_t, 
                    double r_t,
                    std::vector<int> s_prime,
                    bool terminal);

     void Add(std::shared_ptr<MvcEnv> env,int n_step);
     std::shared_ptr<ReplaySample> Sampling(int batch_size);


     std::vector< std::shared_ptr<Graph> > graphs;
     std::vector<int> actions;
     std::vector<double> rewards;
     std::vector< std::vector<int> > states, s_primes;
     std::vector<bool> terminals;

     int current, count, memory_size;
     std::default_random_engine generator; //#随机数 使用默认种子
     std::uniform_int_distribution<int>* distribution; //#生成离散均匀分布
};

#endif