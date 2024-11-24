#ifndef MVC_ENV_H
#define MVC_ENV_H

#include <vector>
#include <set>
#include <memory>
#include "graph.h"
#include "disjoint_set.h"
//#include "utils.h"
//#include "graph_utils.h"

class MvcEnv
{
public:
    MvcEnv(double _norm);

    ~MvcEnv();

    void s0(std::shared_ptr<Graph> _g);

    double step_fake(int a);
    double step_fake_reverse(int a);

    double step(int a);

    void stepWithoutReward(int a);

    std::vector<double> Betweenness(std::vector< std::vector <int> > adj_list);

    int randomAction();

    int betweenAction();

    int influenceAction();

    int decycling_dfs_action_contrast();

    int decycling_dfs_action();
    std::vector<int> decycling_dfs_action_list();
    double decycling_dfs_ratio();

    double decycling_dfs_ratio_absolute();
    double getReward_absolute();

    int decyclingaction();
    int decyclingratio();
    std::vector<int> decyclingaction_list();

    bool isTerminal();// #终态判断

//    double getReward(double oldCcNum);
    double getReward(); //#获取reward

    double getMaxConnectedNodesNum(); //#获得最大连通分支节点数

    double getRemainingCNDScore(); //#获得剩余 CND分数

//    double getinfluencespread();

    void get_degrees();


    void printGraph(); //#输出


    double norm;
    int cycle_node_all;
    int k_num;
    double CNR;
    double CNR_all;
    bool CNR_all_flag;
    std::shared_ptr<Graph> graph;

    std::vector<int> record_cycle;

//    std::shared_ptr<Graph> g_relation; //对应的激活边判断后的graph
    std::vector<int> node_act_flag;// 0或1
    std::vector<int> edge_act_flag; // 0或非0
    std::vector<int> left;
    std::vector<int> right;

    std::vector< std::vector<int> > state_seq; //#状态序列（节点的嵌入表示？）

    std::vector<int> act_seq, action_list; //#动作序列（删除节点）
    std::vector<double> reward_seq, sum_rewards; //#reward奖励序列

    int numCoveredEdges; //#删除节点相关连的边的数目
    int numCoveredNodes;
    std::set<int> covered_set;  //#被操作后的节点集，删除操作
    std::vector<int> avail_list; //#只要graph不是只有孤立节点，则avail的size就非空

    std::vector< double > prob;//激活概率

//    std::vector<double> hda_prob_tem;//hda方法记录的剩余边权情况，删除边后权置0.0

    std::vector<int> node_degrees; //#节点度的分布
    int total_degrees;
};

#endif