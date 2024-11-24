#ifndef PREPAREBATCHGRAPH_H_
#define PREPAREBATCHGRAPH_H_

#include "graph.h"
#include "graph_struct.h"
#include <random>
#include <algorithm>
#include <cstdlib> /** #c的标准库的c++实现**/
#include <memory>
#include <set>
#include <math.h>
#include <stdio.h>

class sparseMatrix
{
 public:
    sparseMatrix();
    ~sparseMatrix();
    std::vector<int> rowIndex;
    std::vector<int> colIndex;
    std::vector<double> value;
    int rowNum;
    int colNum;
};

class PrepareBatchGraph{
public:
    PrepareBatchGraph(int aggregatorID);
    ~PrepareBatchGraph();
//    int SetupGraphInput_hda(std::vector<int> idxes,
//                           std::vector< std::shared_ptr<Graph> > g_list,
//                           std::vector< std::vector<int> > covered,
//                           const int* actions);
//    int SetupTrain_hda(std::vector<int> idxes,
//                           std::vector< std::shared_ptr<Graph> > g_list,
//                           std::vector< std::vector<int> > covered,
//                           const int* actions);
//    int SetupPredAll_hda(std::vector<int> idxes,
//                           std::vector< std::shared_ptr<Graph> > g_list,
//                           std::vector< std::vector<int> > covered);



    void SetupGraphInput(std::vector<int> idxes,
                         std::vector< std::shared_ptr<Graph> > g_list,
                         std::vector< std::vector<int> > covered,
                         const int* actions);
    void SetupTrain(std::vector<int> idxes,
                         std::vector< std::shared_ptr<Graph> > g_list,
                         std::vector< std::vector<int> > covered,
                         const int* actions);
    void SetupPredAll(std::vector<int> idxes,
                         std::vector< std::shared_ptr<Graph> > g_list,
                         std::vector< std::vector<int> > covered);
    int GetStatusInfo(std::shared_ptr<Graph> g, int num, const int* covered, int& counter,int& twohop_number,int& threehop_number, std::vector<int>& idx_map);

    std::shared_ptr<sparseMatrix> act_select; /** # 动作选择记录，子图与其相关的被删除节点的嵌入**/
    std::shared_ptr<sparseMatrix> rep_global; /**#节点所属子图的矩阵记录 **/
    std::shared_ptr<sparseMatrix> n2nsum_param; /**#节点与节点相连的矩阵记录 **/
//    std::shared_ptr<sparseMatrix> n2nsum_param_A;
    std::shared_ptr<sparseMatrix> laplacian_param; /**# 拉普拉斯矩阵 **/
    std::shared_ptr<sparseMatrix> subgsum_param; /** # 子图所含节点矩阵记录**/
   // std::shared_ptr<sparseMatrix> subgsum_param_A;

    std::vector< std::vector<int> > idx_map_list; /**#子图序号idx对应的子图节点序号集 **/
    std::vector<std::pair<int,int>> subgraph_id_span; /**#？？？ 析构中是不是少了对该函数的clear  根据剩余图的各个子图节点数目，排序，每一个pair代表一个子图，0，子图1节点数-1，下一个pair是子图1节点数，子图1+子图2节点数-1... **/
    std::vector< std::vector<double> > aux_feat; /**#记录了四条数据 最后第四位为1 分割两条数据的标记 **/
    GraphStruct graph; /** #图**/
    std::vector<int> avail_act_cnt;
    int aggregatorID; /** #操作的选择变量？**/
};



std::vector<std::shared_ptr<sparseMatrix>> n2n_construct(GraphStruct* graph, int aggregatorID); /**#函数 **/
std::vector<std::shared_ptr<sparseMatrix>> n2n_construct_weight(GraphStruct* graph,int aggregatorID);

std::shared_ptr<sparseMatrix> e2n_construct(GraphStruct* graph);

std::shared_ptr<sparseMatrix> n2e_construct(GraphStruct* graph);

std::shared_ptr<sparseMatrix> e2e_construct(GraphStruct* graph);

std::shared_ptr<sparseMatrix> subg_construct(GraphStruct* graph, std::vector<std::pair<int,int>>& subgraph_id_span);
std::shared_ptr<sparseMatrix> subg_construct_st(GraphStruct* graph, std::vector<std::vector<int>> sol);
#endif