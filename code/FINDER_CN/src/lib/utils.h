#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <set>
#include <memory>
#include "graph.h"
#include "disjoint_set.h"
#include "graph_utils.h"
#include "decrease_strategy.cpp"

class Utils //#工具
{
public:
    Utils(); //#不用析构？ 使用默认的 释放普通的变量内存

//    int influspread_one(const std::vector<std::vector<int> > &backupCompletedAdjListGraph, std::vector<bool>& backupAllVex,const std::vector< std::pair<int, int> > &edge_list,const std::vector< double > &edge_value,std::vector<double>&prob,std::vector<int> &node_select);

    int influspread(const std::vector<std::vector<int> > &backupCompletedAdjListGraph, std::vector<bool>& backupAllVex,const std::vector< std::pair<int, int> > &edge_list,const std::vector< double > &edge_value,std::vector<double>&prob,std::vector<bool> &node_select);

    double influspread_multi(const std::vector<std::vector<int> > &backupCompletedAdjListGraph, std::vector<bool>& backupAllVex,const std::vector< std::pair<int, int> > &edge_list,const std::vector< double > &edge_value,int number);

//    int influspread_one2(std::shared_ptr<Graph> graph, std::vector<bool>& backupAllVex,std::vector<double>& prob,std::vector<int>& node_select);

    int influspread2(std::shared_ptr<Graph> graph, std::vector<bool>& backupAllVex,std::vector<double>& prob,std::vector<bool> &node_select);

    double influspread_multi2(std::shared_ptr<Graph> graph, std::vector<bool>& backupAllVex,int number);

    double getRobustness(std::shared_ptr<Graph> graph, std::vector<int> solution);

    double getRobustnessInflu(std::shared_ptr<Graph> graph, std::vector<int> solution);

//    double getRobustnessdecycle(std::shared_ptr<Graph> graph, std::vector<int> solution);


    std::vector<int> getInfluactions(std::shared_ptr<Graph> graph, std::vector<bool> backupAllVex,std::vector<double> prob); //对应influspread_one函数

    std::vector<int> reInsert(std::shared_ptr<Graph> graph,std::vector<int> solution,const std::vector<int> allVex,int decreaseStrategyID,int reinsertEachStep);

    std::vector<int> reInsert_inner(const std::vector<int> &beforeOutput, std::shared_ptr<Graph> &graph, const std::vector<int> &allVex, std::shared_ptr<decreaseComponentStrategy> &decreaseStrategy,int reinsertEachStep);

    int getMxWccSz(std::shared_ptr<Graph> graph);

    std::vector<double> Betweenness(std::shared_ptr<Graph> _g);

    std::vector<double> Betweenness(std::vector< std::vector <int> > adj_list); //#没有实现？ 可以将adj_list转换为graph再用Betweenness（Graph）

    std::vector<double> MaxWccSzList; //存储ANC值

    std::vector<double> MaxInfluSzList; //影响力延展度/N

//    std::vector<double> MaxdecycleSzList;


};

#endif