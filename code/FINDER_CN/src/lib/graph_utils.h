#ifndef GRAPH_UTILS_H
#define GRAPH_UTILS_H

#include <map>
#include <vector>
#include <memory>
#include <algorithm>
#include <set>
#include "disjoint_set.h"

//# utils 是实用工具的意思 一些常用操作，只有成员函数无成员变量
class GraphUtil
{
public:
    GraphUtil();

    ~GraphUtil();

    void deleteNode(std::vector<std::vector<int> > &adjListGraph, int node);

    void recoverAddNode(const std::vector<std::vector<int> > &backupCompletedAdjListGraph, std::vector<bool>& backupAllVex, std::vector<std::vector<int> > &adjListGraph, int node, Disjoint_Set &unionSet);

    void addEdge(std::vector<std::vector<int> > &adjListGraph, int node0, int node1);

//    int influspread(const std::vector<std::vector<int> > &backupCompletedAdjListGraph, std::vector<bool>& backupAllVex,const std::vector< std::pair<int, int> > &edge_list,const std::vector< int > &edge_value,std::vector<int>&prob，std::vector<vector<int>> &node_select);
//
//    int influspread_multi(const std::vector<std::vector<int> > &backupCompletedAdjListGraph, std::vector<bool>& backupAllVex,const std::vector< std::pair<int, int> > &edge_list,const std::vector< int > &edge_value,int number);
};



#endif