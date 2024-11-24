#include "graph_utils.h"
#include <cassert>
#include <iostream>
#include <random>
#include <iterator>
//#include "stdio.h"

GraphUtil::GraphUtil()
{

}

GraphUtil::~GraphUtil()
{

}

//*************需要删除对应边的权重********************
 void GraphUtil::deleteNode(std::vector<std::vector<int> > &adjListGraph, int node)
{
//    for (auto neighbour : adjListGraph[node])
//    {
//        adjListGraph[neighbour].erase(remove(adjListGraph[neighbour].begin(), adjListGraph[neighbour].end(), node), adjListGraph[neighbour].end());
//    }
    for (int i=0; i<(int)adjListGraph[node].size(); ++i) //#遍历要删除节点的邻居
    {
        int neighbour = adjListGraph[node][i];
        adjListGraph[neighbour].erase(remove(adjListGraph[neighbour].begin(), adjListGraph[neighbour].end(), node), adjListGraph[neighbour].end());//#remove是忽略value值的前移操作，实际没有删除元素 配合erase可以使用，即将新结尾到原结尾元素删除即可
    }
    adjListGraph[node].clear();
}

void GraphUtil::recoverAddNode(const std::vector<std::vector<int> > &backupCompletedAdjListGraph, std::vector<bool>& backupAllVex, std::vector<std::vector<int> > &adjListGraph, int node, Disjoint_Set &unionSet)
{

    for (int i = 0; i < (int)backupCompletedAdjListGraph[node].size(); i++) //# backupcompleted  整张图的连接情况
    {
        int neighbourNode = backupCompletedAdjListGraph[node][i];

        if (backupAllVex[neighbourNode])
        {
            addEdge(adjListGraph, node, neighbourNode);
            //需要改
            unionSet.merge(node, neighbourNode);
        }
    }

    backupAllVex[node] = true;
}

void GraphUtil::addEdge(std::vector<std::vector<int> > &adjListGraph, int node0, int node1)
{
    if (((int)adjListGraph.size() - 1) < std::max(node0, node1)) //#涉及到新的节点 孤立节点？
    {
        adjListGraph.resize(std::max(node0, node1) + 1);
    }

    adjListGraph[node0].push_back(node1);
    adjListGraph[node1].push_back(node0);
    //加入权重 不是真的删除加入 只是一个标志
}

//int GraphUtil::influspread(const std::vector<std::vector<int> > &backupCompletedAdjListGraph, std::vector<bool>& backupAllVex,const std::vector< std::pair<int, int> > &edge_list,const std::vector< int > &edge_value，std::vector<int>&prob，std::vector<vector<int>> &node_select)
//{
//    std::set<int> influ_node;
//    for(int i =0;i<(int)backupAllVex.size();++i){
//        if(backupAllVex[i] == true){
//            influ_node.insert(i);
//        }
//    }
//
//    int num=(int)influ_node.size();//初始
////    int prim_num=(int)influ_node.size();
////    std::vector< int > prob;
//    prob.resize((int)edge_value.size());
//    for(int i=0;i<(int)edge_value.size();++i){
//        prob.push_back(random.uniform(0, 1));//激活概率
//    }
//
////    int num_tep;
////    bool flag = true;
////    std::vector<vector<int>> node_select;//每个vector int 都是下一个vector int的前半部分，后半部分才是下一轮的新标记
//    int n_s=0;//记录上面node_select的第一维
//    while(true){
////    std::set<int> influ_node_tep;
//       std::vector<int> node_tem;
//        for(int i=0;i<num;++i){//这里还可以改进 加速
//            for(int j=0;j<(int)backupCompletedAdjListGraph[influ_node[i]].size();++j){
//                for(int k=0;k<(int)edge_list.size();++k){
//                    if((edge_list[k].first == backupCompletedAdjListGraph[influ_node[i]][j] || edge_list[k].second == backupCompletedAdjListGraph[influ_node[i]][j] )&&( edge_list[k].first == influ_node[i] || edge_list[k].second == influ_node[i])){
//                        if(prob[k]<edge_value[k]){
//                        influ_node.insert(edge_list[k].first);
//                        influ_node.insert(edge_list[k].second);
////                        node_select[n_s].push_back(k);
//                        node_tem.push_back(k);
//                        }else{
//                        continue;
//                        }
//                    }else{
//                    continue;
//                    }
//                }
//            }
//        }
////    influ_node_tep.clear();
//     node_select[n_s].assign(node_tem.begin(),node_tem.end());
//     n_s+=1;
//     if(num==(int)influ_node.size()){
//     break;
//     }
//     num=(int)influ_node.size();
//    }//得到最终的influence spread
//
//return num;
//
//}
//
//int GraphUtil::influspread_multi(const std::vector<std::vector<int> > &backupCompletedAdjListGraph, std::vector<bool>& backupAllVex,const std::vector< std::pair<int, int> > &edge_list,const std::vector< int > &edge_value,int number){
//
//    int num=0;
//    std::vector<bool> backupAllVex_tem; //防止对初始backupAallVex的修改
////    backupAllVex_tem.assign<backupAllVex.begin(),backupAllVex.end()>;
//    std::vector<std::vector<int>> prob_num;//记录number次的每次概率判断
//    std::vector<std::vector<int>> node_select;
//    prob_num.resize(number);
//    for(int i=0;i<number;++i){
//    backupAllVex_tem.assign<backupAllVex.begin(),backupAllVex.end()>;
//    num+=influspread(backupCompletedAdjListGraph,backupAllVex_tem,edge_list,edge_value,prob_num[i],node_select);
//        for(int j=0;j<(int)node_select.size();++j){
//            node_select[j].clear();
//        }
//        node_select.clear();
//    }
//    return (int)(num/number)
//}