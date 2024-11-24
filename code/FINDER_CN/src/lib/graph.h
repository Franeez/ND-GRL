#ifndef GRAPH_H
#define GRAPH_H

#include <map>
#include <vector> /** # 可变长数组**/
#include <memory> /** #内含智能指针**/
#include <algorithm> /** #是所有STL头文件中最大的一个，其中常用到的功能范围涉及到比较、交换等**/
#include <set> /** #集合**/
class Graph
{
public:
    Graph();
//    Graph(Graph &G);
    Graph(const int _num_nodes, const int _num_edges, const int* edges_from, const int* edges_to); /** #加下划线表示是某类的属性，设置const是因为不需要改变值，只在定义时设置参数即可**/
    Graph(const int _num_nodes, const int _num_edges, const int* edges_from, const int* edges_to,const double* edge_value_copy);
    ~Graph();
    int num_nodes; /**#节点数目 **/
    int num_edges; /**#边数目 **/
    std::vector< std::vector< int > > adj_list; /**#每个节点的邻居节点记录 **/
    std::vector< std::pair<int, int> > edge_list; /** #连边的记录 first时from second是to**/
    std::vector< double > edge_value; //边权，即激活概率
    double getTwoRankNeighborsRatio(std::vector<int> covered); /** #不在covered向量中的节点两两之间有共同邻居（方向不计）的数目统计**/

};

class GSet
{
public:
    GSet();
    ~GSet();
    void InsertGraph(int gid, std::shared_ptr<Graph> graph); /**#还未见到定义 **/
    std::shared_ptr<Graph> Sample();
    std::shared_ptr<Graph> Get(int gid);
    std::vector< double > Get_value(int gid);
    void Clear();
    std::map<int, std::shared_ptr<Graph> > graph_pool; /** #整数和指向Graph的智能指针的hash对应成员变量**/
};

extern GSet GSetTrain; /** #extern 是声明**/
extern GSet GSetTest;

#endif