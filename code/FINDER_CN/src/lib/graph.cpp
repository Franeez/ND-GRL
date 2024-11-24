#include "graph.h"
#include <cassert>//#错误处理库的一部分 响应函数：若指定条件false则中断程序
#include <iostream> //#输入输出
#include <random>//#随机数及随机数分布
#include <iterator>//#迭代器 使容器可以被逐个访问，类似于指针，容器和容器操作（算法）之间的中介
//#include "stdio.h"

Graph::Graph() : num_nodes(0), num_edges(0) //#初始化，对于常量 引用 基类有参情况 成员类有参情况下使用冒号初始化，此处初始化0节点0条边
{
    edge_list.clear();//#清空vector中元素    vector::clear()
    adj_list.clear();
    edge_value.clear();
}

//Graph::Graph(Graph &G): num_nodes(G.num_nodes),num_edges(G.num_edges)
//{
//    edge_list = G.edge_list;
//    adj_list = G.adj_list;
//    edge_value = G.edge_value;
//
//}



//from和to是人为规定的顺序，即为了方便使边的记录pair中前项节点idx小于后项节点idx
Graph::Graph(const int _num_nodes, const int _num_edges, const int* edges_from, const int* edges_to)/**指针可以改 内容不能改**/
        : num_nodes(_num_nodes), num_edges(_num_edges)
{
    edge_list.resize(num_edges);// #resize是分配容器中内存
    adj_list.resize(num_nodes);
    edge_value.resize(num_edges);

    std::random_device rd;
    std::default_random_engine generator{rd()}; //#随机数 使用默认种子
    std::uniform_real_distribution<double> distribution(0.0,1.0); //#生成离散均匀分布
//    distribution = new std::uniform_real_distribution<double>(0.0, 1.0);



    for (int i = 0; i < num_nodes; ++i) //#前置++比后置++更快一些
        adj_list[i].clear(); //#清空每个节点的相邻节点记录

    for (int i = 0; i < num_edges; ++i) //连边按序号排序
    {
        int x = edges_from[i], y = edges_to[i];
        adj_list[x].push_back(y);//#邻居记录 push_back是在vector尾部加入数据
        adj_list[y].push_back(x);
        edge_list[i]=std::make_pair(edges_from[i], edges_to[i]);//#make_pair无需指定变量类型 等同于pair< >
        edge_value[i]=distribution(generator);
    }
}

Graph::Graph(const int _num_nodes, const int _num_edges, const int* edges_from, const int* edges_to, const double* edge_value_copy)/**指针可以改 内容不能改**/
        : num_nodes(_num_nodes), num_edges(_num_edges)
{
    edge_list.resize(num_edges);// #resize是分配容器中内存
    adj_list.resize(num_nodes);
    edge_value.resize(num_edges);

//    std::default_random_engine generator; //#随机数 使用默认种子
//    std::uniform_real_distribution<double> distribution(0.0,1.0); //#生成离散均匀分布
//    distribution = new std::uniform_real_distribution<double>(0.0, 1.0);



    for (int i = 0; i < num_nodes; ++i) //#前置++比后置++更快一些
        adj_list[i].clear(); //#清空每个节点的相邻节点记录

    for (int i = 0; i < num_edges; ++i) //连边按序号排序
    {
        int x = edges_from[i], y = edges_to[i];
        adj_list[x].push_back(y);//#邻居记录 push_back是在vector尾部加入数据
        adj_list[y].push_back(x);
        edge_list[i]=std::make_pair(edges_from[i], edges_to[i]);//#make_pair无需指定变量类型 等同于pair< >
        edge_value[i]=edge_value_copy[i];
    }
}




Graph::~Graph()//#将成员变量重置
{
    edge_list.clear();
    adj_list.clear();
    edge_value.clear();
    num_nodes = 0;
    num_edges = 0;
}

//节点间的连通性的一种表现， 剩余图中有共同邻居的节点对数 两节点的组合 不是排列 没有顺序
double Graph::getTwoRankNeighborsRatio(std::vector<int> covered)//#不在covered向量中的节点两两之间有共同邻居（方向不计）的数目统计
{
    std::set<int> tempSet;
    for(int i =0;i<(int)covered.size();++i){ //#去重
        tempSet.insert(covered[i]);//#insert函数插入元素到set中，返回pair<iterator,bool> iterator是插入位置
    }
    double sum  = 0;
    for(int i =0;i<num_nodes;++i){
        if(tempSet.count(i)==0){
        for(int j=i+1;j<num_nodes;++j){
        if(tempSet.count(j)==0){
            std::vector<int> v3;
            std::set_intersection(adj_list[i].begin(),adj_list[i].end(),adj_list[j].begin(),adj_list[j].end(),std::inserter(v3,v3.begin()));//#交集
            if(v3.size()>0){
                sum += 1.0;
            }
        }
        }
        }
    }
    return sum; //#不在covered集中的有公共邻居的节点对数
}

GSet::GSet()
{
    graph_pool.clear();
}

GSet::~GSet()
{
    graph_pool.clear();
}

void GSet::Clear()
{
    graph_pool.clear();
}

void GSet::InsertGraph(int gid, std::shared_ptr<Graph> graph)
{
    assert(graph_pool.count(gid) == 0); //#pool中还未加入gid相应的数据时

    graph_pool[gid] = graph;
}

std::shared_ptr<Graph> GSet::Get(int gid)
{
    assert(graph_pool.count(gid));//#条件为false则程序终止，判断关键字gid出现次数（是否存在）
    return graph_pool[gid];//#返回gid关键字对应的智能指针
}

std::vector< double > GSet::Get_value(int gid)
{
    assert(graph_pool.count(gid));
    return graph_pool[gid]->edge_value;
}


std::shared_ptr<Graph> GSet::Sample()//#取样
{
//    printf("graph_pool_size:%d",graph_pool.size());
    assert(graph_pool.size());//#判断是否为空 非空才可以采样
//    printf("graph_pool_size:%d",graph_pool.size());
    int gid = rand() % graph_pool.size(); //#0到size之间的随机数
    assert(graph_pool[gid]); //#保险期间验证一下是否指针为nullptr
    return graph_pool[gid];
}

GSet GSetTrain, GSetTest; //#定义 FINDER中使用的类的成员变量