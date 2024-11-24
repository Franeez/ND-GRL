#include "graph_struct.h"

template<typename T> //#模板类的成员函数实现格式
LinkedTable<T>::LinkedTable()
{
	n = ncap = 0;
	head.clear();	
}

template<typename T>
LinkedTable<T>::~LinkedTable()
{
	n = ncap = 0;
	head.clear();
}

template<typename T>
void LinkedTable<T>::AddEntry(int head_id, T content) //#id是下标
{			
	if (head_id >= n)
	{				
		if (head_id + 1 > ncap)
		{
			ncap = std::max(ncap * 2, head_id + 1); //#扩大容量
			head.resize(ncap);	
			for (int i = n; i < head_id + 1; ++i)
				head[i].clear(); //#并没有真正释放内存
		}
		n = head_id + 1;
	}
	
	head[head_id].push_back(content);
}

template<typename T>
void LinkedTable<T>::Resize(int new_n) //#重置size 并clear
{
	if (new_n > ncap)
	{
		ncap = std::max(ncap * 2, new_n);
		head.resize(ncap);
	}
	n = new_n;
	for (size_t i = 0; i < (int)head.size(); ++i) //#size_t是一种整数类型，记录下标
		head[i].clear();
}

template class LinkedTable<int>;
template class LinkedTable< std::pair<int, int> >;

GraphStruct::GraphStruct()
{
	out_edges = new LinkedTable< std::pair<int, int> >();
    in_edges = new LinkedTable< std::pair<int, int> >();
	subgraph = new LinkedTable< int >();
    edge_list.clear();
//    edge_value.clear();
}

GraphStruct::~GraphStruct()
{
	delete out_edges;
    delete in_edges;
	delete subgraph;
}

void GraphStruct::AddEdge(int idx, int x, int y)
{
//    std::default_random_engine generator; //#随机数 使用默认种子
//    std::uniform_real_distribution<double> distribution(0.0,1.0); //#生成离散均匀分布

    out_edges->AddEntry(x, std::pair<int, int>(idx, y)); //#节点x的出边连接节点 idx是边的下标，从0递增开始逐渐使用AddEdge
    in_edges->AddEntry(y, std::pair<int, int>(idx, x));         
	num_edges++;
    edge_list.push_back(std::make_pair(x, y));
//    edge_value.push_back(distribution(generator));
//    edge_value.push_back(); 在外面修改该变量
//    assert(edge_value.size() == edge_list.size());
    assert(num_edges == edge_list.size()); //#验证是否加入
    assert(num_edges - 1 == (unsigned)idx); //#验证idx是否符合要求
}

void GraphStruct::AddNode(int subg_id, int n_idx)
{
	subgraph->AddEntry(subg_id, n_idx); //#按照子图的下标加入节点（下标）
}

void GraphStruct::Resize(unsigned _num_subgraph, unsigned _num_nodes)
{
	num_nodes = _num_nodes;
	num_edges = 0;
    edge_list.clear();
//    edge_value.clear();
	num_subgraph = _num_subgraph;
	
	in_edges->Resize(num_nodes);
    out_edges->Resize(num_nodes);
	subgraph->Resize(num_subgraph);
}
