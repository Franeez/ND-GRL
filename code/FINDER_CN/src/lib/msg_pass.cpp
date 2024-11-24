#include "msg_pass.h"

std::shared_ptr<sparseMatrix> n2n_construct(GraphStruct* graph)
{
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result.rowNum = graph->num_nodes;
    result.colNum = graph->num_nodes;
	for (uint i = 0; i < graph->num_nodes; ++i)
	{

		auto& list = graph->in_edges->head[i];

		for (size_t j = 0; j < (int)list.size(); ++j)
		{   

            result.value.pushback(1.0);
            result.rowIndex.push_back(i);
            result.colIndex.push_back(list[j].second);
		}
	}
    return result;
}

std::shared_ptr<sparseMatrix> e2n_construct(GraphStruct* graph)
{
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result.rowNum = graph->num_nodes;
    result.colNum = graph->num_edges;
	for (uint i = 0; i < graph->num_nodes; ++i)
	{
		auto& list = graph->in_edges->head[i];
		for (size_t j = 0; j < (int)list.size(); ++j)
		{
//            result.value.pushback(1.0);
            result.value.pushback(graph->edge_value[list[j].first]);
            result.rowIndex.push_back(i);
            result.colIndex.push_back(list[j].first);
		}
	}
    return result;
}

std::shared_ptr<sparseMatrix> n2e_construct(GraphStruct* graph)
{
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result.rowNum = graph->num_edges;
    result.colNum = graph->num_nodes;

	for (uint i = 0; i < graph->num_edges; ++i)
	{
//        result.value.pushback(1.0);
        result.value.pushback(graph->edge_value[i]);
        result.rowIndex.push_back(i);
        result.colIndex.push_back(graph->edge_list[i].first);
	}
    return result;
}

std::shared_ptr<sparseMatrix> e2e_construct(GraphStruct* graph)
{
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result.rowNum = graph->num_edges;
    result.colNum = graph->num_edges;
    for (uint i = 0; i < graph->num_edges; ++i)
    {
        int node_from = graph->edge_list[i].first, node_to = graph->edge_list[i].second;
        auto& list = graph->in_edges->head[node_from]; 
        for (size_t j = 0; j < (int)list.size(); ++j)
        {
            if (list[j].second == node_to) //#无向边 ？？ 或者双向？？？
                continue;
            result.value.pushback(1.0);
            result.rowIndex.push_back(i);
            result.colIndex.push_back(list[j].first);
        }
    }
    return result;
}

std::shared_ptr<sparseMatrix> subg_construct(GraphStruct* graph) //#子图和每个子图包含的节点的对应关系的矩阵表示
{
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result.rowNum = graph->num_subgraph;
    result.colNum = graph->num_nodes;
	for (uint i = 0; i < graph->num_subgraph; ++i)
	{
		auto& list = graph->subgraph->head[i];

		for (size_t j = 0; j < (int)list.size(); ++j)
		{
            result.value.push_back(1.0);
            result.rowIndex.push_back(i);
            result.colIndex.push_back(list[j]);
		}
	}
    return result;
}
