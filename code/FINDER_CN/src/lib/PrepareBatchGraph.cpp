#include "PrepareBatchGraph.h"

sparseMatrix::sparseMatrix()
{
    rowNum = 0;
    colNum = 0;
}

sparseMatrix::~sparseMatrix()
{
    rowNum = 0;
    colNum = 0;
    rowIndex.clear();
    colIndex.clear();
    value.clear();
}

 PrepareBatchGraph::PrepareBatchGraph(int _aggregatorID)
{
    aggregatorID = _aggregatorID;
}

PrepareBatchGraph::~PrepareBatchGraph()
{
    act_select =nullptr;
    rep_global =nullptr;
    n2nsum_param =nullptr;
//    n2nsum_param_A = nullptr;
    subgsum_param =nullptr;
//    subgsum_param_A = nullptr;
    laplacian_param = nullptr;
    idx_map_list.clear();
    aux_feat.clear();
    avail_act_cnt.clear();
    aggregatorID = -1;
}

//将covered被删节点集合变为影响力激活节点集合 可以考虑孤立节点
int PrepareBatchGraph::GetStatusInfo(std::shared_ptr<Graph> g, int num, const int* covered,int& counter,int& twohop_number,int& threehop_number, std::vector<int>& idx_map)
{ //#获取状态信息函数 covered是已被删除节点记录，num是对应的covered元素总数
    std::set<int> c;

    idx_map.resize(g->num_nodes); //#按照图的节点数目对idx_map进行大小设置

    for (int i = 0; i < g->num_nodes; ++i) //#idx_map的值全部设置为-1 每个节点的标记
        idx_map[i] = -1;

    for (int i = 0; i < num; ++i) //# num 对应 covered序列，对其中元素进行去重放置到集合c
        c.insert(covered[i]);

    counter = 0; //#节点本身 即普通一条边连两节点的结构

    twohop_number = 0; //#二跳  ，类似于三角形不封闭的数目 即三个节点两条边相连的结构

    threehop_number = 0; //#三跳 下面暂时不考虑 猜测可能是通过三条连续边相连节点结构的情况
    std::set<int> node_twohop_set; //#好像没使用上

    int n = 0; //#记录非孤立节点且不在covered中的节点数目（除去连边另一端全是covered中节点的情况，此时也是孤立节点了）
 	std::map<int,int> node_twohop_counter; //#记录每个节点（first）的对应twohop_number（second）值

    for (auto& p : g->edge_list) //#c++11中容器内容遍历的新写法 遍历所有连边（有向）
    {

        if (c.count(p.first) || c.count(p.second))
        {
            counter++;         //#covered集合涉及到的边的数目记录 即已被删除边的数目
        } else {

            if (idx_map[p.first] < 0) //#证明该节点还没有经过操作 即第一次进行下面的操作
                n++; //#记录不在covered中的节点数 不包括孤立节点

            if (idx_map[p.second] < 0)
                n++;
            idx_map[p.first] = 0;//还存在剩余图中
            idx_map[p.second] = 0;

           if(node_twohop_counter.find(p.first) != node_twohop_counter.end()) //#该二跳节点 有多少连边
          {
              twohop_number+=node_twohop_counter[p.first]; //#对一个节点 0+1+2+3...  twohop_number对所有满足条件节点进行0+1+2...
              node_twohop_counter[p.first]=node_twohop_counter[p.first]+1; //#p.second 记录出现次数，即不在covered中节点的 连边不连covered节点集的边数
          }
          else{
              node_twohop_counter.insert(std::make_pair(p.first,1));
          }


          if(node_twohop_counter.find(p.second) != node_twohop_counter.end())
          {
              twohop_number+=node_twohop_counter[p.second];
              node_twohop_counter[p.second]=node_twohop_counter[p.second]+1;
          }
          else{
              node_twohop_counter.insert(std::make_pair(p.second,1));
          }
        }
    }


    for(int i=0;i<g->num_nodes;++i){
        if(c.count(i)){
            continue;
    }

    if(idx_map[i]==0){

        continue;
    }
    n++;//孤立节点
    idx_map[i]=0;

    }

//    assert(n == (int)(g->num_nodes-c.size()));

//    for(int i=0;i<g->num_nodes;++i){
//        printf("%d,",idx_map[i]);
//
//    }
//    printf("\n");
//    printf("(%d,%d) ",g->num_nodes,c.size());

    return g->num_nodes-c.size(); //#不在covered节点集中且涉及连边的节点数  现在包括了孤立节点 直接return n大小会多1？？？？
}


void PrepareBatchGraph::SetupGraphInput(std::vector<int> idxes,
                           std::vector< std::shared_ptr<Graph> > g_list,
                           std::vector< std::vector<int> > covered,
                           const int* actions)
{
    act_select = std::shared_ptr<sparseMatrix>(new sparseMatrix()); //#动作选择
    rep_global = std::shared_ptr<sparseMatrix>(new sparseMatrix());

    idx_map_list.resize(idxes.size()); //#idexes 是涉及的下标 不同的子图的相应idx_map集合 -1和0 标记
    avail_act_cnt.resize(idxes.size()); //#与上面下标序列对应的 不同子图的GetStatusInfo数值  不同子图的当前可删节点数目 剩余节点数

    int node_cnt = 0;
//    int node_test = 0;
//    int sumnum = 0;

    for (int i = 0; i < (int)idxes.size(); ++i) //#不同子图遍历
    {
        std::vector<double> temp_feat;

        auto g = g_list[idxes[i]];  //#与下标idxes相关的图g

        int counter;
        int twohop_number;
        int threehop_number;

        if (g->num_nodes) //#节点数非空

            temp_feat.push_back((double)covered[idxes[i]].size() / (double)g->num_nodes); //#每个图g对应的covered集合节点占总节点数的比例

        avail_act_cnt[i] = GetStatusInfo(g, covered[idxes[i]].size(), covered[idxes[i]].data(), counter,twohop_number,threehop_number, idx_map_list[i]);
//可以直接换为g->num_nodes - (double)covered[idxes[i]].size()
//        if (g->edge_list.size()) //#边非空 不全是孤立节点

//            temp_feat.push_back((double)counter / (double)g->edge_list.size()); //#counter集的边数占总边数比例

//         temp_feat.push_back((double)twohop_number / ((double)g->num_nodes * (double)g->num_nodes)); //#twohop结构占比

         temp_feat.push_back(1.0); //#每条记录的标记分割
//         node_test+=covered[idxes[i]].size();
//         sumnum+=g->num_nodes;
         node_cnt += avail_act_cnt[i]; //#所有子图的总GetStatusInfo  即整体当前可被删节点数目
         aux_feat.push_back(temp_feat); //#存储不同子图的每条temp_feat记录
    }
//    assert(sumnum == node_test+node_cnt);
    graph.Resize(idxes.size(), node_cnt); //#重设有向图graphstruct 子图数和节点总数 此时图重构了

    if (actions) //#非空 [子图数，剩余节点] 重新编号后的（从0开始...）
    {
        act_select->rowNum=idxes.size(); //#设置动作稀疏矩阵
        act_select->colNum=node_cnt;
    } else //剩余节点的所属子图指示 [剩余节点，子图数] 重新编号后的 终止态后的记录（从0开始）
    {
        rep_global->rowNum=node_cnt;
        rep_global->colNum=idxes.size();
    }


    node_cnt = 0; //#为了使所有子图构成的图的所有节点按顺序排序节点
    int edge_cnt = 0; //#排序边

    for (int i = 0; i < (int)idxes.size(); ++i)//#遍历每个子图
    {
        auto g = g_list[idxes[i]];
        std::vector<int> idx_map;
        for(int j =0;j<(int)idx_map_list[i].size();++j){
            idx_map.push_back(idx_map_list[i][j]);

        }
//        auto idx_map = idx_map_list[i];

        int t = 0;
        for (int j = 0; j < (int)g->num_nodes; ++j) //#遍历当前子图每个节点
        {
            if (idx_map[j] < 0)
                continue;
            idx_map[j] = t; //#按照搜索顺序设定节点的指标，0，1，2，...
            graph.AddNode(i, node_cnt + t); //#对应子图加入节点下标 根据子图序号i 加节点
            if (!actions)
            {
                rep_global->rowIndex.push_back(node_cnt + t);
                rep_global->colIndex.push_back(i);
                rep_global->value.push_back(1.0);
            }
            t += 1;
        }
        assert(t == avail_act_cnt[i]); //#判断是否t等于当前子图i剩余节点数目

//        auto idx_map_temp = idx_map;

        if (actions)
        {
            auto act = actions[idxes[i]]; //#act是相应子图对应的那个操作节点下标？
            assert(idx_map[act] >= 0 && act >= 0 && act < g->num_nodes); //#保证选择的act节点是在剩余子图中 且序号满足0到最大之间
            act_select->rowIndex.push_back(i);
            act_select->colIndex.push_back(node_cnt + idx_map[act]);//#新的序号表示 idx_map中存储
            act_select->value.push_back(1.0);
        }
        // g-> edge_value需要考虑设置
        int temp = 0;
        for (auto p : g->edge_list) //#此时的edge_list是否为空 非空，注意g和graph不同
        {
            if (idx_map[p.first] < 0 || idx_map[p.second] < 0)
            {
                temp += 1;
                continue;
            }
            auto x = idx_map[p.first] + node_cnt, y = idx_map[p.second] + node_cnt;
            graph.AddEdge(edge_cnt, x, y);
//            graph.edge_value.push_back(g->edge_value[temp]);
            edge_cnt += 1;
            graph.AddEdge(edge_cnt, y, x);//#双向记录，实际为无向图   人为加方向，记录时一条无向边看作两条双向有向边
//            graph.edge_value.push_back(g->edge_value[temp]);
            edge_cnt += 1;
            temp += 1;
        }
        node_cnt += avail_act_cnt[i];
    }
    assert(node_cnt == (int)graph.num_nodes); //#验证剩余总节点数是否相等
//    assert(graph.edge_value.size() == graph.edge_list.size()); //判断

    auto result_list = n2n_construct(&graph,aggregatorID); //生成的n2n laplacian是对称矩阵，因为将无向图看作双向有向存储记录
    n2nsum_param = result_list[0]; //邻接矩阵A value==1
    laplacian_param = result_list[1]; //拉普拉斯矩阵 D-A 度矩阵-邻接矩阵
//    n2nsum_param_A = result_list[2];
    subgsum_param = subg_construct(&graph,subgraph_id_span);//子图与节点所属对应 value==1
    //subgsum_param_A = subg_construct_st(&graph,sol);


}





void PrepareBatchGraph::SetupTrain(std::vector<int> idxes,
                           std::vector< std::shared_ptr<Graph> > g_list,
                           std::vector< std::vector<int> > covered,
                           const int* actions)
{
    SetupGraphInput(idxes, g_list, covered, actions);
}



void PrepareBatchGraph::SetupPredAll(std::vector<int> idxes,
                           std::vector< std::shared_ptr<Graph> > g_list,
                           std::vector< std::vector<int> > covered)
{
    SetupGraphInput(idxes, g_list, covered, nullptr);
}




//// ~~~ 将A和Laplace变为带权（直接）
//std::vector<std::shared_ptr<sparseMatrix>> n2n_construct_weight(GraphStruct* graph,int aggregatorID)
//{
//    //aggregatorID = 0 sum
//    //aggregatorID = 1 mean
//    //aggregatorID = 2 GCN
//    std::vector<std::shared_ptr<sparseMatrix>> resultList;
//    resultList.resize(3);
//    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix()); //#连边权重稀疏矩阵
//    result->rowNum = graph->num_nodes;
//    result->colNum = graph->num_nodes;
//
//    std::shared_ptr<sparseMatrix> result_A =std::shared_ptr<sparseMatrix>(new sparseMatrix()); //#连边权重稀疏矩阵
//    result_A->rowNum = graph->num_nodes;
//    result_A->colNum = graph->num_nodes;
//
//    std::shared_ptr<sparseMatrix> result_laplacian= std::shared_ptr<sparseMatrix>(new sparseMatrix());
//    result_laplacian->rowNum = graph->num_nodes; //#拉普拉斯矩阵规格是节点数N×N的
//    result_laplacian->colNum = graph->num_nodes;
//
////    std::vector<double> HDA_node;
////    HDA_node.resize(graph->num_nodes);
//
//
//
//	for (int i = 0; i < (int)graph->num_nodes; ++i)//#遍历节点
//	{
//		auto& list = graph->in_edges->head[i]; //#存储着pair<边的标号，相关入节点标号>
//		// i 相关的边及相连节点的id pair类型
//		double weight=0.0;
////		weight.assign(list.size(),0.0);//初始化
//		//x->y  :         y == i  x == list[`].second  edge_id = list[`].first
//
//
//
//
//        if (list.size() > 0)//连边非空
//        {
//
//            for(int j = 0;j<(int)list.size();++j){
//                weight += graph->edge_value[list[j].first];
//
//
//            }
//
////            std::cout << weight<< std::endl;
//            result_laplacian->value.push_back(weight); //#权求和
////            result_laplacian->value.push_back()
//		    result_laplacian->rowIndex.push_back(i);
//		    result_laplacian->colIndex.push_back(i);
////		    HDA_node[i] = weight;
//        }
//
//		for (int j = 0; j < (int)list.size(); ++j) //#遍历每条边
//		{
//		    switch(aggregatorID){
//		       case 0:
//		       {
//		          result->value.push_back(graph->edge_value[list[j].first]); //#边权值？
//		          result_A->value.push_back(1.0);
//		          break;
//		       }
//		       case 1:
//		       {
//		          result->value.push_back(1.0/(double)list.size());
//		          break;
//		       }
//		       case 2:
//		       {
//		          int neighborDegree = (int)graph->in_edges->head[list[j].second].size();//#邻居节点的入度
//		          int selfDegree = (int)list.size();
//		          double norm = sqrt((double)(neighborDegree+1))*sqrt((double)(selfDegree+1));
//		          result->value.push_back(1.0/norm);
//		          break;
//		       }
//		       default:
//		          break;
//		    }
//
//            result->rowIndex.push_back(i);
//            result->colIndex.push_back(list[j].second);
//
//
//            result_A->rowIndex.push_back(i);
//            result_A->colIndex.push_back(list[j].second);
//
//            result_laplacian->value.push_back(-1.0*graph->edge_value[list[j].first]);
//		    result_laplacian->rowIndex.push_back(i);
//		    result_laplacian->colIndex.push_back(list[j].second);
//
//		}
//	}
//	resultList[0]= result;
//	resultList[1] = result_laplacian;
//	resultList[2] = result_A;
//
////	//计算HDA
////	double HDA_result = (*std::max_element(HDA_node.begin(), HDA_node.begin() + graph->num_nodes));
//////	int HDA_index=0;
////	for(int i=0;i<graph->num_nodes;i++){
////	    if(HDA_node[i]==HDA_result){
////	        *HDA_index = i;
////	        break;
////	    }
////
////	}
//
//
//    return resultList;
//}













std::vector<std::shared_ptr<sparseMatrix>> n2n_construct(GraphStruct* graph,int aggregatorID)
{
    //aggregatorID = 0 sum
    //aggregatorID = 1 mean
    //aggregatorID = 2 GCN
    std::vector<std::shared_ptr<sparseMatrix>> resultList;
    resultList.resize(2);
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix()); //#连边权重稀疏矩阵
    result->rowNum = graph->num_nodes;
    result->colNum = graph->num_nodes;

    std::shared_ptr<sparseMatrix> result_laplacian= std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result_laplacian->rowNum = graph->num_nodes; //#拉普拉斯矩阵规格是节点数N×N的
    result_laplacian->colNum = graph->num_nodes;

	for (int i = 0; i < (int)graph->num_nodes; ++i)//#遍历节点
	{
		auto& list = graph->in_edges->head[i]; //#存储着pair<边的标号，相关入节点标号>


        if (list.size() > 0)
        {
            result_laplacian->value.push_back(list.size()); //#度值
//            result_laplacian->value.push_back()
		    result_laplacian->rowIndex.push_back(i);
		    result_laplacian->colIndex.push_back(i);
        }

		for (int j = 0; j < (int)list.size(); ++j) //#遍历每条边
		{
		    switch(aggregatorID){
		       case 0:
		       {
		          result->value.push_back(1.0); //#边权值？
		          break;
		       }
		       case 1:
		       {
		          result->value.push_back(1.0/(double)list.size());
		          break;
		       }
		       case 2:
		       {
		          int neighborDegree = (int)graph->in_edges->head[list[j].second].size();//#邻居节点的入度
		          int selfDegree = (int)list.size();
		          double norm = sqrt((double)(neighborDegree+1))*sqrt((double)(selfDegree+1));
		          result->value.push_back(1.0/norm);
		          break;
		       }
		       default:
		          break;
		    }

            result->rowIndex.push_back(i);
            result->colIndex.push_back(list[j].second);

            result_laplacian->value.push_back(-1.0);
		    result_laplacian->rowIndex.push_back(i);
		    result_laplacian->colIndex.push_back(list[j].second);

		}
	}
	resultList[0]= result;
	resultList[1] = result_laplacian;
    return resultList;
}

std::shared_ptr<sparseMatrix> e2n_construct(GraphStruct* graph)
{
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_nodes;
    result->colNum = graph->num_edges; //#边数
	for (int i = 0; i < (int)graph->num_nodes; ++i)
	{
		auto& list = graph->in_edges->head[i];
		for (int j = 0; j < (int)list.size(); ++j)
		{
            result->value.push_back(1.0);
//            result->value.push_back(graph->edge_value[list[j].first]);
            result->rowIndex.push_back(i);
            result->colIndex.push_back(list[j].first);
		}
	}
    return result;
}

std::shared_ptr<sparseMatrix> n2e_construct(GraphStruct* graph)
{
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_edges;
    result->colNum = graph->num_nodes;

	for (int i = 0; i < (int)graph->num_edges; ++i)
	{
        result->value.push_back(1.0);
//        result->value.push_back(graph->edge_value[i]);
        result->rowIndex.push_back(i);
        result->colIndex.push_back(graph->edge_list[i].first);
	}
    return result;
}

std::shared_ptr<sparseMatrix> e2e_construct(GraphStruct* graph)
{
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_edges;
    result->colNum = graph->num_edges;
    for (int i = 0; i < (int)graph->num_edges; ++i)
    {
        int node_from = graph->edge_list[i].first, node_to = graph->edge_list[i].second;
        auto& list = graph->in_edges->head[node_from]; //#node_from的相连的边的入节点列表
        for (int j = 0; j < (int)list.size(); ++j)
        {
            if (list[j].second == node_to)
                continue;
            result->value.push_back(1.0);
            result->rowIndex.push_back(i);
            result->colIndex.push_back(list[j].first);//#边的指标
        }
    }
    return result;
}

std::shared_ptr<sparseMatrix> subg_construct(GraphStruct* graph, std::vector<std::pair<int,int>>& subgraph_id_span)
{
   std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_subgraph; //上面resize时赋值了
    result->colNum = graph->num_nodes;

    subgraph_id_span.clear();
    int start = 0;
    int end = 0;
	for (int i = 0; i < (int)graph->num_subgraph; ++i)//#遍历每个子图
	{

		auto& list = graph->subgraph->head[i]; //#子图涉及的节点集 上面AddNode时已经设置好
        end  = start + list.size() - 1;
		for (int j = 0; j < (int)list.size(); ++j)
		{
            result->value.push_back(1.0);
            result->rowIndex.push_back(i);
            result->colIndex.push_back(list[j]);
		}
		if(list.size()>0){
		    subgraph_id_span.push_back(std::make_pair(start,end));
		}
		else{
		    subgraph_id_span.push_back(std::make_pair(graph->num_nodes,graph->num_nodes));//#子图为空时，用(总节点数，总节点数)标记
		}
		start = end +1 ;
	}
    return result;
}

std::shared_ptr<sparseMatrix> subg_construct_st(GraphStruct* graph, std::vector<std::vector<int>> sol)
{
   std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_subgraph; //上面resize时赋值了
    result->colNum = graph->num_nodes;



	for (int i = 0; i < (int)graph->num_subgraph; ++i)//#遍历每个子图
	{

//		auto& list = graph->subgraph->head[i]; //#子图涉及的节点集 上面AddNode时已经设置好
		auto& list = sol[i];

		for (int j = 0; j < (int)list.size(); ++j)
		{
		    if(list[j]==1){
		        result->value.push_back(1.0);
                result->rowIndex.push_back(i);
                result->colIndex.push_back(j);

		    }

		}

	}
    return result;
}