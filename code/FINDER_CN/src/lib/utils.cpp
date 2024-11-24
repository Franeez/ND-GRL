#include "utils.h"
#include "graph.h" //#不重复？
#include <cassert>
#include <random>
#include <algorithm>
#include <set>
#include "stdio.h"
#include <queue>
#include <stack>



Utils::Utils()
{
MaxWccSzList.clear();
MaxInfluSzList.clear();
//MaxdecycleSzList.clear();
}


std::vector<int> Utils::reInsert(std::shared_ptr<Graph> graph,std::vector<int> solution,const std::vector<int> allVex,int decreaseStrategyID,int reinsertEachStep){ //#重插入
    std::shared_ptr<decreaseComponentStrategy> decreaseStrategy; //#父指针

    switch(decreaseStrategyID){
        case 1:{
            decreaseStrategy = std::shared_ptr<decreaseComponentStrategy>(new decreaseComponentCount());
            break;
        }
        case 2:{
            decreaseStrategy = std::shared_ptr<decreaseComponentStrategy>(new decreaseComponentRank());
            break;
        }
        case 3:{
            decreaseStrategy = std::shared_ptr<decreaseComponentStrategy>(new decreaseComponentMultiple());
            break;
        }
        default:
        {
            decreaseStrategy = std::shared_ptr<decreaseComponentStrategy>(new decreaseComponentRank()); //#默认是Rank操作
            break;
        }
    }

        return reInsert_inner(solution,graph,allVex,decreaseStrategy,reinsertEachStep);

}


std::vector<int> Utils::reInsert_inner(const std::vector<int> &beforeOutput, std::shared_ptr<Graph> &graph, const std::vector<int> &allVex, std::shared_ptr<decreaseComponentStrategy> &decreaseStrategy,int reinsertEachStep)
{
    std::shared_ptr<GraphUtil> graphutil =std::shared_ptr<GraphUtil>(new GraphUtil()); //#实用操作相关指针

    std::vector<std::vector<int> > currentAdjListGraph; //#当前图的连接情况        //###是否需要初始化当前连接情况

    std::vector<std::vector<int>> backupCompletedAdjListGraph = graph->adj_list; //#前一步图的连接情况

    std::vector<bool> currentAllVex(graph->num_nodes, false); //#当前的判断bool

    for (int eachV : allVex) //#allvex表示的是剩余图中的节点
    {
        currentAllVex[eachV] = true;
    }

    std::unordered_set<int> leftOutput(beforeOutput.begin(), beforeOutput.end()); //#前一输出去重

    std::vector<int> finalOutput;

    Disjoint_Set disjoint_Set =  Disjoint_Set(graph->num_nodes); //#按图的节点规模建立一个不相交集合表达  #####少了重构连通分支具体规模？

    while (leftOutput.size() != 0)
    {
//        printf (" reInsertCount:%d\n", leftOutput.size());

        std::vector<std::pair<long long, int> >  batchList;

        for (int eachNode : leftOutput)
        {
            //min is better
            long long decreaseValue = decreaseStrategy->decreaseComponentNumIfAddNode(backupCompletedAdjListGraph, currentAllVex, disjoint_Set, eachNode); //#修改disjoint set
            batchList.push_back(make_pair(decreaseValue, eachNode));
        }


        if (reinsertEachStep >= (int)batchList.size())
        {
            reinsertEachStep = (int)batchList.size(); //#超出能够重新插入的最大数
        }
        else
        {
            std::nth_element(batchList.begin(), batchList.begin() + reinsertEachStep, batchList.end()); //#只能确定第n大的元素
        }

        for (int i = 0; i < reinsertEachStep; i++) //#两种情况 第一是遍历batchlist  第二是遍历batchlist中的前reinserteachstep部分
        {
            finalOutput.push_back(batchList[i].second); //#更新输出
            leftOutput.erase(batchList[i].second); //#循环条件更新
            graphutil->recoverAddNode(backupCompletedAdjListGraph, currentAllVex, currentAdjListGraph, batchList[i].second, disjoint_Set);//#更新了disjoint currentlist currentvex
        }

    }

    std::reverse(finalOutput.begin(), finalOutput.end()); //#反转 即第一个元素变为最后一个元素 第二个变为倒数第二个 ...

    return finalOutput; //#重新插入 多次取比第reinsertEachStep小的 reinsertEachStep个节点 直到全部solution重插完毕
}

//int Utils::influspread_one(const std::vector<std::vector<int> > &backupCompletedAdjListGraph, std::vector<bool>& backupAllVex,const std::vector< std::pair<int, int> > &edge_list,const std::vector< double > &edge_value,std::vector<double>& prob,std::vector<int> &node_select)
//{//需要保证至少有一个节点已经被激活
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
////    prob.resize((int)edge_value.size());
////    for(int i=0;i<(int)edge_value.size();++i){
////        prob.push_back(random.uniform(0, 1));//激活概率
////    }
//
////    int num_tep;
////    bool flag = true;
////    std::vector<vector<int>> node_select;//每个vector int 都是下一个vector int的前半部分，后半部分才是下一轮的新标记
//
//
////    std::set<int> influ_node_tep;
//       std::vector<int> node_tem;
//
////       std::set<int>::iterator influ=influ_node.begin();
//
//       std::set<int> influ_node2;
//
//
//
//        for(int i=0;i<num;++i){//这里还可以改进 加速
//            std::set<int>::iterator influ=influ_node.begin();
//            for(int t=0;t<i;++t){
//                influ++;
//            }
////            influ = influ_node.begin()+i;
//            for(int j=0;j<(int)backupCompletedAdjListGraph[*influ].size();++j){
//                for(int k=0;k<(int)edge_list.size();++k){
//                    if((edge_list[k].first == backupCompletedAdjListGraph[*influ][j] || edge_list[k].second == backupCompletedAdjListGraph[*influ][j] )&&( edge_list[k].first == *influ || edge_list[k].second == *influ)){
//                        if(prob[k]<edge_value[k]){
////                          influ_node.insert(edge_list[k].first);
////                          influ_node.insert(edge_list[k].second);
//                            influ_node2.insert(edge_list[k].first);
//                            influ_node2.insert(edge_list[k].second);
////                          node_select[n_s].push_back(k);
//                            node_tem.push_back(k);//是否会有重复 边号
//                        }else{
//                        continue;
//                        }
//                    }else{
//                    continue;
//                    }
//                }
//            }
//        }
//        influ_node.insert(influ_node2.begin(),influ_node2.end());
//        influ_node2.clear();
//
//
////    influ_node_tep.clear();
//    std::sort(node_tem.begin(),node_tem.end());
//    node_tem.erase(std::unique(node_tem.begin(), node_tem.end()), node_tem.end());//去重
//
//    //下面边号转换为节点
//    std::vector<int> node_tem_num;
//    for(int i=0;i<(int)node_tem.size();++i){
//        node_tem_num.push_back(edge_list[node_tem[i]].first);
//        node_tem_num.push_back(edge_list[node_tem[i]].second);
//    }
//    std::sort(node_tem_num.begin(),node_tem_num.end());
//    node_tem_num.erase(std::unique(node_tem_num.begin(), node_tem_num.end()), node_tem_num.end());//去重
//
//    for(int i=0;i<(int)backupAllVex.size();++i){
//        if(backupAllVex[i] == true){//idx 0可删 -1以删  true以删 false 未删
////            node_tem.erase()
//            for(vector<int>::iterator iter=node_tem_num.begin();iter!=node_tem_num.end();iter++){        //从vector中删除指定的某一个元素
//                if(*iter==i){
//                    node_tem_num.erase(iter);
//                    break;
//                    }
//            }
//        }
//    }
//
//
//     node_select.assign(node_tem_num.begin(),node_tem_num.end());
//
//     num=(int)influ_node.size();
//
//
//
//    //得到一轮扩散后的influence spread
//
//return num;
//
//}

int Utils::influspread(const std::vector<std::vector<int> > &backupCompletedAdjListGraph, std::vector<bool>& backupAllVex,const std::vector< std::pair<int, int> > &edge_list,const std::vector< double > &edge_value,std::vector<double>& prob,std::vector<bool> &node_select)
{
    std::set<int> influ_node;
    for(int i =0;i<(int)backupAllVex.size();++i){
        if(backupAllVex[i] == true){
            influ_node.insert(i);
        }
    }

    int num=(int)influ_node.size();//初始
//    int prim_num=(int)influ_node.size();
//    std::vector< int > prob;


//    prob.resize((int)edge_value.size());
//    std::default_random_engine generator; //#随机数 使用默认种子
//    std::uniform_real_distribution<double> distribution(0.0,1.0); //#生成离散均匀分布
//    for(int i=0;i<(int)edge_value.size();++i){
//        prob.push_back(distribution(generator));//激活概率
//    }

//    int num_tep;
//    bool flag = true;
//    std::vector<vector<int>> node_select;//每个vector int 都是下一个vector int的前半部分，后半部分才是下一轮的新标记
//    int n_s=0;//记录上面node_select的第一维
    while(true){
//    std::set<int> influ_node_tep;
//       std::vector<int> node_tem;
//       std::set<int>::iterator influ=influ_node.begin();

       std::set<int> influ_node2;


        for(int i=0;i<num;++i){//这里还可以改进 加速
            std::set<int>::iterator influ=influ_node.begin();
            for(int t=0;t<i;++t){
                influ++;
            }
//            influ = influ_node.begin()+i;
            for(int j=0;j<(int)backupCompletedAdjListGraph[*influ].size();++j){
                for(int k=0;k<(int)edge_list.size();++k){
                    if((edge_list[k].first == backupCompletedAdjListGraph[*influ][j] || edge_list[k].second == backupCompletedAdjListGraph[*influ][j] )&&( edge_list[k].first == *influ || edge_list[k].second == *influ)){
                        if(prob[k]<edge_value[k]){
//                            influ_node.insert(edge_list[k].first);
//                            influ_node.insert(edge_list[k].second);
                            influ_node2.insert(edge_list[k].first);
                            influ_node2.insert(edge_list[k].second);
//                          node_select[n_s].push_back(k);
//                            node_tem.push_back(k);
                        }else{
                            continue;
                        }
                    }else{
                        continue;
                    }
                }
            }
        }

        influ_node.insert(influ_node2.begin(),influ_node2.end());
        influ_node2.clear();

//        std::sort(node_tem.begin(),node_tem.end());
//        node_tem.erase(std::unique(node_tem.begin(), node_tem.end()), node_tem.end());//去重
//
//
//        std::vector<int> node_tem_num;
//        for(int i=0;i<(int)node_tem.size();++i){
//            node_tem_num.push_back(edge_list[node_tem[i]].first);
//            node_tem_num.push_back(edge_list[node_tem[i]].second);
//        }
//        std::sort(node_tem_num.begin(),node_tem_num.end());
//        node_tem_num.erase(std::unique(node_tem_num.begin(), node_tem_num.end()), node_tem_num.end());//去重



//        for(int i=0;i<(int)backupAllVex.size();++i){
//        if(backupAllVex[i] == true){//idx 0可删 -1以删  true以删 false 未删
////            node_tem.erase()
//            for(vector<int>::iterator iter=node_tem_num.begin();iter!=node_tem_num.end();iter++){        //从vector中删除指定的某一个元素
//                if(*iter==i){
//                    node_tem_num.erase(iter);
//                    break;
//                    }
//               }
//           }
//        }



//    influ_node_tep.clear();

//     node_select[n_s].assign(node_tem_num.begin(),node_tem_num.end()); //记录每一轮的当前轮新增被激活节点
//     n_s+=1;
     if(num==(int)influ_node.size()){
     break;
     }
     num=(int)influ_node.size();
    }//得到最终的influence spread


    node_select.assign(backupCompletedAdjListGraph.size(),true);
    for(std::set<int>::iterator ite=influ_node.begin();ite !=influ_node.end();++ite){
        int p=0;
        for(std::vector<bool>::iterator ite2=node_select.begin();ite2 !=node_select.end();++ite2){
            if(*ite == p){
                *ite2=false;
                break;
            }
            p+=1;
        }

    }





return num;

}




//int Utils::influspread(const std::vector<std::vector<int> > &backupCompletedAdjListGraph, std::vector<bool>& backupAllVex,const std::vector< std::pair<int, int> > &edge_list,const std::vector< double > &edge_value,std::vector<double>& prob,std::vector<std::vector<int>> &node_select)
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
//    std::default_random_engine generator; //#随机数 使用默认种子
//    std::uniform_real_distribution<double> distribution(0.0,1.0); //#生成离散均匀分布
//    for(int i=0;i<(int)edge_value.size();++i){
//        prob.push_back(distribution(generator));//激活概率
//    }
//
////    int num_tep;
////    bool flag = true;
////    std::vector<vector<int>> node_select;//每个vector int 都是下一个vector int的前半部分，后半部分才是下一轮的新标记
//    int n_s=0;//记录上面node_select的第一维
//    while(true){
////    std::set<int> influ_node_tep;
//       std::vector<int> node_tem;
////       std::set<int>::iterator influ=influ_node.begin();
//
//       std::set<int> influ_node2;
//
//
//        for(int i=0;i<num;++i){//这里还可以改进 加速
//            std::set<int>::iterator influ=influ_node.begin();
//            for(int t=0;t<i;++t){
//                influ++;
//            }
////            influ = influ_node.begin()+i;
//            for(int j=0;j<(int)backupCompletedAdjListGraph[*influ].size();++j){
//                for(int k=0;k<(int)edge_list.size();++k){
//                    if((edge_list[k].first == backupCompletedAdjListGraph[*influ][j] || edge_list[k].second == backupCompletedAdjListGraph[*influ][j] )&&( edge_list[k].first == *influ || edge_list[k].second == *influ)){
//                        if(prob[k]<edge_value[k]){
////                            influ_node.insert(edge_list[k].first);
////                            influ_node.insert(edge_list[k].second);
//                            influ_node2.insert(edge_list[k].first);
//                            influ_node2.insert(edge_list[k].second);
////                          node_select[n_s].push_back(k);
//                            node_tem.push_back(k);
//                        }else{
//                            continue;
//                        }
//                    }else{
//                        continue;
//                    }
//                }
//            }
//        }
//
//        influ_node.insert(influ_node2.begin(),influ_node2.end());
//        influ_node2.clear();
//
//        std::sort(node_tem.begin(),node_tem.end());
//        node_tem.erase(std::unique(node_tem.begin(), node_tem.end()), node_tem.end());//去重
//
//
//        std::vector<int> node_tem_num;
//        for(int i=0;i<(int)node_tem.size();++i){
//            node_tem_num.push_back(edge_list[node_tem[i]].first);
//            node_tem_num.push_back(edge_list[node_tem[i]].second);
//        }
//        std::sort(node_tem_num.begin(),node_tem_num.end());
//        node_tem_num.erase(std::unique(node_tem_num.begin(), node_tem_num.end()), node_tem_num.end());//去重
//
//
//
//        for(int i=0;i<(int)backupAllVex.size();++i){
//        if(backupAllVex[i] == true){//idx 0可删 -1以删  true以删 false 未删
////            node_tem.erase()
//            for(vector<int>::iterator iter=node_tem_num.begin();iter!=node_tem_num.end();iter++){        //从vector中删除指定的某一个元素
//                if(*iter==i){
//                    node_tem_num.erase(iter);
//                    break;
//                    }
//               }
//           }
//        }
//
//
//
////    influ_node_tep.clear();
//
//     node_select[n_s].assign(node_tem_num.begin(),node_tem_num.end()); //记录每一轮的当前轮新增被激活节点
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

//int Utils::influspread_multi(const std::vector<std::vector<int> > &backupCompletedAdjListGraph, std::vector<bool>& backupAllVex,const std::vector< std::pair<int, int> > &edge_list,const std::vector< double > &edge_value,int number){
//
//    int num=0;
//    std::vector<bool> backupAllVex_tem; //防止对初始backupAallVex的修改
////    backupAllVex_tem.assign<backupAllVex.begin(),backupAllVex.end()>;
//    std::vector<std::vector<double>> prob_num;//记录number次的每次概率判断
//    std::vector<std::vector<int>> node_select;
//    prob_num.resize(number);
//    for(int i=0;i<number;++i){
//        backupAllVex_tem.assign(backupAllVex.begin(),backupAllVex.end());
//        num+=influspread(backupCompletedAdjListGraph,backupAllVex_tem,edge_list,edge_value,prob_num[i],node_select);
//        for(int j=0;j<(int)node_select.size();++j){
//            node_select[j].clear();
//        }
//        node_select.clear();
//    }
//    return (int)(num/number);
//}

double Utils::influspread_multi(const std::vector<std::vector<int> > &backupCompletedAdjListGraph, std::vector<bool>& backupAllVex,const std::vector< std::pair<int, int> > &edge_list,const std::vector< double > &edge_value,int number)
{

    int num=0;
    std::vector<bool> backupAllVex_tem; //防止对初始backupAallVex的修改
//    backupAllVex_tem.assign<backupAllVex.begin(),backupAllVex.end()>;
    std::vector<std::vector<double>> prob_num;//记录number次的每次概率判断
    std::vector<std::vector<bool>> node_select;
    node_select.assign(number,std::vector <bool> (backupCompletedAdjListGraph.size(), true));
//    prob_num.assign(number,std::vector <double> (edge_value.size(), 0.0));
    prob_num.resize(number);

    std::random_device rd;
    std::default_random_engine generator{rd()}; //#随机数 使用默认种子
    std::uniform_real_distribution<double> distribution(0.0,1.0); //#生成离散均匀分布
    for(int i=0;i<number;++i){
        for(int j=0;j<(int)edge_value.size();++j){
            prob_num[i].push_back(distribution(generator));
        }

    }


    for(int i=0;i<number;++i){
        backupAllVex_tem.assign(backupAllVex.begin(),backupAllVex.end());
        num+=influspread(backupCompletedAdjListGraph,backupAllVex_tem,edge_list,edge_value,prob_num[i],node_select[i]);
//        for(int j=0;j<number;++j){
//            node_select[j].clear();
//        }
//        node_select.clear();
    }
    return (double)(num/number);
}













//int Utils::influspread_one2(std::shared_ptr<Graph> graph, std::vector<bool>& backupAllVex,std::vector<double>& prob,std::vector<int> &node_select){
//
//return (int)influspread_one(graph->adj_list,backupAllVex,graph->edge_list,graph->edge_value,prob,node_select);
//
//}


int Utils::influspread2(std::shared_ptr<Graph> graph, std::vector<bool>& backupAllVex,std::vector<double>& prob,std::vector<bool> &node_select){

return (int)influspread(graph->adj_list,backupAllVex,graph->edge_list,graph->edge_value,prob,node_select);

}

double Utils::influspread_multi2(std::shared_ptr<Graph> graph, std::vector<bool>& backupAllVex,int number){

return (double)influspread_multi(graph->adj_list, backupAllVex,graph->edge_list, graph->edge_value,number);

}










double Utils::getRobustness(std::shared_ptr<Graph> graph, std::vector<int> solution) //#图和节点序列标号
{
    assert(graph);
    MaxWccSzList.clear();
    std::vector<std::vector<int>> backupCompletedAdjListGraph = graph->adj_list;
    std::vector<std::vector<int>> current_adj_list;
    std::shared_ptr<GraphUtil> graphutil =std::shared_ptr<GraphUtil>(new GraphUtil());
    Disjoint_Set disjoint_Set =  Disjoint_Set(graph->num_nodes);
    std::vector<bool> backupAllVex(graph->num_nodes, false);
    double totalMaxNum = 0.0;
    double temp = 0.0;
    double norm = (double)graph->num_nodes * (double)(graph->num_nodes-1) / 2.0; //#节点全连接的边数目
    //printf("Norm:%.8f\n", norm);
    for (std::vector<int>::reverse_iterator it = solution.rbegin(); it != solution.rend(); ++it) //#你想迭代，即原end 到原begin逆向
    {
        int Node =(*it);
        graphutil->recoverAddNode(backupCompletedAdjListGraph, backupAllVex, current_adj_list, Node, disjoint_Set);
        // calculate the remaining components and its nodes inside
        //std::set<int> lccIDs;
        //for(int i =0;i< graph->num_nodes; i++){
        //    lccIDs.insert(disjoint_Set.unionSet[i]);
        //}
        //double CCDScore = 0.0;
        //for(std::set<int>::iterator it=lccIDs.begin(); it!=lccIDs.end(); it++)
       // {
        //    double num_nodes = (double) disjoint_Set.getRank(*it);
        //    CCDScore += (double) num_nodes * (num_nodes-1) / 2;
       // }

        totalMaxNum += disjoint_Set.CCDScore / norm; //#ccdscore是当前 pair connectivity 的值，即全部连通分支的全全连接边数求和 totalmax == ANC公式的R值？
        MaxWccSzList.push_back(disjoint_Set.CCDScore / norm); //#存储当前的ANC值
        temp = disjoint_Set.CCDScore / norm;
    }

    totalMaxNum = totalMaxNum - temp; //#去除最后一步  即还没开始删除节点的情况 最后一步是1，即考虑全部节点
    std::reverse(MaxWccSzList.begin(), MaxWccSzList.end()); //#为什么要反转 反转后第一个是最大的值 1 然后逐渐减小 即删除一个节点的情况 删除两个节点的情况...

    return (double)totalMaxNum / (double)graph->num_nodes;//#除以节点数N 得到了R值
}


double Utils::getRobustnessInflu(std::shared_ptr<Graph> graph, std::vector<int> solution)//solution是按照从大到小指标排序节点
{
    assert(graph);
    MaxInfluSzList.clear();
    std::vector<std::vector<int>> backupCompletedAdjListGraph = graph->adj_list;
//    std::vector<std::vector<int>> current_adj_list;
//    std::shared_ptr<GraphUtil> graphutil =std::shared_ptr<GraphUtil>(new GraphUtil());
//    Disjoint_Set disjoint_Set =  Disjoint_Set(graph->num_nodes);
    std::vector<bool> backupAllVex(graph->num_nodes, false);
    double totalMaxNum = 0.0;
    double temp = 0.0;

    double norm = (double)graph->num_nodes; //#节点全连接的边数目

    for (std::vector<int>::iterator it = solution.begin(); it != solution.end(); ++it) //#你想迭代，即原end 到原begin逆向
    {
        int Node =(*it);
//        graphutil->recoverAddNode(backupCompletedAdjListGraph, backupAllVex, current_adj_list, Node, disjoint_Set);
        backupAllVex[Node]=true;
        temp=(double)influspread_multi(backupCompletedAdjListGraph, backupAllVex,graph->edge_list,graph->edge_value,10);

        totalMaxNum+=temp/norm;

//        totalMaxNum += disjoint_Set.CCDScore / norm; //#ccdscore是当前 pair connectivity 的值，即全部连通分支的全全连接边数求和 totalmax == ANC公式的R值？
        MaxInfluSzList.push_back(temp/norm); //#存储当前的ANC值
//        temp = disjoint_Set.CCDScore / norm;
    }

//    totalMaxNum = totalMaxNum - temp; //#去除最后一步  即还没开始删除节点的情况 最后一步是1，即考虑全部节点
    std::reverse(MaxInfluSzList.begin(), MaxInfluSzList.end()); //#为什么要反转 反转后第一个是最大的值 1 然后逐渐减小 即删除一个节点的情况 删除两个节点的情况...

    return (double)totalMaxNum / (double)graph->num_nodes;//#除以节点数N 得到了R值

}


//double Utils::getRobustnessdecycle(std::shared_ptr<Graph> graph, std::vector<int> solution)
//{
//    assert(graph);
//    MaxdecycleSzList.clear();
//    std::vector<std::vector<int>> backupCompletedAdjListGraph = graph->adj_list;
//
//
//
//}






//每个阶段节点选择 需要输入prob 将prob固定 每一轮统一 返回指示向量
std::vector<int> Utils::getInfluactions(std::shared_ptr<Graph> graph, std::vector<bool> backupAllVex,std::vector<double> prob){

    std::vector<std::vector<int>> backupCompletedAdjListGraph = graph->adj_list;
//    std::shared_ptr<GraphUtil> graphutil =std::shared_ptr<GraphUtil>(new GraphUtil());
    std::vector<bool> node_select;
//    std::vector<double> prob;
    node_select.assign(backupCompletedAdjListGraph.size(),true);
//    prob.assign(graph->edge_value.size(),0.0);
//    int n_s=0;
//    std::vector< int > prob;
    int num=0;//当前激活节点数目
    num=influspread(backupCompletedAdjListGraph,backupAllVex,graph->edge_list,graph->edge_value,prob,node_select);

    int one_dim = node_select.size();
//    std:vector<int> two_dim ;
//    two_dim.resize(one_dim);

//    std::vector<int> node_sort;//二维转换为一维
    std::vector<int> idx_map_tem;//指示向量
    idx_map_tem.assign(graph->num_nodes,0);

//    for(int i=0;i<(int)backupAllVex.size();++i){
//        if(backupAllVex[i]==true){ //目前是已经激活true  未激活false
//            idx_map_tem[i]=0;
//        }else{
//
//            idx_map_tem[i]=0;
//        }
//
//    }


    for(int i=0;i<one_dim;++i){
        if(node_select[i] == true){
            idx_map_tem[i]=1;

        }else{

            idx_map_tem[i]=0;
        }
//        idx_map_tem[i]=1;
    }

//    bool flag=false;
//    for(int i=0;i<one_dim;++i){
//        for(int j=0;j<(int)node_select[i].size();++j){
//            if(i==0){
//            node_sort.push_back(node_select[i][j]);
//
//            }else{
//            if(std::count(node_select[i].begin(),node_select[i].end(),) == 1){ //有问题
//
//            }
//            }
//                node_sort.push_back(node_select[i][j]);
//        }
//    }



    return idx_map_tem;


}






int Utils::getMxWccSz(std::shared_ptr<Graph> graph) //#这个W是指什么含义？
{
    assert(graph);
    Disjoint_Set disjoint_Set =  Disjoint_Set(graph->num_nodes);
    for (int i = 0; i < (int)graph->adj_list.size(); i++)
    {
        for (int j = 0; j < (int)graph->adj_list[i].size(); j++)
        {
            disjoint_Set.merge(i, graph->adj_list[i][j]); //#节点i和其第j个邻居合并 更新disjoint
        }
    }
    return disjoint_Set.maxRankCount; //#最大连通分支规模
}


std::vector<double> Utils::Betweenness(std::shared_ptr<Graph> _g) { //#计算介数中心性值

	int i, j, u, v;
	int Long_max = 4294967295;
	int nvertices = _g->num_nodes;	// The number of vertices in the network
	std::vector<double> CB;
    double norm=(double)(nvertices-1)*(double)(nvertices-2);

	CB.resize(nvertices);

	std::vector<int> d;								// A vector storing shortest distance estimates
	std::vector<int> sigma;							// sigma is the number of shortest paths
	std::vector<double> delta;							// A vector storing dependency of the source vertex on all other vertices
	std::vector< std::vector <int> > PredList;			// A list of predecessors of all vertices

	std::queue <int> Q;								// A priority queue soring vertices
	std::stack <int> S;								// A stack containing vertices in the order found by Dijkstra's Algorithm

	// Set the start time of Brandes' Algorithm

	// Compute Betweenness Centrality for every vertex i
	for (i=0; i < nvertices; i++) {
		/* Initialize */
		PredList.assign(nvertices, std::vector <int> (0, 0));
		d.assign(nvertices, Long_max);
		d[i] = 0;
		sigma.assign(nvertices, 0);
		sigma[i] = 1;
		delta.assign(nvertices, 0);
		Q.push(i);

		// Use Breadth First Search algorithm
		while (!Q.empty()) {
			// Get the next element in the queue
			u = Q.front();
			Q.pop();
			// Push u onto the stack S. Needed later for betweenness computation
			S.push(u);
			// Iterate over all the neighbors of u
			for (j=0; j < (int) _g->adj_list[u].size(); j++) {
				// Get the neighbor v of vertex u
				// v = (ui64) network->vertex[u].edge[j].target;
				v = (int) _g->adj_list[u][j];

				/* Relax and Count */
				if (d[v] == Long_max) {
					 d[v] = d[u] + 1;
					 Q.push(v);
				}
				if (d[v] == d[u] + 1) {
					sigma[v] += sigma[u];
					PredList[v].push_back(u);
				}
			} // End For

		} // End While

		/* Accumulation */
		while (!S.empty()) {
			u = S.top();
			S.pop();
			for (j=0; j < (int)PredList[u].size(); j++) {
				delta[PredList[u][j]] += ((double) sigma[PredList[u][j]]/sigma[u]) * (1+delta[u]);
			}
			if (u != i)
				CB[u] += delta[u];
		}

		// Clear data for the next run
		PredList.clear();
		d.clear();
		sigma.clear();
		delta.clear();
	} // End For

	// End time after Brandes' algorithm and the time difference

    for(int i =0; i<nvertices;++i){
        CB[i]=CB[i]/norm;
    }

	return CB;

} // End of BrandesAlgorithm_Unweighted


