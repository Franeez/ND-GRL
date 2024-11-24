#include "mvc_env.h"
#include "graph.h"
#include <iostream>
#include <cassert>
#include <random>
#include <algorithm>
#include <set>
#include <queue> //#队列
#include <stack> //#堆栈
#include <numeric>
#include <exception>
//#include <cstring>



struct edge
{
    int u;
    int v;
};
//void clear_edge(std::vector<struct edge>& vt){
//    std::vector<struct edge> vtemp;
//    vtemp.swap(vt);
//}


int ln; int nodeN;
int bn; int num;




MvcEnv::MvcEnv(double _norm) //#这里的norm是否等于(n-1)*(n-2) 介数？...
{
norm = _norm;
k_num = 8;
CNR = 0.0;
CNR_all = 1.0;
CNR_all_flag = false;
graph = nullptr;
//g_relation = nullptr;

numCoveredEdges = 0;
numCoveredNodes = 0;
cycle_node_all = 0;

node_act_flag.clear();
edge_act_flag.clear();
left.clear();
right.clear();

record_cycle.clear();

state_seq.clear();
act_seq.clear();
action_list.clear();
reward_seq.clear();
sum_rewards.clear();
covered_set.clear();
avail_list.clear();
prob.clear();
node_degrees.clear();

}

MvcEnv::~MvcEnv()
{
    norm = 0;
    k_num = 0;
    CNR = 0.0;
    CNR_all = 0.0;
    CNR_all_flag = false;
    graph = nullptr;
//    g_relation = nullptr;

    numCoveredEdges = 0;
    numCoveredNodes = 0;

    cycle_node_all = 0;

    node_act_flag.clear();
    edge_act_flag.clear();
    left.clear();
    right.clear();

    record_cycle.clear();

    state_seq.clear();
    act_seq.clear();
    action_list.clear();
    reward_seq.clear();
    sum_rewards.clear();
    covered_set.clear();
    avail_list.clear();
    prob.clear();
    node_degrees.clear();

}

void MvcEnv::s0(std::shared_ptr<Graph> _g)  //#初始状态s0？
{
    graph = _g;
    covered_set.clear();
    action_list.clear();
    numCoveredEdges = 0; //被覆盖的
    numCoveredNodes = 0;

    cycle_node_all = 0;
    k_num = 10;
    CNR = 1.0;
    CNR_all = 1.0;
    CNR_all_flag = false;

    record_cycle.assign(_g->num_nodes,1);

    state_seq.clear();
    act_seq.clear();
    reward_seq.clear();
    sum_rewards.clear();

    edge_act_flag.clear();
    edge_act_flag.assign(_g->num_edges,0);// 0 :没激活  1：激活

    node_act_flag.clear();
    node_act_flag.assign(_g->num_nodes,0);

    std::random_device rd;
    std::default_random_engine generator{rd()}; //#随机数 使用默认种子
    std::uniform_real_distribution<double> distribution(0.0,1.0); //#生成离散均匀分布
    for(int i=0;i<(int)_g->edge_value.size();++i){
        prob.push_back(distribution(generator));//激活概率
    }


    for(int i =0; i<(int)_g->num_edges; ++i){
        if(_g->edge_value[i]>prob[i]){
            left.push_back(_g->edge_list[i].first);
            right.push_back(_g->edge_list[i].second);
            edge_act_flag[i] = i+1;
        }
    }
//    int *e_from = new int[left.size()];
//    int *e_to = new int[right.size()];
//    for(int i =0;i<left.size();++i){
//        e_from[i] = left[i];
//        e_to[i] = right[i];
//    }
//    Graph g_tem(_g->num_nodes,_g->num_edges,e_from,e_to,);


}


void MvcEnv::get_degrees()
{
    node_degrees.resize(graph->num_nodes);
    //每轮初始化原始图的分布
    for(int i = 0; i<graph->adj_list.size();++i)
    {
        node_degrees[i] = graph->adj_list[i].size();
    }
    //参照剩余图修改度  自适应

    for(std::set<int>::iterator it = covered_set.begin();it!=covered_set.end();++it)
    {
        for(int i = 0; i<graph->adj_list[*it].size();++i)
        {
            node_degrees[graph->adj_list[*it][i]]--;
        }
    }
    for(std::set<int>::iterator it = covered_set.begin();it!=covered_set.end();++it)
    {
        node_degrees[*it]=0;
    }

}








//节点a可被删 变为 可变为激活节点
//double MvcEnv::step(int a)
//
//{
//    //k_num--;
//    assert(graph); //#非空判断
//    assert(covered_set.count(a) == 0); //#节点a不在covered集合里
//    state_seq.push_back(action_list); //#action_list用来表示state？    删除a节点这一步骤前已删除节点记录（含顺序）表示为一个state   已删除节点按顺序排列 表示当前图的状态
//    act_seq.push_back(a); //#删除节点顺序记录
//    covered_set.insert(a);
//    int sum1 = std::accumulate(node_act_flag.begin(),node_act_flag.end(),0);
//    int sum2 = sum1;
//    action_list.push_back(a);//#为了赋给state seq 所设立的变量
//
//    if(node_act_flag[a] == 0)
//    {
//
//
//        node_act_flag[a] = 1;
//
//
//        std::queue<int> Q;
//        Q.push(a);
//
//        //do{
//            //sum1 = sum2;
//            while(!Q.empty()){
//
//             //sum1 = sum2;
//             int u = Q.front();
//             Q.pop();
//             for(int i =0 ; i<(int)graph->adj_list[u].size();++i){
//                    if(node_act_flag[graph->adj_list[u][i]] == 1){
//                        continue;
//                    }
//                    if(u<graph->adj_list[u][i]){
//                    //make_pair(u,graph->adj_list[u][i])
//                    //std::vector<int>::iterator it = find(graph->edge_list.begin(), graph->edge_list.end(), make_pair(u,graph->adj_list[u][i]));
//                    for(int j =0 ;j<(int)graph->edge_list.size();++j){
//                        if(graph->edge_list[j] == std::make_pair(u,graph->adj_list[u][i])){
//                            if(edge_act_flag[j]!=0){
//                                node_act_flag[graph->adj_list[u][i]] = 1;
//                                Q.push(graph->adj_list[u][i]);
//                                break;
//                            }else{
//                                break;
//                            }
//
//                        }
//
//                    }
//
//                    }else{
//                    for(int j =0 ;j<(int)graph->edge_list.size();++j){
//                        if(graph->edge_list[j] == std::make_pair(graph->adj_list[u][i],u)){
//                            if(edge_act_flag[j]!=0){
//                                node_act_flag[graph->adj_list[u][i]] = 1;
//                                Q.push(graph->adj_list[u][i]);
//                                break;
//                            }else{
//                                break;
//                            }
//
//                        }
//
//                    }
//
//                    }
//
//            }
//
//
//
//            }
//
//
//            //sum2 = accumulate(node_act_flag.begin(),node_act_flag.end(),0);
//        //}while(sum1 != sum2);
//
//    }
//    sum2 = std::accumulate(node_act_flag.begin(),node_act_flag.end(),0);
//
//
//
//
//
//    for (auto neigh : graph->adj_list[a])
//        if (covered_set.count(neigh) == 0) //#同时避免了重复计算
//            numCoveredEdges++; //#代表删除节点的同时也删除相关的连边
//    numCoveredNodes++;
////    double r_t = getReward(oldCcNum);
////    double r_t = getReward();//#当前的ANC值 即删除节点a这一步执行后的ANC
//    double r_t = (double)sum2 - sum1;
//    reward_seq.push_back(r_t);
//    sum_rewards.push_back(r_t);   //#为啥有两个reward记录？？？？？？
//
//    return r_t;
//}



double MvcEnv::step_fake(int a)
{
    assert(graph); //#非空判断
    assert(covered_set.count(a) == 0); //#节点a不在covered集合里
    covered_set.insert(a);

    double r_t = 0.0;//#当前的ANC值 即删除节点a这一步执行后的ANC
    if(record_cycle[a]==0){
        r_t = 0.0;
    }else{
        r_t = getReward_absolute(); //需要改一下record_cycle 下面的reverse也是要注意
    }
    return r_t;
}


double MvcEnv::step_fake_reverse(int a)
{
    assert(graph);
    assert(covered_set.count(a) == 1);
    for(std::set<int>::iterator it=covered_set.begin();it!=covered_set.end();++it){
        if(*it == a){
            covered_set.erase(it);
        }
    }
    getReward_absolute(); //恢复record_cycle有关于节点a的标记



}


//节点a可被删 变为 可变为激活节点
double MvcEnv::step(int a)
{
    k_num--;
    assert(graph); //#非空判断
    get_degrees();

    assert(covered_set.count(a) == 0); //#节点a不在covered集合里
    state_seq.push_back(action_list); //#action_list用来表示state？    删除a节点这一步骤前已删除节点记录（含顺序）表示为一个state   已删除节点按顺序排列 表示当前图的状态
    act_seq.push_back(a); //#删除节点顺序记录
    covered_set.insert(a);

    action_list.push_back(a);//#为了赋给state seq 所设立的变量


    for (auto neigh : graph->adj_list[a])
        if (covered_set.count(neigh) == 0) //#同时避免了重复计算
            numCoveredEdges++; //#代表删除节点的同时也删除相关的连边
    numCoveredNodes++;
//    double r_t = getReward(oldCcNum);
    double r_t = 0.0;//#当前的ANC值 即删除节点a这一步执行后的ANC
    double r_tem = 0.0;
    if(record_cycle[a]==0){//非环中节点
        r_t = -1.0;
    }else{
        r_tem = getReward();
        r_t = -0.5/(1+node_degrees[a])+0.5*r_tem;
    }

 //  double r_t = (double)sum2 - sum1;
    reward_seq.push_back(r_t);
    sum_rewards.push_back(r_t);   //#为啥有两个reward记录？？？？？？
    std::cout<<"step function"<<std::endl;

    return r_t;
}


void MvcEnv::stepWithoutReward(int a) // #取节点并不考虑reward的情况

{
    k_num--;
    assert(graph); //#非空
    assert(covered_set.count(a) == 0);// #节点a还没被删除
    covered_set.insert(a);
    action_list.push_back(a); //#记录了顺序 同时也为了赋给state seq

//    if(node_act_flag[a] == 0)
//    {
//
//
//        node_act_flag[a] = 1;
//        //int sum1 = accumulate(node_act_flag.begin(),node_act_flag.end(),0);
//        //int sum2 = sum1;
//        std::queue<int> Q;
//        Q.push(a);
//
//        //do{
//            //sum1 = sum2;
//            while(!Q.empty()){
//
//             //sum1 = sum2;
//             int u = Q.front();
//             Q.pop();
//             for(int i =0 ; i<(int)graph->adj_list[u].size();++i){
//                    if(node_act_flag[graph->adj_list[u][i]] == 1){
//                        continue;
//                    }
//                    if(u<graph->adj_list[u][i]){
//                    //make_pair(u,graph->adj_list[u][i])
//                    //std::vector<int>::iterator it = find(graph->edge_list.begin(), graph->edge_list.end(), make_pair(u,graph->adj_list[u][i]));
//                    for(int j =0 ;j<(int)graph->edge_list.size();++j){
//                        if(graph->edge_list[j] == std::make_pair(u,graph->adj_list[u][i])){
//                            if(edge_act_flag[j]!=0){
//                                node_act_flag[graph->adj_list[u][i]] = 1;
//                                Q.push(graph->adj_list[u][i]);
//                                break;
//                            }else{
//                                break;
//                            }
//
//                        }
//
//                    }
//
//                    }else{
//                    for(int j =0 ;j<(int)graph->edge_list.size();++j){
//                        if(graph->edge_list[j] == std::make_pair(graph->adj_list[u][i],u)){
//                            if(edge_act_flag[j]!=0){
//                                node_act_flag[graph->adj_list[u][i]] = 1;
//                                Q.push(graph->adj_list[u][i]);
//                                break;
//                            }else{
//                                break;
//                            }
//
//                        }
//
//                    }
//
//                    }
//
//            }
//
//
//
//            }
//
//
//            //sum2 = accumulate(node_act_flag.begin(),node_act_flag.end(),0);
//        //}while(sum1 != sum2);
//
//    }


    for (auto neigh : graph->adj_list[a])
        if (covered_set.count(neigh) == 0)
            numCoveredEdges++;// #没被删除的邻居节点的连边记录为被删，这个变量为了判断是否达到终态terminal
    numCoveredNodes++;
}


// random
int MvcEnv::randomAction() //#随机选取一个非孤立节点
{
    assert(graph);
    avail_list.clear(); //#下面找出还没删除的节点中的非孤立节点

    for(int i = 0; i < graph->num_nodes; ++i)
        if(covered_set.count(i) == 0)
        {

//            bool useful = false;
//            for (auto neigh : graph->adj_list[i])
//                if (covered_set.count(neigh) == 0) //#除非所有邻居都是covered，否则可以取i节点到avail_list
//                {
//                    useful = true;
//                    break;
//                }
//            if (useful)
            avail_list.push_back(i);  //#非孤立节点加入avail_list中
        }

    assert(avail_list.size()); //#如果均为孤立节点 则结束程序？
    int idx = rand() % avail_list.size();  //#随机取一个指标
    return avail_list[idx];
}


int MvcEnv::betweenAction() //#返回剩余图中介数中心性最大的节点编号
{
    assert(graph);

    std::map<int,int> id2node;
    std::map<int,int> node2id;

    std::map <int,std::vector<int>> adj_dic_origin; //#记录初始时没被选取的节点 及其没被选取的邻居节点集
    std::vector<std::vector<int>> adj_list_reID; //#将上面原节点编号改为重新定义的编号 记录没被删除节点间的链接关系（邻居关系）
    std::vector<std::vector<int>> adj_list_reID_noisol;//不含孤立节点的reID

    for (int i = 0; i < graph->num_nodes; ++i)
    {
        if (covered_set.count(i) == 0)  //#i没被选取
        {
            for (auto neigh : graph->adj_list[i]) //#遍历邻居
            {
                if (covered_set.count(neigh) == 0) //#邻居没被选取
                {
                   if(adj_dic_origin.find(i) != adj_dic_origin.end()) //#origin中有节点i
                   {
                       adj_dic_origin[i].push_back(neigh); //#将邻居加入节点i所指的vector
                   }
                   else{
                       std::vector<int> neigh_list;
                       neigh_list.push_back(neigh);
                       adj_dic_origin.insert(std::make_pair(i,neigh_list)); //#创建一个新的origin中的项
                   }
                }
            }
        }

    }

    std::vector<int> map_key;
    map_key.resize(adj_dic_origin.size());

    std::map<int, std::vector<int>>::iterator iter;
    iter = adj_dic_origin.begin(); //#记录开始位置
    while(iter != adj_dic_origin.end())
     {
        map_key.push_back(iter->first);
        iter++;
     }

    int star=adj_dic_origin.size();
     adj_list_reID_noisol.resize(adj_dic_origin.size());

    //将剩余图中的孤立节点加入
    for(int i=0;i<graph->num_nodes;++i){
        if(covered_set.count(i)==0 && std::count(map_key.begin(),map_key.end(),i)==0){
        adj_dic_origin.insert(std::make_pair(i,std::vector <int> (0, 0)));
        }
    }


//     std::map<int, std::vector<int>>::iterator iter;
     iter = adj_dic_origin.begin(); //#记录开始位置

     int numrealnodes = 0;
     while(iter != adj_dic_origin.end())
     {
        id2node[numrealnodes] = iter->first;// #节点记录在id2node里
        node2id[iter->first] = numrealnodes; //#反向记录 关键字——值
        numrealnodes += 1;
        iter++;
     }

     adj_list_reID.resize(adj_dic_origin.size());


     int stop=0;
     iter = adj_dic_origin.begin();
     while(iter != adj_dic_origin.end())
     {
        for(int i=0;i<(int)iter->second.size();++i){ //#对每个节点i 遍历其符合条件的邻居节点
            adj_list_reID[node2id[iter->first]].push_back(node2id[iter->second[i]]); //#使用上面循环中设置的节点id重新进行节点及邻居的对应存储
        }
        if((int)iter->second.size()==0){
        stop+=1;
        adj_list_reID[node2id[iter->first]].assign(0,0);
        }
        iter++;
     }
//     std::vector<std::vector<int>>::const_iterator it1=adj_list_reID.begin();
//    std::vector<std::vector<int>>::iterator it1=adj_list_reID.begin();
    assert(star+stop ==adj_dic_origin.size());
    adj_list_reID_noisol.assign(adj_list_reID.begin(),adj_list_reID.begin()+star);


    std::vector<double> BC = Betweenness(adj_list_reID_noisol); //#计算节点的介数中心性
    std::vector<double>::iterator biggest_BC = std::max_element(std::begin(BC), std::end(BC)); //#取最大值函数max_element
    int maxID = std::distance(std::begin(BC), biggest_BC); //#返回迭代器间元素个数 区间类型[·，·)
    int idx = id2node[maxID];// #因为下标从0开始记录，转换为原节点编号记录idx
//    printGraph();
//    printf("\n maxBetID:%d, value:%.6f\n",idx,BC[maxID]);
    return idx; //#最大介数中心性节点的编号
}

//影响力
int MvcEnv::influenceAction(){
    assert(graph);
//    std::shared_ptr<Utils> graphutil =std::shared_ptr<Utils>(new Utils());
    std::vector<bool> backupAllVex(graph->num_nodes, false); //当前已被选出的节点为true 其他false

    for(std::set<int>::iterator it_set=covered_set.begin();it_set != covered_set.end();++it_set){ //set里面自动排序 从小到大
        int tem=0;//记录下面循环中covered set中的节点序号
        for(std::vector<bool>::iterator iter_back=backupAllVex.begin();iter_back != backupAllVex.end();++iter_back){
            if(tem == *it_set){
                *iter_back=true;
                break;
            }
            tem+=1;
        }
//        backupAllVex[covered_set[i]]=true;
    }



    std::vector<int> diff_num_node;
    diff_num_node.resize(graph->num_nodes);
    int sum1;
    int sum2;
    std::vector<int> node_act_flag_tem;
    node_act_flag_tem = node_act_flag;
    for(int s=0;s<(int)backupAllVex.size();++s){
        if(backupAllVex[s] == true){
            diff_num_node[s] = 0;
            continue;
        }else{


            sum1 = std::accumulate(node_act_flag_tem.begin(),node_act_flag_tem.end(),0);
            sum2 = sum1;

        if(node_act_flag_tem[s] == 0)
            {


                node_act_flag_tem[s] = 1;

                std::queue<int> Q;
                Q.push(s);


                    while(!Q.empty()){


                    int u = Q.front();
                    Q.pop();
                    for(int i =0 ; i<(int)graph->adj_list[u].size();++i){
                            if(node_act_flag_tem[graph->adj_list[u][i]] == 1){
                                continue;
                            }
                            if(u<graph->adj_list[u][i]){
                            //make_pair(u,graph->adj_list[u][i])
                            //std::vector<int>::iterator it = find(graph->edge_list.begin(), graph->edge_list.end(), make_pair(u,graph->adj_list[u][i]));
                            for(int j =0 ;j<(int)graph->edge_list.size();++j){
                                if(graph->edge_list[j] == std::make_pair(u,graph->adj_list[u][i])){
                                    if(edge_act_flag[j]!=0){
                                        node_act_flag_tem[graph->adj_list[u][i]] = 1;
                                        Q.push(graph->adj_list[u][i]);
                                        break;
                                    }else{
                                        break;
                                    }

                                }

                            }

                            }else{
                            for(int j =0 ;j<(int)graph->edge_list.size();++j){
                                if(graph->edge_list[j] == std::make_pair(graph->adj_list[u][i],u)){
                                    if(edge_act_flag[j]!=0){
                                        node_act_flag_tem[graph->adj_list[u][i]] = 1;
                                        Q.push(graph->adj_list[u][i]);
                                        break;
                                    }else{
                                        break;
                                    }

                                }

                            }

                            }

                    }



                    }




            }




        }
        sum2 = std::accumulate(node_act_flag_tem.begin(),node_act_flag_tem.end(),0);
        diff_num_node[s] = sum2 -sum1;
    }

    int node_select = *(std::max_element(diff_num_node.begin(),diff_num_node.end()));

    return node_select;

}


//int dfs(int u, int father, edge* loop, int* first, int* Next, edge* a,
//    edge* brige, int* dfn, int* invis, int root, int* low)      //The father into the stack, avoid the inverse father-son edges
//{
//    invis[u] = 1;
//    int child = 0; //树中节点u的孩子节点的数目
//    dfn[u] = low[u] = ++num; //Define the dfn
//    for(int e = first[u]; e != -1; e = Next[e]){//对节点u遍历相关边，先遍历节点在记录中为左节点再是记录中是右节点（有向图） 无向图则是先树边再back edge
//
//        int v = a[e].v; //边e的邻居节点
//        if(!dfn[v]){ //father-son edges  如果没有被标记过
//
//            dfs(v, u, loop, first, Next, a, brige, dfn, invis, root, low);
//
//            child++;
//            low[u] = low[u] < low[v] ? low[u] : low[v];
//            if(dfn[u] < low[v]){
//
//                brige[bn].u = u;
//                brige[bn++].v = v;
//            }else{  //loop
//
//                loop[ln].u = u;
//                loop[ln++].v = v;
//            }
//
//        }else if(v != father && invis[v]){//Back-edges if(v != father): make sure the back father-son edges was forbidden.
//
//            loop[ln].u = u, loop[ln++].v = v;
//            low[u] = low[u] < dfn[v] ? low[u] : dfn[v];
//        }
//    }
//
//    invis[u] = 0;
//    return child;
//}

int dfs(int u, int father, std::vector<struct edge>& loop, std::vector<int>& first, std::vector<int>& Next, std::vector<struct edge>& a,
    std::vector<struct edge>& brige, std::vector<int>& dfn, std::vector<int>& invis, std::vector<int>& low)      //The father into the stack, avoid the inverse father-son edges
{
//    std::cout<< "start dfs"<<std::endl;
    invis[u] = 1;
    int child = 0; //树中节点u的孩子节点的数目
    dfn[u] = low[u] = num++; //Define the dfn
    for(int e = first[u]; e != -1; e = Next[e]){//对节点u遍历相关边，先遍历节点在记录中为左节点再是记录中是右节点（有向图） 无向图则是先树边再back edge

        int v = a[e].v; //边e的邻居节点
        if(dfn[v]==-1){ //father-son edges  如果没有被标记过
//            try{
//                dfs(v, u, loop, first, Next, a, brige, dfn, invis, root, low);
//            }catch(const std::exception &e){
//
//                std::cout<<e.what()<<std::endl;
//            }

//            std::cout<< "1"<<std::endl;
            dfs(v, u, loop, first, Next, a, brige, dfn, invis, low);
//            std::cout<< "2"<<std::endl;

            child++;
            low[u] = low[u] < low[v] ? low[u] : low[v];
            //下面判断loop或者brige
            if(dfn[u] < low[v]){
                assert(u!=v);
                brige[bn].u = u;
                brige[bn].v = v;
                bn++;
            }else{  //loop
                assert(u!=v);
                loop[ln].u = u;
                loop[ln].v = v;
                ln++;
//                record_cycle[u] = 1;
//                record_cycle[v] = 1;
            }

        }else if(v != father && invis[v]){//Back-edges if(v != father): make sure the back father-son edges was forbidden.
            assert(u!=v);
            loop[ln].u = u;
            loop[ln].v = v;
            ln++;
//            record_cycle[u] = 1;
//            record_cycle[v] = 1;
            low[u] = low[u] < dfn[v] ? low[u] : dfn[v];

//            std::cout<< "3"<<std::endl;
        }
    }
//    std::cout<< "end dfs"<<std::endl;
    invis[u] = 0;
    return child;
}


//void addedge(int u, int v, int e, int* first, int* Next, edge* all){//add edges. all是全部边的记录 e是边的编号
//
//    Next[e] = first[u];
//    all[e].u = u;
//    all[e].v = v;
//    first[u] = e;
//}

void addedge(int u, int v, int e, std::vector<int>& first, std::vector<int>& Next, std::vector<struct edge>& all){//add edges. all是全部边的记录 e是边的编号

    Next[e] = first[u];
    all[e].u = u;
    all[e].v = v;
    first[u] = e;
}
//对比算法 对比一下效果 可作为基准
int MvcEnv::decycling_dfs_action_contrast(){
    assert(graph);
    std::vector<bool> backupAllVex(graph->num_nodes, false); //当前已被选出的节点为true 其他false

    for(std::set<int>::iterator it_set=covered_set.begin();it_set != covered_set.end();++it_set){ //set里面自动排序 从小到大
        int tem=0;//记录下面循环中covered set中的节点序号
        for(std::vector<bool>::iterator iter_back=backupAllVex.begin();iter_back != backupAllVex.end();++iter_back){
            if(tem == *it_set){
                *iter_back=true;
                break;
            }
            tem+=1;
        }

    }
    for(int i = 0 ; i < record_cycle.size() ; ++i){
        if((covered_set.size() != 0) && (record_cycle[i] != 1)){
            backupAllVex[i] = true;
        }
    }
    std::vector<int> decy_num_node; //通过拓扑排序后 再根据度最大选出action
    decy_num_node.resize(graph->num_nodes);

    int N = graph->num_nodes;
    std::vector<int> deg(N);

    for(int i = 0; i < N; ++i){
        deg[i] = graph->adj_list[i].size();
    }

    for(int s=0;s<(int)backupAllVex.size();++s){
        if(backupAllVex[s] == true){
            decy_num_node[s] = 0;
            deg[s] = 0;//已经被删除 所以其度为0
            for(int i = 0;i<graph->adj_list[s].size();++i){
                deg[graph->adj_list[s][i]]--;
            }
            continue;
        }
    }
    for(int i = 0;i<N;++i){

        if(deg[i]<0){
            deg[i] = 0;
        }
    }


    std::vector< std::pair<int, int> > tem;
    tem = graph->edge_list;
    for(int s=0;s<(int)backupAllVex.size();++s){
        if(backupAllVex[s] == true){
            int flag = 0;
            int len = tem.size();
            for(int k = 0;k<len;++k){
                int u,v;
                u = tem[k].first;
                v = tem[k].second;
                if(v == s && u != s){


                    tem.erase(remove(tem.begin(),tem.end(),std::make_pair(u,v)),tem.end());
                    k--;
                    len--;


                }else if(u==s){
                    flag = 1;
                    tem.erase(remove(tem.begin(),tem.end(),std::make_pair(u,v)),tem.end());
                    k--;
                    len--;
                }

                if(flag == 1 && u != s){
                    break;
                }


            }


            continue;
        }
    }
    std::cout<<"action edge(in contrast function): "<<std::endl;
    for(int i = 0;i<tem.size(); ++i){
        std::cout<<tem[i].first<<"->"<<tem[i].second<<"   ";
    }
    std::cout<<std::endl;

// tem 已经处理过 现在只剩下剩余图中的连边情况
    int S = tem.size(); //边数
    if (S == 0){
        return -1;
    }
    int MAXM = 2 * S;
//    int MAXN = 2 * S;
//    int* first = new int[MAXN];
//    int* Next = new int[MAXM];
//    edge* all = new edge[MAXM];
//    memset(first, -1, sizeof(int) * MAXN);//initialization
    int max_index = 0;
    for(int i = 0;i<S;++i){
        int x = tem[i].first;
        int y = tem[i].second;
        max_index = max_index > x ? max_index : x;
        max_index = max_index > y ? max_index : y;
    }
    max_index++;



    std::vector<int> first(max_index,-1);
    std::vector<int> Next(MAXM);
    std::vector<struct edge> all(MAXM);
//    all.resize(MAXM);

//    struct edge tnum;
//    tnum.u = 0;
//    tnum.v = 0;
//    for(int i = 0; i<MAXM;++i){
//       all.push_back(tnum);
//    }





//    std::cout<<"tete"<<std::endl;


    num = 0;
    ln = 0;
    bn = 0;
    nodeN = 0;
    for(int j = 0; j < S; j++){

        int x = tem[j].first;
        int y = tem[j].second;
        addedge(x, y, j, first, Next, all);
        addedge(y, x, j + S, first, Next, all);
        nodeN = nodeN < x ? x : nodeN;
        nodeN = nodeN < y ? y : nodeN;//record the nodes number
    }
//    std::cout<<"tete2"<<std::endl;

    nodeN++; //从0开始标号，如果从1开始就不需要++
//    edge* loop = new edge[MAXM];
//    edge* brige = new edge[MAXM];

    std::vector<int> tem_flag(nodeN,0);
    for(int i = 0; i<S ; ++i){
        tem_flag[tem[i].first] = 1;
        tem_flag[tem[i].second] = 1;
    }



    std::vector<struct edge> loop(MAXM);
//    loop.resize(MAXM);


//    for(int i = 0; i<MAXM;++i){
//       loop.push_back(tnum);
//    }
    std::vector<struct edge> brige(MAXM);
//    brige.resize(MAXM);


//    for(int i = 0; i<MAXM;++i){
//       brige.push_back(tnum);
//    }

//    ln = 0;
//    int* dfn = new int[MAXN];
//    int* low = new int[MAXN];
//    memset(dfn, 0, sizeof(int) * MAXN);
//    int* invis = new int[MAXN];
//    memset(invis, 0, sizeof(int) * MAXN);//initialization

    std::vector<int> dfn(max_index,-1);
    std::vector<int> low(max_index,-1);
    std::vector<int> invis(max_index);

//    int root = 0;// 为了阅读者方便 标记dfs的树根节点    其实没有什么用

    for(int i = 0; i < nodeN; i++){ //nodeN从1开始标号，first(以maxn为最大的数组)下标从1开始考虑 next all(以maxM为最大的数组)从0开始
        std::cout<< "test"<< "   ";
        if(tem_flag[i] == 0){
            continue;
        }
        std::cout<< "real_test"<< std::endl;

        if(dfn[i] == -1){//如果有多个连通分支则多次dfs

            dfs(i, i, loop, first, Next, all, brige, dfn, invis, low);
//            try{
//                dfs(i, i, loop, first, Next, all, brige, dfn, invis, root, low);
//
//
//            }catch(const std::exception &e){
//                std::cout<<e.what()<<std::endl;
//            }
        }
    }//DFS the networks

//    std::cout<<"tete3"<<std::endl;

    std::set <int> lp;
    lp.clear();
//    record_cycle.assign(graph->num_nodes,0);
    for(int i = 0; i < record_cycle.size() ; ++i){
        record_cycle[i] = 0;
    }
    std::cout<<std::endl;
    std::cout<<"cycle: "<<std::endl;
    for(int k = 0; k < ln; k++){//extract the loop edges
        lp.insert(loop[k].u);
        lp.insert(loop[k].v);
        std::cout<<loop[k].u<<"->"<<loop[k].v<<"   ";
        record_cycle[loop[k].u] = 1;
        record_cycle[loop[k].v] = 1;
    }
    std::cout<<std::endl;
    std::cout<<"not cycle: "<<std::endl;
    for(int k = 0 ; k <bn ; k++){
        std::cout<<brige[k].u<<"->"<<brige[k].v<<"   ";
    }
    std::cout<<std::endl;
    assert(CNR>=(double)lp.size()/graph->num_nodes*1);
    CNR = (double)lp.size()/graph->num_nodes*1;

//    std::cout << CNR << std::endl;


//    delete[] first;
//    delete[] Next;
//    delete[] all;
//    delete[] loop;
//    delete[] brige;
//    delete[] dfn;
//    delete[] low;
//    delete[] invis;

//    clear_edge(all);
//    clear_edge(loop);
//    clear_edge(brige);



//    std::vector<struct edge>().swap(all);
//    std::vector<struct edge>().swap(loop);
//    std::vector<struct edge>().swap(brige);


    if(lp.size()==0){

        std::cout << "decycling node is none"<<std::endl;
        return -1;//表示没有圈
    }else{
        for(std::set<int>::iterator it = lp.begin();it!=lp.end();++it){

            decy_num_node[*it] = deg[*it];

        }
        int node_select = std::max_element(decy_num_node.begin(),decy_num_node.end())-decy_num_node.begin(); //下标
        return node_select;
//        return 1;
    }

}

//基于dfs方法--Tarjan算法
int MvcEnv::decycling_dfs_action(){
    assert(graph);

    std::vector<bool> backupAllVex(graph->num_nodes, false); //当前已被选出的节点为true 其他false

    for(std::set<int>::iterator it_set=covered_set.begin();it_set != covered_set.end();++it_set){ //set里面自动排序 从小到大
        int tem=0;//记录下面循环中covered set中的节点序号
        for(std::vector<bool>::iterator iter_back=backupAllVex.begin();iter_back != backupAllVex.end();++iter_back){
            if(tem == *it_set){
                *iter_back=true;
                break;
            }
            tem+=1;
        }

    }
    for(int i = 0 ; i < record_cycle.size() ; ++i){
        if((covered_set.size() != 0) && (record_cycle[i] != 1)){
            backupAllVex[i] = true;
        }
    }




//    std::vector<int> decy_num_node; //通过拓扑排序后 再根据度最大选出action
//    decy_num_node.resize(graph->num_nodes);
//
//    int N = graph->num_nodes;
//    std::vector<int> deg(N);
//
//    for(int i = 0; i < N; ++i){
//        deg[i] = graph->adj_list[i].size();
//    }
//
//    for(int s=0;s<(int)backupAllVex.size();++s){
//        if(backupAllVex[s] == true){
//            decy_num_node[s] = 0;
//            deg[s] = 0;//已经被删除 所以其度为0
//            for(int i = 0;i<graph->adj_list[s].size();++i){
//                deg[graph->adj_list[s][i]]--;
//            }
//            continue;
//        }
//    }
//    for(int i = 0;i<N;++i){
//
//        if(deg[i]<0){
//            deg[i] = 0;
//        }
//    }


    std::vector< std::pair<int, int> > tem;
    tem = graph->edge_list;
    for(int s=0;s<(int)backupAllVex.size();++s){
        if(backupAllVex[s] == true){
            int flag = 0;
            int len = tem.size();
            for(int k = 0;k<len;++k){
                int u,v;
                u = tem[k].first;
                v = tem[k].second;
                if(v == s && u != s){


                    tem.erase(remove(tem.begin(),tem.end(),std::make_pair(u,v)),tem.end());
                    k--;
                    len--;


                }else if(u==s){
                    flag = 1;
                    tem.erase(remove(tem.begin(),tem.end(),std::make_pair(u,v)),tem.end());
                    k--;
                    len--;
                }

                if(flag == 1 && u != s){
                    break;
                }


            }


            continue;
        }
    }
    std::cout<<"action edge(in action function, i.e. isterminal): "<<std::endl;
    for(int i = 0;i<tem.size(); ++i){
        std::cout<<tem[i].first<<"->"<<tem[i].second<<"   ";
    }
    std::cout<<std::endl;

// tem 已经处理过 现在只剩下剩余图中的连边情况
    int S = tem.size(); //边数
    if (S == 0){
        return -1;
    }
    int MAXM = 2 * S;
//    int MAXN = 2 * S;
//    int* first = new int[MAXN];
//    int* Next = new int[MAXM];
//    edge* all = new edge[MAXM];
//    memset(first, -1, sizeof(int) * MAXN);//initialization
    int max_index = 0;
    for(int i = 0;i<S;++i){
        int x = tem[i].first;
        int y = tem[i].second;
        max_index = max_index > x ? max_index : x;
        max_index = max_index > y ? max_index : y;
    }
    max_index++;


    std::vector<int> first(max_index,-1);
    std::vector<int> Next(MAXM);
    std::vector<struct edge> all(MAXM);
//    all.resize(MAXM);

//    struct edge tnum;
//    tnum.u = 0;
//    tnum.v = 0;
//    for(int i = 0; i<MAXM;++i){
//       all.push_back(tnum);
//    }

//    std::cout<<"tete"<<std::endl;


    num = 0;
    ln = 0;
    bn = 0;
    nodeN = 0;
    for(int j = 0; j < S; j++){

        int x = tem[j].first;
        int y = tem[j].second;
        addedge(x, y, j, first, Next, all);
        addedge(y, x, j + S, first, Next, all);
        nodeN = nodeN < x ? x : nodeN;
        nodeN = nodeN < y ? y : nodeN;//record the nodes number
    }
//    std::cout<<"tete2"<<std::endl;
    nodeN++; //从0开始标号，如果从1开始就不需要++
//    edge* loop = new edge[MAXM];
//    edge* brige = new edge[MAXM];


    std::vector<int> tem_flag(nodeN,0);
    for(int i = 0; i<S ; ++i){
        tem_flag[tem[i].first] = 1;
        tem_flag[tem[i].second] = 1;
    }

    std::vector<struct edge> loop(MAXM);
//    loop.resize(MAXM);

//    for(int i = 0; i<MAXM;++i){
//       loop.push_back(tnum);
//    }
    std::vector<struct edge> brige(MAXM);
//    brige.resize(MAXM);

//    for(int i = 0; i<MAXM;++i){
//       brige.push_back(tnum);
//    }

//    ln = 0;
//    int* dfn = new int[MAXN];
//    int* low = new int[MAXN];
//    memset(dfn, 0, sizeof(int) * MAXN);
//    int* invis = new int[MAXN];
//    memset(invis, 0, sizeof(int) * MAXN);//initialization

    std::vector<int> dfn(max_index,-1);
    std::vector<int> low(max_index,-1);
    std::vector<int> invis(max_index);

//    int root = 0;// 为了阅读者方便 标记dfs的树根节点    其实没有什么用

    for(int i = 0; i < nodeN; i++){ //nodeN从1开始标号，first(以maxn为最大的数组)下标从1开始考虑 next all(以maxM为最大的数组)从0开始
        if(tem_flag[i] == 0){
            continue;
        }

        if(dfn[i] == -1){//如果有多个连通分支则多次dfs

            dfs(i, i, loop, first, Next, all, brige, dfn, invis, low);
//            try{
//                dfs(i, i, loop, first, Next, all, brige, dfn, invis, root, low);
//
//
//            }catch(const std::exception &e){
//                std::cout<<e.what()<<std::endl;
//            }
        }
    }//DFS the networks

//    std::cout<<"tete3"<<std::endl;
    std::set <int> lp;
    lp.clear();
//    record_cycle.assign(graph->num_nodes,0);
    for(int i = 0; i < record_cycle.size() ; ++i){
        record_cycle[i] = 0;
    }
    for(int k = 0; k < ln; k++){//extract the loop edges
        lp.insert(loop[k].u);
        lp.insert(loop[k].v);
        record_cycle[loop[k].u] = 1;
        record_cycle[loop[k].v] = 1;
    }
    assert(CNR>=(double)lp.size()/graph->num_nodes*1);
    CNR = (double)lp.size()/graph->num_nodes*1;
    if(CNR_all_flag == false){
        CNR_all = CNR;
        CNR_all_flag = true;
    }

//    std::cout << CNR << std::endl;


//    delete[] first;
//    delete[] Next;
//    delete[] all;
//    delete[] loop;
//    delete[] brige;
//    delete[] dfn;
//    delete[] low;
//    delete[] invis;

//    clear_edge(all);
//    clear_edge(loop);
//    clear_edge(brige);



//    std::vector<struct edge>().swap(all);
//    std::vector<struct edge>().swap(loop);
//    std::vector<struct edge>().swap(brige);


    if(lp.size()==0){

        std::cout << "decycling is end"<<std::endl;
        return -1;//表示没有圈
    }else{
//        for(std::set<int>::iterator it = lp.begin();it!=lp.end();++it){
//
//            decy_num_node[*it] = deg[*it];
//
//        }
//        int node_select = std::max_element(decy_num_node.begin(),decy_num_node.end())-decy_num_node.begin(); //下标
//        return node_select;
        return 1;
    }

}


std::vector<int> MvcEnv::decycling_dfs_action_list(){
    assert(graph);

    std::vector<bool> backupAllVex(graph->num_nodes, false); //当前已被选出的节点为true 其他false

    for(std::set<int>::iterator it_set=covered_set.begin();it_set != covered_set.end();++it_set){ //set里面自动排序 从小到大
        int tem=0;//记录下面循环中covered set中的节点序号
        for(std::vector<bool>::iterator iter_back=backupAllVex.begin();iter_back != backupAllVex.end();++iter_back){
            if(tem == *it_set){
                *iter_back=true;
                break;
            }
            tem+=1;
        }

    }
    for(int i = 0 ; i < record_cycle.size() ; ++i){
        if((covered_set.size() != 0) && (record_cycle[i] != 1)){
            backupAllVex[i] = true;
        }
    }

    std::vector< std::pair<int, int> > tem;
    tem = graph->edge_list;
    for(int s=0;s<(int)backupAllVex.size();++s){
        if(backupAllVex[s] == true){
            int flag = 0;
            int len = tem.size();
            for(int k = 0;k<len;++k){
                int u,v;
                u = tem[k].first;
                v = tem[k].second;
                if(v == s && u != s){


                    tem.erase(remove(tem.begin(),tem.end(),std::make_pair(u,v)),tem.end());
                    k--;
                    len--;


                }else if(u==s){
                    flag = 1;
                    tem.erase(remove(tem.begin(),tem.end(),std::make_pair(u,v)),tem.end());
                    k--;
                    len--;
                }
                if(flag == 1 && u != s){
                    break;
                }


            }


            continue;
        }
    }

// tem 已经处理过 现在只剩下剩余图中的连边情况
    int S = tem.size(); //边数
    int MAXM = 2 * S;
//    int MAXN = 2 * S;
//    int* first = new int[MAXN];
//    int* Next = new int[MAXM];
//    edge* all = new edge[MAXM];
//    memset(first, -1, sizeof(int) * MAXN);//initialization

    int max_index = 0;
    for(int i = 0;i<S;++i){
        int x = tem[i].first;
        int y = tem[i].second;
        max_index = max_index > x ? max_index : x;
        max_index = max_index > y ? max_index : y;
    }
    max_index++;

    std::vector<int> first(max_index,-1);
    std::vector<int> Next(MAXM);
    std::vector<struct edge> all(MAXM);
//    all.resize(MAXM);


//    struct edge tnum;
//    tnum.u = 0;
//    tnum.v = 0;
//    for(int i = 0; i<MAXM;++i){
//       all.push_back(tnum);
//    }

    num = 0;
    ln = 0;
    bn = 0;
    nodeN = 0;
    for(int j = 0; j < S; j++){

        int x = tem[j].first;
        int y = tem[j].second;
        addedge(x, y, j, first, Next, all);
        addedge(y, x, j + S, first, Next, all);
        nodeN = nodeN < x ? x : nodeN;
        nodeN = nodeN < y ? y : nodeN;//record the nodes number
    }
    nodeN++; //从0开始标号，如果从1开始就不需要++
//    edge* loop = new edge[MAXM];
//    edge* brige = new edge[MAXM];


    std::vector<int> tem_flag(nodeN,0);
    for(int i = 0; i<S ; ++i){
        tem_flag[tem[i].first] = 1;
        tem_flag[tem[i].second] = 1;
    }

    std::vector<struct edge> loop(MAXM);
//    loop.resize(MAXM);

//    for(int i = 0; i<MAXM;++i){
//       loop.push_back(tnum);
//    }
    std::vector<struct edge> brige(MAXM);
//    brige.resize(MAXM);

//    for(int i = 0; i<MAXM;++i){
//       brige.push_back(tnum);
//    }

//    ln = 0;
//    int* dfn = new int[MAXN];
//    int* low = new int[MAXN];
//    memset(dfn, 0, sizeof(int) * MAXN);
//    int* invis = new int[MAXN];
//    memset(invis, 0, sizeof(int) * MAXN);//initialization
    std::vector<int> dfn(max_index,-1);
    std::vector<int> low(max_index,-1);
    std::vector<int> invis(max_index);

//    int root = 0;// 为了阅读者方便 标记dfs的树根节点

    for(int i = 0; i < nodeN; i++){ //nodeN从1开始标号，first(以maxn为最大的数组)下标从1开始考虑 next all(以maxM为最大的数组)从0开始
        if(tem_flag[i] == 0){
            continue;
        }

        if(dfn[i] == -1){//如果有多个连通分支则多次dfs


            dfs(i, i, loop, first, Next, all, brige, dfn, invis, low);
//            try{
//                dfs(i, i, loop, first, Next, all, brige, dfn, invis, root, low);
//
//
//            }catch(const std::exception &e){
//                std::cout<<e.what()<<std::endl;
//            }
        }
    }//DFS the networks

    std::set <int> lp;
    lp.clear();
//    record_cycle.assign(graph->num_nodes,0);
    for(int i = 0; i < record_cycle.size() ; ++i){
        record_cycle[i] = 0;
    }
    for(int k = 0; k < ln; k++){//extract the loop edges

        lp.insert(loop[k].u);
        lp.insert(loop[k].v);
        record_cycle[loop[k].u] = 1;
        record_cycle[loop[k].v] = 1;
    }
    CNR = (double)lp.size()/graph->num_nodes*1;
    std::vector<int> result(graph->num_nodes,0);
//    delete[] first;
//    delete[] Next;
//    delete[] all;
//    delete[] loop;
//    delete[] brige;
//    delete[] dfn;
//    delete[] low;
//    delete[] invis;


//    clear_edge(all);
//    clear_edge(loop);
//    clear_edge(brige);


//    std::vector<struct edge>().swap(all);
//    std::vector<struct edge>().swap(loop);
//    std::vector<struct edge>().swap(brige);


    if(lp.size()==0){

        std::cout << "decycling is end"<<std::endl;
        return result;
    }else{
        for(std::set<int>::iterator it = lp.begin();it!=lp.end();++it){
            result[*it] = 1;
        }
        std::cout << "decycling is not end"<<std::endl;
        return result;
    }

}


double MvcEnv::decycling_dfs_ratio(){
    assert(graph);

    std::vector<bool> backupAllVex(graph->num_nodes, false); //当前已被选出的节点为true 其他false

    for(std::set<int>::iterator it_set=covered_set.begin();it_set != covered_set.end();++it_set){ //set里面自动排序 从小到大
        int tem=0;//记录下面循环中covered set中的节点序号
        for(std::vector<bool>::iterator iter_back=backupAllVex.begin();iter_back != backupAllVex.end();++iter_back){
            if(tem == *it_set){
                *iter_back=true;
                break;
            }
            tem+=1;
        }

    }

    for(int i = 0 ; i < record_cycle.size() ; ++i){
        if((covered_set.size() != 0) && (record_cycle[i] != 1)){
            backupAllVex[i] = true;
        }
    }

    std::vector< std::pair<int, int> > tem;
    tem = graph->edge_list;
    for(int s=0;s<(int)backupAllVex.size();++s){
        if(backupAllVex[s] == true){
            int flag = 0;
            int len = tem.size();
            for(int k = 0;k<len;++k){
                int u,v;
                u = tem[k].first;
                v = tem[k].second;
                if(v == s && u != s){


                    tem.erase(remove(tem.begin(),tem.end(),std::make_pair(u,v)),tem.end());
                    k--;
                    len--;


                }else if(u==s){
                    flag = 1;
                    tem.erase(remove(tem.begin(),tem.end(),std::make_pair(u,v)),tem.end());
                    k--;
                    len--;
                }
                if(flag == 1 && u != s){
                    break;
                }


            }


            continue;
        }
    }
    std::cout<<"action edge(in ratio function, i.e. step): "<<std::endl;
    for(int i = 0;i<tem.size(); ++i){
        std::cout<<tem[i].first<<"->"<<tem[i].second<<"   ";
    }
    std::cout<<std::endl;

// tem 已经处理过 现在只剩下剩余图中的连边情况
    int S = tem.size(); //边数
    if (S == 0){
        return 0.0;
    }
    int MAXM = 2 * S;
//    int MAXN = 2 * S;
//    int* first = new int[MAXN];
//    int* Next = new int[MAXM];
//    edge* all = new edge[MAXM];
//    memset(first, -1, sizeof(int) * MAXN);//initialization
    int max_index = 0;
    for(int i = 0;i<S;++i){
        int x = tem[i].first;
        int y = tem[i].second;
        max_index = max_index > x ? max_index : x;
        max_index = max_index > y ? max_index : y;
    }
    max_index++;

    std::vector<int> first(max_index,-1);
    std::vector<int> Next(MAXM);
    std::vector<struct edge> all(MAXM);
//    all.resize(MAXM);


//    struct edge tnum;
//    tnum.u = 0;
//    tnum.v = 0;
//    for(int i = 0; i<MAXM;++i){
//       all.push_back(tnum);
//    }


    num = 0;
    ln = 0;
    bn = 0;

    nodeN = 0;
    for(int j = 0; j < S; j++){

        int x = tem[j].first;
        int y = tem[j].second;
        addedge(x, y, j, first, Next, all);
        addedge(y, x, j + S, first, Next, all);
        nodeN = nodeN < x ? x : nodeN;
        nodeN = nodeN < y ? y : nodeN;//record the nodes number
    }
    nodeN++; //从0开始标号，如果从1开始就不需要++
//    edge* loop = new edge[MAXM];
//    edge* brige = new edge[MAXM];

    std::vector<int> tem_flag(nodeN,0);
    for(int i = 0; i<S ; ++i){
        tem_flag[tem[i].first] = 1;
        tem_flag[tem[i].second] = 1;
    }

    std::vector<struct edge> loop(MAXM);
//    loop.resize(MAXM);


//    for(int i = 0; i<MAXM;++i){
//       loop.push_back(tnum);
//    }
    std::vector<struct edge> brige(MAXM);
//    brige.resize(MAXM);

//    for(int i = 0; i<MAXM;++i){
//       brige.push_back(tnum);
//    }

//    ln = 0;
//    int* dfn = new int[MAXN];
//    int* low = new int[MAXN];
//    memset(dfn, 0, sizeof(int) * MAXN);
//    int* invis = new int[MAXN];
//    memset(invis, 0, sizeof(int) * MAXN);//initialization

    std::vector<int> dfn(max_index,-1);
    std::vector<int> low(max_index,-1);
    std::vector<int> invis(max_index);


//    int root = 0;// 为了阅读者方便 标记dfs的树根节点
//    std::cout<<"arrive here? "<<std::endl;
    for(int i = 0; i < nodeN; i++){ //nodeN从1开始标号，first(以maxn为最大的数组)下标从1开始考虑 next all(以maxM为最大的数组)从0开始
        if(tem_flag[i] == 0){
            continue;
        }

        if(dfn[i] == -1){//如果有多个连通分支则多次dfs


            dfs(i, i, loop, first, Next, all, brige, dfn, invis, low);

//            try{
//                dfs(i, i, loop, first, Next, all, brige, dfn, invis, root, low);
//
//
//            }catch(const std::exception &e){
//                std::cout<<e.what()<<std::endl;
//            }
        }
    }//DFS the networks
//    std::cout<<"dfs has end"<<std::endl;

    std::set <int> lp;
    lp.clear();
//    record_cycle.assign(graph->num_nodes,0);
    for(int i = 0; i < record_cycle.size() ; ++i){
        record_cycle[i] = 0;
    }
    std::cout<<"ring: "<<std::endl;
    for(int k = 0; k < ln; k++){//extract the loop edges

        lp.insert(loop[k].u);
        lp.insert(loop[k].v);
        std::cout<<loop[k].u<<"->"<<loop[k].v<<"   ";
        record_cycle[loop[k].u] = 1;
        record_cycle[loop[k].v] = 1;
    }
    std::cout<<std::endl;
    std::cout<<"Acyclic: "<<std::endl;
    for(int k = 0 ; k< bn ; k++){
        std::cout<<brige[k].u<<"->"<<brige[k].v<<"   ";
    }
    std::cout<<std::endl;
//    if(bn==0){
//    std::cout<<"none edge"<<std::endl;
//    }
//    delete[] first;
//    delete[] Next;
//    delete[] all;
//    delete[] loop;
//    delete[] brige;
//    delete[] dfn;
//    delete[] low;
//    delete[] invis;
//        double CNR = lp.size() / (nodeN-covered_set.size());


//    clear_edge(all);
//    clear_edge(loop);
//    clear_edge(brige);


//    std::cout<<"run"<<std::endl;
//    std::vector<struct edge>().swap(all);
//    std::vector<struct edge>().swap(loop);
//    std::vector<struct edge>().swap(brige);
    for(int i = 0; i<record_cycle.size() ;++i){
        std::cout<<record_cycle[i]<<" ";
    }
    std::cout<<std::endl;
//    double cnr_tem = CNR;//original value
    CNR = (double)lp.size()/graph->num_nodes*1;//current value
    std::cout<<"step is running once"<<std::endl;
//    return CNR - cnr_tem;
//    return CNR/CNR_all;
    return CNR;
}


double MvcEnv::decycling_dfs_ratio_absolute(){
    assert(graph);

    std::vector<bool> backupAllVex(graph->num_nodes, false); //当前已被选出的节点为true 其他false

    for(std::set<int>::iterator it_set=covered_set.begin();it_set != covered_set.end();++it_set){ //set里面自动排序 从小到大
        int tem=0;//记录下面循环中covered set中的节点序号
        for(std::vector<bool>::iterator iter_back=backupAllVex.begin();iter_back != backupAllVex.end();++iter_back){
            if(tem == *it_set){
                *iter_back=true;
                break;
            }
            tem+=1;
        }

    }

//    for(int i = 0 ; i < record_cycle.size() ; ++i){
//        if((covered_set.size() != 0) && (record_cycle[i] != 1)){
//            backupAllVex[i] = true;
//        }
//    }

    std::vector< std::pair<int, int> > tem;
    tem = graph->edge_list;
    for(int s=0;s<(int)backupAllVex.size();++s){
        if(backupAllVex[s] == true){
            int flag = 0;
            int len = tem.size();
            for(int k = 0;k<len;++k){
                int u,v;
                u = tem[k].first;
                v = tem[k].second;
                if(v == s && u != s){


                    tem.erase(remove(tem.begin(),tem.end(),std::make_pair(u,v)),tem.end());
                    k--;
                    len--;


                }else if(u==s){
                    flag = 1;
                    tem.erase(remove(tem.begin(),tem.end(),std::make_pair(u,v)),tem.end());
                    k--;
                    len--;
                }
                if(flag == 1 && u != s){
                    break;
                }


            }


            continue;
        }
    }
    std::cout<<"action edge(in ratio function, i.e. step): "<<std::endl;
    for(int i = 0;i<tem.size(); ++i){
        std::cout<<tem[i].first<<"->"<<tem[i].second<<"   ";
    }
    std::cout<<std::endl;

// tem 已经处理过 现在只剩下剩余图中的连边情况
    int S = tem.size(); //边数
    if (S == 0){
        return 0.0;
    }
    int MAXM = 2 * S;


    int max_index = 0;
    for(int i = 0;i<S;++i){
        int x = tem[i].first;
        int y = tem[i].second;
        max_index = max_index > x ? max_index : x;
        max_index = max_index > y ? max_index : y;
    }
    max_index++;

    std::vector<int> first(max_index,-1);
    std::vector<int> Next(MAXM);
    std::vector<struct edge> all(MAXM);




    num = 0;
    ln = 0;
    bn = 0;

    nodeN = 0;
    for(int j = 0; j < S; j++){

        int x = tem[j].first;
        int y = tem[j].second;
        addedge(x, y, j, first, Next, all);
        addedge(y, x, j + S, first, Next, all);
        nodeN = nodeN < x ? x : nodeN;
        nodeN = nodeN < y ? y : nodeN;//record the nodes number
    }
    nodeN++; //从0开始标号，如果从1开始就不需要++


    std::vector<int> tem_flag(nodeN,0);
    for(int i = 0; i<S ; ++i){
        tem_flag[tem[i].first] = 1;
        tem_flag[tem[i].second] = 1;
    }

    std::vector<struct edge> loop(MAXM);



    std::vector<struct edge> brige(MAXM);


    std::vector<int> dfn(max_index,-1);
    std::vector<int> low(max_index,-1);
    std::vector<int> invis(max_index);



    for(int i = 0; i < nodeN; i++){ //nodeN从1开始标号，first(以maxn为最大的数组)下标从1开始考虑 next all(以maxM为最大的数组)从0开始
        if(tem_flag[i] == 0){
            continue;
        }

        if(dfn[i] == -1){//如果有多个连通分支则多次dfs


            dfs(i, i, loop, first, Next, all, brige, dfn, invis, low);


        }
    }//DFS the networks
//    std::cout<<"dfs has end"<<std::endl;

    std::set <int> lp;
    lp.clear();
//    record_cycle.assign(graph->num_nodes,0);
    for(int i = 0; i < record_cycle.size() ; ++i){
        record_cycle[i] = 0;
    }
    std::cout<<"ring: "<<std::endl;
    for(int k = 0; k < ln; k++){//extract the loop edges

        lp.insert(loop[k].u);
        lp.insert(loop[k].v);
        std::cout<<loop[k].u<<"->"<<loop[k].v<<"   ";
        record_cycle[loop[k].u] = 1;
        record_cycle[loop[k].v] = 1;
    }
    std::cout<<std::endl;
    std::cout<<"Acyclic: "<<std::endl;
    for(int k = 0 ; k< bn ; k++){
        std::cout<<brige[k].u<<"->"<<brige[k].v<<"   ";
    }
    std::cout<<std::endl;

    for(int i = 0; i<record_cycle.size() ;++i){
        std::cout<<record_cycle[i]<<" ";
    }
    std::cout<<std::endl;
    double cnr_tem = CNR;//original value
    CNR = (double)lp.size()/graph->num_nodes*1;//current value
    std::cout<<"step is running once"<<std::endl;
    return CNR - cnr_tem;
//    return CNR;
}



int MvcEnv::decyclingaction(){
    assert(graph);
    std::vector<int> core;

    std::vector<bool> backupAllVex(graph->num_nodes, false); //当前已被选出的节点为true 其他false

    for(std::set<int>::iterator it_set=covered_set.begin();it_set != covered_set.end();++it_set){ //set里面自动排序 从小到大
        int tem=0;//记录下面循环中covered set中的节点序号
        for(std::vector<bool>::iterator iter_back=backupAllVex.begin();iter_back != backupAllVex.end();++iter_back){
            if(tem == *it_set){
                *iter_back=true;
                break;
            }
            tem+=1;
        }
//        backupAllVex[covered_set[i]]=true;
    }
    std::vector<int> decy_num_node; //通过拓扑排序后 再根据度最大选出action
    decy_num_node.resize(graph->num_nodes);
    //排除已经删除的节点

    int md = -1; //max degree
    int N = graph->num_nodes;
    std::vector<int> deg(N);

    for (int i = 0; i < N; ++i) {
        deg[i] = graph->adj_list[i].size();
    }

    for(int s=0;s<(int)backupAllVex.size();++s){
        if(backupAllVex[s] == true){
            decy_num_node[s] = 0;
            deg[s] = 0;//已经被删除 所以其度为0
            for(int i = 0;i<graph->adj_list[s].size();++i){
                deg[graph->adj_list[s][i]]--;
            }
            continue;
        }
    }
    for(int i = 0;i<N;++i){
        md = std::max(md, deg[i]);
        if(deg[i]<0){
            deg[i] = 0;
        }
    }
    std::vector<int> deg_r;
    deg_r = deg;
    std::vector<int> bin(N);//节点度分布统计，递增排序

    for (int i = 0; i < N; ++i) {
        bin[i] = 0;
    }
    for (int i = 0; i < N; ++i) {
        ++bin[deg[i]]; //md后面都是0 前面是度分布
    }
    int start = 0;
    for (int i = 0; i <= md; ++i) {
        int num = bin[i];
        bin[i] = start;
        start += num;
    }
    std::vector<int> pos(N); //元素值是idx标号的节点的下标所在
    std::vector<int> vert(N);//对应pos的节点放置
    for (int i = 0; i < N; ++i) {
        int v = i;
        pos[v] = bin[deg[v]];
        vert[pos[v]] = v;
        ++bin[deg[v]];
    }
    //此时bin[i]表示度数为i+1的第一个点，所以后面要进行左移操作

    for (int i = md; i > 0; --i) {
        bin[i] = bin[i - 1];
    }
    bin[0] = 0;


    std::vector<std::vector<int>> tem;
    tem = graph->adj_list;
    //删除度小于2的
    //vector<int>::iterator it;
    std::queue <int> Q;
    std::set<int> record;
    for (int i = 0; i < bin[2]; ++i) {
        if(i<bin[1] && tem[vert[i]].size()!=0){
            record.insert(vert[i]);
            continue;
        }else if(i<bin[1]){
            Q.push(vert[i]);
            record.insert(vert[i]);
        }else{
            Q.push(vert[i]);
            record.insert(vert[i]);
        }

        //deg[vert[i]] = 0;
    }
    int u;
    while (!Q.empty()) {
        u = Q.front();
        Q.pop();
        for (int i = 0; i < tem[u].size(); ++i) {
            deg[tem[u][i]]--;
            if (deg[tem[u][i]] == 1) {
                Q.push(tem[u][i]);
                record.insert(tem[u][i]);
            }

        }
    }
    if (record.size() == N) {
        std::cout << "无环"<<std::endl;
        core.clear();
        if(covered_set.size()==0){
            cycle_node_all = 0;
        }

    }
    else {
//        std::cout << "有环"<<std::endl;
        int kk = 0;
        for (int i = 0; i < N; ++i) {
            std::set<int>::iterator it;
            it = record.find(i);
            if (it != record.end()) {
                continue;
            }
            else {
                kk++;
                core.push_back(i); //未删除的节点

            }

        }
        if(covered_set.size()==0){
            cycle_node_all = core.size();
        }


    }
    if(core.size()==0){
        return -1;//表示没有圈
    }else{
        for(int i = 0;i<core.size();++i){
            decy_num_node[core[i]]=deg_r[core[i]];
        }
        int node_select = std::max_element(decy_num_node.begin(),decy_num_node.end())-decy_num_node.begin(); //下标
        return node_select;
    }



}


std::vector<int> MvcEnv::decyclingaction_list(){
    assert(graph);
    std::vector<int> core;

    std::vector<bool> backupAllVex(graph->num_nodes, false); //当前已被选出的节点为true 其他false

    for(std::set<int>::iterator it_set=covered_set.begin();it_set != covered_set.end();++it_set){ //set里面自动排序 从小到大
        int tem=0;//记录下面循环中covered set中的节点序号
        for(std::vector<bool>::iterator iter_back=backupAllVex.begin();iter_back != backupAllVex.end();++iter_back){
            if(tem == *it_set){
                *iter_back=true;
                break;
            }
            tem+=1;
        }
//        backupAllVex[covered_set[i]]=true;
    }
    std::vector<int> decy_num_node; //通过拓扑排序后 再根据度最大选出action
    decy_num_node.resize(graph->num_nodes);
    //排除已经删除的节点

    int md = -1; //max degree
    int N = graph->num_nodes;
    std::vector<int> deg(N);

    for (int i = 0; i < N; ++i) {
        deg[i] = graph->adj_list[i].size();
    }

    for(int s=0;s<(int)backupAllVex.size();++s){
        if(backupAllVex[s] == true){
            decy_num_node[s] = 0;
            deg[s] = 0;//已经被删除 所以其度为0
            for(int i = 0;i<graph->adj_list[s].size();++i){
                deg[graph->adj_list[s][i]]--;
            }
            continue;
        }
    }
    for(int i = 0;i<N;++i){
        md = std::max(md, deg[i]);
        if(deg[i]<0){
            deg[i] = 0;
        }
    }
    std::vector<int> deg_r;
    deg_r = deg;
    std::vector<int> bin(N);//节点度分布统计，递增排序

    for (int i = 0; i < N; ++i) {
        bin[i] = 0;
    }
    for (int i = 0; i < N; ++i) {
        ++bin[deg[i]]; //md后面都是0 前面是度分布
    }
    int start = 0;
    for (int i = 0; i <= md; ++i) {
        int num = bin[i];
        bin[i] = start;
        start += num;
    }
    std::vector<int> pos(N); //元素值是idx标号的节点的下标所在
    std::vector<int> vert(N);//对应pos的节点放置
    for (int i = 0; i < N; ++i) {
        int v = i;
        pos[v] = bin[deg[v]];
        vert[pos[v]] = v;
        ++bin[deg[v]];
    }
    //此时bin[i]表示度数为i+1的第一个点，所以后面要进行左移操作

    for (int i = md; i > 0; --i) {
        bin[i] = bin[i - 1];
    }
    bin[0] = 0;


    std::vector<std::vector<int>> tem;
    tem = graph->adj_list;
    //删除度小于2的
    //vector<int>::iterator it;
    std::queue <int> Q;
    std::set<int> record;
    for (int i = 0; i < bin[2]; ++i) {
        if(i<bin[1] && tem[vert[i]].size()!=0){
            record.insert(vert[i]);
            continue;
        }else if(i<bin[1]){
            Q.push(vert[i]);
            record.insert(vert[i]);
        }else{
            Q.push(vert[i]);
            record.insert(vert[i]);
        }

        //deg[vert[i]] = 0;
    }
    int u;
    while (!Q.empty()) {
        u = Q.front();
        Q.pop();
        for (int i = 0; i < tem[u].size(); ++i) {
            deg[tem[u][i]]--;
            if (deg[tem[u][i]] == 1) {
                Q.push(tem[u][i]);
                record.insert(tem[u][i]);
            }

        }
    }
    if (record.size() == N) {
        std::cout << "无环"<<std::endl;
        core.clear();
        if(covered_set.size()==0){
            cycle_node_all = 0;
        }

    }
    else {
        std::cout << "有环"<<std::endl;
        int kk = 0;
        for (int i = 0; i < N; ++i) {
            std::set<int>::iterator it;
            it = record.find(i);
            if (it != record.end()) {
                continue;
            }
            else {
                kk++;
                core.push_back(i);

            }

        }
        if(covered_set.size()==0){
            cycle_node_all = core.size();
        }


    }
    std::vector<int> result(graph->num_nodes,0);

    if(core.size()==0){
        return result;//表示没有圈
    }else{
        for(int i = 0;i<core.size();++i){
            decy_num_node[core[i]]=deg_r[core[i]];
        }
        for(int i = 0;i<decy_num_node.size();++i){
            if(decy_num_node[i]==0)
                result[i] = 0;
            else
                result[i] = 1;

        }
//        int node_select = std::max_element(decy_num_node.begin(),decy_num_node.end())-decy_num_node.begin(); //下标
        return result;
    }



}



//计算reward使用
int MvcEnv::decyclingratio(){
    assert(graph);
    std::vector<int> core;

    std::vector<bool> backupAllVex(graph->num_nodes, false); //当前已被选出的节点为true 其他false

    for(std::set<int>::iterator it_set=covered_set.begin();it_set != covered_set.end();++it_set){ //set里面自动排序 从小到大
        int tem=0;//记录下面循环中covered set中的节点序号
        for(std::vector<bool>::iterator iter_back=backupAllVex.begin();iter_back != backupAllVex.end();++iter_back){
            if(tem == *it_set){
                *iter_back=true;
                break;
            }
            tem+=1;
        }
//        backupAllVex[covered_set[i]]=true;
    }
    std::vector<int> decy_num_node; //通过拓扑排序后 再根据度最大选出action
    decy_num_node.resize(graph->num_nodes);
    //排除已经删除的节点

    int md = -1; //max degree
    int N = graph->num_nodes;
    std::vector<int> deg(N);

    for (int i = 0; i < N; ++i) {
        deg[i] = graph->adj_list[i].size();
    }

    for(int s=0;s<(int)backupAllVex.size();++s){
        if(backupAllVex[s] == true){
            decy_num_node[s] = 0;
            deg[s] = 0;//已经被删除 所以其度为0
            for(int i = 0;i<graph->adj_list[s].size();++i){
                deg[graph->adj_list[s][i]]--;
            }
            continue;
        }
    }
    for(int i = 0;i<N;++i){
        md = std::max(md, deg[i]);
        if(deg[i]<0){
            deg[i] = 0;
        }
    }
    std::vector<int> deg_r;
    deg_r = deg;
    std::vector<int> bin(N);//节点度分布统计，递增排序

    for (int i = 0; i < N; ++i) {
        bin[i] = 0;
    }
    for (int i = 0; i < N; ++i) {
        ++bin[deg[i]]; //md后面都是0 前面是度分布
    }
    int start = 0;
    for (int i = 0; i <= md; ++i) {
        int num = bin[i];
        bin[i] = start;
        start += num;
    }
    std::vector<int> pos(N); //元素值是idx标号的节点的下标所在
    std::vector<int> vert(N);//对应pos的节点放置
    for (int i = 0; i < N; ++i) {
        int v = i;
        pos[v] = bin[deg[v]];
        vert[pos[v]] = v;
        ++bin[deg[v]];
    }
    //此时bin[i]表示度数为i+1的第一个点，所以后面要进行左移操作

    for (int i = md; i > 0; --i) {
        bin[i] = bin[i - 1];
    }
    bin[0] = 0;


    std::vector<std::vector<int>> tem;
    tem = graph->adj_list;
    //删除度小于2的
    //vector<int>::iterator it;
    std::queue <int> Q;
    std::set<int> record;
    for (int i = 0; i < bin[2]; ++i) {
        if(i<bin[1] && tem[vert[i]].size()!=0){
            record.insert(vert[i]);
            continue;
        }else if(i<bin[1]){
            Q.push(vert[i]);
            record.insert(vert[i]);
        }else{
            Q.push(vert[i]);
            record.insert(vert[i]);
        }

        //deg[vert[i]] = 0;
    }
    int u;
    while (!Q.empty()) {
        u = Q.front();
        Q.pop();
        for (int i = 0; i < tem[u].size(); ++i) {
            deg[tem[u][i]]--;
            if (deg[tem[u][i]] == 1) {
                Q.push(tem[u][i]);
                record.insert(tem[u][i]);
            }

        }
    }
    if (record.size() == N) {
//        std::cout << "无环"<<std::endl;
        core.clear();

    }
    else {
//        std::cout << "有环"<<std::endl;
        int kk = 0;
        for (int i = 0; i < N; ++i) {
            std::set<int>::iterator it;
            it = record.find(i);
            if (it != record.end()) {
                continue;
            }
            else {
                kk++;
                core.push_back(i);

            }

        }


    }
    return core.size();



}


//如果转换为影响力扩散后，最后一步
bool MvcEnv::isTerminal()
{
    assert(graph);
    int judge = 0;
//    std::shared_ptr<Utils> graphutil =std::shared_ptr<Utils>(new Utils());
//    printf ("num edgeds:%d\n", graph->num_edges);
//    printf ("numCoveredEdges:%d\n", numCoveredEdges);
//    std::vector<bool> backupAllVex(graph->num_nodes, false);
//
//
//    for(std::set<int>::iterator it_set=covered_set.begin();it_set != covered_set.end();++it_set){ //set里面自动排序 从小到大
//        int tem=0;//记录下面循环中covered set中的节点序号
//        for(std::vector<bool>::iterator iter_back=backupAllVex.begin();iter_back != backupAllVex.end();++iter_back){
//            if(tem == *it_set){
//                *iter_back=true;
//                break;
//            }
//            tem+=1;
//        }
//
//    }

//    std::vector<int> idx_map_tem;//指示向量
//    idx_map_tem.swap(graphutil->getInfluactions(graph,backupAllVex,prob)); //swap c++ vector中的操作-> 交换容器
//    std::vector<int>::iterator it = find(idx_map_tem.begin(), idx_map_tem.end(), 1);

//    return (graph->num_edges == numCoveredEdges)||(numCoveredNodes == graph->num_nodes)||(it == idx_map_tem.end());  //#当coverededges数目等于图的边数时 达到终态， 即删除节点时同时记录相关的边 保证了碎片化程度最大？
//    return (it == idx_map_tem.end());



//    std::vector<int>::iterator it = find(node_act_flag.begin(), node_act_flag.end(), 0);
//    return (k_num ==0 || numCoveredNodes == graph->num_nodes || it == node_act_flag.end());


//    judge = decycling_dfs_action(); //不会影响环境中的covered_set 因为没有接着运行step or stepwithoutreward 函数

//    if(covered_set.size()==0){
//
//        judge = decycling_dfs_action();
//    }else{
//
//
//        int num = 0;
//        for(int i = 0; i< record_cycle.size(); ++i){
//            if(record_cycle[i] == 1)
//                num++;
//        }
//
//        if(num==0)
//            judge = -1;
//        else
//            judge = 1;
//
//    }

    judge = decycling_dfs_action();
    std::cout<<"isterminal function"<<std::endl;
    std::cout<<"judge is: "<<judge<<"  "<<"k_num is: "<<k_num<<std::endl;
//    std::cout<<judge<<" "<<k_num<<std::endl;
    if(judge ==-1){
        std::cout << "end: is terminal" << std::endl;
        return 1;
    }else{
        return 0;
    }
}


double MvcEnv::getReward_absolute()
{
    return -(double)decycling_dfs_ratio_absolute();

}



double MvcEnv::getReward()
{
//    int num = 0;
//    for(int i = 0; i< record_cycle.size(); ++i){
//        if(record_cycle[i] == 1)
//            num++;
//    }

    return -(double)decycling_dfs_ratio();
//    return -(double)num/graph->num_nodes;




//    if(reward_seq.size() == 0){
//        return (double)(cycle_node_all-decyclingratio())/graph->num_nodes;
//
//    }else{
////        std::vector<double>::iterator prime_ele = reward_seq.begin();
//        double prime_num = 0;
//        for(std::vector<double>::iterator prime_ele = reward_seq.begin();prime_ele != reward_seq.end();++prime_ele){
//            prime_num += *prime_ele;
//        }
//        return ((double)cycle_node_all-decyclingratio())/graph->num_nodes-prime_num;
//
//
//    }
}



void MvcEnv::printGraph()
{
    printf("edge_list:\n");
    printf("[");
    for (int i = 0; i < (int)graph->edge_list.size();i++)
    {
    printf("[%d,%d,%f],",graph->edge_list[i].first,graph->edge_list[i].second,graph->edge_value[i]);
    }
    printf("]\n");


    printf("covered_set:\n");

    std::set<int>::iterator it;
    printf("[");
    for (it=covered_set.begin();it!=covered_set.end();it++)
    {
        printf("%d,",*it);
    }
    printf("]\n");

}



double MvcEnv::getRemainingCNDScore()
{
    assert(graph);
    Disjoint_Set disjoint_Set =  Disjoint_Set(graph->num_nodes);// #不相交集 大小为图的节点数

    for (int i = 0; i < graph->num_nodes; i++)
    {
        if (covered_set.count(i) == 0)
        {
            for (auto neigh : graph->adj_list[i])
            {
                if (covered_set.count(neigh) == 0)
                {
                    disjoint_Set.merge(i, neigh); //#合并两节点，即相连
                }
            }
        }
    }

    std::set<int> lccIDs; //#记录根节点（去重）
    for(int i =0;i< graph->num_nodes; i++){
        lccIDs.insert(disjoint_Set.unionSet[i]);
    }

    double CCDScore = 0.0;
    for(std::set<int>::iterator it=lccIDs.begin(); it!=lccIDs.end(); it++)
    {
        double num_nodes = (double) disjoint_Set.getRank(*it);// #连通分支的节点数
        CCDScore += (double) num_nodes * (num_nodes-1) / 2; //#即每个连通分支的互相相连的边数求和，即pairwise connectivity
    }

    return CCDScore;
}

//double MvcEnv::getinfluencespread()
//{
//    assert(graph);
//    std::shared_ptr<Utils> graphutil =std::shared_ptr<Utils>(new Utils());
//    std::vector<bool> backupAllVex(graph->num_nodes, false);
//
//    for(std::set<int>::iterator it_set=covered_set.begin();it_set != covered_set.end();++it_set){ //set里面自动排序 从小到大
//        int tem=0;
//        for(std::vector<bool>::iterator iter_back=backupAllVex.begin();iter_back != backupAllVex.end();++iter_back){
//            if(tem == *it_set){
//                *iter_back=true;
//                break;
//            }
//            tem+=1;
//        }
////        backupAllVex[covered_set[i]]=true;
//    }
//
//
//
////    for(int i=0;i<(int)covered_set.size();++i){
////
////    backupAllVex[covered_set[i]]=true;
////
////    }
//    double CCDScore = 0.0;
//    CCDScore=(double)graphutil->influspread_multi2(graph,backupAllVex,10);
//
//
//    return CCDScore;
//}



double MvcEnv::getMaxConnectedNodesNum()
{
    assert(graph);
    Disjoint_Set disjoint_Set =  Disjoint_Set(graph->num_nodes);

    for (int i = 0; i < graph->num_nodes; i++) //#保证两节点同时不是被删除节点，然后通过merge合并节点，合并最终结果就是具有最大连通分支的结果，同时计算出最大连通分支的规模
    {
        if (covered_set.count(i) == 0)
        {
            for (auto neigh : graph->adj_list[i])
            {
                if (covered_set.count(neigh) == 0)
                {
                    disjoint_Set.merge(i, neigh);
                }
            }
        }
    }
    return (double)disjoint_Set.maxRankCount;
}


std::vector<double> MvcEnv::Betweenness(std::vector< std::vector <int> > adj_list) { //#计算每个节点的介数（平均介数？因为除以norm 不是2）

	int i, j, u, v;
	int Long_max = 4294967295;
	int nvertices = adj_list.size();	// The number of vertices in the network 即节点数
	std::vector<double> CB;

    double norm=(double)(nvertices-1)*(double)(nvertices-2); //#排除图节点数等于1和2的情况，因为介数的计算至少涉及到3个节点；另外就是两个节点的对数，norm值是下面每个节点计算时考虑的所有两端节点选择总数

	CB.resize(nvertices);

	std::vector<int> d;								// A vector storing shortest distance estimates 最短路径长度
	std::vector<int> sigma;							// sigma is the number of shortest paths 最短路径数目
	std::vector<double> delta;							// A vector storing dependency of the source vertex on all other vertices 源节点对其他节点的依赖 单个节点的介数
	std::vector< std::vector <int> > PredList;			// A list of predecessors of all vertices 节点的前置节点，可能有多条路径 所以有多个可能

	std::queue <int> Q;								// A priority queue soring vertices 优先队列 存节点
	std::stack <int> S;								// A stack containing vertices in the order found by Dijkstra's Algorithm 按dijk算法存节点的栈，即起点到所有点的最短距离

	// Set the start time of Brandes' Algorithm

	// Compute Betweenness Centrality for every vertex i  介数中心性
	for (i=0; i < nvertices; i++) {
		/* Initialize */
		PredList.assign(nvertices, std::vector <int> (0, 0)); //#初始化predlist 每个节点的vector是空 即元素个数为0的空vector
		d.assign(nvertices, Long_max); //#初始化为无穷大
		d[i] = 0; //#对于节点i 考虑其最短距离为0
		sigma.assign(nvertices, 0); //#数目置为0
		sigma[i] = 1; //#考虑的节点i数目初始为1 即自身
		delta.assign(nvertices, 0); //#初始为0
		Q.push(i); //#第一步将节点i加入队列

		// Use Breadth First Search algorithm  BFS算法   边无权时可以用
		while (!Q.empty()) {
			// Get the next element in the queue
			int u = Q.front(); //#取队首元素
			Q.pop(); //#删除队首元素
			// Push u onto the stack S. Needed later for betweenness computation
			S.push(u); //#将元素u加入栈
			// Iterate over all the neighbors of u
			for (j=0; j < (int) adj_list[u].size(); j++) {
				// Get the neighbor v of vertex u
				// v = (ui64) network->vertex[u].edge[j].target;
				v = (int) adj_list[u][j]; //#u的邻居遍历

				/* Relax and Count */
				if (d[v] == Long_max) { //#第一次遇到的节点才加入队列Q
					 d[v] = d[u] + 1;
					 Q.push(v); //#加入队列
				} //#此判断结束后进入下面的判断

				if (d[v] == d[u] + 1) {
					sigma[v] += sigma[u]; //#节点v的最短路径数增加
					PredList[v].push_back(u); //#前置节点记录
				}
			} // End For

		} // End While

		/* Accumulation */
		while (!S.empty()) {
			u = S.top(); //#获得栈顶值 即从上述求最短路径及数目步骤的最后往前考虑
			S.pop(); //#删除栈顶
			for (j=0; j < (int)PredList[u].size(); j++) {
				delta[PredList[u][j]] += ((double) sigma[PredList[u][j]]/sigma[u]) * (1+delta[u]); //#计算delta_i,·(pre)
			}
			if (u != i) //#排除最短路径的固定起始节点i
				CB[u] += delta[u]; //# 因为是栈，所以倒着取节点，取的节点不作为其他的前置节点 因此delta在之后不会改变
		}

		// Clear data for the next run
		PredList.clear();
		d.clear();
		sigma.clear();
		delta.clear();
	} // End For

	// End time after Brandes' algorithm and the time difference

    for(int i =0; i<nvertices;++i){
        if (norm == 0)
        {
            CB[i] = 0;
        }
        else
        {
            CB[i]=CB[i]/norm; //#每个节点平均介数中心性？  不应该除以norm 应该是2吧？？
        }
    }

	return CB; //#除以norm后的各个节点的介数中心性

} // End of BrandesAlgorithm_Unweighted