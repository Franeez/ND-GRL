#include "graph.h"
#include <cassert>
#include <iostream>
#include <random>
#include <iterator>
#include "disjoint_set.h"
//#include "stdio.h"

Disjoint_Set::Disjoint_Set(int graphSize)
{
    unionSet.resize(graphSize);
    rankCount.resize(graphSize);
    for (int i = 0; i < (int)unionSet.size(); i++)//#i++可以换成++i，更快
    {
        unionSet[i] = i; //节点序号
        rankCount[i] = 1;
    }
    maxRankCount = 1;
    CCDScore = 0.0;
}

Disjoint_Set::~Disjoint_Set()
{
    unionSet.clear();
    rankCount.clear();
    maxRankCount = 1;
    CCDScore = 0.0;
}

int Disjoint_Set::findRoot(int node)//#找节点node的根节点
{
    if (node != unionSet[node])
    {
        int rootNode = findRoot(unionSet[node]);
        unionSet[node] = rootNode;
        return rootNode;
    }
    else
    {
        return node;
    }
}

void Disjoint_Set::merge(int node1, int node2)//#合并
{
    int node1Root = findRoot(node1);
    int node2Root = findRoot(node2);
    if (node1Root != node2Root)//#如果两分支没合并
    {
        double node1Rank = (double)rankCount[node1Root];//#根据根节点的rankCount确定合并方式
        double node2Rank = (double)rankCount[node2Root];
        CCDScore = CCDScore - node1Rank*(node1Rank-1)/2.0 - node2Rank*(node2Rank-1)/2.0;//#去除两根节点单独对score的影响
        CCDScore = CCDScore + (node1Rank+node2Rank)*(node1Rank+node2Rank-1)/2.0;//#加入两节点共同的混合影响到score中

        if (rankCount[node2Root] > rankCount[node1Root])
        {
            unionSet[node1Root] = node2Root;
            rankCount[node2Root] += rankCount[node1Root];//#rankCount代表节点数（因为初始化均是1）

            if (rankCount[node2Root] > maxRankCount)
            {
                maxRankCount = rankCount[node2Root];//#maxRankCount则是合并后最大连通分支的rank，代表节点数
            }
        }
        else
        {
            unionSet[node2Root] = node1Root;
            rankCount[node1Root] += rankCount[node2Root];

            if (rankCount[node1Root] > maxRankCount)
            {
                maxRankCount = rankCount[node1Root];
            }

        }
    }
}


double Disjoint_Set::getBiggestComponentCurrentRatio() const
{
    return double(maxRankCount) / double(rankCount.size());//#最大rank对节点数平均，即最大连通分支节点数占总节点比例
}


int Disjoint_Set::getRank(int rootNode) const
{
    return rankCount[rootNode];
}

