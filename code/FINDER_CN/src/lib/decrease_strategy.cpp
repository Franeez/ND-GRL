#include <vector>
#include <algorithm>
#include <iterator>
#include <unordered_set>
#include <set>
#include "stdio.h"
#include "disjoint_set.h"
#include "math.h"
using namespace std;

class decreaseComponentStrategy /**#减少连通分支的步骤，加节点后连通分支数减小的情况 **/
{
public:

	decreaseComponentStrategy()
	{
//		printf("decreaseComponentStrategy:%s\n",description);
	}

	virtual long long decreaseComponentNumIfAddNode(const vector<vector<int> >& backupCompletedAdjListGraph, const vector<bool>& currentAllVex, Disjoint_Set &unionSet, int node) = 0; //#纯虚函数 类只能被继承

};


class decreaseComponentRank :public decreaseComponentStrategy //#加节点后受影响的新连通分支节点数
{

public:
	decreaseComponentRank() :decreaseComponentStrategy(){}

	virtual long long decreaseComponentNumIfAddNode(const vector<vector<int> >& backupCompletedAdjListGraph, const vector<bool>& currentAllVex, Disjoint_Set &unionSet, int node)
	{
		unordered_set<int> componentSet;

		for (int i = 0; i < (int)backupCompletedAdjListGraph[node].size(); i++)
		{
			int neighbourNode = backupCompletedAdjListGraph[node][i];

			if (currentAllVex[neighbourNode]) //#暂时不知道这个什么时候是1 什么时候是0 False      如果是剩余节点则为1 True
			{
				componentSet.insert(unionSet.findRoot(neighbourNode)); //#找到与node有关的连通分支对应的根root
			}
		}

		long long sum = 1;

		for (int eachNode : componentSet)
		{
			sum += unionSet.getRank(eachNode); //#sum从一开始加 1代表node本身 其他代表通过node连接的原连通分支的大小
		}
		return sum;
	}
};

class decreaseComponentCount : public decreaseComponentStrategy //#计数count
{
public:

	decreaseComponentCount() :decreaseComponentStrategy(){}

	virtual long long decreaseComponentNumIfAddNode(const vector<vector<int> >& backupCompletedAdjListGraph, const vector<bool>& currentAllVex, Disjoint_Set &unionSet, int node) override
	{
		unordered_set<int> componentSet;

		for (int i = 0; i < (int)backupCompletedAdjListGraph[node].size(); i++)
		{
			int neighbourNode = backupCompletedAdjListGraph[node][i];

			if (currentAllVex[neighbourNode])
			{
				componentSet.insert(unionSet.findRoot(neighbourNode));
			}
		}

		return (long long)componentSet.size(); //#返回受node 节点影响的连通分支对应的root集合
	}
};

class decreaseComponentMultiple : public decreaseComponentStrategy
{
public:

	decreaseComponentMultiple() :decreaseComponentStrategy(){}

	virtual long long decreaseComponentNumIfAddNode(const vector<vector<int> >& backupCompletedAdjListGraph, const vector<bool>& currentAllVex, Disjoint_Set &unionSet, int node) override
	{
		unordered_set<int> componentSet;

		for (int i = 0; i < (int)backupCompletedAdjListGraph[node].size(); i++)
		{
			int neighbourNode = backupCompletedAdjListGraph[node][i];

			if (currentAllVex[neighbourNode])
			{
				componentSet.insert(unionSet.findRoot(neighbourNode));
			}
		}

		long long sum = 1;

		for (int eachNode : componentSet)
		{
			sum += unionSet.getRank(eachNode);
		}

		sum *= componentSet.size(); //#类似于权重？ 乘以受影响的原连通分支数（即不同的root数目）

		return sum;
	}
};

