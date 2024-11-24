#ifndef DISJOINT_SET_H //#防止一个cpp中多次include该头文件
#define DISJOINT_SET_H

#include <map> //#标准模板库STL中的容器，提供一对一的hash，即 关键字——值
#include <vector> //#向量（动态 有序序列）
#include <memory> //#包含了一些指针，比如auto_ptr shared_ptr ...
#include <algorithm> //#包含一些对容器的常用操作
#include <set> //#集合 排异性
class Disjoint_Set //#不相交集合，找树的根？
{
public:
    Disjoint_Set(); //#未定义？
    Disjoint_Set(int graphSize);
    ~Disjoint_Set();
    int findRoot(int node);
    void merge(int node1, int node2);
    double getBiggestComponentCurrentRatio() const; //#this 指针是const，即this 指向的内容是const，表示函数无法改变数据，只读
    int getRank(int rootNode) const;
	std::vector<int> unionSet;//#节点集，表示每个下标节点中元素代表节点所在分支的根
	std::vector<int> rankCount;//#每个节点的等级rank，根据这个来合并节点的根
	int maxRankCount;//#所有rank中max
    double CCDScore;//#合并后的总评判分数
};



#endif
