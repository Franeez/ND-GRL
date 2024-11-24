'''
file:graph.pxd 类graph的定义文件对应graph.h
'''
#Cython已经编译了C++的std模板库，位置在~/Cython/Includes/lincpp/    lincpp中是pxd文件组成    pxd文件中不能有python的语法定义变量，比如def 因为python无需声明，这是语言的区别所在
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp.map cimport map #关联容器，提供一对一的hash
from libcpp.pair cimport pair

cdef extern from "./src/lib/graph.h":  #在cython中引入c/c++内容 并封装，使cython可以使用  下面的类定义必须嵌套在extern from这个模块下
    cdef cppclass Graph:     #cppclass告诉cython编译器封装的外部代码是c++，类名必须与c++一致 ，没有必要所有的类成员 属性都声明 因为cython不支持全部的c++内容，除非自行封装
        Graph()except+   #except+可以识别并容纳c++的错误，可安全地引发因内存分配产生的异常等

        Graph(const int _num_nodes, const int _num_edges, const int* edges_from, const int* edges_to) except+
        Graph(const int _num_nodes, const int _num_edges, const int* edges_from, const int* edges_to, const double* edge_value_copy) except+
        int num_nodes
        int num_edges
        vector[vector[int]] adj_list
        vector[pair[int,int]] edge_list
        vector[double] edge_value

cdef extern from "./src/lib/graph.h":
    cdef cppclass GSet:
        GSet()except+
        void InsertGraph(int gid, shared_ptr[Graph] graph)except+
        shared_ptr[Graph] Sample()except+
        shared_ptr[Graph] Get(int gid)except+
        vector[double] Get_value(int gid)except+
        void Clear()except+
        map[int, shared_ptr[Graph]] graph_pool

