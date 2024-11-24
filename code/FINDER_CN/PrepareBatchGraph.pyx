from cython.operator import dereference as deref #cython.operator库下包含了一些特殊的c++运算符替代函数，用于重载C++运算符，该处是解引用 比如*a 也可用索引使用 比如[0]
from libcpp.memory cimport shared_ptr   #是否和pxd中的重复了
import numpy as np

import graph   # 应该是等待pyd文件产生后import  pyd需要运行setup

from libc.stdlib cimport malloc
from libc.stdlib cimport free
from graph cimport Graph #.pxd文件  是否和pxd中的重复了
import tensorflow as tf
from scipy.sparse import coo_matrix #稀疏矩阵的坐标形式存储
# import gc
#自动同名pxd文件导入

cdef class py_sparseMatrix:  #cdef class 扩展类型的声明
    cdef shared_ptr[sparseMatrix] inner_sparseMatrix  #扩展类型的cdef属性 指针
    def __cinit__(self):
        self.inner_sparseMatrix =shared_ptr[sparseMatrix](new sparseMatrix()) #默认构造
    # def __dealloc__(self):
    #     if self.inner_sparseMatrix != NULL:
    #         self.inner_sparseMatrix.reset() #清空内存?
    #         gc.collect()

    @property   #一种装饰器 修饰方法，将方法转换为同名称的只读属性
    def rowIndex(self):
        return deref(self.inner_sparseMatrix).rowIndex   #deref函数是什么
    @property
    def colIndex(self):
        return deref(self.inner_sparseMatrix).colIndex
    @property
    def value(self):
        return deref(self.inner_sparseMatrix).value
    @property
    def rowNum(self):
        return deref(self.inner_sparseMatrix).rowNum
    @property
    def colNum(self):
        return deref(self.inner_sparseMatrix).colNum


cdef class py_PrepareBatchGraph:
    cdef shared_ptr[PrepareBatchGraph] inner_PrepareBatchGraph
    cdef sparseMatrix matrix  #这里用cdef一个sparsematrix可能是因为c语言处理更快？
    def __cinit__(self,aggregatorID):
        self.inner_PrepareBatchGraph =shared_ptr[PrepareBatchGraph](new PrepareBatchGraph(aggregatorID))
    # def __dealloc__(self):
    #     if self.inner_PrepareBatchGraph != NULL:
    #         self.inner_PrepareBatchGraph.reset()
    #         gc.collect()

    def SetupTrain(self,idxes,g_list,covered,list actions):  #这里参数不写类型吗？ 比如int 为什么只写list actions一个，其中list是python中对应的数据类型
        cdef shared_ptr[Graph] inner_Graph #下面循环中的每一项单独考虑
        cdef vector[shared_ptr[Graph]] inner_glist #存储g_list的相应子图  转化为cdef来操作，输入的参数是通过python的接口，无法直接使用 需要转换
        for _g in g_list:
            # inner_glist.push_back(_g.inner_Graph)
            inner_Graph = shared_ptr[Graph](new Graph()) #这里是构造 不是malloc 所以不需要free 循环结束自动消除
            deref(inner_Graph).num_nodes = _g.num_nodes
            deref(inner_Graph).num_edges = _g.num_edges
            deref(inner_Graph).edge_list = _g.edge_list
            deref(inner_Graph).adj_list = _g.adj_list
            deref(inner_Graph).edge_value = _g.edge_value
            inner_glist.push_back(inner_Graph)

        cdef int *refint = <int*>malloc(len(actions)*sizeof(int)) #对应actions 的cdef ，因为函数参数说明了list 所以需要手动转换
        cdef int i
        for i in range(len(actions)):
            refint[i] = actions[i]
        deref(self.inner_PrepareBatchGraph).SetupTrain(idxes,inner_glist,covered,refint)
        free(refint)

    def SetupPredAll(self,idxes,g_list,covered):
        cdef shared_ptr[Graph] inner_Graph
        cdef vector[shared_ptr[Graph]] inner_glist
        for _g in g_list:
            # inner_glist.push_back(_g.inner_Graph)
            inner_Graph = shared_ptr[Graph](new Graph())
            deref(inner_Graph).num_nodes = _g.num_nodes
            deref(inner_Graph).num_edges = _g.num_edges
            deref(inner_Graph).edge_list = _g.edge_list
            deref(inner_Graph).adj_list = _g.adj_list
            deref(inner_Graph).edge_value = _g.edge_value
            inner_glist.push_back(inner_Graph)
        deref(self.inner_PrepareBatchGraph).SetupPredAll(idxes,inner_glist,covered)








    @property
    def act_select(self):
        self.matrix = deref(deref(self.inner_PrepareBatchGraph).act_select)
        return self.ConvertSparseToTensor(self.matrix)
    @property
    def rep_global(self):
        matrix = deref(deref(self.inner_PrepareBatchGraph).rep_global)
        return self.ConvertSparseToTensor(matrix)
        #return coo_matrix((data, (rowIndex,colIndex)), shape=(rowNum,colNum))
    @property
    def n2nsum_param(self):
        matrix = deref(deref(self.inner_PrepareBatchGraph).n2nsum_param)
        return self.ConvertSparseToTensor(matrix)



    @property
    def laplacian_param(self):
        matrix = deref(deref(self.inner_PrepareBatchGraph).laplacian_param)
        return self.ConvertSparseToTensor(matrix)
    @property
    def subgsum_param(self):
        matrix = deref(deref(self.inner_PrepareBatchGraph).subgsum_param)
        return self.ConvertSparseToTensor(matrix)




    @property
    def idx_map_list(self):
        return deref(self.inner_PrepareBatchGraph).idx_map_list #vector自动转化tensor？
    @property
    def subgraph_id_span(self):
        return deref(self.inner_PrepareBatchGraph).subgraph_id_span
    @property
    def aux_feat(self):
        return deref(self.inner_PrepareBatchGraph).aux_feat
    @property
    def aggregatorID(self):
        return deref(self.inner_PrepareBatchGraph).aggregatorID
    @property
    def avail_act_cnt(self):
        return deref(self.inner_PrepareBatchGraph).avail_act_cnt

    cdef ConvertSparseToTensor(self,sparseMatrix matrix):  #供内部使用，非python成员函数接口  cpdef同时提供c和python接口使用

        rowIndex= matrix.rowIndex
        colIndex= matrix.colIndex
        data= matrix.value
        rowNum= matrix.rowNum
        colNum= matrix.colNum
        indices = np.mat([rowIndex, colIndex]).transpose()  #np的函数mat
        return tf.SparseTensorValue(indices, data, (rowNum,colNum)) #tf的函数sparsetensorvalue   表示稀疏矩阵，需要转化为dense才能print出矩阵形式数据表达






