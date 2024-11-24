from cython.operator import dereference as deref
from libcpp.memory cimport shared_ptr
import numpy as np
import graph #是pyx吗？
from graph cimport Graph
import gc
from libc.stdlib cimport free

cdef class py_Utils: #下面是python封装接口
    cdef shared_ptr[Utils] inner_Utils
    cdef shared_ptr[Graph] inner_Graph
    def __cinit__(self):
        self.inner_Utils = shared_ptr[Utils](new Utils())
    # def __dealloc__(self):
    #     if self.inner_Utils != NULL:
    #         self.inner_Utils.reset()
    #         gc.collect()
    def influspread2(self,_g,backupAllVex,prob,node_select):
        self.inner_Graph =shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes = _g.num_nodes
        deref(self.inner_Graph).num_edges = _g.num_edges
        deref(self.inner_Graph).edge_list = _g.edge_list
        deref(self.inner_Graph).adj_list = _g.adj_list
        deref(self.inner_Graph).edge_value = _g.edge_value
        return deref(self.inner_Utils).influspread2(self.inner_Graph,backupAllVex,prob,node_select)

    def influspread_multi2(self,_g,backupAllVex,number):
        self.inner_Graph =shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes = _g.num_nodes
        deref(self.inner_Graph).num_edges = _g.num_edges
        deref(self.inner_Graph).edge_list = _g.edge_list
        deref(self.inner_Graph).adj_list = _g.adj_list
        deref(self.inner_Graph).edge_value = _g.edge_value
        return deref(self.inner_Utils).influspread_multi2(self.inner_Graph,backupAllVex,number)

    def getRobustness(self,_g,solution):
        self.inner_Graph =shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes = _g.num_nodes
        deref(self.inner_Graph).num_edges = _g.num_edges
        deref(self.inner_Graph).edge_list = _g.edge_list
        deref(self.inner_Graph).adj_list = _g.adj_list
        deref(self.inner_Graph).edge_value = _g.edge_value
        return deref(self.inner_Utils).getRobustness(self.inner_Graph,solution)

    def getRobustnessInflu(self,_g,solution):
        self.inner_Graph =shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes = _g.num_nodes
        deref(self.inner_Graph).num_edges = _g.num_edges
        deref(self.inner_Graph).edge_list = _g.edge_list
        deref(self.inner_Graph).adj_list = _g.adj_list
        deref(self.inner_Graph).edge_value = _g.edge_value
        return deref(self.inner_Utils).getRobustnessInflu(self.inner_Graph,solution)

    def getInfluactions(self,_g,backupAllVex,prob):
        self.inner_Graph=shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes = _g.num_nodes
        deref(self.inner_Graph).num_edges = _g.num_edges
        deref(self.inner_Graph).edge_list = _g.edge_list
        deref(self.inner_Graph).adj_list = _g.adj_list
        deref(self.inner_Graph).edge_value = _g.edge_value
        return deref(self.inner_Utils).getInfluactions(self.inner_Graph,backupAllVex,prob)

    def reInsert(self,_g,solution,allVex,int decreaseStrategyID,int reinsertEachStep):
        self.inner_Graph =shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes = _g.num_nodes
        deref(self.inner_Graph).num_edges = _g.num_edges
        deref(self.inner_Graph).edge_list = _g.edge_list
        deref(self.inner_Graph).adj_list = _g.adj_list
        deref(self.inner_Graph).edge_value = _g.edge_value
        return deref(self.inner_Utils).reInsert(self.inner_Graph,solution,allVex,decreaseStrategyID,reinsertEachStep)

    def getMxWccSz(self, _g):
        self.inner_Graph =shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes = _g.num_nodes
        deref(self.inner_Graph).num_edges = _g.num_edges
        deref(self.inner_Graph).edge_list = _g.edge_list
        deref(self.inner_Graph).adj_list = _g.adj_list
        deref(self.inner_Graph).edge_value = _g.edge_value
        return deref(self.inner_Utils).getMxWccSz(self.inner_Graph)

    def Betweenness(self,_g):
        self.inner_Graph =shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes = _g.num_nodes
        deref(self.inner_Graph).num_edges = _g.num_edges
        deref(self.inner_Graph).edge_list = _g.edge_list
        deref(self.inner_Graph).adj_list = _g.adj_list
        deref(self.inner_Graph).edge_value = _g.edge_value
        return deref(self.inner_Utils).Betweenness(self.inner_Graph)

    @property
    def MaxWccSzList(self):
        return deref(self.inner_Utils).MaxWccSzList

    @property
    def MaxInfluSzList(self):
        return deref(self.inner_Utils).MaxInfluSzList