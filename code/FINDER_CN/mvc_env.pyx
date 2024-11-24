from cython.operator import dereference as deref
from libcpp.memory cimport shared_ptr #与pxd中重复？
import numpy as np

import graph  #pyx文件，主要是为了导入相应类，以便能实现比如返回python接口类型的变量等操作 /应该是setup后的graph文件 动态链接库
# import utils #不确定是否需要

from graph cimport Graph #pxd中的封装后Graph   与pxd中重复？
# from utils cimport Utils #不确定是否需要
import gc
from libc.stdlib cimport free

cdef class py_MvcEnv:
    cdef shared_ptr[MvcEnv] inner_MvcEnv
    cdef shared_ptr[Graph] inner_Graph
    # cdef shared_ptr[Utils] inner_Utils
    def __cinit__(self,double _norm):
        self.inner_MvcEnv = shared_ptr[MvcEnv](new MvcEnv(_norm))
        self.inner_Graph =shared_ptr[Graph](new Graph())
    #    self.inner_Utils =shared_ptr[Utils](new Utils())
    # def __dealloc__(self):
    #     if self.inner_MvcEnv != NULL:
    #         self.inner_MvcEnv.reset()
    #         gc.collect()
    #     if self.inner_Graph != NULL:
    #         self.inner_Graph.reset()
    #         gc.collect()
    def s0(self,_g):
        self.inner_Graph =shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes = _g.num_nodes
        deref(self.inner_Graph).num_edges = _g.num_edges
        deref(self.inner_Graph).edge_list = _g.edge_list
        deref(self.inner_Graph).adj_list = _g.adj_list
        deref(self.inner_Graph).edge_value = _g.edge_value
        deref(self.inner_MvcEnv).s0(self.inner_Graph)

    def step(self,int a):
        return deref(self.inner_MvcEnv).step(a)

    def step_fake(self,int a):
        return deref(self.inner_MvcEnv).step_fake(a)

    def step_fake_reverse(self,int a):
        return deref(self.inner_MvcEnv).step_fake_reverse(a)

    def stepWithoutReward(self,int a):
        deref(self.inner_MvcEnv).stepWithoutReward(a)

    def randomAction(self):
        return deref(self.inner_MvcEnv).randomAction()

    def betweenAction(self):
        return deref(self.inner_MvcEnv).betweenAction()

    def influenceAction(self):
        return deref(self.inner_MvcEnv).influenceAction()

    def decyclingaction(self):
        return deref(self.inner_MvcEnv).decyclingaction()

    def decyclingratio(self):
        return deref(self.inner_MvcEnv).decyclingratio()

    def decyclingaction_list(self):
        return deref(self.inner_MvcEnv).decyclingaction_list()

    def decycling_dfs_action_contrast(self):
        return deref(self.inner_MvcEnv).decycling_dfs_action_contrast()

    def decycling_dfs_action(self):
        return deref(self.inner_MvcEnv).decycling_dfs_action()

    def decycling_dfs_action_list(self):
        return deref(self.inner_MvcEnv).decycling_dfs_action_list()

    def decycling_dfs_ratio(self):
        return deref(self.inner_MvcEnv).decycling_dfs_ratio()

    def decycling_dfs_ratio_absolute(self):
        return deref(self.inner_MvcEnv).decycling_dfs_ratio_absolute()


    def isTerminal(self):
        return deref(self.inner_MvcEnv).isTerminal()

    def getReward(self):
        return deref(self.inner_MvcEnv).getReward()

    def getReward_absolute(self):
        return deref(self.inner_MvcEnv).getReward_absolute()

    def getMaxConnectedNodesNum(self):
        return deref(self.inner_MvcEnv).getMaxConnectedNodesNum()

    def getRemainingCNDScore(self):
        return deref(self.inner_MvcEnv).getRemainingCNDScore()

    # def getinfluencespread(self):
       # return deref(self.inner_MvcEnv).getinfluencespread()



    @property
    def norm(self):
        return deref(self.inner_MvcEnv).norm

    @property
    def cycle_node_all(self):
        return deref(self.inner_MvcEnv).cycle_node_all

    @property
    def k_num(self):
        return deref(self.inner_MvcEnv).k_num

    @property
    def CNR(self):
        return deref(self.inner_MvcEnv).CNR

    @property
    def CNR_all(self):
        return deref(self.inner_MvcEnv).CNR_all

    @property
    def CNR_all_flag(self):
        return deref(self.inner_MvcEnv).CNR_all_flag

    @property
    def record_cycle(self):
        return deref(self.inner_MvcEnv).record_cycle

    @property
    def graph(self):
        # temp_innerGraph=deref(self.inner_Graph)   #得到了Graph 对象
        return self.G2P(deref(self.inner_Graph))

    @property
    def state_seq(self):
        return deref(self.inner_MvcEnv).state_seq

    @property
    def act_seq(self):
        return deref(self.inner_MvcEnv).act_seq

    @property
    def action_list(self):
        return deref(self.inner_MvcEnv).action_list

    @property
    def reward_seq(self):
        return deref(self.inner_MvcEnv).reward_seq

    @property
    def sum_rewards(self):
        return deref(self.inner_MvcEnv).sum_rewards

    @property
    def numCoveredEdges(self):
        return deref(self.inner_MvcEnv).numCoveredEdges

    @property
    def numCoveredNodes(self):
        return deref(self.inner_MvcEnv).numCoveredNodes

    @property
    def covered_set(self):
        return deref(self.inner_MvcEnv).covered_set

    @property
    def avail_list(self):
        return deref(self.inner_MvcEnv).avail_list

    @property
    def prob(self):
        return deref(self.inner_MvcEnv).prob

    @property
    def node_act_flag(self):
        return deref(self.inner_MvcEnv).node_act_flag

    @property
    def edge_act_flag(self):
        return deref(self.inner_MvcEnv).edge_act_flag

    @property
    def left(self):
        return deref(self.inner_MvcEnv).left

    @property
    def right(self):
        return deref(self.inner_MvcEnv).right

    @property
    def node_degrees(self):
        return deref(self.inner_MvcEnv).node_degrees


    cdef G2P(self,Graph graph1):
        num_nodes = graph1.num_nodes     #得到Graph对象的节点个数
        num_edges = graph1.num_edges    #得到Graph对象的连边个数
        edge_list = graph1.edge_list
        edge_value = graph1.edge_value
        cint_edges_from = np.zeros([num_edges],dtype=np.int)
        cint_edges_to = np.zeros([num_edges],dtype=np.int)
        cdouble_edge_value_copy = np.zeros([num_edges],dtype=np.double)
        for i in range(num_edges):
            cint_edges_from[i]=edge_list[i].first
            cint_edges_to[i] =edge_list[i].second
            cdouble_edge_value_copy[i] = edge_value[i]
        return graph.py_Graph(num_nodes,num_edges,cint_edges_from,cint_edges_to,cdouble_edge_value_copy)


    # cdef reshape_Graph(self, int _num_nodes, int _num_edges, int[:] edges_from, int[:] edges_to):
    #     cdef int *cint_edges_from = <int*>malloc(_num_edges*sizeof(int))
    #     cdef int *cint_edges_to = <int*>malloc(_num_edges*sizeof(int))
    #     cdef int i
    #     for i in range(_num_edges):
    #         cint_edges_from[i] = edges_from[i]
    #     for i in range(_num_edges):
    #         cint_edges_to[i] = edges_to[i]
    #     free(cint_edges_from)
    #     free(cint_edges_to)
    #     return  new Graph(_num_nodes,_num_edges,&cint_edges_from[0],&cint_edges_to[0]) #已经free了 参数不能使用了...