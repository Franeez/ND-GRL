
from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp.memory cimport shared_ptr #和pyx重复？
from libcpp cimport bool
from graph cimport Graph

cdef extern from "./src/lib/utils.h":
    cdef cppclass Utils:
        Utils()
        int influspread2(shared_ptr[Graph] graph, vector[bool] backupAllVex,vector[double] prob,vector[bool] node_select)except+
        double influspread_multi2(shared_ptr[Graph] graph, vector[bool] backupAllVex,int number)except+
        double getRobustness(shared_ptr[Graph] graph,vector[int] solution)except+
        double getRobustnessInflu(shared_ptr[Graph] graph, vector[int] solution)except+
        vector[int] getInfluactions(shared_ptr[Graph] graph,vector[bool] backupAllVex,vector[double] prob)except+
        vector[int] reInsert(shared_ptr[Graph] graph,vector[int] solution,vector[int] allVex,int decreaseStrategyID,int reinsertEachStep)except+
        int getMxWccSz(shared_ptr[Graph] graph)
        vector[double] Betweenness(shared_ptr[Graph] graph)
        vector[double] MaxWccSzList
        vector[double] MaxInfluSzList

