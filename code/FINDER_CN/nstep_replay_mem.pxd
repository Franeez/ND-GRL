from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp cimport bool
from graph cimport Graph
from mvc_env cimport MvcEnv #pyx文件中需要使用mvcenv


cdef extern from "./src/lib/nstep_replay_mem.h":
    cdef cppclass ReplaySample:
        ReplaySample(int batch_size)except+
        vector[shared_ptr[Graph]] g_list
        vector[vector[int]] list_st
        vector[vector[int]] list_s_primes
        vector[int] list_at
        vector[double] list_rt
        vector[bool] list_term


cdef extern from "./src/lib/nstep_replay_mem.h":
    cdef cppclass NStepReplayMem:
        NStepReplayMem(int memory_size)except+
        # void Add(shared_ptr[Graph] g,vector[int]& s_t,int a_t,double r_t,vector[int]& s_prime,bool terminal) #内部的单条加入可以隐藏起来不给接口
        void Add(shared_ptr[MvcEnv] env,int n_step)except+
        shared_ptr[ReplaySample] Sampling(int batch_size)except+ #存储
        void Clear()except+ #清空？
        vector[shared_ptr[Graph]] graphs
        vector[int] actions
        vector[double] rewards
        vector[vector[int]] states
        vector[vector[int]] s_primes
        vector[bool] terminals
        int current
        int count
        int memory_size

