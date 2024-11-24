
from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp.memory cimport shared_ptr
from libcpp cimport bool
from graph cimport Graph
# from utils cimport Utils

cdef extern from "./src/lib/mvc_env.h":
    cdef cppclass MvcEnv:
        MvcEnv(double _norm)
        void s0(shared_ptr[Graph] _g)except+
        double step(int a)except+


        double step_fake(int a)except+
        double step_fake_reverse(int a)except+


        void stepWithoutReward(int a)except+
        int randomAction()except+
        int betweenAction()except+
        int influenceAction()except+

        int decyclingaction()except+
        int decyclingratio()except+
        vector[int] decyclingaction_list()except+

        int decycling_dfs_action_contrast()except+

        int decycling_dfs_action()except+
        vector[int] decycling_dfs_action_list()except+
        double decycling_dfs_ratio()except+

        double decycling_dfs_ratio_absolute()except+


        bool isTerminal()except+
        # double getReward(double oldCcNum)except+
        double getReward()except+
        double getReward_absolute()except+

        double getMaxConnectedNodesNum()except+
        double getRemainingCNDScore()except+
        # double getinfluencespread()except+;



        double norm
        int cycle_node_all
        int k_num
        double CNR
        double CNR_all;
        bool CNR_all_flag;

        vector[int] record_cycle

        # double CcNum #还是不知道是什么
        shared_ptr[Graph] graph
        vector[vector[int]]  state_seq
        vector[int] act_seq
        vector[int] action_list
        vector[double] reward_seq
        vector[double] sum_rewards
        int numCoveredEdges
        int numCoveredNodes
        set[int] covered_set
        vector[int] avail_list
        vector[double] prob

        vector[int] node_act_flag
        vector[int] edge_act_flag
        vector[int] left
        vector[int] right

        vector[int] node_degrees

