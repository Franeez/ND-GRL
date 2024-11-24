#include "nstep_replay_mem.h"
#include "i_env.h" //#没有cpp？ 文件中只有一个纯虚类
#include <cassert>
#include <algorithm>
#include <time.h>
#include <math.h>

#define max(x, y) (x > y ? x : y)
#define min(x, y) (x > y ? y : x)


 ReplaySample::ReplaySample(int batch_size){
    g_list.resize(batch_size);
    list_st.resize(batch_size);
    list_s_primes.resize(batch_size);
    list_at.resize(batch_size);
    list_rt.resize(batch_size);
    list_term.resize(batch_size);
 }
 NStepReplayMem::NStepReplayMem(int _memory_size)
{
    memory_size = _memory_size;
    graphs.resize(memory_size);
    actions.resize(memory_size);
    rewards.resize(memory_size);
    states.resize(memory_size);
    s_primes.resize(memory_size);
    terminals.resize(memory_size);

    current = 0; //#内存块当前所指位置，即下一个要存的位置指示
    count = 0; //#存储数目
    distribution = new std::uniform_int_distribution<int>(0, memory_size - 1); //#按照内存块编号
}

void NStepReplayMem::Add(std::shared_ptr<Graph> g, 
                        std::vector<int> s_t,
                        int a_t, 
                        double r_t,
                        std::vector<int> s_prime,
                        bool terminal)
{
    graphs[current] = g;
    actions[current] = a_t;
    rewards[current] = r_t;
    states[current] = s_t;
    s_primes[current] = s_prime;
    terminals[current] = terminal; //#是否终态

    count = max(count, current + 1); //#已存内容数目的计数更新 直到达到memory_size 不变
    current = (current + 1) % memory_size; 
}

void NStepReplayMem::Add(std::shared_ptr<MvcEnv> env,int n_step) //#将env记录的信息 即num_steps条消息 存n_step给mem中
{
    assert(env->isTerminal()); //#达到终止态
    int num_steps = env->state_seq.size(); //#删除节点次数
    assert(num_steps); //#非初始 即已经有删除节点

    env->sum_rewards[num_steps - 1] = env->reward_seq[num_steps - 1]; //#最后一步的奖励值初始化sum_rewards
    for (int i = num_steps - 1; i >= 0; --i)
        if (i < num_steps - 1)
            env->sum_rewards[i] = env->sum_rewards[i + 1] + env->reward_seq[i]; //#求和的过程

    for (int i = 0; i < num_steps; ++i) //#遍历每一步删除过程 每一步存一次
    {
        bool term_t = false;
        double cur_r; //#当前第i步的reward
        std::vector<int> s_prime;
        if (i + n_step >= num_steps)//？？？？？？？？ 向后不足n步时
        {
            cur_r = env->sum_rewards[i];
            s_prime = (env->action_list); //#整个图的最后的状态表示 即num_steps步操作后的state
            term_t = true; //#终态  因为已经不足取完整的n_step了
        } else { //
            cur_r = env->sum_rewards[i] - env->sum_rewards[i + n_step]; //#当前step i 向后n步累计reward
            s_prime = (env->state_seq[i + n_step]);
        }
        Add(env->graph, env->state_seq[i], env->act_seq[i], cur_r, s_prime, term_t); //#记录env图g的每一步 总共num_steps条记录   n step DQN prime是n步后状态
    }
}

std::shared_ptr<ReplaySample> NStepReplayMem::Sampling(int batch_size)
{
//    std::shared_ptr<ReplaySample> result {new ReplaySample(batch_size)};
    std::shared_ptr<ReplaySample> result =std::shared_ptr<ReplaySample>(new ReplaySample(batch_size));
    assert(count >= batch_size); //#不能超出内存大小   memory中每一条是 图g的一步操作？

    result->g_list.resize(batch_size);
    result->list_st.resize(batch_size);
    result->list_at.resize(batch_size);
    result->list_rt.resize(batch_size);
    result->list_s_primes.resize(batch_size);
    result->list_term.resize(batch_size);
    auto& dist = *distribution; //#dist就是分布的别名
    for (int i = 0; i < batch_size; ++i)
    {
        int idx = dist(generator) % count; //#随机取下标
        result->g_list[i] = graphs[idx];
        result->list_st[i] = (states[idx]);
        result->list_at[i] = actions[idx];
        result->list_rt[i] = rewards[idx];
        result->list_s_primes[i] = (s_primes[idx]);
        result->list_term[i] = terminals[idx];
    }
    return result;
}
