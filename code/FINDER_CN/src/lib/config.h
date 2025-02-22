#ifndef cfg_H
#define cfg_H

#include <iostream> /** #输入输出流**/
#include <cstring>
#include <fstream> /** #文件流**/
#include <set>
#include <map>

typedef float Dtype;

struct cfg
{
    static int embed_dim;
    static int batch_size;
    static int max_n, min_n;
    static int mem_size;
    static int node_dim;
    static int aux_dim;
    static bool msg_average;

    static void LoadParams(const int argc, const char** argv) //从1开始 -embed_dim 64 -max_n 50 ....
    {
        for (int i = 1; i < argc; i += 2) /**#为什么i不从0开始？？ **/
        {
            if (strcmp(argv[i], "-embed_dim") == 0)
                embed_dim = atoi(argv[i + 1]); 
            if (strcmp(argv[i], "-max_n") == 0)
                max_n = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-min_n") == 0)
                min_n = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-mem_size") == 0)
                mem_size = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-batch_size") == 0)
                batch_size = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-msg_average") == 0)
                msg_average = atoi(argv[i + 1]);                 

        }

        if (n_step <= 0)  /**#n_step在哪? **/
            n_step = max_n;

    }
};

#endif
