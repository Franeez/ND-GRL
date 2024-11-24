
from distutils.core import setup #框架，说明要链接外部库，这里是C++
from distutils.extension import Extension #一种链接第三方扩展包的方法，可以指明语言 编辑器等选项，涉及动态链接库and相对路径
# from Cython.Build import cythonize #将其他语言编写的程序转化为C语言编写的程序
from Cython.Distutils import build_ext #指明python生成C/C++的扩展模块

setup(
    cmdclass = {'build_ext':build_ext},

    #################for ubuntu compile
    ext_modules = [
                    Extension('PrepareBatchGraph', sources = ['PrepareBatchGraph.pyx','src/lib/PrepareBatchGraph.cpp','src/lib/graph.cpp','src/lib/graph_struct.cpp',  'src/lib/disjoint_set.cpp'],language='c++',extra_compile_args=['-std=c++11']),
                    Extension('graph', sources=['graph.pyx', 'src/lib/graph.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('mvc_env', sources=['mvc_env.pyx', 'src/lib/mvc_env.cpp', 'src/lib/graph.cpp','src/lib/disjoint_set.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('utils', sources=['utils.pyx', 'src/lib/utils.cpp', 'src/lib/graph.cpp', 'src/lib/graph_utils.cpp', 'src/lib/disjoint_set.cpp', 'src/lib/decrease_strategy.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('nstep_replay_mem', sources=['nstep_replay_mem.pyx', 'src/lib/nstep_replay_mem.cpp', 'src/lib/graph.cpp', 'src/lib/mvc_env.cpp', 'src/lib/disjoint_set.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('nstep_replay_mem_prioritized',sources=['nstep_replay_mem_prioritized.pyx', 'src/lib/nstep_replay_mem_prioritized.cpp','src/lib/graph.cpp', 'src/lib/mvc_env.cpp', 'src/lib/disjoint_set.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('graph_struct', sources=['graph_struct.pyx', 'src/lib/graph_struct.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('FINDER', sources = ['FINDER.pyx'])
                   ])