'''
#file:graph.pyx类graph的实现文件
#可以自动导入相同路径下相同名称的.pxd的文件
#可以省略cimport graph命令
#需要重新设计python调用的接口，此文件            pyx是将 封装c++文件的pxd文件 设计给python使用
'''
from cython.operator cimport dereference as deref
cimport cpython.ref as cpy_ref  #cimport 导入pxd文件     ref指的是什么 ?引用？
from libcpp.memory cimport shared_ptr  #是否和pxd中重复了？？？
from libc.stdlib cimport malloc
from libc.stdlib cimport free
import numpy as np  #矩阵处理
# import gc

cdef class py_Graph: #外部是供给python使用的接口
    cdef shared_ptr[Graph] inner_graph #使用unique_ptr优于shared_ptr    cdef指针 可以使用Graph的非默认构造函数创建Graph实例
    #__cinit__会在__init__之前被调用
    def __cinit__(self,*arg):  #与c相关的初始化，比如new   self相当于this
        '''doing something before python calls the __init__.
        cdef 的C/C++对象必须在__cinit__里面完成初始化，否则没有为之分配内存
        可以接收参数，使用python的变参数模型实现类似函数重载的功能。 ？？？？'''
        #print("doing something before python calls the __init__")
        # if len(arg)==0:
        #     print("num of parameter is 0")
        self.inner_graph = shared_ptr[Graph](new Graph())   #内部调用的c++封装后的构造函数 Graph() 默认使用无参数构造，如果arg数目是4，则使用cdef的reshape_graph函数
        cdef int _num_nodes  #cinit内部可以使用cdef 来处理类中cdef的初始化，这里是inner_graph
        cdef int _num_edges
        cdef int[:] edges_from
        cdef int[:] edges_to
        cdef double[:] edge_value_copy
        if len(arg)==0:
            #这两行代码为了防止内存没有初始化，没有实际意义
            deref(self.inner_graph).num_edges=0  #deref 就等于 取指针指向的内容（值）
            deref(self.inner_graph).num_nodes=0
        elif len(arg)==4:   #cython的语法是if -- elif ？？
            _num_nodes=arg[0]
            _num_edges=arg[1]
            edges_from = np.array([int(x) for x in arg[2]], dtype=np.int32)
            edges_to = np.array([int(x) for x in arg[3]], dtype=np.int32)
            self.reshape_Graph(_num_nodes,  _num_edges,  edges_from,  edges_to)
        elif len(arg)==5:
            _num_nodes=arg[0]
            _num_edges=arg[1]
            edges_from = np.array([int(x) for x in arg[2]], dtype=np.int32)
            edges_to = np.array([int(x) for x in arg[3]], dtype=np.int32)
            edge_value_copy = np.array([float(x) for x in arg[4]], dtype=np.double)
            self.reshape_Graph_prob(_num_nodes,  _num_edges,  edges_from,  edges_to, edge_value_copy)
        # elif len(arg)==1:
            # self.inner_graph=arg[0].inner_graph

            # _num_nodes=arg[0].num_nodes
            # _num_edges=arg[0].num_edges
            # edges_from = np.array([int(x) for x in arg[0].], dtype=np.int32)
            # edges_to = np.array([int(x) for x in arg[3]], dtype=np.int32)
            # edge_value_copy = np.array([float(x) for x in arg[4]], dtype=np.double)
            # self.reshape_Graph_prob(_num_nodes,  _num_edges,  edges_from,  edges_to, edge_value_copy)

        else:
            print('Error：py_Graph类未被成功初始化，因为提供参数数目不匹配，参数个数应为0或4或5。')
    # def __dealloc__(self):
    #     if self.inner_graph != NULL:
    #         self.inner_graph.reset()
    #         gc.collect()  #清理内存

    @property
    def num_nodes(self):
        return deref(self.inner_graph).num_nodes

    # @num_nodes.setter  #设置变量的值
    # def num_nodes(self):
    #     def __set__(self,num_nodes):
    #         self.setadj(adj_list)   #这里写的有问题

    @property
    def num_edges(self):
        return deref(self.inner_graph).num_edges

    @property
    def adj_list(self):
        return deref(self.inner_graph).adj_list

    @property
    def edge_list(self):
        return deref(self.inner_graph).edge_list

    @property
    def edge_value(self):
        return deref(self.inner_graph).edge_value

    cdef reshape_Graph(self, int _num_nodes, int _num_edges, int[:] edges_from, int[:] edges_to): #为了处理cinit中的情况所需构造的函数 内部使用
        cdef int *cint_edges_from = <int*>malloc(_num_edges*sizeof(int))
        cdef int *cint_edges_to = <int*>malloc(_num_edges*sizeof(int))
        cdef int i
        for i in range(_num_edges):
            cint_edges_from[i] = edges_from[i]
        for i in range(_num_edges):
            cint_edges_to[i] = edges_to[i]
        self.inner_graph = shared_ptr[Graph](new Graph(_num_nodes,_num_edges,&cint_edges_from[0],&cint_edges_to[0]))  #使用有参数的构造函数
        free(cint_edges_from)
        free(cint_edges_to)



    cdef reshape_Graph_prob(self, int _num_nodes, int _num_edges, int[:] edges_from, int[:] edges_to, double[:] edge_value_copy): #为了处理cinit中的情况所需构造的函数 内部使用
        cdef int *cint_edges_from = <int*>malloc(_num_edges*sizeof(int))
        cdef int *cint_edges_to = <int*>malloc(_num_edges*sizeof(int))
        cdef double *cdouble_edge_value_copy = <double*>malloc(_num_edges*sizeof(double))
        cdef int i
        for i in range(_num_edges):
            cint_edges_from[i] = edges_from[i]
        for i in range(_num_edges):
            cint_edges_to[i] = edges_to[i]
        for i in range(_num_edges):
            cdouble_edge_value_copy[i] = edge_value_copy[i]
        self.inner_graph = shared_ptr[Graph](new Graph(_num_nodes,_num_edges,&cint_edges_from[0],&cint_edges_to[0],&cdouble_edge_value_copy[0]))  #使用有参数的构造函数
        free(cint_edges_from)
        free(cint_edges_to)
        free(cdouble_edge_value_copy)


    def reshape(self,int _num_nodes, int _num_edges, int[:] edges_from, int[:] edges_to):  #留给python的相应功能接口（设置的功能），对应上面cdef的函数
        self.reshape_Graph(_num_nodes, _num_edges, edges_from, edges_to)

    def reshape_prob(self,int _num_nodes, int _num_edges, int[:] edges_from, int[:] edges_to, double[:] edge_value_copy):
        self.reshape_Graph_prob(_num_nodes, _num_edges, edges_from, edges_to, edge_value_copy)


cdef class py_GSet:
    cdef shared_ptr[GSet] inner_gset
    def __cinit__(self):
        self.inner_gset = shared_ptr[GSet](new GSet()) #默认构造
    # def __dealloc__(self):
    #     if self.inner_gset != NULL:
    #         self.inner_gset.reset()
    #         gc.collect()
    def InsertGraph(self,int gid,py_Graph graph):
        deref(self.inner_gset).InsertGraph(gid,graph.inner_graph) #使用封装的方法
        #self.InsertGraph(gid,graph.inner_graph)

        # deref(self.inner_gset).InsertGraph(gid,graph.inner_graph)
         #self.Inner_InsertGraph(gid,graph.inner_graph)

    def Sample(self): #返回一个Py_Graph类
        temp_innerGraph=deref(deref(self.inner_gset).Sample())   #得到了Graph 对象
        return self.G2P(temp_innerGraph)  #cdef函数G2P

    def Get(self,int gid):
        temp_innerGraph=deref(deref(self.inner_gset).Get(gid))   #得到了Graph 对象
        return self.G2P(temp_innerGraph)

    def Get_value(self,int gid):
        return deref(self.inner_gset).Get_value(gid)

    def Clear(self):
        deref(self.inner_gset).Clear()

    cdef G2P(self,Graph graph):  #Graph是封装的Graph 也就是pxd文件中的   返回py_Graph类（内含Graph指针）
        num_nodes = graph.num_nodes     #得到Graph对象的节点个数 不用int定义吗？
        num_edges = graph.num_edges    #得到Graph对象的连边个数
        edge_list = graph.edge_list
        edge_value = graph.edge_value
        cint_edges_from = np.zeros([num_edges],dtype=np.int)
        cint_edges_to = np.zeros([num_edges],dtype=np.int)
        cdouble_edge_value_copy = np.zeros([num_edges],dtype=np.double)
        for i in range(num_edges):
            cint_edges_from[i]=edge_list[i].first
            cint_edges_to[i] =edge_list[i].second
            cdouble_edge_value_copy[i] = edge_value[i]
        return py_Graph(num_nodes,num_edges,cint_edges_from,cint_edges_to,cdouble_edge_value_copy)


