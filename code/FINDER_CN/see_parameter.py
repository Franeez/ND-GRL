import os
from tensorflow.python import pywrap_tensorflow

# checkpoint_path = os.path.join("<你的模型的目录>", "./model.ckpt-11000")
# # Read data from checkpoint file
# reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
# # Print tensor name and values
# for key in var_to_shape_map:
# 	print("tensor_name: ", key)
# 	print(reader.get_tensor(key))


def main():
    # model_file_ckpt = 'nrange_30_50_iter_300.ckpt'
    # GetSolution(0.01, model_file_ckpt)
    # GetSolution_contrast(0.01, model_file_ckpt) #相同的图结构 相同的激活概率 不同的激活判断 多次试验进行对比
    # GetSolution_contrast_hda(0.01, model_file_ckpt)
    # EvaluateSolution(0.01, model_file_ckpt, 0)
    checkpoint_path = os.path.join("D:\Beihang_graduate_stage\git_warehouse\FINDER\code\FINDER_CN\models\Model_barabasi_albert", "./nrange_30_50_iter_0.ckpt")
    # Read data from checkpoint file
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    # Print tensor name and values
    for key in var_to_shape_map:
        print("tensor_name: ", key)
        print(reader.get_tensor(key))
        print("\n**************************\n")

    print("\n")
    print("===================================================================")
    print("\n")

    checkpoint_path2 = os.path.join("D:\Beihang_graduate_stage\git_warehouse\FINDER\code\FINDER_CN\models\Model_barabasi_albert", "./nrange_30_50_iter_600.ckpt")
    # Read data from checkpoint file
    reader2 = pywrap_tensorflow.NewCheckpointReader(checkpoint_path2)
    var_to_shape_map2 = reader2.get_variable_to_shape_map()
    # Print tensor name and values
    for key in var_to_shape_map2:
        print("tensor_name: ", key)
        print(reader.get_tensor(key))
        print("\n**************************\n")




if __name__=="__main__":
    main()