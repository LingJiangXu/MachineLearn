import pickle
import numpy as np
import pandas as pd
from Samples_Node import Samples, Node

# 属性值判断
def compare_value(value1, value2):
    # 暂不支持ndarray对象，因为ndarray里整数元素类型不为int，而是nd.int32等类型(solved),pd.series不得不解决掉
    # 穷尽分支判断，还没找到更巧妙的方法
    iter_list = [list, tuple, set, dict]
    ty_1, ty_2 = type(value1), type(value2)
    int_ls = [int, np.int32, np.int64]
    if ty_1 in int_ls and ty_2 in int_ls:
        return value1 == value2
    elif ty_1 == str and ty_2 == str:
        return value1 == value2
    elif ty_1 in int_ls and ty_2 == str:
        return eval(str(value1) + value2)
    elif ty_2 in int_ls and ty_1 == str:
        return eval(str(value2) + value1)
    elif ty_1 in iter_list or ty_2 in iter_list:
        if ty_1 in iter_list and ty_2 in iter_list:
            pass
        elif ty_1 in iter_list:
            value2 = [value2 for i in range(len(value1))]
        elif ty_2 in iter_list:
            value1 = [value1 for i in range(len(value2))]
        return tuple(map( lambda x, y: compare_value(x,y), value1, value2))
    elif ty_1 == pd.Series or ty_2 == pd.Series:
        if ty_1 in int_ls or ty_2 in int_ls:
            return value1 == value2
        elif ty_1 == str:
            return value2.map( lambda x: eval(str(x) + value1) )
        elif ty_2 == str:
            return value1.map( lambda x: eval(str(x) + value2) )
        elif ty_1 == pd.Series and ty_2 == pd.Series:
            return value1 == value2
    else:
        print("WARNING IN COMPARE_VALUE! ", ty_1, ty_2)

# 根据sample_set生成决策树
def GenerateDecisionTree(sample_set):
    features_exit = list(Samples(sample_set).features)  # 未被作为划分依据的属性列表
    root_feature_space = Samples(sample_set).feature_space  # 属性的所有取值组成的空间
    Tree = []  # 决策树，将节点和叶子存放于此

    # 寻找最优属性划分的属性
    def find_divide_feature(node, features, method="Gini_index"):
        temp = []
        for fea in features:
            if method == "Gini_index":
                temp.append(node.gini_index(fea, continue_pro=True))
            elif method == "Gain":
                temp.append(node.gain(fea))
            elif method == "Gain_ratio":
                temp.append(node.gain_ratio(fea))
            else:
                print("候选的选取划分属性的方法为Gini_index、Gain、Gain_ratio")
        result = dict(zip(features, temp))
        return max(result, key=result.get) if method in ["Gian", "Gain_ratio"] else min(result, key=result.get)

    # 决策树生成(嵌套在main函数中，一方面每次迭代更改feature_exit，另一方面保持root_feature_space不变)
    node = Node(sample_set)
    Tree.append(node)  # 根节点就放入节点列表，以供返回

    def GenTree(node, features_exit, root_feature_space):
        # 第一种情况
        if node.is_same_label():
            label = node.most_label()
            node.node2leaf(label)
        # 第二种情况
        elif node.is_same_feature() or node.is_empty_feature(features_exit):
            label = node.most_label()
            node.node2leaf(label)
        else:
            feature = find_divide_feature(node, features_exit)
            node.set_divide_feature(feature)  # 给被划分的节点标记划分依据
            features_exit.remove(feature)  # 从为作为划分依据的属性列表中去除feature
            for cond_samp in node.groupBy(feature, root_feature_space, continue_pro=True):
                condition, samp = cond_samp
                child_node = Node(samp.samples)
                node.add_child(child_node)  # 将子节点添加到父节点的“子节点列表”
                Tree.append(child_node)  # 每生成一个字节点就放入节点列表，以供返回
                child_node.set_divide_condition(condition)  # 给被划分的节点标记划分条件
                # 第三种情况
                if child_node.is_empty():
                    label = node.most_label()
                    child_node.node2leaf(label)
                else:
                    GenTree(child_node, features_exit, root_feature_space)

    GenTree(node, features_exit, root_feature_space)

    return Tree


# 决策树可视化函数


# 预测函数
def prediction(valdata, tre):
    root = tre[0]
    while root.leaf is None:
        # "while not root.leaf"会在root.leaf 为0时陷入死循环
        feature = root.divFeat
        value = valdata[feature].iloc[0]
        origin_address = id(root)
        for chilNode in root.child:
            # !!如果value值没有对应的子节点属性值相等，会进入死循环。
            # 例如，训练集中feature上取值为1，2，3，而预测样本在该feature上取值为4，则会进入遍历死循环
            # 解决方案1：在生成决策树时放弃取值不满的属性
            # 方案2：在此处遇到此类情况，报错
            # 方案3：属性值不离散，转化为连续值处理
            if compare_value(chilNode.divCon, value):
                root = chilNode
        if id(root) == origin_address:
            # 如果进行到这，root未被子节点赋值
            print(f'预测样本feature {feature} 的value值 {value} 没有对应的子节点属性值相等')
            for chilNode in root.child:
                print(chilNode.divCon)
            break
    pred_label = root.leaf
    return pred_label

if __name__ == '__main__':
    # dataset = pd.read_csv("./test_data.csv").iloc[:, 1:]
    # valdata = pd.read_csv("./val.csv").iloc[:, 1:]

    m = eval(input("请输入训练得样本量m: "))
    n = eval(input("请输入训练和预测得样本属性量n: "))
    # 读取数据并整理
    train_x = pd.read_csv("./data/fashion-mnist_train.csv").iloc[:m, 1:n]
    train_y = pd.read_csv("./data/fashion-mnist_train.csv").iloc[:m, [0]]
    dataset = pd.concat([train_x, train_y], axis=1)
    # 训练决策树并写入文件
    tree = GenerateDecisionTree(dataset)
    with open('./data.pickle', 'wb') as f:
        pickle.dump(tree, f)
    # # 读取决策树数据
    # with open("./data.pickle",'rb') as f:
    #     tree = pickle.load(f)
    # 预测
    valdata = pd.read_csv("./data/fashion-mnist_test.csv")
    outcome = {"prediction_label": [],"truth_label": []}
    test_index = []
    for i in range(len(valdata)):
        test_x = valdata.iloc[[i], 1:n]
        test_y = valdata.iloc[i, 0]
        pred = prediction(test_x, tree)
        outcome["prediction_label"].append(pred)
        outcome["truth_label"].append(test_y)
        test_index.append(i)
    with open('./outcomes.csv',"w", encoding="utf-8") as f:
        outcome = pd.DataFrame(outcome, index=test_index)
        outcome.to_csv(f,index=True)
    print(np.sum(outcome["prediction_label"] == outcome["truth_label"])/len(outcome))
    # print("pre:", pred)
    # print("tru:", test_y.iloc[0, 0])
