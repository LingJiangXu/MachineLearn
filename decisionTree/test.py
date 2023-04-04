import numpy as np
import pandas as pd


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



# 样本集类
class Samples:
    def __init__(self, dataframe):
        self.samples = dataframe  # 样本集（包括属性空间和标记空间）
        self.samples_features = dataframe.iloc[:, :-1]  # 样本集的属性空间
        self.samples_labels = dataframe.iloc[:, [-1]]  # 样本集的标记空间
        self.features = dataframe.columns[:-1]  # 样本集的属性
        self.labels = dataframe.columns[-1]  # 样本集的标签名
        self.label_space = self.samples_labels.drop_duplicates()  # 样本集的标签类型
        self.feature_space = dict(zip(self.features, [self.samples[a].drop_duplicates() for a in self.features]))
        self.size = len(self.samples)  # 样本量

    # 更改属性的属性值
    def change_featureValue(self, feature, value):
        # 原本计划是更改某一属性的取值空间，并在样本数据中对应更改，例如 色泽=[白、黑、绿]改成[红、蓝、紫]
        # 样本上在色泽的取值也对应发生变化
        # 但考虑到处理连续属性值时，每次划分属性并非固有属性。见西瓜书P85
        # 因此此函数采用第一种思路，即用于属性空间“彻底”改变
        self.feature_space[feature] = value

    def __continue_featureValue_divie(self, feature, t):
        # 此函数采用第二种思路，即用于连续值处理，属性空间被划分，但原本属性值不变
        self.feature_space[feature] = ["<"+str(t), ">="+str(t)]
        # 根据分割值t将连续数据划分成二值属性

    # 由属性feature划分样本集
    def groupBy(self, feature, feature_space=None, continue_pro=False):
        # 在生成决策树GenTree中，子节点在划分子节点可能遇到划分属性feature在被划分子节点上取值不全，此时需要指定属性空间（即包含所有属性值的属性空间）。默认不指定，方便后面计算子节点的信息熵、基尼指数、IV等。
        result = []
        if feature_space and not continue_pro:
            # 仅当传入feature_space且不指定属性值为连续
            for aStar in feature_space[feature]:
                bolen = compare_value(self.samples[feature], aStar)   #1、传入compare得时list 2、bool列表要转为pd.series（改进compare函数，否者下面bool切片会出现错误，因为没有保存index信息，忽略此条）
                # if not any(bolen):  # 若全为false，即在在属性没有这个属性值时
                #     samples_by_aStart = None
                samples_by_aStart = Samples(self.samples.loc[bolen])
                result.append((aStar, samples_by_aStart))  # feature值不全时，返回值与划分集合的元组
        else:
            for aStar in self.feature_space[feature]:
                bolen = compare_value(self.samples[feature], aStar)
                samples_by_aStart = Samples(self.samples.loc[bolen])
                if feature_space and continue_pro:
                    result.append((aStar, samples_by_aStart))
                else:
                    result.append(samples_by_aStart)
        return result

    # 样本中属性或标记 为 属性值aSatr或标记值label占比
    def _prob(self, aStar_or_labelStar, features_or_labels="labels"):
        if features_or_labels == "labels" or features_or_labels == self.labels:
            # 默认为对标签
            bolen = compare_value(self.samples_labels.iloc[:,0], aStar_or_labelStar)
            labelStar_num = np.sum(bolen)       # 样本集中标记为label的样本数
            return labelStar_num / self.size
        else:
            # 指定对某一属性
            bolen = compare_value(self.samples_features[features_or_labels], aStar_or_labelStar)
            aStar_num = np.sum(bolen)  # 样本集中给定属性中某属性值的样本数
            return aStar_num / self.size

    # 判断样本集是否为空
    def is_empty(self):
        return len(self.samples_features) == 0

    # 判断样本集是否类别相同
    def is_same_label(self):
        return len(self.label_space) == 1

    # 判断样本集(未被划分的)属性是否为空
    def is_empty_feature(self, feature_list=None):
        # 与groupBy类似，在生成决策树GenTree中，feature属性组成的列表在变化，随每次划分而减少一个
        if feature_list is None:
            # attention！ "if not feature_list:"，当传入参数为“{}”时，任然会视为没有传参的情况，
            # 即把self.features作为fearture_list，从而产生错误
            feature_list = self.features
        return len(feature_list) == 0

    # 判断样本集是否属性取值相同
    def is_same_feature(self):
        return len(self.samples_features.drop_duplicates()) == 1

    # 返回标签类别最多的类别
    def most_label(self):
        if self.is_same_label():
            return self.label_space.iloc[0, 0]
        else:
            labels = np.array(self.samples_labels.stack())  # 将标签列（dataframe）铺平转化为ndarray
            label_list = list(labels)  # 将ndarray转化为list
            return max(label_list, key=lambda x: label_list.count(x))  # 返回list中出现次数最多的元素

    # 样本集的基尼指数
    def gini(self):
        temp_result = []
        for label in self.label_space.iloc[:, -1]:
            temp_result.append(self._prob(label))
        result = 1 - np.sum(np.power(temp_result, 2))
        return np.round(result, 3)

    # 样本集中属性a的基尼指数
    def gini_index(self, feature, continue_pro=False):
        if continue_pro:
            temp = []
            result = -1
            result_t = -1
            iters = self.samples[feature].drop_duplicates()[1:] \
                if len(self.samples[feature].drop_duplicates()) > 1 \
                else [0]
            for t in  iters:
                self.__continue_featureValue_divie(feature, t)
                for sample in self.groupBy(feature):
                    temp.append(sample.size / self.size * sample.gini())
                temp_result = np.round(np.sum(temp), 3)
                if temp_result > result:
                    result = temp_result
                    result_t = t
            self.__continue_featureValue_divie(feature,result_t)
            return result
        else:
            temp_result = []
            for sample in self.groupBy(feature):
                temp_result.append(sample.size / self.size * sample.gini())
            return np.round(np.sum(temp_result), 3)

    # 样本集的信息熵
    def ent(self):
        temp_result = []
        for label in self.label_space.iloc[:, -1]:
            temp_result.append(self._prob(label))
        result = 0 - np.sum(list(map(lambda p: p * np.log2(p), temp_result)))
        return np.round(result, 3)

    # 样本集中属性a的信息增益
    def gain(self, feature):
        ent = self.ent()
        temp_result = []
        for samp in self.groupBy(feature):
            temp_result.append(samp.ent() * samp.size / self.size)
        result = ent - np.sum(temp_result)
        return np.round(result, 3)

    # 样本集中属性a的信息增益率
    def gain_ratio(self, feature):
        gain = self.gain(feature)
        temp_result = []
        for aStar in self.feature_space[feature]:
            temp_result.append(self._prob(aStar, feature))
        IV = 0 - np.sum(list(map(lambda p: p * np.log2(p), temp_result)))
        result = gain / IV
        return np.round(result, 3)


# 节点类
class Node(Samples):
    def __init__(self, dataframe):
        super().__init__(dataframe)
        self.child = []  # 当前节点的子节点列表
        self.divFeat = None
        self.divCon = None
        self.leaf = None

    # 将child_node添加为子节点列表
    def add_child(self, child_node):
        if self.leaf:
            print("当前节点为叶节点，无法添加子节点")
        else:
            self.child.append(child_node)

    # 节点作为父节点，指定划分为子节点的依据（即按什么属性划分）
    def set_divide_feature(self, feature):
        self.divFeat = feature

    # 节点作为子节点，指定被划分的条件（即满足划分属性的什么条件）
    def set_divide_condition(self, condition):
        self.divCon = condition

    # 将节点设置为叶节点
    def node2leaf(self, label):
        if self.child:
            print("当前节点已有子节点，无法成为叶节点")
        else:
            self.leaf = label


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
        value = valdata[feature][0]
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
    train_x = pd.read_csv("./data/fashion-mnist_train.csv").iloc[:100, 1:30]
    train_y = pd.read_csv("./data/fashion-mnist_train.csv").iloc[:100, [0]]
    dataset = pd.concat([train_x, train_y], axis=1)
    test_x = pd.read_csv("./data/fashion-mnist_test.csv").iloc[[0], 1:30]
    test_y = pd.read_csv("./data/fashion-mnist_test.csv").iloc[[0], [0]]
    valdata = test_x
    tree = GenerateDecisionTree(dataset)
    pred = prediction(valdata, tree)
    print("pre:", pred)
    print("tru:", test_y.iloc[0, 0])
