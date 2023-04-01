import numpy as np
import pandas as pd


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
        self.feature_space[feature] = value

    # 由属性feature划分样本集
    def groupBy(self, feature, feature_space=None):
        # 在生成决策树GenTree中，子节点在划分子节点可能遇到划分属性feature在被划分子节点上取值不全，此时需要指定属性空间（即包含所有属性值的属性空间）。默认不指定，方便后面计算子节点的信息熵、基尼指数、IV等。
        result = []
        if feature_space:
            for aStar in feature_space[feature]:
                bolen = self.samples[feature] == aStar
                # if not any(bolen):  # 若全为false，即在在属性没有这个属性值时
                #     samples_by_aStart = None
                samples_by_aStart = Samples(self.samples.loc[bolen])
                result.append((aStar, samples_by_aStart))  # feature值不全时，返回值与划分集合的元组
        else:
            for aStar in self.feature_space[feature]:
                samples_by_aStart = Samples(self.samples.loc[self.samples[feature] == aStar])
                result.append(samples_by_aStart)
        return result

    # 样本中属性或标记 为 属性值aSatr或标记值label占比
    def _prob(self, aStar_or_labelStar, features_or_labels="labels"):
        if features_or_labels == "labels" or features_or_labels == self.labels:
            labelStar_num = np.sum(self.samples_labels == aStar_or_labelStar)  # 样本集中标记为label的样本数
            return labelStar_num / self.size
        else:
            aStar_num = np.sum(self.samples_features[features_or_labels] == aStar_or_labelStar)  # 样本集中给定属性中某属性值的样本数
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
        if not feature_list:
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
    def gini_index(self, feature):
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
                temp.append(node.gini_index(fea))
            elif method == "Gain":
                temp.append(node.gain(fea))
            elif method == "Gain_ratio":
                temp.append(node.gain_ratio(fea))
            else:
                print("候选的选取划分属性的方法为Gini_index、Gain、Gain_ratio")
        result = dict(zip(features, temp))
        return max(result, key=result.get)

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
            for cond_samp in node.groupBy(feature, root_feature_space):
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
    while not root.leaf:
        feature = root.divFeat
        value = valdata[feature][0]
        for chilNode in root.child:
            if chilNode.divCon == value:
                root = chilNode
    pred_label = root.leaf
    return pred_label


if __name__ == '__main__':
    dataset = pd.read_csv("./test_data.csv").iloc[:, 1:]
    valdata = pd.read_csv("./val.csv").iloc[:, 1:]
    tree = GenerateDecisionTree(dataset)
    pred = prediction(valdata, tree)
    print(pred)

    import pandas as pd
    import numpy as np
