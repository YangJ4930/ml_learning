from sklearn.datasets import load_iris
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from math import log
from graphviz import Digraph
import matplotlib.pyplot as plt

dot = Digraph(comment='My Tree')

class TreeNode:
    def __init__(self):
        self.is_end = False
        self.label_name = -1
        self.label_value = -1
        self.left_node = None
        self.right_node = None
        self.result = -1


def process():
    iris_dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size=0.30,random_state=0)
    train_set = list()
    for i in range(y_train.shape[0]):
        curr = list()
        for j in range(4):
            curr.append(X_train[i][j])
        curr.append(y_train[i])
        train_set.append(curr)
    test_set = list()
    for i in range(y_test.shape[0]):
        curr = list()
        for j in range(4):
            curr.append(X_test[i][j])
        curr.append(y_test[i])
        test_set.append(curr)
    return train_set, test_set


# 计算信息熵
def calculate_entropy(dataset):
    entropy = 0.0
    dataset_length = len(dataset)
    p = [0, 0, 0]

    for i in range(dataset_length):
        p[dataset[i][4]] += 1
    p_sum = p[0] + p[1] + p[2]
    for i in range(3):
        if p[i] / p_sum == 0.0:
            entropy += 0
        else:
            entropy += -1 * (p[i] / p_sum) * log(p[i] / p_sum, 2)
    return entropy


# 计算信息增益率
# 返回了index与特征
def calculate_gain_ratio(dataset):
    ratio = -999
    res_index = -1
    res_class = -1
    sum_en = calculate_entropy(dataset)
    for i in range(4):
        dataset.sort(key=lambda e: e[i])
        dataset_size = len(dataset)

        for j in range(1, dataset_size):
            dataset_left = dataset[0:j]
            dataset_right = dataset[j:]
            left_en = calculate_entropy(dataset_left)
            right_en = calculate_entropy(dataset_right)
            # 条件信息熵怎么计算在这里
            condition_en = (j / dataset_size) * left_en + (dataset_size - j) / dataset_size * right_en
            gain = sum_en - condition_en
            # 交叉熵
            cross_en = -1 * (j / dataset_size) * log((j / dataset_size), 2) + -1 * (
                    dataset_size - j) / dataset_size * log(
                (dataset_size - j) / dataset_size, 2)
            if ratio < gain / cross_en:
                res_class = i
                res_index = j
                ratio = gain / cross_en
    dataset.sort(key=lambda e: e[res_class])
    return res_index, res_class


def check_is_end(data_set):
    data_class = data_set[0][4]
    data_len = len(data_set)
    for i in range(1, data_len):
        if data_class != data_set[i][4]:
            return False
    return True



def create_tree(data_set):
    # 选择最好的标签，构建一个节点
    node = TreeNode()
    node.is_end = check_is_end(data_set)
    if node.is_end:
        node.result = data_set[0][4]
        return node
    if not node.is_end:
        tem_index, tem_class = calculate_gain_ratio(data_set)
        data_set.sort(key=lambda e: e[tem_class])
        data_set_left = data_set[0: tem_index]
        data_set_right = data_set[tem_index:]

        node.left_node = create_tree(data_set_left)

        node.right_node = create_tree(data_set_right)
        node.label_name = tem_class
        node.label_value = (data_set[tem_index - 1][tem_class] + data_set[tem_index][tem_class]) / 2
    return node

def dfs(node, data):
    if node.is_end:
        return node.result
    if node.label_value >= data[node.label_name]:
        return dfs(node.left_node, data)
    else:
        return dfs(node.right_node, data)
    return -1

def draw_graph(node, label_str):
    str_name = ""
    if node.is_end:
        str_name = "res " + str(node.result)
    else:
        str_name = "label " + str(node.label_name) + " is < " + str(node.label_value)
    dot.node(label_str, str_name)
    if node.left_node is not None:
        draw_graph(node.left_node, label_str + "l")
        dot.edge(label_str, label_str+"l", "yes")
    if node.right_node is not None:
        draw_graph(node.right_node, label_str + "r")
        dot.edge(label_str, label_str+"r", "no")
    return

if __name__ == '__main__':
    train_set, test_set = process()
    node = create_tree(train_set)
    test_set_len = len(test_set)
    true_list = list()
    predict_list = list()
    for i in range(test_set_len):
        true_list.append(test_set[i][4])
        predict_list.append(dfs(node, test_set[i]))

    micro_f1=f1_score(y_true=true_list,y_pred=predict_list,average="micro")
    macro_f1=f1_score(y_true=true_list,y_pred=predict_list,average="macro")

    print('Micro F1: {}'.format(micro_f1))
    print('Macro F1: {}'.format(macro_f1))
    draw_graph(node, "t")
    dot.view()
    dot.render('test-output/my_tree.gv', view=True)
