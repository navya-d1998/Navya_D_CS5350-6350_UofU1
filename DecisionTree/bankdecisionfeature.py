
import numpy as np
import pandas as pd

def get_sub_tree(train_data, label_name, label, feature, max_depth):
    attribute_tree_dict = {}
    value_count_feat = train_data[feature].value_counts(sort=False)

    for value, count in value_count_feat.items():
        sub_feature_data = train_data[train_data[feature] == value]
        max_type_count = 0
        max_type = None
        is_data_pure = False

        # Find the majority class for the current feature value
        for lbl in label:
            type_count = sub_feature_data[sub_feature_data[label_name] == lbl].shape[0]
            if type_count > max_type_count:
                max_type_count = type_count
                max_type = lbl
            if type_count == count:
                is_data_pure = True
                attribute_tree_dict[value] = lbl
                train_data = train_data[train_data[feature] != value]

        if not is_data_pure:
            attribute_tree_dict[value] = "multi" if max_depth > 1 else max_type

    return train_data, attribute_tree_dict

def get_tree(train_data, label_name, label, pre_feature, node, max_depth, method):
    if train_data.shape[0] > 0 and max_depth > 0:
        best_feature = get_best_feature(train_data, label_name, label, method)
        train_data, tree_dict = get_sub_tree(train_data, label_name, label, best_feature, max_depth)

        # Initialize next_node properly
        if pre_feature is None:
            node[best_feature] = tree_dict
            next_node = node[best_feature]
        else:
            # Ensure that node[pre_feature] is a dictionary
            if pre_feature not in node or not isinstance(node[pre_feature], dict):
                node[pre_feature] = {}
            node[pre_feature][best_feature] = tree_dict
            next_node = node[pre_feature][best_feature]

        # Traverse down the tree if necessary
        for value, tree_type in next_node.items():
            if tree_type == "multi":
                sub_feature_data = train_data[train_data[best_feature] == value]
                get_tree(sub_feature_data, label_name, label, value, next_node, max_depth - 1, method)

def get_id3_tree(train_data, label_name, label, max_depth, method):
    tree_dict = {}
    get_tree(train_data, label_name, label, None, tree_dict, max_depth, method)
    return tree_dict

def calculate_impurity(train_data, label_name, label, method):
    if method == "entropy":
        return get_entropy(train_data, label_name, label)
    elif method == "gini":
        return get_GI(train_data, label_name, label)
    elif method == "me":
        return get_ME(train_data, label_name, label)

def get_GI(train_data, label_name, label):
    total_size = train_data.shape[0]
    gini_index = 1 - sum((train_data[train_data[label_name] == lbl].shape[0] / total_size) ** 2 for lbl in label)
    return gini_index

def get_ME(train_data, label_name, label):
    total_size = train_data.shape[0]
    max_label_proportion = max(train_data[train_data[label_name] == lbl].shape[0] / total_size for lbl in label)
    return 1 - max_label_proportion

def get_entropy(train_data, label_name, label):
    total_size = train_data.shape[0]
    return -sum(
        (count / total_size) * np.log2(count / total_size)
        for count in (train_data[train_data[label_name] == lbl].shape[0] for lbl in label) if count > 0
    )

def get_info_gain(train_data, label_name, label, feature, method):
    total_size = train_data.shape[0]
    total_impurity = calculate_impurity(train_data, label_name, label, method)
    feature_info_gain = total_impurity - sum(
        (train_data[train_data[feature] == value].shape[0] / total_size) *
        calculate_impurity(train_data[train_data[feature] == value], label_name, label, method)
        for value in train_data[feature].unique()
    )
    return feature_info_gain

def get_best_feature(train_data, label_name, label, method):
    features = train_data.columns.drop(label_name)
    return max(features, key=lambda feature: get_info_gain(train_data, label_name, label, feature, method))

def predict(id3_tree, instance):
    if not isinstance(id3_tree, dict):
        return id3_tree  # If it's a leaf node, return the label
    else:
        node = next(iter(id3_tree))  # Get the feature to split on
        feature_value = instance.get(node, None)  # Safely get the feature value from the instance
        
        if feature_value is None:
            return None  # Handle the case where the feature is missing
        
        subtree = id3_tree[node].get(feature_value, None)  # Get the corresponding subtree
        
        if subtree is None:
            return None  # If there's no subtree for the feature value, return None
        
        return predict(subtree, instance)  # Recurse with the subtree

def fetch_error(tree, test_data, label_name):
    predictions = test_data.apply(lambda row: predict(tree, row), axis=1)
    error_rate = (predictions != test_data[label_name]).mean()
    return error_rate

def convert_feature(data_set, num_feature_list):
    for feature in num_feature_list:
        median_value = data_set[feature].median()
        data_set[feature] = np.where(data_set[feature] <= median_value, "neg", "pos")
    return data_set


if __name__ == '__main__':
    actions = ["entropy", "gini", "me"]
    columns_names = ["age", "job", "marital", "education", "default", "balance", "housing",
                    "loan", "contact", "day", "month", "duration", "campaign", "pdays",
                    "previous", "poutcome", "class"]

    bank_numerous = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    labels = ["yes", "no"]
    banktraindata = pd.read_csv("./bank/train.csv", names=columns_names, header=None)
    bank_test = pd.read_csv("./bank/test.csv", names=columns_names, header=None)

    banktraindata = convert_feature(banktraindata, bank_numerous)
    bank_test = convert_feature(bank_test, bank_numerous)

    for method in actions:
        for depth in range(1, 17):
            decision_tree = get_id3_tree(banktraindata, "class", labels , depth, method)
            test_error = fetch_error(decision_tree, bank_test, 'class')
            train_error = fetch_error(decision_tree, banktraindata, 'class')

            print(f"Test Error:\tMethod: {method}\tDepth: {depth}\tError: {test_error}")
            print(f"Train Error:\tMethod: {method}\tDepth: {depth}\tError: {train_error}")
