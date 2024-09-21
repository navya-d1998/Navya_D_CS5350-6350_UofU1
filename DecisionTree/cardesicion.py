
import numpy as np
import pandas as pd

def build_sub_tree(dataset, target_col, target_labels, feature, max_depth):
    feature_tree = {}
    feature_values_count = dataset[feature].value_counts(sort=False)

    for value, count in feature_values_count.items():
        subset_data = dataset[dataset[feature] == value]
        highest_count = 0
        majority_label = None
        is_pure = False

        # Find the majority class for the current feature value
        for target in target_labels:
            label_count = subset_data[subset_data[target_col] == target].shape[0]
            if label_count > highest_count:
                highest_count = label_count
                majority_label = target
            if label_count == count:
                is_pure = True
                feature_tree[value] = target
                dataset = dataset[dataset[feature] != value]

        if not is_pure:
            feature_tree[value] = "multiple" if max_depth > 1 else majority_label

    return dataset, feature_tree

def build_decision_tree(dataset, target_col, target_labels, previous_feature, current_node, max_depth, impurity_method):
    if dataset.shape[0] > 0 and max_depth > 0:
        best_feature = find_best_feature(dataset, target_col, target_labels, impurity_method)
        dataset, subtree = build_sub_tree(dataset, target_col, target_labels, best_feature, max_depth)

        # Initialize next_node properly
        if previous_feature is None:
            current_node[best_feature] = subtree
            next_node = current_node[best_feature]
        else:
            # Ensure that current_node[previous_feature] is a dictionary
            if previous_feature not in current_node or not isinstance(current_node[previous_feature], dict):
                current_node[previous_feature] = {}
            current_node[previous_feature][best_feature] = subtree
            next_node = current_node[previous_feature][best_feature]

        # Traverse down the tree if necessary
        for value, subtree_type in next_node.items():
            if subtree_type == "multiple":
                subset_data = dataset[dataset[best_feature] == value]
                build_decision_tree(subset_data, target_col, target_labels, value, next_node, max_depth - 1, impurity_method)

def train_id3_tree(dataset, target_col, target_labels, max_depth, impurity_method):
    tree_structure = {}
    build_decision_tree(dataset, target_col, target_labels, None, tree_structure, max_depth, impurity_method)
    return tree_structure

def calculate_impurity(dataset, target_col, target_labels, impurity_method):
    if impurity_method == "entropy":
        return compute_entropy(dataset, target_col, target_labels)
    elif impurity_method == "gini":
        return compute_gini_index(dataset, target_col, target_labels)
    elif impurity_method == "me":
        return compute_majority_error(dataset, target_col, target_labels)

def compute_gini_index(dataset, target_col, target_labels):
    total_size = dataset.shape[0]
    gini_index = 1 - sum((dataset[dataset[target_col] == target].shape[0] / total_size) ** 2 for target in target_labels)
    return gini_index

def compute_majority_error(dataset, target_col, target_labels):
    total_size = dataset.shape[0]
    max_label_proportion = max(dataset[dataset[target_col] == target].shape[0] / total_size for target in target_labels)
    return 1 - max_label_proportion

def compute_entropy(dataset, target_col, target_labels):
    total_size = dataset.shape[0]
    return -sum(
        (count / total_size) * np.log2(count / total_size)
        for count in (dataset[dataset[target_col] == target].shape[0] for target in target_labels) if count > 0
    )

def compute_info_gain(dataset, target_col, target_labels, feature, impurity_method):
    total_size = dataset.shape[0]
    total_impurity = calculate_impurity(dataset, target_col, target_labels, impurity_method)
    info_gain = total_impurity - sum(
        (dataset[dataset[feature] == value].shape[0] / total_size) *
        calculate_impurity(dataset[dataset[feature] == value], target_col, target_labels, impurity_method)
        for value in dataset[feature].unique()
    )
    return info_gain

def find_best_feature(dataset, target_col, target_labels, impurity_method):
    features = dataset.columns.drop(target_col)
    return max(features, key=lambda feature: compute_info_gain(dataset, target_col, target_labels, feature, impurity_method))

def predict_class(decision_tree, instance):
    if not isinstance(decision_tree, dict):
        return decision_tree  # If it's a leaf node, return the label
    else:
        node = next(iter(decision_tree))  # Get the feature to split on
        feature_value = instance.get(node, None)  # Safely get the feature value from the instance
        
        if feature_value is None:
            return None  # Handle the case where the feature is missing
        
        subtree = decision_tree[node].get(feature_value, None)  # Get the corresponding subtree
        
        if subtree is None:
            return None  # If there's no subtree for the feature value, return None
        
        return predict_class(subtree, instance)  # Recurse with the subtree

def compute_error_rate(tree, test_data, target_col):
    predictions = test_data.apply(lambda row: predict_class(tree, row), axis=1)
    error_rate = (predictions != test_data[target_col]).mean()
    return error_rate

def transform_numeric_features(dataset, numeric_features):
    for feature in numeric_features:
        median_value = dataset[feature].median()
        dataset[feature] = np.where(dataset[feature] <= median_value, "low", "high")
    return dataset

def handle_unknown_values(dataset, feature_list):
    for feature in feature_list:
        if 'unknown' in dataset[feature].unique():
            most_common_value = dataset[feature].mode()[0]
            dataset[feature] = dataset[feature].replace('unknown', most_common_value)
    return dataset

if __name__ == '__main__':
    columns_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    car_train = pd.read_csv("./car/train.csv", names=columns_names, header=None)
    car_test = pd.read_csv("./car/test.csv", names=columns_names, header=None)
    target_labels = ["unacc", "acc", "good", "vgood"]
    impurity_methods = ["entropy", "gini", "me"]

    for method in impurity_methods:
        for depth in range(1, 7):
            decision_tree = train_id3_tree(car_train, "class", target_labels, depth, method)
            test_error_data = compute_error_rate(decision_tree, car_test, 'class')
            train_error_data = compute_error_rate(decision_tree, car_train, 'class')

            print(f"Test Error:\tMethod: {method}\tDepth: {depth}\tError: {test_error_data}")
            print(f"Train Error:\tMethod: {method}\tDepth: {depth}\tError: {train_error_data}")
