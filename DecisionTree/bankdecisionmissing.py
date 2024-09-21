import numpy as np
import pandas as pd

def get_sub_tree(train_data, label_name, label, feature,max_depth):
    attribute_tree_dict = {}
    value_count_feat = train_data[feature].value_counts(sort=False)
    #print(value_count_feat)
    for i,count in value_count_feat.items():
        sub_feature_data = train_data[train_data[feature] == i]

        is_data_pure = False
        max_type_count = 0
        max_type = None
        for type in label:
            type_count = sub_feature_data[sub_feature_data[label_name] == type].shape[0]
            if type_count > max_type_count:
                max_type_count = type_count
                max_type = type
            if type_count == count:
                is_data_pure = True
                attribute_tree_dict[i] = type
                train_data = train_data[train_data[feature] != i]
        if not is_data_pure:
            if max_depth > 1:
                attribute_tree_dict[i] = "multi"
            else:
                attribute_tree_dict[i] = max_type
        #print(i, attribute_tree_dict[i])
    return train_data, attribute_tree_dict

def get_tree(train_data, label_name, label, pre_feature, node, max_depth, method):
    if train_data.shape[0] > 0 and max_depth > 0:
        best_feature = get_best_feature(train_data, label_name, label, method)
        #print("get tree best feature" + best_feature + "\t" + str(max_depth))
        train_data, tree_dict = get_sub_tree(train_data, label_name, label, best_feature, max_depth)
        next_node = None

        if pre_feature == None:
            node[best_feature] = tree_dict
            next_node = node[best_feature]
        else:
            node[pre_feature] = dict()
            node[pre_feature][best_feature] = tree_dict
            next_node = node[pre_feature][best_feature]

        for i,type in list(next_node.items()):
            if type == "multi":
                sub_feature_data = train_data[train_data[best_feature] == i]
                get_tree(sub_feature_data, label_name, label, i, next_node, max_depth-1, method)

def get_id3_tree(train_data, label_name, label,max_depth, method):
    tree_dict = {}
    get_tree(train_data, label_name, label, None, tree_dict, max_depth, method)
    return tree_dict
def get_GI(train_data, label_name, label):
    gini_index = 0
    total_size = train_data.shape[0]
    for i in label:
        label_count = train_data[train_data[label_name] == i].shape[0]
        label_gini_index = pow(label_count/total_size,2)
        gini_index += label_gini_index
        #print(i,label_count,label_gini_index,gini_index)
    return 1-gini_index

def get_ME(train_data, label_name, label):
    total_size = train_data.shape[0]
    label_ME = 0
    for i in label:
        label_count = train_data[train_data[label_name] == i].shape[0]
        label_ME = max(label_count/total_size, label_ME)
        #print(i,label_count,label_ME)
    ME = 1-label_ME
    return ME

def get_entropy(train_data, label_name, label):
    entropy = 0
    total_size = train_data.shape[0]
    for i in label:
        label_entropy = 0
        label_count = train_data[train_data[label_name] == i].shape[0]
        if label_count != 0:
            label_entropy = -(label_count/total_size)*np.log2(label_count/total_size)
        entropy += label_entropy
        #print(i,label_count,label_entropy,entropy)
    return entropy

def get_info_gain(train_data, label_name, label, feature, method):
    sub_data = train_data[feature].unique()
    total_size = train_data.shape[0]
    #print(sub_data)
    sum_feat_infogain = 0
    total_cal = 0
    if method == "entropy":
        total_cal = get_entropy(train_data,label_name, label)
    elif method == "gini":
        total_cal = get_GI(train_data, label_name, label)
    elif method == "me":
        total_cal = get_ME(train_data, label_name, label)
    #print(total_cal)

    for i in sub_data:
        filtered_data = train_data[train_data[feature] == i]
        filtered_data_count = filtered_data.shape[0]
        filtered_data_cal = 0
        if method == "entropy":
            filtered_data_cal = get_entropy(filtered_data,label_name, label)
        elif method == "gini":
            filtered_data_cal = get_GI(filtered_data, label_name, label)
        elif method == "me":
            filtered_data_cal = get_ME(filtered_data, label_name, label)
        sum_feat_infogain += filtered_data_count/total_size*filtered_data_cal

    #print(sum_feat_infogain)
    info_gain = total_cal-sum_feat_infogain
    #print(info_gain)
    return info_gain  # Press Ctrl+F8 to toggle the breakpoint.

def get_best_feature(train_data, label_name, label,method):
    feature_only = train_data.columns.drop(label_name)
    #print(feature_only)
    max_info_gain = -1
    best_feature = None
    for i in feature_only:
        feature_gain = get_info_gain(train_data,label_name,label, i, method)
        if max_info_gain < feature_gain:
            max_info_gain = feature_gain
            best_feature = i
    #print(best_feature + "\tget_best_feature")
    return best_feature

def predict(id3_tree, instance):
    if not isinstance(id3_tree, dict):
        return id3_tree
    else:
        node = next(iter(id3_tree))
        feature = instance[node]
        if feature in id3_tree[node]:
            return predict(id3_tree[node][feature], instance)
        else:
            return None

def fetch_error(tree, test_data, label_name):
    wrong_count = 0
    correct_count = 0
    for index, row in test_data.iterrows():
        result = predict(tree, test_data.iloc[index])
        if result == test_data[label_name].iloc[index]:
            correct_count += 1
        else:
            wrong_count += 1
    error_rate = wrong_count / (correct_count + wrong_count)

    return error_rate

def convert_feature(data_set, num_feature_list):
    for num_feature in num_feature_list:
        cur_num_column = data_set.get(num_feature)
        temp_column = cur_num_column.copy(deep=True)
        mid_val = np.median(data_set.get(num_feature).to_numpy())
        temp_column.where(cur_num_column <= mid_val, "neg", inplace=True)
        temp_column.where(cur_num_column > mid_val, "pos", inplace=True)
        data_set[num_feature] = temp_column
    return data_set

def convert_unknown(data_set, feature_list):
    for feature in feature_list:
        if (data_set[feature].unique().__contains__('unknown')):
            cur_column = data_set.get(feature)
            temp_column = cur_column.copy(deep=True)
            sub_data = data_set[feature].unique()
            max_count = 0
            max_feature = None
            for i in sub_data:
                if i != "unknown":
                    filtered_data = data_set[data_set[feature] == i]
                    filtered_data_count = filtered_data.shape[0]
                    if filtered_data_count > max_count:
                        max_count = filtered_data_count
                        max_feature = i
            temp_column.where(cur_column != 'unknown', max_feature, inplace=True)
            data_set[feature] = temp_column
    return data_set


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    train_car_values = pd.read_csv("./car/train.csv", names=columns, header=None)
    test_car_values = pd.read_csv("./car/test.csv", names=columns, header=None)
    label_car = ["unacc", "acc", "good", "vgood"]
    actions = ["entropy", "gini", "me"]


    for method in actions:
        for depth in range(1, 7):
            decision_tree = get_id3_tree(train_car_values, "class", label_car, depth, method)
            #print(decision_tree)
            test_error = fetch_error(decision_tree, test_car_values, 'class')
            train_error = fetch_error(decision_tree, train_car_values, 'class')


    bank_columns_names = ["age", "job", "marital", "education", "default", "balance", "housing",
                    "loan", "contact", "day", "month", "duration", "campaign", "pdays",
                    "previous", "poutcome", "class"]

    bank_numerous = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    banklabels = ["yes", "no"]
    train_data_bank = pd.read_csv("./bank/train.csv", names=bank_columns_names, header=None)
    test_data_bank = pd.read_csv("./bank/test.csv", names=bank_columns_names, header=None)

    train_data_bank = convert_feature(train_data_bank, bank_numerous)
    test_data_bank = convert_feature(test_data_bank, bank_numerous)
    train_data_bank.head(5)

    for method in actions:
        for depth in range(1, 17):
            decision_tree = get_id3_tree(train_data_bank, "class", banklabels , depth, method)
            test_error = fetch_error(decision_tree, test_data_bank, 'class')
            train_error = fetch_error(decision_tree, train_data_bank, 'class')


    train_bank_unknown = convert_unknown(train_data_bank, bank_columns_names)
    test_bank_unknown = convert_unknown(test_data_bank, bank_columns_names)
    train_bank_unknown.head(5)

    for method in actions:
        for depth in range(1, 17):
            decision_tree_data = get_id3_tree(train_bank_unknown, "class", banklabels , depth, method)
            # print(decision_tree)
            test_error = fetch_error(decision_tree_data, test_bank_unknown, 'class')
            train_error = fetch_error(decision_tree_data, train_bank_unknown, 'class')

            print(f"Test Error:\tMethod: {method}\tDepth: {depth}\tError: {test_error}")
            print(f"Train Error:\tMethod: {method}\tDepth: {depth}\tError: {train_error}")

