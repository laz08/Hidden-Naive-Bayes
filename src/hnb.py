import csv
import random
import math
import sys
import collections


def load_data(filename):
    lines = csv.reader(open(filename, 'rt'), delimiter=';')
    dataset = []
    headers = []
    for i, row in enumerate(lines):
        if i == 0:
            headers = row
        if i != 0:
            line = []
            for j, value in enumerate(row):
                if value == ' ':
                    line = []
                    break

                column_value = int(value) if value.isdigit() else value
                line.append(column_value)

            if len(line) > 0:
                dataset.append(line)
    return headers, dataset


def train_test_data(dataset, split_ratio):
    train_data = int(len(dataset) * split_ratio)
    train_set = []
    copy = list(dataset)
    while len(train_set) < train_data:
        index = random.randrange(len(copy))
        train_set.append(copy.pop(index))
    return [train_set, copy]


def class_probabilites(dataset, predict_column):
    total_amount = len(dataset)
    predict_column_values = list(map(lambda row: row[predict_column], dataset))
    counter_values = collections.Counter(predict_column_values)
    class_probs = list(map(lambda class_value: class_value/total_amount, counter_values.values()))
    return counter_values.keys(), class_probs, counter_values.values()


def attribute_unique_keys(dataset, attribute_column):
    counter = collections.Counter(list(map(lambda row: row[attribute_column], dataset)))
    attribute_unique_values = counter.keys()
    attribute_values = counter.values()
    return attribute_unique_values, attribute_values


def get_attribute_index(attribute_keys, a):
    for el in attribute_keys:
        if a in el:
            return el[a]
    return -1


def attributes_count(dataset, class_keys, predict_column, class_probs):
    total_number_attribute_values = 0
    keys_value_index = []

    for column_index, column in enumerate(dataset[0]):
        keys, values = attribute_unique_keys(dataset, column_index)
        keys_value_index.append([{key: total_number_attribute_values + index} for index, key in enumerate(keys)])
        total_number_attribute_values += len(keys)

    attribute_count = [[[0 for i in range(0, total_number_attribute_values)]
                        for i in range(0, total_number_attribute_values)] for k in range(0, len(list(class_keys)))]

    for row_index, row in enumerate(dataset):
        class_index = list(class_keys).index(row[predict_column])
        for column_index, column in enumerate(row):
            if column_index != predict_column:
                index = get_attribute_index(keys_value_index[column_index], column)
                for column_index_2, column_2 in enumerate(row):
                    if column_index_2 != predict_column:
                        index_2 = get_attribute_index(keys_value_index[column_index_2], column_2)
                        attribute_count[class_index][index][index_2] += 1

    num_attr = len(dataset[0])
    conditional_mutual_info = [[0] * num_attr] * num_attr

    for son in range(0, num_attr):
        if son != predict_column:
            for parent in range(0, num_attr):
                if parent != predict_column and parent != son:
                    conditional_mutual_info[son][parent] = \
                        compute_conditional_mutual_info(dataset, son, parent, keys_value_index, attribute_count, class_probs)

    return keys_value_index, conditional_mutual_info, attribute_count


def compute_conditional_mutual_info(dataset, son, parent, key_value_index, attr_count, class_probs):

    conditional_mutual_info = 0

    start_index_son = list(key_value_index[son][0].values())[0]
    start_index_parent = list(key_value_index[parent][0].values())[0]
    num_total_rows = len(dataset)

    for c in range(0, len(class_probs)):
        for attr_values_parent in range(0, len(key_value_index[parent])):
            p_class_parent = attr_count[c][start_index_parent + attr_values_parent][
                                 start_index_parent + attr_values_parent] / num_total_rows
            for attr_values_son in range(0, len(key_value_index[son])):

                p_class_parent_son = attr_count[c][start_index_parent + attr_values_parent][start_index_son + attr_values_son] / num_total_rows
                p_class_son = attr_count[c][start_index_son + attr_values_son][start_index_son + attr_values_son]/num_total_rows

                div = p_class_parent * p_class_son
                dividend = p_class_parent_son * class_probs[c]

                log = 1
                if div == 0 or dividend == 0:
                    log = 0
                else:
                    log = math.log2(dividend/div)
                conditional_mutual_info += p_class_parent_son * log

    return conditional_mutual_info


def predict(dataset, conditional_mutual_info, test_set, attribute_count, key_value_index, class_values, class_keys, predict_column):
    num_classes = len(class_values)
    num_rows = len(dataset)
    num_attr = len(dataset[0])

    total_test_set = len(test_set)
    total_true = 0

    counter_ok = [0] * num_classes
    counter_fail = [0] * num_classes
    
    for i, instance in enumerate(test_set):
        probs = [0] * num_classes
        for class_index in range(0, num_classes):
            probs[class_index] = (list(class_values)[class_index] + 1.0 / num_classes) / (num_rows + 1.0)
            for attr in range(0, num_attr):
                if attr == predict_column:
                    continue
                prob = 0
                index_attr = get_attribute_index(key_value_index[attr], instance[attr])
                conditional_mutual_info_sum = 0

                for attr2 in range(0, num_attr):
                    if attr2 == predict_column:
                        continue
                    index_attr2 = get_attribute_index(key_value_index[attr2], instance[attr2])
                    conditional_mutual_info_sum += conditional_mutual_info[attr][attr2]
                    prob += conditional_mutual_info[attr][attr2] * (
                            attribute_count[class_index][index_attr2][index_attr] + 1.0 / len(key_value_index[attr])) /\
                            (attribute_count[class_index][index_attr2][index_attr2] + 1.0)
                if conditional_mutual_info_sum > 0:
                    prob = prob / conditional_mutual_info_sum
                    probs[class_index] *= prob

                else:
                    prob = (attribute_count[class_index][index_attr][index_attr] + 1.0 / len(key_value_index[attr]) /
                            (num_classes + 1.0))
                    probs[class_index] *= prob
        max_prob = max(probs)
        max_prob_index = probs.index(max_prob)
        # print(instance, list(class_keys)[max_prob_index])

        if instance[predict_column] == list(class_keys)[max_prob_index]:
            total_true += 1
            counter_ok[max_prob_index] += 1
        else:
            counter_fail[max_prob_index] += 1

    print("Accuracy: " + str(round((total_true / total_test_set) * 100.00, 2)))
    print("-------")
    for i, cl in enumerate(class_keys):
        ok = counter_ok[i]
        fail = counter_fail[i]
        acc = 100
        if ok+fail != 0:
            acc = round((ok/(ok+fail) * 100.0), 2)
        print(" * " + cl + "  || OK: " + str(ok) + "  || FAIL: " + str(fail) + "  || acc: " + str(acc))
        

def hnb(dataset, split_ratio, predict_column, headers):
    training_set, test_set = train_test_data(dataset, split_ratio)
    print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(training_set), len(test_set)))
    class_keys, class_probs, class_values = class_probabilites(dataset, predict_column)
    key_value_index, conditional_mutual_info, attribute_count = \
        attributes_count(dataset, class_keys, predict_column, class_probs)

    num_attr = len(dataset[0])
    weight = [0] * num_attr
    weights = [[0] * num_attr] * num_attr

    print("-------")
    for attr in range(0, num_attr):
        for attr2 in range(0, num_attr):
            if attr2 != attr:
                weight[attr] += conditional_mutual_info[attr][attr2]
        for attr2 in range(0, num_attr):
            if attr2 != attr:
                weights[attr][attr2] = conditional_mutual_info[attr][attr2] / weight[attr]
                print(" {0} -> {1} : {2}".format(headers[attr], headers[attr2], round(weights[attr][attr2],2)))

    print("-------")

    predict(dataset, conditional_mutual_info, test_set, attribute_count, key_value_index, class_values, class_keys, predict_column)


def main():

    if len(sys.argv) < 3:
        print('Usage: ' + sys.argv[0] + ' dataset_name column_class_index [split_ratio]')
        exit()

    filename = str(sys.argv[1])
    predict_column = int(sys.argv[2])

    split_ratio = 0.8
    if(len(sys.argv) == 4):
        split_ratio = float(sys.argv[3])
    
    headers, dataset = load_data(filename)
    hnb(dataset, split_ratio, predict_column, headers)


main()
