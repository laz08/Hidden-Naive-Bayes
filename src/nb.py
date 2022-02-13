# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math
import sys


def load_data(filename):
        lines = csv.reader(open(filename, 'rt'), delimiter=';')
        dataset = []
        for i, row in enumerate(lines):
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
        return dataset


def train_test_data(dataset, split_ratio):
    train_data = int(len(dataset) * split_ratio)
    train_set = []
    copy = list(dataset)
    while len(train_set) < train_data:
        index = random.randrange(len(copy))
        train_set.append(copy.pop(index))
    return [train_set, copy]


def split_by_class(dataset, predict_column):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if vector[predict_column] not in separated:
            separated[vector[predict_column]] = []
        separated[vector[predict_column]].append(vector)
    return separated


def statistics_continuous(numbers):
    mean = sum(numbers) / float(len(numbers))
    variance = sum([pow(x - mean, 2) for x in numbers]) / float(len(numbers) - 1)
    return [sum(numbers) / float(len(numbers)), math.sqrt(variance)]


def statistics_discrete(feature):
    n = len(feature)
    vals = list(set(feature))
    feature_stats = {}
    for val in vals:
        n_i = feature.count(val)
        p_i = n_i/n  # probability for a specific object
        feature_stats[val] = p_i
    return feature_stats


def summarize(dataset, categorical_list, predict_column):
    unpack_data = [x for x in zip(*dataset)]
    summaries = []
    if len(categorical_list) > 0:
        for cat in categorical_list:
            summaries.insert(cat, statistics_discrete(unpack_data[cat]))
            # summaries.insert(cat, ())
    continuous_var = list(set([x for x in range(len(unpack_data))]) - set(categorical_list))
    continuous_data = [x for i, x in enumerate(unpack_data) if i in continuous_var]
    for i in range(len(continuous_data)):
        mean, stdev = statistics_continuous(continuous_data[i])
        summaries.insert(continuous_var[i], (mean, stdev))
    del summaries[predict_column]  # Decision variable must be the last column of the dataset
    return summaries


def summarize_by_class(dataset, categoricals, predict_column):
    separated = split_by_class(dataset, predict_column)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] \
            = summarize(instances, categoricals, predict_column)  # [] is the index of categorical variable(s)
    return summaries


def estimate_probability_gaussian(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculate_class_probabilities(summaries, input_vector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            try:

                mean, stdev = classSummaries[i]
                x = input_vector[i]
                if isinstance(x, str):
                    category = input_vector[i]
                    try:
                        probability_categorical = classSummaries[i][category]
                        probabilities[classValue] *= probability_categorical
                    except KeyError:  # If the key or category has not observed yet
                        probabilities[classValue] *= 0.0001
                else:
                    probabilities[classValue] *= estimate_probability_gaussian(x, mean, stdev)
            except ValueError:
                pass

    return probabilities


def predict(summaries, input_vector):
    probabilities = calculate_class_probabilities(summaries, input_vector)
    best_label, best_prob = None, -1
    for classValue, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = classValue
    return best_label


def get_predictions(summaries, test_set, predict_column):
    predictions = []
    for i in range(len(test_set)):
        new_test_set = [column_value for (i, column_value) in enumerate(test_set[i]) if i != predict_column]
        result = predict(summaries, new_test_set)
        predictions.append(result)
    return predictions


def get_accuracy(test_set, predictions, predict_column):
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][predict_column] == predictions[i]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0


def main():

    if len(sys.argv) < 3:
        print('Usage: ' + sys.argv[0] + ' dataset_name column_class_index [split_ratio]')
        exit()

    filename = str(sys.argv[1])
    predict_column = int(sys.argv[2])

    split_ratio = 0.8
    if(len(sys.argv) == 4):
        split_ratio = float(sys.argv[3])

    dataset = load_data(filename)
    training_set, test_set = train_test_data(dataset, split_ratio)
    categorical_variables = [i for (i, value) in list(enumerate(dataset[0])) if isinstance(value, str)]
    print(categorical_variables)
    print(' #Initial rows: {0} splitted. \n #Train rows: {1} \n #Test rows: {2}'.format(len(dataset), len(training_set), len(test_set)))
    
    # prepare model
    summaries = summarize_by_class(training_set, categorical_variables, predict_column)
    # test model
    predictions = get_predictions(summaries, test_set, predict_column)
    accuracy = get_accuracy(test_set, predictions, predict_column)
    print('Accuracy: {0}%'.format(accuracy))


main()
