import re
from collections import Counter

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


def read_csv(path, columns=None):
    data = pd.read_csv(path, usecols=columns, sep=',', encoding='latin-1')

    first_column_data = data[columns[0]]
    second_column_data = data[columns[1]]

    return data, first_column_data, second_column_data


def plot_top_words(top_words, group):
    dictionary = dict(top_words)
    plt.title("Top 15 words from %s" % group)
    plt.bar(dictionary.keys(), dictionary.values())
    plt.show()


def get_top_words(counter, top_words_number):
    return counter.most_common(top_words_number)


def count_words(texts):
    counter = Counter()
    for text in texts:
        words = re.sub('\W', ' ', text.lower()).split()
        for word in words:
            counter[word] += 1

    return counter


def print_line(count):
    print('*' * (count + 2))


def print_header(string):
    l = len(string)
    print_line(l)
    print(' ' + string)
    print_line(l)


def print_top_words(top_words):
    for item in top_words:
        print("\t%10s : %5d" % ("\'" + item[0] + "\'", item[1]))


def print_classif_results(actual_y, predicted_y, particular_subset):
    classif_report = classification_report(actual_y, predicted_y)
    conf_matrix = confusion_matrix(actual_y, predicted_y)

    print_header('Classification Results, %s set' % particular_subset)
    print('Classification Report:')
    print(classif_report)
    print()
    print('Confusion Matrix:')
    print(conf_matrix)


def main():
    label_column = 'v1'
    message_column = 'v2'

    spam_msjs_path = '../data/spam.csv'

    # content_msgs -> x, labels -> y
    data, labels, content_msgs = read_csv(spam_msjs_path, (label_column, message_column))

    count = len(content_msgs)

    #####
    # EDA
    #####

    print_header("\'Messages\' column")
    print("\n***************")
    print(content_msgs)
    print("***************\n")
    print()
    print("Column description:\n%s" % content_msgs.describe())
    print()

    print_header("\'Labels\' column")
    print("\n***************")
    print(labels)
    print("***************\n")
    print("Column description:\n%s" % labels.describe())
    print()

    print_header("Data set grouped by label")
    grouped = data.groupby('v1')
    print("Groups:\n%s" % grouped.describe())
    print()

    ham = grouped.get_group('ham')
    spam = grouped.get_group('spam')
    print("Ham percentage: %.2f%%" % (ham.shape[0] * 100 / count))
    print("Spam percentage: %.2f%%" % (spam.shape[0] * 100 / count))
    print()

    # Plotting count plot for categorical values
    categorical_attributes = data.select_dtypes(include=['object'])
    # v1 Count plot
    plt.title("Distribution")
    sns.countplot(data=categorical_attributes, x="v1")
    plt.show()

    # Make BOW
    ham_words = count_words(ham.loc[:, 'v2'].values)
    ham_words_top = get_top_words(ham_words, 15)
    spam_words = count_words(spam.loc[:, 'v2'].values)
    spam_words_top = get_top_words(spam_words, 15)

    print("Top 15 words from hpam:")
    print_top_words(ham_words_top)
    plot_top_words(ham_words_top, "ham")
    print()
    print("Top 15 words from spam:")
    print_top_words(spam_words_top)
    plot_top_words(spam_words_top, "spam")
    print()

    ####################
    # Model Construction
    ####################
    # Label encoding
    # https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Split the data set
    x_train, x_test, y_train, y_test = train_test_split(content_msgs, labels_encoded, test_size=0.3, random_state=42)

    # Using the documents we are creating feature vectors
    vectorizer = CountVectorizer()
    vectorizer.fit_transform(content_msgs)
    x_train_vectors = vectorizer.transform(x_train)
    x_test_vectors = vectorizer.transform(x_test)

    # Create model
    model = SVC(kernel="rbf", C=100, gamma=1)
    # model = SVC(gamma='scale')

    print("Training model...")
    # Train model
    # The model is trained on the training set
    model.fit(x_train_vectors, y_train)

    # The model is tested on the training and test sets
    pred_y_train = model.predict(x_train_vectors)
    pred_y_test = model.predict(x_test_vectors)

    ##########################################################
    # Cálculos con classification_report, para ambos conjuntos
    ##########################################################
    print_classif_results(y_train, pred_y_train, "training", )
    print()
    print_classif_results(y_test, pred_y_test, "test")
    print()

    ##################
    # Parameter Tuning
    ##################

    ###############################
    # Parameter Tuning, grid search
    ###############################
    tuned_parameters = [
        {
            'kernel': ['rbf'],
            'gamma': [1e-3, 1e-4],
            'C': [1, 10, 100, 1000]
        },
        {
            'kernel': ['linear'],
            'C': [1, 10, 100, 1000]
        }
    ]
    scores = ['precision', 'recall', 'accuracy']

    for score in scores:
        print_header("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            SVC(), tuned_parameters, scoring=(score if score == 'accuracy' else '%s_macro' % score)
        )
        clf.fit(x_train_vectors, y_train)

        print("Best parameters set found on train set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on train set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_pred = clf.predict(x_test_vectors)
        print(classification_report(y_test, y_pred))
        print()


main()

