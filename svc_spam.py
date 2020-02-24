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
from sklearn.metrics import precision_score, recall_score, accuracy_score  # precision, recall and accuracy
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


def extract_features(complete_texts_set, train_texts, test_texts):
    vectorizer = CountVectorizer()
    vectorizer.fit_transform(complete_texts_set)
    return vectorizer.transform(train_texts), vectorizer.transform(test_texts)


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


def print_classif_results(actual_y, predicted_y, particular_subset, model_number=None):
    classif_report = classification_report(actual_y, predicted_y, digits=4)
    conf_matrix = confusion_matrix(actual_y, predicted_y)

    print_header('Classification Results%s, %s set' % (", model " + model_number if model_number is not None else "", particular_subset))
    print('Classification Report:')
    print(classif_report)
    print()
    print('Confusion Matrix:')
    print(conf_matrix)

global metrics, models
def main():
    global metrics, models
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
    # An initial approach should tell you that you could use the default parameters, and measure the performance metrices, including the Confusion Matrix
    # For the Report use default parameters on training set and measure performance metrics

    # En esta parte se exploran cuáles son los parámetros por defecto

    # Label encoding
    # https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Split the data set
    x_train, x_test, y_train, y_test = train_test_split(content_msgs, labels_encoded, test_size=0.3, random_state=42)

    # Using the documents we are creating feature vectors
    x_train_vectors, x_test_vectors = extract_features(content_msgs, x_train, x_test)

    # Create model

    # Default parameter values according to documentation:
    #   C -> default=1.0
    #   kernel  -> default=’rbf’
    #   gamma -> default=’scale’

    print("Default parameters, gamma='auto'")
    # Esto es equivalente a model = SVC() o model = SVC(gamma='auto')
    model_1 = SVC(kernel="rbf", C=1, gamma='auto')

    print("Training model...")
    # The model is trained on the training set
    model_1.fit(x_train_vectors, y_train)
    # The model is tested on the training and test sets
    pred_y_train = model_1.predict(x_train_vectors)
    pred_y_test = model_1.predict(x_test_vectors)

    ###################################################
    # Resultados de clasificación, para ambos conjuntos
    ###################################################
    print_classif_results(y_train, pred_y_train, "training", "1")
    print()
    print_classif_results(y_test, pred_y_test, "test", "1")
    print()

    print("Default parameters, gamma='scale'")
    # The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features
    # Esto es equivalente a model = SVC(gamma='scale')
    model_2 = SVC(kernel="rbf", C=1, gamma='scale')

    print("Training model...")
    model_2.fit(x_train_vectors, y_train)
    pred_y_train = model_2.predict(x_train_vectors)
    pred_y_test = model_2.predict(x_test_vectors)

    print_classif_results(y_train, pred_y_train, "training", "2")
    print()
    print_classif_results(y_test, pred_y_test, "test", "2")
    print()

    ###############
    # Training loop
    ###############
    # Afterwards, you must change the values of C and gamma in the training loop
    # You must train the model and measure the indicated metrics (precision, recall and accuracy)
    # Ask yourself, can this values be improved? Start exploring with the C and gamma parameters.
    # Include the test set inside this training loop

    C = (1, 10, 100, 1000, 10000)
    gamma = (1, 1e-1, 1e-2, 1e-3, 1e-4)
    models = list()
    metrics = pd.DataFrame(columns=['precision', 'recall', 'accuracy'])

    # Next vary C (regularization parameter) as 10, 100, 1000, 10000.
    for C_ind in range(len(C)):
            model = SVC(kernel="rbf", C=C[C_ind])
            model.fit(x_train_vectors, y_train)
            pred_y_test = model.predict(x_test_vectors)
            precision = precision_score(y_test, pred_y_test, average='macro')
            recall = recall_score(y_test, pred_y_test, average='macro')
            accuracy = accuracy_score(y_test, pred_y_test)
            print("Results for model kernel='rbf', C=%d -> Precision: %0.4f, Recall: %0.4f, Accuracy: %0.4f" % (C[C_ind], precision, recall, accuracy))
            models.append(model)
            metrics = metrics.append({'precision': precision, 'recall': recall, 'accuracy': accuracy}, ignore_index=True)

    model_ind_precision, model_ind_recall, model_ind_accuracy = metrics.idxmax()

    if model_ind_precision == model_ind_recall and model_ind_precision == model_ind_accuracy:
        print("Same model for C")

    model_ind = model_ind_accuracy
    selected_C = C[model_ind]
    models = list()
    metrics = pd.DataFrame(columns=['precision', 'recall', 'accuracy'])

    # At last, lets play with gamma. Add one more parameter gamma = 1.0. Use values 0.1, 0.01, 0.001.
    for gamma_ind in range(len(gamma)):
        model = SVC(kernel="rbf", C=selected_C, gamma=gamma[gamma_ind])
        model.fit(x_train_vectors, y_train)
        pred_y_test = model.predict(x_test_vectors)
        precision = precision_score(y_test, pred_y_test, average='macro')
        recall = recall_score(y_test, pred_y_test, average='macro')
        accuracy = accuracy_score(y_test, pred_y_test)
        print("Results for model kernel='rbf', C=%d, gamma=%.4f -> Precision: %0.4f, Recall: %0.4f, Accuracy: %0.4f" %
              (selected_C, gamma[gamma_ind], precision, recall, accuracy))
        models.append(model)
        metrics = metrics.append({'precision': precision, 'recall': recall, 'accuracy': accuracy}, ignore_index=True)

    model_ind_precision, model_ind_recall, model_ind_accuracy = metrics.idxmax()

    if model_ind_precision == model_ind_recall and model_ind_precision == model_ind_accuracy:
        print("Same model for gamma")

    #################################################################################################
    # Resultados de clasificación, para ambos conjuntos, utilizando el modelo que dio mejor precision
    #################################################################################################
    model_ind = model_ind_precision

    pred_y_train = models[model_ind].predict(x_train_vectors)
    pred_y_test = models[model_ind].predict(x_test_vectors)

    print_classif_results(y_train, pred_y_train, "training", "best precision")
    print()
    print_classif_results(y_test, pred_y_test, "test", "best precision")
    print()

    print("Model best precision parameters:\n%s" % models[model_ind].get_params())

    ##############################################################################################
    # Resultados de clasificación, para ambos conjuntos, utilizando el modelo que dio mejor recall
    ##############################################################################################
    model_ind = model_ind_recall

    pred_y_train = models[model_ind].predict(x_train_vectors)
    pred_y_test = models[model_ind].predict(x_test_vectors)

    print_classif_results(y_train, pred_y_train, "training", "best recall")
    print()
    print_classif_results(y_test, pred_y_test, "test", "best recall")
    print()

    print("Model best recall parameters:\n%s" % models[model_ind].get_params())

    ################################################################################################
    # Resultados de clasificación, para ambos conjuntos, utilizando el modelo que dio mejor accuracy
    ################################################################################################
    model_ind = model_ind_accuracy

    pred_y_train = models[model_ind].predict(x_train_vectors)
    pred_y_test = models[model_ind].predict(x_test_vectors)

    print_classif_results(y_train, pred_y_train, "training", "best accuracy")
    print()
    print_classif_results(y_test, pred_y_test, "test", "best accuracy")
    print()

    print("Model best accuracy parameters:\n%s" % models[model_ind].get_params())
    print()

    ###############################
    # Parameter Tuning, grid search
    ###############################

    # Para esta parte se adaptaron las variables que se utilizan en "Parameter estimation using grid search with cross-validation"
    # Fuente del código original: scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html

    tuned_parameters = [
        {
            'kernel': ['rbf'],
            'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4],
            'C': [1, 10, 100, 1000]
        },
        {
            'kernel': ['linear'],
            'C': [1, 10, 100, 1000]
        }
    ]
    scores = ['precision', 'recall', 'accuracy']

    for score in scores:
        print_header("Tuning hyper-parameters for %s" % score)
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
        print(classification_report(y_test, y_pred, digits=4))
        print()


main()

