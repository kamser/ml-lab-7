import re

import torch
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn import model_selection , metrics, svm
from sklearn.model_selection import train_test_split


def read_csv(path, columns):
    data = pd.read_csv(path, usecols=columns, sep=',', encoding='latin-1')

    first_column_data = data[columns[0]]
    second_column_data = data[columns[1]]

    return first_column_data, second_column_data

def find_amount_spam(label_vector):
    label_target = 'spam'
    mask_vector = label_vector == label_target
    aux_tensor = torch.tensor(mask_vector)
    index_tensor = aux_tensor.nonzero()
    amount_spam = index_tensor.size()[0]
    return amount_spam

def find_amount_ham(label_vector):
    label_target = 'ham'
    mask_vector = label_vector == label_target
    aux_tensor = torch.tensor(mask_vector)
    index_tensor = aux_tensor.nonzero()
    amount_spam = index_tensor.size()[0]
    return amount_spam

def polt_histogram(amounts):
    pass

def make_BOW(msj_vector, label_vector):
    cnt_spam = Counter()
    cnt_ham = Counter()
    index = 0
    for msj in msj_vector:
        aux_vec_phrase_split = re.sub('\W', ' ', msj.lower()).split()
        for word in aux_vec_phrase_split:
            if label_vector[index] == 'spam':
                cnt_spam[word] += 1
            else:
                cnt_ham[word] += 1
        index += 1
    return cnt_spam, cnt_ham

def get_top_words(counter, top_words_number):
    return counter.most_common()[:top_words_number]

def main():
    label_column = 'v1'
    message_column = 'v2'

    spam_msjs_path = 'spam.csv'

    labels, content_msjs = read_csv(spam_msjs_path, (label_column, message_column))

    amount_content = 5572


    print("***************")
    print(content_msjs)

    print("***************")
    print(find_amount_spam(labels))
    print(find_amount_ham(labels))

    percent_spam = (find_amount_spam(labels) * 100)/(amount_content)
    percent_ham = (find_amount_ham(labels) * 100)/(amount_content)

    print(percent_ham)

    x = np.arange(2)
    percents = [find_amount_spam(labels), find_amount_ham(labels)]
    fig, ax = plt.subplots()
    plt.bar(x, percents)
    plt.xticks(x, ('Spam', 'Ham'))
    plt.show()

    #*****************

    print(labels)
    s, h = make_BOW(content_msjs, labels)

    print(s)
    print(h)

    print(get_top_words(s, 15))

    #*****************

    # probar maxfeatures
    vectorizer = CountVectorizer(stop_words='english')
    features_matrix = vectorizer.fit_transform(content_msjs)
    print(vectorizer.get_feature_names())
    print(features_matrix.toarray())

    # Creando el modelo:

    # Spliteando los set; 70% para el del training y 30% para test
    # Se debe poner de esta forma para que funcione
    # TODO: Pasar el vector de labels a binario con un tensor y luego volverlo lista para pasarlo al split
    training_set_labels, training_set_messages, test_set_labels, test_set_messages = train_test_split(labels, content_msjs, test_size=0.30, random_state=42)
    print("------------------------------------------")
    print(training_set_labels.shape[0])
    print(test_set_labels.shape[0])

    model = svm.SVC(kernel="rbf", C=100, gamma=1)
    print("Training model.")
    # train model
    model.fit(features_matrix, labels)


main()

