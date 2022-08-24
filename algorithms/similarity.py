import numpy as np
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from pprint import pprint


def get_cosine_similarity(X, Y, isNormalized=False):

    X_WEIGHTED_KEY_PHRASES = []
    for i in range(0, len(X)):
        X_WEIGHTED_KEY_PHRASES.append(X[i][1])

    Y_WEIGHTED_KEY_PHRASES = [1] * len(X)
    for i in range(0, len(X)):
        for j in range(0, len(Y)):
            # if X[i][0].strip() == Y[j][0].strip():
            if X[i][0] == Y[j][0]:
                Y_WEIGHTED_KEY_PHRASES[i] = Y[j][1]

    X_WEIGHTED_KEY_PHRASES_NP_ARRAY = np.array(
        X_WEIGHTED_KEY_PHRASES).reshape(-1, 1)
    Y_WEIGHTED_KEY_PHRASES_NP_ARRAY = np.array(
        Y_WEIGHTED_KEY_PHRASES).reshape(-1, 1)

    if not isNormalized:
        mm = MinMaxScaler()
        X_WEIGHTED_KEY_PHRASES_NORMS = mm.fit_transform(
            X_WEIGHTED_KEY_PHRASES_NP_ARRAY)
        Y_WEIGHTED_KEY_PHRASES_NORMS = mm.fit_transform(
            Y_WEIGHTED_KEY_PHRASES_NP_ARRAY)

        numerator = np.dot(X_WEIGHTED_KEY_PHRASES_NORMS.T,
                           Y_WEIGHTED_KEY_PHRASES_NORMS)[0][0]

        denominator = np.sqrt(np.sum(X_WEIGHTED_KEY_PHRASES_NORMS**2)) * \
            np.sqrt(np.sum(Y_WEIGHTED_KEY_PHRASES_NORMS**2))

        if denominator == 0:
            return 0

        cosine_similarity = numerator / denominator
        return cosine_similarity
    else:
        numerator = np.dot(X_WEIGHTED_KEY_PHRASES_NP_ARRAY.T,
                           Y_WEIGHTED_KEY_PHRASES_NP_ARRAY)[0][0]

        denominator = np.sqrt(np.sum(X_WEIGHTED_KEY_PHRASES_NP_ARRAY**2)) * \
            np.sqrt(np.sum(Y_WEIGHTED_KEY_PHRASES_NP_ARRAY**2))

        if denominator == 0:
            return 0

        cosine_similarity = numerator / denominator
        return cosine_similarity


def get_jaccard_similarity(X, Y):
    INTERSECTION_COUNT = 0
    for i in range(0, len(X)):
        for j in range(0, len(Y)):
            # if X[i][0].strip() == Y[j][0].strip():
            if X[i][0].strip() == Y[j][0]:

                INTERSECTION_COUNT += 1

    UNION_COUNT = len(X) + len(Y) - INTERSECTION_COUNT

    jaccard_similarity = INTERSECTION_COUNT / UNION_COUNT
    return jaccard_similarity
