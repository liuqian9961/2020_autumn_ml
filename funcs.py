from gensim.models.doc2vec import TaggedDocument
import numpy as np


def pre_treatment(reviews):
    a = []
    for review in reviews:
        review = review.lower()
        review = review.replace('\n', '')
        review = review.replace('<br />', ' ')
        #   用空格隔开单词和标点
        puncs = """()[]{}.,:;*?!"""
        for punc in puncs:
            review = review.replace(punc, ' ' + punc + ' ')
        review = review.split()
        a.append(review)

    return a


def lable_reviews(reviews, labels):
    a = []
    for i, review in enumerate(reviews):
        a.append(TaggedDocument(review, [labels[i]]))
    return a


def combine(dm, dbow, reviews, labels):
    dm_vec = np.zeros((len(reviews), 100))
    dbow_vec = np.zeros((len(reviews), 100))
    for i in range(len(reviews)):
        dm_vec[i] = dm.docvecs[labels[i]]
        dbow_vec[i] = dbow.docvecs[labels[i]]
    vecs = np.hstack((dm_vec, dbow_vec))

    return vecs
