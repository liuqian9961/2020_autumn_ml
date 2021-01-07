from gensim.models.doc2vec import Doc2Vec
import pandas as pd
from sklearn.svm import LinearSVC
import funcs


#   读入数据
train = pd.read_csv('/Users/liuqian/Downloads/public_data/train.csv', index_col=0)
test = pd.read_csv('/Users/liuqian/Downloads/public_data/test_data.csv', index_col=0)

#   分离数据
train_labels = train.index
x_train = train['review']
y_train = train['sentiment']

test_labels = test.index
x_test = test['review']

#   对数据进行处理,去除无关符号以及用空格分隔单词
x_train = funcs.pre_treatment(x_train)
x_test = funcs.pre_treatment(x_test)

#   对数据进行标记
x_train_with_lable = list(funcs.lable_reviews(x_train, train_labels))
x_test_with_lable = list(funcs.lable_reviews(x_test, test_labels))
all_train = x_train_with_lable.copy()
all_train.extend(x_test_with_lable)

#   初始化模型
dm = Doc2Vec(vector_size=100, min_count=5, window=10, alpha=0.05, negative=5, epochs=10, dm=1)
dbow = Doc2Vec(vector_size=100, min_count=5, negative=5, epochs=10, dm=0)
dm.build_vocab(all_train)
dbow.build_vocab(all_train)

for i in range(1, 11):
    dm.train(all_train, total_examples=dm.corpus_count, epochs=1)
    dbow.train(all_train, total_examples=dbow.corpus_count, epochs=1)

#   训练完成，分离训练集和测试集
x_train_vec = funcs.combine(dm, dbow, x_train, train_labels)
x_test_vec = funcs.combine(dm, dbow, x_test, test_labels)

#   支持向量机
model = LinearSVC(dual=False, C=0.5)
model.fit(x_train_vec, y_train)
print('score:', model.score(x_train_vec, y_train))
test_data = pd.read_csv('/Users/liuqian/Desktop/机器学习大作业/IMDB评论情感预测/submission.csv', index_col=0)
test_data.loc[:, 'sentiment'] = model.predict(x_test_vec)
test_data.to_csv('/Users/liuqian/Desktop/机器学习大作业/IMDB评论情感预测/submission.csv')

