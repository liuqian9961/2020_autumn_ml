import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split as holdout
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
import warnings
import linear
import randomforst
import poly

warnings.filterwarnings('ignore')

df = pd.read_csv('/Users/liuqian/Desktop/机器学习大作业/医疗花费预测/public_dataset/train.csv')

# 查看charge的分布情况
plt.figure(figsize=(12, 5))
sns.distplot(np.log10(df.charges), color='b')
plt.show()

# 查看age的分布情况
sns.distplot(df.age, color='b').set_title('Distribution of age')
plt.show()

# 查看bmi的分布情况
sns.distplot(df.bmi, color='b').set_title('Distribution of BMI')
plt.show()

df_num = df[['age', 'bmi', 'charges']]
sns.pairplot(df_num, diag_kind='kde')
plt.show()

plt.figure(figsize=(25, 10))
plt.subplot(2, 2, 1)
sns.boxplot(x='region', y='charges', data=df)

plt.subplot(2, 2, 2)
sns.boxplot(x='children', y='charges', data=df)

plt.subplot(2, 2, 3)
sns.boxplot(x='sex', y='charges', data=df)

plt.subplot(2, 2, 4)
sns.boxplot(x='smoker', y='charges', data=df)
plt.show()

ax = sns.lmplot(x='age', y='charges', data=df, hue='smoker', palette='Set1')
plt.title('Age vs Medical Charges')
plt.show()
ax = sns.lmplot(x='bmi', y='charges', data=df, hue='smoker', palette='Set1')
plt.title('BMI vs Medical Charges')
plt.show()
ax = sns.lmplot(x='children', y='charges', data=df, hue='smoker', palette='Set1')
plt.title('Number of children vs Medical Charges')
plt.show()

# 模型训练处理数据
df[['region', 'sex', 'smoker']] = df[['region', 'sex', 'smoker']].astype('category')
df.dtypes

label = LabelEncoder()

label.fit(df.region.drop_duplicates())  # 去重
df.region = label.transform(df.region)  # 分组

label.fit(df.sex.drop_duplicates())
df.sex = label.transform(df.sex)

label.fit(df.smoker.drop_duplicates())
df.smoker = label.transform(df.smoker)

df.dtypes
sns.pairplot(df)

features = df.drop(['charges'], axis=1)  # 删除charges,留下剩余的
targets = df['charges']  # 得到charges
x_train, x_test, y_train, y_test = holdout(features, targets, test_size=0.1, random_state=0)

#   线性回归
Lin_reg_model = linear.linear(x_train, y_train, x_test, y_test)

# 随机森林
RFR = randomforst.randomforst(x_train, y_train, x_test, y_test)

# 多项式拟合
features = df.drop(['charges', 'sex', 'region'], axis=1)
target = df.charges

pol = PolynomialFeatures(degree=2)
x_pol = pol.fit_transform(features)
x_train, x_test, y_train, y_test = holdout(x_pol, target, test_size=0.2, random_state=0)

Pol_reg = poly.poly(x_train, y_train, x_test, y_test)

# 使用多项式拟合的结果对测试集进行预测
df1 = pd.read_csv('/Users/liuqian/Desktop/机器学习大作业/医疗花费预测/public_dataset/test_sample.csv')

df1_num = df1[['age', 'bmi', 'charges']]

df1[['region', 'sex', 'smoker']] = df1[['region', 'sex', 'smoker']].astype('category')
df1.dtypes

label = LabelEncoder()

label.fit(df1.region.drop_duplicates())  # 去重
df1.region = label.transform(df1.region)  # 分组

label.fit(df1.sex.drop_duplicates())
df1.sex = label.transform(df1.sex)

label.fit(df1.smoker.drop_duplicates())
df1.smoker = label.transform(df1.smoker)

df1.dtypes

features = df1.drop(['charges', 'sex', 'region'], axis=1)  # 删除charges,留下剩余的
targets = df1['charges']  # 得到charges
pol = PolynomialFeatures(degree=2)
x_pol = pol.fit_transform(features)
y_test_predic = Pol_reg.predict(x_pol)
df1['charges'] = y_test_predic
df1.to_csv('/Users/liuqian/Desktop/机器学习大作业/医疗花费预测/public_dataset/test_sample.csv', index=False, encoding='utf-8')
