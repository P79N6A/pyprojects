import os
import tarfile
import urllib
import pandas as pd
import numpy as np
import hashlib
import matplotlib.pyplot as plt
# https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


def fetch_housing_data(housing_url = HOUSING_URL,housing_path = HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)


# 会出现的问题：每次载入时会重新分配数据集，这样训练集可能会见到所有的测试集
# 解决方法：每次把训练集和测试集划分后存起来，便可以解决上述问题；
#   但是会有新的问题：数据集更新之后划分的数据集会失效，也需要重新分配
def split_train_test(data,test_ratio):
    shufffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shufffled_indices[:test_set_size]
    train_indices = shufffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]


# 数据划分的解决方法：给每一个实例一个标记，判断其是否进入测试集
# 具体方法：用某一个属性做hash值，然后取hash值的最后一个字节，如果该值小于等于51（51约为256的20%），
#   则放入测试集，否则放入训练集
def test_set_check(identifier,test_ratio,hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


# 索引的生成有两种方式：可以额外的添加一个索引列，也可以使用已有属性来构造索引(需要选择一个稳定的属性，即属性值不会经常变化的属性)，只要是均匀分布即可
# 额外添加索引列 ： data_with_id = data.reset_index()
# 使用已有属性构造索引列：data_with_id["id"] = data["longitude"] * 1000 + data["latitude"]
def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

# sklearn 提供的划分数据集的方法
# from sklearn.model_selection import train_test_split
# train_set,test_set = train_test_split(data,test_size=0.2,random_state=42)


def analysis_data_set(train_set_copy):
    train_set_copy.plot(kind='scatter', x="longitude", y="latitude", alpha=0.1)
    plt.show()


def plot_attribute_pandas(train_set_copy):
    from pandas.plotting import scatter_matrix
    attributes = ["median_house_value","median_income","total_rooms","housing_median_age"]
    scatter_matrix(train_set_copy[attributes],figsize=(12,8),alpha=0.2)
    train_set_copy.plot(kind = "scatter",x = "median_income",y = "median_house_value",alpha = 0.1)
    # 没画出图形来，是因为没有show出来，要这样哦
    plt.show()


def fill_unknown(train_set_copy):
    from sklearn.preprocessing import Imputer
    # 本例中imputer称为估算器，估算策略由strategy参数指定，由fit方法进行执行
    imputer = Imputer(strategy="median")
    # 中位数只能在数值属性上计算
    train_set_copy_num = train_set_copy.drop("ocean_proximity", axis=1)
    imputer.fit(train_set_copy_num)
    print(imputer.statistics_)
    # 转换器函数出来的数据是numpy数组或者scipy稀疏矩阵
    x = imputer.transform(train_set_copy_num)
    data_df = pd.DataFrame(x, columns=train_set_copy_num.columns)
    print(data_df.head(10))
    return data_df


def text_attri_representation(train_set_copy):
    # LabelEncoder 也是一个转换器
    from sklearn.preprocessing import LabelEncoder
    # 属性按照自增id增长的方式为文本属性进行编码，该方法的缺点：各个属性之间没有关联
    encoder = LabelEncoder()
    train_set_copy_cat = train_set_copy["ocean_proximity"]
    train_set_copy_cat_encoded = encoder.fit_transform(train_set_copy_cat)
    # print(train_set_copy_cat_encoded)

    # one hot 编码: 是在自增编码的基础上进行one hot编码的
    from sklearn.preprocessing import OneHotEncoder
    onehot_encoder = OneHotEncoder()
    # print("train_set_copy_cat_encoded shape:" + str(len(train_set_copy_cat_encoded)))
    # 返回的数据是稀疏矩阵
    train_set_copy_cat_onehot_encoded = onehot_encoder.fit_transform(train_set_copy_cat_encoded.reshape(-1,1))
    print(train_set_copy_cat_onehot_encoded)
    print("=================分割线================")
    # 可以使用LabelBinarizer转换器直接进行one hot 编码
    from sklearn.preprocessing import LabelBinarizer
    binary_encoder = LabelBinarizer()
    # 此时返回的数据是numpy密集型数组
    train_set_copy_cat_1hot_encoded = binary_encoder.fit_transform(train_set_copy_cat)
    print(train_set_copy_cat_1hot_encoded)
    return train_set_copy_cat_1hot_encoded
    # 还可以自定义转换器，实现fit transform之类的方法


def scala_attri(train_set_copy):
    # 最大最小缩放：先减去最小值，然后处于（max-min）；缺点：受噪声点影响比较大
    from sklearn.preprocessing import MinMaxScaler
    train_set_copy_in_come = train_set_copy["median_income"]
    print(train_set_copy_in_come)
    print("=================分割线================")
    scalar_encoder = MinMaxScaler()
    # scala 的encoder 接收的是numpy的时二维array
    # 从train_set_copy索引出来的列是series，所以需要先转换为numpy的array，操作：train_set_copy_in_come.values，该方法也适用于dataFrame转换为numpy array
    # 然后reshape，操作：train_set_copy_in_come.values.reshape(-1, 1)
    train_set_copy_in_come_scala = scalar_encoder.fit_transform(train_set_copy_in_come.values.reshape(-1, 1))
    print(train_set_copy_in_come_scala)

    print("=================another 分割线================")
    # 标准化缩放：首先减去均值，然后除以方差
    from sklearn.preprocessing import StandardScaler
    standard_scala_encoder = StandardScaler()
    train_set_copy_in_come_scala_standard = standard_scala_encoder.fit_transform(train_set_copy_in_come.values.reshape(-1, 1))
    print(train_set_copy_in_come_scala_standard)
    return train_set_copy_in_come_scala_standard


# pipeline
def sklearn_pipline_defined(train_copy_num):
    from sklearn.pipeline import Pipeline
    from sklearn.pipeline import FeatureUnion
    from sklearn.preprocessing import Imputer
    from sklearn.preprocessing import StandardScaler
    from house_predict import selectors_defined

    num_attribute = list(train_copy_num)
    cat_attribute = ["ocean_proximity"]
    num_pipeline = Pipeline([
        ('selector',selectors_defined.DataFrameSelector(num_attribute)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', selectors_defined.CombinedAttributesAdder()),
        ('std_scala',StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('selector',selectors_defined.DataFrameSelector(cat_attribute)),
        ('label_binarizer', selectors_defined.MyLabelBinarizer()),
    ])
    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline",num_pipeline),
        ("cat_pipeline",cat_pipeline)
    ])
    # prepared_data = full_pipeline.fit_transform(data)
    return full_pipeline


# 线性回归
def model_lr(train_data,train_label,test_data,test_label):
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(train_data,train_label)
    predicted_labels = lin_reg.predict(test_data[:5])
    print(predicted_labels)
    print("===============")
    print("ground_labels:")
    print(test_label[:5])
    from sklearn.metrics import mean_squared_error
    lin_mse = mean_squared_error(test_label[:5],predicted_labels[:5])
    print(lin_mse)
    lin_rmse = np.sqrt(lin_mse)
    print(lin_rmse)


def show_scores(scores):
    print("Scores:",scores)
    print("Mean:",scores.mean())
    print("standard deviation:",scores.std())


# 交叉验证
def cross_validate(train_data,train_label,test_data,test_label):
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.externals import joblib
    # 决策树
    print("decision tree")
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(train_data,train_label)
    tree_scores = cross_val_score(tree_reg,train_data,train_label,scoring="neg_mean_squared_error",cv=10)
    tree_rmse_score = np.sqrt(-tree_scores)
    show_scores(tree_rmse_score)
    # 保存模型
    print("decision tree loaded")
    joblib.dump(tree_reg,"tree_reg.pkl")
    #later
    loaded_tree_model = joblib.load("tree_reg.pkl")
    tree_scores = cross_val_score(tree_reg, train_data, train_label, scoring="neg_mean_squared_error", cv=10)
    tree_rmse_score = np.sqrt(-tree_scores)
    show_scores(tree_rmse_score)
    # 随机森林
    print("random forest")
    from sklearn.ensemble import RandomForestRegressor
    forest_reg = RandomForestRegressor()
    forest_reg.fit(train_data, train_label)
    forest_scores = cross_val_score(forest_reg, train_data, train_label, scoring="neg_mean_squared_error", cv=10)
    forest_rmse_score = np.sqrt(-forest_scores)
    show_scores(forest_rmse_score)


def predata_train_pipeline(data):
    data_with_id = data
    data_with_id["id"] = data["longitude"] * 1000 + data["latitude"]
    train_set, test_set = split_train_test_by_id(data_with_id, 0.2, "id")
    train_set_feature = train_set.drop("median_house_value", axis=1)
    train_set_labels = train_set["median_house_value"]
    test_set_feature = test_set.drop("median_house_value",axis = 1)
    test_set_labels = test_set["median_house_value"]
    train_set_copy = train_set_feature.copy()
    train_copy_num = train_set_copy.drop("ocean_proximity", axis = 1)
    full_pipeline = sklearn_pipline_defined(train_copy_num)
    prepared_train_data = full_pipeline.fit_transform(train_set_feature)
    prepared_test_data = full_pipeline.fit_transform(test_set_feature)
    cross_validate(prepared_train_data,train_set_labels,prepared_test_data,test_set_labels)


def main():
    fetch_housing_data()
    data = load_housing_data()
    # print(data.head())
    # print(data.info()) # 输出每个属性的信息，包括总行数、每个属性的非空数量和类型
    # print(data["ocean_proximity"].value_counts()) # 查看属性有多少种分类
    # print(data.describe()) # 查看属性的摘要：总数、均值、标准差（用来测量数据的离散程度）、最小值、最大值、百分位值（75% 50%  25%）

    # 绘制直方图
    import matplotlib.pyplot as plt
    data.hist(bins=50,figsize=(20,15))
    plt.show()
    # data_with_id = data
    # data_with_id["id"] = data["longitude"] * 1000 + data["latitude"]
    # train_set, test_set = split_train_test_by_id(data_with_id, 0.2, "id")
    # train_set_copy = train_set.copy()
    # train_copy_num = train_set_copy.drop["ocean_proximity"]
    # analysis_data_set(train_set_copy)
    # plot_attribute_pandas(train_set_copy)
    # corr_matrix = data.corr()
    # print(corr_matrix["median_house_value"].sort_values(ascending=False))
    # train_set_copy_text = text_attri_representation(train_set_copy)
    # train_set_copy_unknown = fill_unknown(train_set_copy_text)
    # print(train_set_copy_unknown.info())
    # train_set_copy_scala = scala_attri(train_set_copy_unknown)
    # print(train_set_copy_scala.info())

    predata_train_pipeline(data)


if __name__ == "__main__":
    main()

