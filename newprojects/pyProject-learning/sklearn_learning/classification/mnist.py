from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from sklearn.base import BaseEstimator
import numpy as np

import matplotlib
# import matplotlib.pyplot as plt

class Never5Classifier(BaseEstimator):
    def __init__(self):
        pass

    def fit(self,X,y):
        pass

    def predict(self,X):
        return np.zeros((len(X),1),dtype=bool)


def print_data_set(x):
    digit_shape = x.reshape(8,8)
    plt.imshow(digit_shape,cmap=matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")
    plt.show()


def split_data():
    mnist = load_digits()
    print("data shape",mnist.data.shape)
    print("target shape",mnist.target.shape)
    # print_data_set(mnist.data[1000])
    return train_test_split(mnist.data,mnist.target)


# py3.6环境下plot有问题，在这个例子中画不了图，切到py3.5下画（以后有时间细查）
def plot_presicion_recall_vs_threshold(precisions,recalls,thresholds):
    plt.plot(thresholds,precisions[:-1],"b--",label="Precision")
    plt.plot(thresholds,recalls[:-1],"g-",label="Recall")
    plt.xlabel("threshold")
    plt.legend(loc="upper left")
    plt.ylim([0,1])
    plt.show()


def plot_roc_curve(fpr,tpr,label=None):
    plt.plot(fpr,tpr,linewidth=2,label=label)
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


def cross_val(train_x,train_y):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone
    from sklearn.linear_model import SGDClassifier
    skfolds = StratifiedKFold(n_splits=3,random_state=42)

    sgd_clf = SGDClassifier()

    for train_index,test_index in skfolds.split(train_x,train_y):
        clone_clf = clone(sgd_clf)
        train_x_folds = train_x[train_index]
        train_y_folds = train_y[train_index]
        test_x_folds = train_x[train_index]
        test_y_folds = train_y[train_index]
        clone_clf.fit(train_x_folds,train_y_folds)
        pred_y = clone_clf.predict(test_x_folds)
        n_correct = sum(pred_y == test_y_folds)
        print(n_correct/len(test_y_folds))

        score_y_some = clone_clf.decision_function(test_x_folds[200].reshape(1, -1))
        print("score",score_y_some)

    # another implement
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(sgd_clf,train_x,train_y,cv = 3,scoring="accuracy")
    print("accuracies",accuracies)

    # confuse matrix
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix
    train_y_predict = cross_val_predict(sgd_clf,train_x,train_y,cv=3)
    print(confusion_matrix(train_y,train_y_predict))
    from sklearn.metrics import precision_score,recall_score
    print("precision score:",precision_score(train_y,train_y_predict))
    print("recall score:",recall_score(train_y,train_y_predict))
    from sklearn.metrics import f1_score
    print("f1 score:",f1_score(train_y,train_y_predict))

    # 绘制曲线
    scores_y = cross_val_predict(sgd_clf,train_x,train_y,cv=3,method="decision_function")
    from sklearn.metrics import precision_recall_curve
    # precisions,recalls,thresholds = precision_recall_curve(train_y,scores_y)
    # plot_presicion_recall_vs_threshold(precisions,recalls,thresholds)

    from sklearn.metrics import roc_curve
    fpr,tpr,thresholds = roc_curve(train_y,scores_y)
    # plot_roc_curve(fpr,tpr,label="SGD")

    # 曲线上再加一个随机森林的曲线嘞
    from sklearn.ensemble import RandomForestClassifier
    forest_clf = RandomForestClassifier(random_state=42)
    probas_y = cross_val_predict(forest_clf,train_x,train_y,cv=3,method="predict_proba")
    print(probas_y.shape)
    score_y_forest = probas_y[:,-1]
    fpr_forest, tpr_forest, thresholds_forest = roc_curve(train_y, scores_y)
    plt.plot(fpr,tpr,label="SGD")
    plot_roc_curve(fpr_forest, tpr_forest, label="Random Forest")
    plt.show()

    # 非5分类判断
    # never_5 = Never5Classifier()
    # accuracies_never_5 = cross_val_score(never_5,train_x,train_y,cv =3,scoring="accuracy")
    # print("accuracies_never_5:",accuracies_never_5)


def sgd_clf(trian_x,train_y,test_x,test_y):
    from sklearn.linear_model import SGDClassifier
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(trian_x,train_y)
    predicted_labels = sgd_clf.predict(test_x[:20])
    print("predict labels:",predicted_labels)
    print("ground labels:",test_y[:20])


def main():
    train_x,test_x,train_y,test_y = split_data()
    print("train ",train_x.shape)
    print("test ",test_x.shape)
    # 二分类
    train_y_5 = (train_y == 5)
    test_y_5 = (test_y == 5)
    # sgd_clf(train_x,train_y_5,test_x,test_y_5)
    # cross_val(train_x,train_y_5)

    # 多分类
    from sklearn.linear_model import SGDClassifier
    sgd_clf = SGDClassifier()
    sgd_clf.fit(train_x,train_y)
    pred_y = sgd_clf.decision_function(test_x[50].reshape(1,-1))
    print(pred_y)


if __name__ == '__main__':
    main()