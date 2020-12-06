import pandas as pd
from hdbscan import HDBSCAN
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


def prep_for_class(txt_col, f, r):
    """
    prepare dataframe for clustering
    :param txt_col:
    :param f:
    :param r:
    :return:
    """
    ans_ = pd.concat([pd.DataFrame(txt_col), f], axis=1)
    ans_ = ans_[ans_[-1] == 0]
    ans_ = ans_.drop(-1, axis=1)
    ans_ = ans_.melt(id_vars='text', var_name='cluster3', value_name='value')
    ans_ = ans_[ans_['value'] == 1]
    ans_ = ans_.drop('value', axis=1)
    ans_svm = ans_[ans_['cluster3'] > -1]
    for_x = pd.concat([pd.DataFrame(txt_col), pd.DataFrame(r)], axis=1)
    authorssvm = for_x.merge(ans_svm, left_on='text', right_on='text', how='right')
    y = list(authorssvm['cluster3'])
    X = authorssvm.drop(['cluster3', 'text'], axis=1)
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    return X_train, X_test, y_train, y_test


def cluster_hdbscan(svmx, hdb, txt_col, f, r):
    """
    cluster noise
    :param svmx:
    :param hdb:
    :param txt_col:
    :param f:
    :param r:
    :return:
    """
    X_train, X_test, y_train, y_test = prep_for_class(txt_col, f, r)
    if svmx:
        clf = svm.SVC(C=1000, kernel='rbf', gamma=0.7, random_state=12)
    else:
        clf = RandomForestClassifier(min_samples_split=2, max_features=3, max_depth=10000, random_state=12)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    nsvm = clf.predict(r)
    f = pd.get_dummies(nsvm)

    with_catsvm = pd.Series([str(i) for i in list(zip(txt_col, nsvm))], name='text')
    answerssvm = pd.concat([with_catsvm, f], axis=1)
    answers = [answerssvm, nsvm, hdb]
    return answers


def return_hdbscansvm(df, txt_col, rf=False, clust_size=15,  samp_size=5, svmx=False, clust_metric='braycurtis'):
    """
    complete pipeline
    :param df:
    :param txt_col:
    :param rf:
    :param clust_size:
    :param samp_size:
    :param svmx:
    :param clust_metric:
    :return:
    """
    if rf or svmx:
        cluster = True
    else:
        cluster = False

    super_flat = pd.DataFrame(df)
    r = super_flat

    hdb = HDBSCAN(min_cluster_size=clust_size, min_samples=samp_size, metric=clust_metric,
                  cluster_selection_method='leaf')
    n = hdb.fit_predict(r)
    f = pd.get_dummies(n)

    with_cat = pd.Series([str(i) for i in list(zip(txt_col, n))], name='text')
    answers = pd.concat([with_cat, f], axis=1)

    answers = [answers, n, hdb]
    if cluster:
        answers = cluster_hdbscan(svmx, hdb, txt_col, f, r)

    return answers
