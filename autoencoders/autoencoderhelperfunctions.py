# basic imports
import pandas as pd
import numpy as np
import math
# ml stuff
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Model
from keras.models import Sequential
# metrics + sklearn
from scipy import spatial
from hdbscan import HDBSCAN
from sklearn import metrics
from sklearn import decomposition
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# data visualization
import plotly.graph_objs as go
import plotly
import hiplot as hip


# this function can extract any named layer from a given model
# takes str, keras model, original dataframe
def layer_extract(layer_name, mod, v_dat):
    layer_name = layer_name
    intermediate_layer_model = Model(inputs=mod.input, outputs=mod.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(v_dat)
    es = intermediate_output
    
    yp = es[0]
    for n in es[1:len(es)]:
        yp = np.vstack((yp, n))
        
    return yp


# this function will return a list of cosine distances between matched rows of two given dataframes
# takes two data frames
def get_cos(dft, dfp):
    cos_val = []
    for i, j in enumerate(dft):
        p = dfp[i]
        dist = spatial.distance.cosine(j, p)
        if math.isnan(dist) is False:
            cos_val.append(1 - dist)
            
    return cos_val


# this saves a graph illustrating points within a 3 dimensional space (if input is more than 3 dim, takes first 3)
# takes data, labels for data, color scheme, can specify color scale or name if desired
def make_graph(X1, hovert, marking, colors='Inferno', name='150_80_30_8np_712tf.html', pca_bool=False):
    
    if pca_bool:
        pca = decomposition.PCA(n_components=3, random_state=12)
        pca.fit(X1)
        X1 = pca.transform(X1)
    
    x = X1[:, 0]
    y = X1[:, 1]
    z = X1[:, 2]


    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        hovertext=hovert,
        mode='markers+text',
        marker=dict(
            size=3,
       # color =  n,
            color =  marking,                # set color to an array/list of desired values
            colorscale=colors,   # choose a colorscale
            opacity=0.8,
            showscale = True
        )
    )])


# tight layout
    fig.update_layout(width = 1000, height = 500, margin=dict(l=0, r=0, b=0, t=0))

    plotly.offline.plot(fig, filename= name)
    
    
def make_graph2d(X1, hovert, marking, colors = 'Inferno', name = '150_80_30_8np_712tf.html', PCA = False):
    
    if PCA == True:
        pca = decomposition.PCA(n_components= 2, random_state = 12)
        pca.fit(X1)
        X1 = pca.transform(X1)
    
    x = X1[:,0]
    y = X1[:,1]



    fig = go.Figure(data=[go.Scatter(
        x=x,
        y=y,
        hovertext = hovert,
        mode='markers+text',
        marker=dict(
            size=3,
       # color =  n,
            color =  marking,                # set color to an array/list of desired values
            colorscale=colors,   # choose a colorscale
            opacity=0.8,
            showscale = True
        )
    )])


# tight layout
    fig.update_layout(width = 1000, height = 500, margin=dict(l=0, r=0, b=0, t=0))

    plotly.offline.plot(fig, filename= name)

# this function does not actually create a hiplot object, but converts keras history into a list of dictionaries interperable
#by hiplot.  takes the history object and a series of relevant labels
def create_hiplot(hist_ob, activ, lr,CS, CSR, encoder, KI, lf):
    hist_df = pd.DataFrame(hist_ob.history)
    hist = hist_df.to_dict('records')
    trial_num = 0
    for i in hist:
        dict1 = {'activation': activ, 'lr': lr,'CS': CS, 'CSR': CSR, 'encoder': encoder, 'KI': KI, 'lossfxn': lf}
        i = i.update(dict1)
    return(hist)

#this allows you to add multiple history objects to the input for hiplot (desirable for comparing models)       
def update_hiplot(hipl, hist_ob, activ, lr,encoder, KI, lf,CS = None, CSR = None):
    hist_df = pd.DataFrame(hist_ob.history)
    hist = hist_df.to_dict('records')
    trial_num = 0
    for i in hist:
        dict1 = {'activation': activ, 'lr': lr,'CS': CS, 'CSR': CSR, 'encoder': encoder, 'KI': KI, 'lossfxn': lf}
        i = i.update(dict1)
    return(hipl + hist)

#this allows you to remove content from specific iterations 
def delete_from_hiplot(hip1, activ, encoder):
    hi2 = []
    for i in hip1:
        if i['activation'] != activ or i['encoder'] != encoder:
            hi2.append(i)
    return(hi2)

def return_hdbscansvm(df, txt_col, rf = False, clust_size = 15,  samp_size = 5, svmx = False, svmc = 100, clust_metric = 'braycurtis'):

    super_flat =pd.DataFrame(df)
    r = super_flat
    
    hdb = HDBSCAN(min_cluster_size = clust_size, min_samples= samp_size, metric = clust_metric, cluster_selection_method = 'leaf')
    n = hdb.fit_predict(r)
    f = pd.get_dummies(n)

    with_cat = pd.Series([str(i) for i in list(zip(txt_col, n))], name = 'text')
    answers = pd.concat([with_cat, f], axis = 1 )
    
    answers = [answers, n, hdb]
    if svmx == True:

        ans_ = pd.concat([pd.DataFrame(txt_col), f], axis = 1 )
        ans_ = ans_[ans_[-1] == 0]
        ans_ = ans_.drop(-1, axis = 1)
        ans_ = ans_.melt(id_vars='text', var_name='cluster3', value_name='value')
        ans_ = ans_[ans_['value'] == 1]
        ans_ = ans_.drop('value', axis = 1)
        ans_svm = ans_[ans_['cluster3'] > -1]
        for_x = pd.concat([pd.DataFrame(txt_col),pd.DataFrame(r)], axis = 1)
        authorssvm = for_x.merge(ans_svm,left_on = 'text', right_on= 'text', how = 'right')
        y = list(authorssvm['cluster3'])
        X = authorssvm.drop(['cluster3', 'text'], axis = 1)
        X = X.fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        clf = svm.SVC(C = svmc, kernel = 'rbf', gamma = 0.7, random_state = 12)
        print('done')

        clf.fit(X_train,y_train)
        print('done')
        y_pred= clf.predict(X_test)

        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


        nsvm = clf.predict(r)
        f = pd.get_dummies(nsvm)

        with_catsvm = pd.Series([str(i) for i in list(zip(txt_col, nsvm))], name = 'text')
        answerssvm = pd.concat([with_catsvm, f], axis = 1 )
        answers = [answerssvm, nsvm, hdb]
    if rf == True:

        ans_ = pd.concat([pd.DataFrame(txt_col), f], axis = 1 )
        ans_ = ans_[ans_[-1] == 0]
        ans_ = ans_.drop(-1, axis = 1)
        ans_ = ans_.melt(id_vars='text', var_name='cluster3', value_name='value')
        ans_ = ans_[ans_['value'] == 1]
        ans_ = ans_.drop('value', axis = 1)
        ans_svm = ans_[ans_['cluster3'] > -1]
        for_x = pd.concat([pd.DataFrame(txt_col),pd.DataFrame(r)], axis = 1)
        authorssvm = for_x.merge(ans_svm,left_on = 'text', right_on= 'text', how = 'right')
        y = list(authorssvm['cluster3'])
        X = authorssvm.drop(['cluster3', 'text'], axis = 1)
        X = X.fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        clf = RandomForestClassifier(min_samples_split =2,max_features = 3, max_depth = 10000, random_state=12)
        print('done')

        clf.fit(X_train,y_train)
        print('done')
        y_pred= clf.predict(X_test)

        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


        nsvm = clf.predict(r)
        f = pd.get_dummies(nsvm)

        with_catsvm = pd.Series([str(i) for i in list(zip(txt_col, nsvm))], name = 'text')
        answerssvm = pd.concat([with_catsvm, f], axis = 1 )
        answers = [answerssvm, nsvm, hdb]
    
    return(answers)
