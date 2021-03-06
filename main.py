import numpy as np
import surprise
import pandas as pd
import io
from pandas import Series
from surprise import Dataset
from sklearn.preprocessing import MinMaxScaler

'''
# Normalize time series data
from pandas import Series
from sklearn.preprocessing import MinMaxScaler

# load the dataset and print the first 5 rows
series = Series.from_csv('daily-minimum-temperatures-in-me.csv', header=0)
print(series.head())

# prepare data for normalization
values = series.values
values = values.reshape((len(values), 1))

# train the normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(values)
print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))

# normalize the dataset and print the first 5 rows
normalized = scaler.transform(values)
for i in range(5):
    print(normalized[i])

# inverse transform and print the first 5 rows
inversed = scaler.inverse_transform(normalized)
for i in range(5):
	print(inversed[i])
'''

'''
def recommend_item(dataset, uid):
    item = []
    for i in range(len(dataset['uid'])):
        if uid == dataset['uid'][i]:
            if dataset['rating'][i] == 0:
                item.append(dataset['iid'][i])
    return item

def recom_testset(alg, uid, items, rating):
    testset = [[uid, iid, rating] for iid in items]
    #[[2, 2, 4.0], [2, 4, 4.0]]
    print testset
    predictions = alg.test(testset)

    return predictions

def best_prediction(predictions, recommend_items):
    pred_ratings = np.array([pred.est for pred in predictions])
    imax = pred_ratings.argmax()
    return [recommend_items[imax], pred_ratings[imax]]
'''

if __name__ == "__main__":
    #df = pd.read_csv("./bj.csv")
    df = pd.read_csv("./t.csv")
    print df.head(100)

    # prepare data for normalization
    scaler = MinMaxScaler(feature_range=(0, 1))

    # train the normalization
    # normalize the dataset
    #df[['a','b','c','d']] = scaler.fit_transform(df[['a','b','c','d']])
    #df[['rating']] = scaler.fit_transform(df[['rating']])
    #dfTest[['A','B']] = dfTest[['A','B']].apply(
    #                       lambda x: MinMaxScaler().fit_transform(x))

    #dfd = df[['a','b','c','d']]
    #df_norm = (dfd - dfd.min()) / (dfd.max() - dfd.min())
    df[['a','b','c','d']] = scaler.fit_transform(df[['a','b','c','d']])
    
    #print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
    print df.head(100)

    # A reader is still needed but only the rating_scale param is requiered.
    reader = surprise.Reader(rating_scale=(0, 1))

    # The columns must correspond to user id, item id and ratings (in that order).
    #dataset = surprise.Dataset.load_from_df(df[['uid', 'iid', 'rating']], reader)
    dataset = surprise.Dataset.load_from_df(df[['a', 'b', 'c', 'd']], reader)
    exit()

    alg = surprise.SVDpp(lr_all=.001)
    #alg = surprise.SVDpp()
    output = alg.fit(dataset.build_full_trainset())

    print output
    #exit()

    '''
    pred  = alg.predict(uid='3562446', iid='2982938')
    score = pred.est
    print score
    '''

    while True:
        '''
        print 'input uid =>'
        puid=str(input())
        if puid == 'exit' and not puid: break
        if puid == '\r\n': continue

        iids = df['iid'].unique()
        print "unique =", iids
        iids2= df.loc[df['uid'] == int(puid), 'iid']
        print "loc =", iids2
        iids_to_pred = np.setdiff1d(iids, iids2)
        print "pred = ", iids_to_pred
        '''

        #testset = [[puid, iid, .5] for iid in iids_to_pred]
        predictions = alg.test(df)
        print 'pred_test:'
        print predictions
        exit()

        print 'max predictions processing ... '

        pred_ratings = np.array([pred.est for pred in predictions])
        print 'pred_ratings:'
        print pred_ratings

        i_max = pred_ratings.argmax()
        iid = iids_to_pred[i_max]
        print iid, pred_ratings[i_max]

# End of File
