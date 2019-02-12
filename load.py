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

if __name__ == "__main__":
    df = pd.read_csv("./bjsam.csv")
    print df.head(100)

    # prepare data for normalization
    scaler = MinMaxScaler(feature_range=(0, 1))

    # train the normalization
    # normalize the dataset
    df[['rating']] = scaler.fit_transform(df[['rating']])
    print df.head(100)

    # A reader is still needed but only the rating_scale param is requiered.
    reader = surprise.Reader(rating_scale=(0, 1))

    # The columns must correspond to user id, item id and ratings (in that order).
    dataset = surprise.Dataset.load_from_df(df[['uid', 'iid', 'rating']], reader)

    alg = surprise.SVDpp(lr_all=.001)
    output = alg.fit(dataset.build_full_trainset())
    print output

    '''
    pred  = alg.predict(uid='3562446', iid='2982938')
    score = pred.est
    print score
    '''

    while True:
        print 'input uid =>'
        puid=str(input())
        if puid == 'exit' and not puid: break
        if puid == '\r': continue

        iids = df['iid'].unique()
        print "unique =", iids
        iids2= df.loc[df['uid'] == int(puid), 'iid']
        print "loc =", iids2
        iids_to_pred = np.setdiff1d(iids, iids2)
        print "pred = ", iids_to_pred

        testset = [[puid, iid, .5] for iid in iids_to_pred]
        #testset = [[puid, iid] for iid in iids_to_pred]
        predictions = alg.test(testset)
        print 'pred_test:'
        print predictions[0]

        print 'max predictions processing ... '

        pred_ratings = np.array([pred.est for pred in predictions])
        print 'pred_ratings:'
        print pred_ratings

        i_max = pred_ratings.argmax()
        iid = iids_to_pred[i_max]
        print iid, pred_ratings[i_max]

# End of File
