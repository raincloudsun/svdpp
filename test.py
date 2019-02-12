import numpy as np
import surprise
import pandas as pd
import io
from surprise import Dataset

'''
    f = open("bj.csv", 'r')
    while True:
        line = f.readline()
        if not line: break

        r = line.split(',')
        r[2] = r[2].split('\r\n')[0]

        
        r[1] = 


        if int(r[0]) > 10 and int(r[1]) < 500:
        #if int(r[0]) > 10:
            xUser.append(r[1])
            yUser.append(r[0])
        else:
            print r
    f.close()
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
    print '--'
    print predictions
    pred_ratings = np.array([pred.est for pred in predictions])
    print '--'
    print pred_ratings
    imax = pred_ratings.argmax()
    print '--'
    print imax
    return [recommend_items[imax], pred_ratings[imax]]

if __name__ == "__main__":
    '''
    5,3,0,1
    4,0,0,1
    1,1,0,5
    1,0,0,4
    '''

    # Creation of the dataframe. Column names are irrelevant.
    ratings_dict = {'uid'   : [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
                    'iid'   : [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
                    'rating': [5, 3, 0, 1, 4, 0, 0, 1, 1, 1, 0, 5, 1, 0, 0, 4]}
    ratings_pred = ratings_dict
    df = pd.DataFrame(ratings_dict)
    print df.head(10)

    # A reader is still needed but only the rating_scale param is requiered.
    reader = surprise.Reader(rating_scale=(1, 5))

    # The columns must correspond to user id, item id and ratings (in that order).
    dataset = surprise.Dataset.load_from_df(df[['uid', 'iid', 'rating']], reader)

    alg = surprise.SVDpp()
    output = alg.fit(dataset.build_full_trainset())
    print output
    #exit()

    p = 0
    for u in range(3):
        for i in range(5):
            if ratings_dict['rating'][p] == 0:
                pred  = alg.predict(uid=(u+1), iid=(i+1))
                score = pred.est
                ratings_pred['rating'][p] = score 
            p += 1

    df_pred = pd.DataFrame(ratings_pred)
    print ''
    print 'Pred-Score:'
    print df_pred

    ###
    # Recommend Item for User
    #
    ruid = 3
    test_score = 3.

    re_items = recommend_item(df, ruid)
    predictions = recom_testset(alg, ruid, re_items, test_score)
    #predictions = alg.test(df)
    print 'predictions ... '
    print predictions
    print ''

    result = best_prediction(predictions, re_items)
    #print result
    print 'Top item for user[{0}] => {0} with predicted rating {1}'.format(ruid, result[0], result[1])

# End of File
