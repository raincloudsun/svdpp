import numpy as np
import surprise
import pandas as pd
from surprise import Dataset

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
    #print test
    predictions = alg.test(testset)

    return predictions

def best_prediction(predictions, recommend_items):
    pred_ratings = np.array([pred.est for pred in predictions])
    imax = pred_ratings.argmax()
    return [recommend_items[imax], pred_ratings[imax]]

if __name__ == "__main__":
    # Creation of the dataframe. Column names are irrelevant.
    ratings_dict = {'uid'   : [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
                    'iid'   : [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
                    'rating': [0, 4, 0, 3, 0, 4, 0, 4, 0, 1, 5, 2, 0, 3, 0]}
    ratings_pred = ratings_dict

    df = pd.DataFrame(ratings_dict)
    print df

    # A reader is still needed but only the rating_scale param is requiered.
    reader = surprise.Reader(rating_scale=(1, 5))

    # The columns must correspond to user id, item id and ratings (in that order).
    dataset = surprise.Dataset.load_from_df(df[['uid', 'iid', 'rating']], reader)

    alg = surprise.SVDpp()
    output = alg.fit(dataset.build_full_trainset())

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

    # Recommend Item for User
    ruid = 3
    test_score = 3.

    re_items = recommend_item(df, ruid)
    predictions = recom_testset(alg, ruid, re_items, test_score)
    result = best_prediction(predictions, re_items)
    #print result
    print 'Top item for user[{0}] => {0} with predicted rating {1}'.format(ruid, result[0], result[1])

# End of File




