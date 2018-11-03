import numpy as np
import pandas as pd
from sklearn import preprocessing
from ultimate.mlp import MLP

import time

INPUT_DIR = "input/"

def feature_engineering(is_train=True):
    if is_train:
        print("processing train.csv")
        df = pd.read_csv(INPUT_DIR + 'train_V2.csv')

        df = df[df['maxPlace'] > 1]
    else:
        print("processing test.csv")
        df = pd.read_csv(INPUT_DIR + 'test_V2.csv')

    print("Adding Features")
    df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]
    df['headshotrate'] = df['kills'] / df['headshotKills']
    df['killStreakrate'] = df['killStreaks'] / df['kills']
    df['healthitems'] = df['heals'] + df['boosts']
    df['killPlace_over_maxPlace'] = df['killPlace'] / df['maxPlace']
    df['headshotKills_over_kills'] = df['headshotKills'] / df['kills']
    df['distance_over_weapons'] = df['totalDistance'] / df['weaponsAcquired']
    df['walkDistance_over_heals'] = df['walkDistance'] / df['heals']
    df['walkDistance_over_kills'] = df['walkDistance'] / df['kills']
    df['killsPerWalkDistance'] = df['kills'] / df['walkDistance']
    df["skill"] = df["headshotKills"]+df["roadKills"]

    df['longestKillWithHead'] = df['longestKill']*df['headshotKills']/df['kills']
    df['getItems'] = df['boosts'] + df['heals'] + df['weaponsAcquired']
    # df['totalTime'] = df['walkDistance']/8 + df['rideDistance']/70 + df['swimDistance']/8 + df['heals']*7 + df['boosts']*11
    df['totalTime'] = df['walkDistance']/6 + df['rideDistance']/30 + df['swimDistance']/2 + df['heals']*6 + df['boosts']*6

    df[df == np.Inf] = np.NaN
    df[df == np.NINF] = np.NaN

    print("Removing Na's From DF")
    df.fillna(0, inplace=True)

    print(df.isnull().any().any())

    print("remove some columns")
    target = 'winPlacePerc'
    features = list(df.columns)
    features.remove("Id")
    features.remove("matchId")
    features.remove("groupId")

    features.remove("matchType")

    y = None

    if is_train:
        print("get target")
        y = np.array(df.groupby(['matchId','groupId'])[target].agg('mean'), dtype=np.float64)
        features.remove(target)

    print("get group mean feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('mean')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()

    if is_train: df_out = agg.reset_index()[['matchId','groupId']]
    else: df_out = df[['matchId','groupId']]

    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])

    print("get group median feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('median')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_median", "_median_rank"], how='left', on=['matchId', 'groupId'])

    print("get group max feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('max')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])

    print("get group min feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('min')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])

    print("get group sum feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('sum')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_sum", "_sum_rank"], how='left', on=['matchId', 'groupId'])

    print("get group size feature")
    agg = df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
    df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])

    print("get match mean feature")
    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()
    df_out = df_out.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])

    print("get match sum feature")
    agg = df.groupby(['matchId'])[features].agg('sum').reset_index()
    df_out = df_out.merge(agg, suffixes=["", "_match_sum"], how='left', on=['matchId'])

    print("get match median feature")
    agg = df.groupby(['matchId'])[features].agg('median').reset_index()
    df_out = df_out.merge(agg, suffixes=["", "_match_median"], how='left', on=['matchId'])

#    print("get match size feature")
#    agg = df.groupby(['matchId']).size().reset_index(name='match_size')
#    df_out = df_out.merge(agg, how='left', on=['matchId'])

    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)

    X = np.array(df_out, dtype=np.float64)

    feature_names = list(df_out.columns)

    return X, y

X_train, y = feature_engineering(True)
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False).fit(X_train)

print("X_train", X_train.shape, X_train.max(), X_train.min())
scaler.transform(X_train)
print("X_train", X_train.shape, X_train.max(), X_train.min())

y = y * 2 - 1
print("y", y.shape, y.max(), y.min())

#np.save('X_train_hok_msm.npy', X_train)
#np.save('y_hok_msm.npy', y)
#X_train = np.load('X_train_hok_msm.npy')
#y = np.load('y_hok_msm.npy')

epoch_decay = 2
epoch_train = epoch_decay * 18
rate_init=0.008
hidden_size = 32
verbose=1
activation='a2m2l'
leaky = -0.001

mlp = MLP(layer_size=[X_train.shape[1], hidden_size, hidden_size, hidden_size, 1],activation=activation, leaky=leaky,bias_rate=[], regularization=1,importance_mul=0.0001, output_shrink=0.1, output_range=[-1,1], loss_type="hardmse")
mlp.train(X_train, y, verbose=verbose, importance_out=True, iteration_log=20000, rate_init=rate_init, rate_decay=0.8, epoch_train=epoch_train, epoch_decay=epoch_decay)

X_test, _ = feature_engineering(False)
scaler.transform(X_test)
print("X_test", X_test.shape, X_test.max(), X_test.min())
np.clip(X_test, out=X_test, a_min=-1, a_max=1)
print("X_test", X_test.shape, X_test.max(), X_test.min())
#np.save('X_test_hok_msm.npy', X_test)

pred = mlp.predict(X_test)

pred = pred.reshape(-1)
pred = (pred + 1) / 2

df_test = pd.read_csv(INPUT_DIR + 'test_V2.csv')

print("fix winPlacePerc")
for i in range(len(df_test)):
    winPlacePerc = pred[i]
    maxPlace = int(df_test.iloc[i]['maxPlace'])
    if maxPlace == 0:
        winPlacePerc = 0.0
    elif maxPlace == 1:
        winPlacePerc = 1.0
    else:
        gap = 1.0 / (maxPlace - 1)
        winPlacePerc = round(winPlacePerc / gap) * gap

    if winPlacePerc < 0: winPlacePerc = 0.0
    if winPlacePerc > 1: winPlacePerc = 1.0
    pred[i] = winPlacePerc

    if (i + 1) % 100000 == 0:
        print(i, flush=True, end=" ")


df_test['winPlacePerc'] = pred

submission = df_test[['Id', 'winPlacePerc']]
submission.to_csv('submission_hok_msm.csv', index=False)

submission.head(10)