import numpy as np
import lightgbm as lgb

np.random.seed(1234)
Y = (np.random.rand(1000000) - 0.5) / 1000.
X = np.random.rand(1000000, 100)
X_train = X[:800000]
Y_train = Y[:800000]
X_test = X[800000:]
Y_test = Y[800000:]

params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'min_data_in_leaf': 10000,
        'max_bin': 10,
        'num_threads': 10,
        'verbose': -1,
}

lgb_train = lgb.Dataset(X_train, Y_train)
gbm = lgb.train(params, lgb_train)
Y_pred = gbm.predict(X_test)
print(Y_pred)