import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor as rfr


def randomforst(x_train, y_train, x_test, y_test):
    RFR = rfr(n_estimators=100, criterion='mse', random_state=0, n_jobs=-1)
    RFR.fit(x_train, y_train)
    print(x_train)
    x_train_predic = RFR.predict(x_train)
    x_test_predic = RFR.predict(x_test)

    # Mean squared error
    train_mse = metrics.mean_squared_error(x_train_predic, y_train)
    test_mse = metrics.mean_squared_error(x_test_predic, y_test)
    print('Mean Squared Error train data: %.3f\nMean Squared Error test data: %.3f\n' % (train_mse, test_mse))

    # Root Mean Squares error
    print('RMSE train data: {:.3}, \nRMSE test data: {:.3}\n'.format(
        np.sqrt(np.absolute(train_mse)),
        np.sqrt(np.absolute(train_mse))))

    # R^2 - coefficient of determination
    print('R2 train data: %.3f\nR2 test data: %.3f\n' % (
        metrics.r2_score(x_train_predic, y_train), metrics.r2_score(x_test_predic, y_test)))

    # Model Score
    print('Model Score:', RFR.score(x_test, y_test))

    return RFR
