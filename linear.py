from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


def linear(x_train, y_train, x_test, y_test):
    Lin_reg_model = LinearRegression()
    Lin_reg_model.fit(x_train, y_train)

    print('Intercept:', Lin_reg_model.intercept_)
    print('Coefficients:', Lin_reg_model.coef_)

    Lin_reg_model_train_pred = Lin_reg_model.predict(x_train)
    Lin_reg_model_test_pred = Lin_reg_model.predict(x_test)

    # Mean squared error
    Lin_reg_model_train_mse = mean_squared_error(y_train, Lin_reg_model_train_pred)
    Lin_reg_model_test_mse = mean_squared_error(y_test, Lin_reg_model_test_pred)
    print('MSE train data: {:.3}, \nMSE test data: {:.3}\n'.format(Lin_reg_model_train_mse, Lin_reg_model_test_mse))

    # Root Mean Squared error
    print('RMSE train data: {:.3}, \nRMSE test data: {:.3}\n'.format(
        np.sqrt(np.absolute(Lin_reg_model_train_mse)),
        np.sqrt(np.absolute(Lin_reg_model_train_mse))))

    # R^2 - coefficient of determination
    print('R2 train data: {:.3}, \nR2 test data: {:.3}\n'.format(
        r2_score(y_train, Lin_reg_model_train_pred),

        r2_score(y_test, Lin_reg_model_test_pred)))

    # Model Score
    print('Model Score:', Lin_reg_model.score(x_test, y_test))

    return Lin_reg_model
