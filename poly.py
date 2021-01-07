from sklearn.linear_model import LinearRegression


def poly(x_train, y_train, x_test, y_test):
    Pol_reg = LinearRegression()
    Pol_reg.fit(x_train, y_train)
    # Pol_reg.fit(features, targets)

    y_train_predic = Pol_reg.predict(x_train)
    y_test_predic = Pol_reg.predict(x_test)

    print('Intercept:', Pol_reg.intercept_)
    print('Coefficients:', Pol_reg.coef_)
    # Model Score
    print('\nModel Score:', Pol_reg.score(x_test, y_test))

    return Pol_reg
