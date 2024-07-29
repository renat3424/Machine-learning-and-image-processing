import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
def solve_matrix(n, A, B, C, F, mu1, kappa1, alpha, betta):
  y = np.zeros(n + 1)


  for i in range(n - 1, 0, -1):
    alpha[i] = A[i] / (C[i] - alpha[i + 1] * B[i])
    betta[i] = (B[i] * betta[i + 1] + F[i]) / (C[i] - alpha[i + 1] * B[i])

  y[0] = (mu1 + kappa1 * betta[1]) / (1 - alpha[1] * kappa1)

  for i in range(0, n):
    y[i + 1] = alpha[i + 1] * y[i] + betta[i + 1]

  return y


def dataset(n, R, sigma_values, E_2_values, H_R_2_values, kappa1, mu1):
    h=R/n
    A = np.zeros(n + 1)
    B = np.zeros(n + 1)
    C = np.zeros(n + 1)

    r = np.linspace(0, R, n + 1)
    X=[]
    Y=[]
    for sigma in sigma_values:
        v = np.full(n + 1, 1 / sigma)
        for E_2 in E_2_values:
            f = 2 * sigma * E_2
            F = -r * f * (h ** 2)

            for H_R_2 in H_R_2_values:
                alpha = np.zeros(n + 1)
                betta = np.zeros(n + 1)

                alpha[n] = 0
                betta[n] = H_R_2

                for i in range(1, n):
                    r_minus = r[i] - h / 2
                    r_plus = r[i] + h / 2
                    v_minus = (v[i] + v[i - 1]) / 2
                    v_plus = (v[i] + v[i + 1]) / 2

                    A[i] = r_minus * v_minus
                    B[i] = r_plus * v_plus
                    C[i] = r_minus * v_minus + r_plus * v_plus

                y_values = solve_matrix(n, A, B, C, F, mu1, kappa1, alpha, betta)
                y_values = np.sqrt(y_values)

                X.append([sigma, E_2, H_R_2])
                Y.append(y_values[0])



    return np.array(X), np.array(Y)





if __name__=="__main__":
    # m = 50
    # n = 100
    # R = 0.014
    # sigma_values = np.linspace(0.9, 1.8, m)*30
    # E_2_values = np.linspace(1400 ** 2, 2300 ** 2, m)
    # H_R_2_values = np.linspace(4400 ** 2, 6200 ** 2, m)
    # kappa1 = 1
    # mu1 = 0
    # X, Y=dataset(n, R, sigma_values, E_2_values, H_R_2_values, kappa1, mu1)
    # np.save("X_data.npy", X)
    # np.save("Y_data.npy", Y)
    X, Y=np.load("X_data.npy"), np.load(""
                                        "Y_data.npy")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)

    plt.plot(y_test, y_pred_lin, label="Linear Regression")


    print(f"Mseloss for linear regression={mean_squared_error(y_test, y_pred_lin)}")

    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(X_train, y_train)
    y_pred_tree = tree_reg.predict(X_test)

    plt.plot(y_test, y_pred_tree, label="Decision tree", c='orange')


    print(f"Mseloss for desicion tree={mean_squared_error(y_test, y_pred_tree)}")

    forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    forest_reg.fit(X_train, y_train)
    y_pred_forest = forest_reg.predict(X_test)

    plt.plot(y_test, y_pred_forest, label="Random Forest", c='green')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
    plt.legend()
    print(f"Mseloss for random forest={mean_squared_error(y_test, y_pred_forest)}")
    # print(y_train.shape[0])
    # regr = MLPRegressor(solver="sgd", learning_rate="adaptive", hidden_layer_sizes=(10,10, 10), max_iter=2000, random_state=42, learning_rate_init=1)
    # regr.fit(X_train, y_train)
    # y_pred_regr = regr.predict(X_test)
    # plt.scatter(y_test, y_pred_regr, c='orange')
    # plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
    # print(f"Mseloss for multilayer_perceptrone={mean_squared_error(y_test, y_pred_regr)}")

    plt.show()