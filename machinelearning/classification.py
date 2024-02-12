import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



mu0 = 175
sig0 = 15
mu1 = 90
sig1 = 15
test_amount = 1000
edge = 185


def predict(X, edge):
    return X > edge


def calcArea(X, Y):
    result = 0
    for i in range(len(X) - 1):
        result += (X[i+1] - X[i]) * (Y[i] + Y[i+1])/2
    return result


basket_players = np.random.normal(mu0, sig0, test_amount)
soccer_players = np.random.normal(mu1, sig1, test_amount)


TP = sum(predict(soccer_players, edge))
FP = sum(predict(basket_players, edge))
FN = soccer_players.size - TP
TN = basket_players.size - FP
Accuracy = (TP+TN)/(basket_players.size + soccer_players.size)

Recall = TP/(TP+FN)

Alpha = FP/(FP+TN)
Beta = FN/(TP+FN)




FPRs = []
TPRs = []

bestEdge = edge
bestAccuracy = Accuracy

for t in range(0,edge * 10):

    TP = sum(predict(soccer_players, t))
    FP = sum(predict(basket_players, t))
    FN = soccer_players.size - TP
    TN = basket_players.size - FP

    Recall = TP/(TP+FN)    
    Alpha = FP/(FP+TN)
    Accuracy = (TP+TN)/(basket_players.size + soccer_players.size)
    if Accuracy > bestAccuracy:
        bestEdge = t
    


    frp = Alpha
    tpr = Recall 
    FPRs.append(frp)
    TPRs.append(tpr)


TP = sum(predict(soccer_players,bestEdge))
FP = sum(predict(basket_players, bestEdge))
FN = soccer_players.size - TP
TN = basket_players.size - FP
Accuracy = (TP+TN)/(basket_players.size + soccer_players.size)

Recall = TP/(TP+FN)

Alpha = FP/(FP+TN)
Beta = FN/(TP+FN)



print(f"Best results: edge = {bestEdge}, TP = {TP}, FP = {FP}, FN = {FN}, TN = {TN}, " + 
    f"Accuarcy = {Accuracy}, Recall = {Recall}, " +
    f"Alpha = {Alpha}, Beta = {Beta}")

print(f"AUC = {calcArea(TPRs, FPRs)}")

plt.plot(TPRs, FPRs)
plt.show()
