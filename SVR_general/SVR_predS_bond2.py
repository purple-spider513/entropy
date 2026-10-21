import pandas as pd
import numpy as np
test_data=pd.read_csv("./hybrid_std6074.csv")
X_test=test_data[["ring","heavy","H","s_heavy_bond","MW","B-C.3","B-C.ar","B-O.3","Br-C.1","Br-C.2","Br-C.3","Br-C.ar","Br-H","Br-N.4","Br-N.am","Br-N.ar","Br-N.pl3","C-C.2","C-C.3","C-C.ar","C-H","C-N.4","C-N.am","C-N.pl3","C.1-C.1","C.1-C.2","C.1-C.3","C.1-C.ar","C.1-Cl","C.1-F","C.1-H","C.1-N.1","C.1-N.2","C.1-N.3","C.1-N.am","C.1-N.ar","C.1-N.pl3","C.1-O.2","C.1-O.3","C.1-P.3","C.1-S.2","C.1-S.3","C.1-S.O","C.1-S.O2","C.1-Si","C.2-C.2","C.2-C.3","C.2-C.ar","C.2-C.cat","C.2-Cl","C.2-F","C.2-Fe","C.2-H","C.2-I","C.2-N.1","C.2-N.2","C.2-N.3","C.2-N.4","C.2-N.am","C.2-N.ar","C.2-N.pl3","C.2-O.2","C.2-O.3","C.2-O.co2","C.2-P.3","C.2-S.2","C.2-S.3","C.2-S.O","C.2-S.O2","C.2-S.o","C.2-S.o2","C.2-Si","C.3-C.3","C.3-C.ar","C.3-C.cat","C.3-Cl","C.3-F","C.3-Fe","C.3-H","C.3-I","C.3-N.1","C.3-N.2","C.3-N.3","C.3-N.4","C.3-N.am","C.3-N.ar","C.3-N.pl3","C.3-O","C.3-O.2","C.3-O.3","C.3-P.3","C.3-S.3","C.3-S.O","C.3-S.O2","C.3-S.o","C.3-S.o2","C.3-Si","C.ar-C.ar","C.ar-C.cat","C.ar-Cl","C.ar-F","C.ar-H","C.ar-I","C.ar-N.1","C.ar-N.2","C.ar-N.3","C.ar-N.4","C.ar-N.am","C.ar-N.ar","C.ar-N.pl3","C.ar-O.2","C.ar-O.3","C.ar-O.co2","C.ar-P.3","C.ar-S.2","C.ar-S.3","C.ar-S.O","C.ar-S.O2","C.ar-S.o","C.ar-S.o2","C.ar-Si","C.cat-N.2","C.cat-N.am","C.cat-N.pl3","C.cat-O.3","C.cat-S.3","Cl-N.3","Cl-N.am","Cl-N.ar","Cl-N.pl3","Cl-O.3","F-N.3","F-N.am","F-N.ar","F-N.pl3","F-O.3","F-S.3","H-N.2","H-N.3","H-N.4","H-N.am","H-N.ar","H-N.pl3","H-O","H-O.2","H-O.3","H-P.3","H-S.3","H-S.O","H-S.O2","H-Si","I-N.3","I-N.am","I-N.ar","N.1-N.1","N.1-N.2","N.1-N.pl3","N.2-N.2","N.2-N.3","N.2-N.4","N.2-N.am","N.2-N.ar","N.2-N.pl3","N.2-O.2","N.2-O.3","N.2-O.co2","N.2-P.3","N.2-S.3","N.2-S.O","N.2-S.O2","N.2-S.o2","N.3-N.3","N.3-N.4","N.3-N.am","N.3-N.ar","N.3-N.pl3","N.3-O.2","N.3-O.3","N.3-O.co2","N.3-P.3","N.3-S.3","N.3-S.O","N.3-S.O2","N.3-S.o2","N.3-Si","N.4-N.4","N.4-N.am","N.4-N.pl3","N.4-O.3","N.4-O.co2","N.4-P.3","N.4-S.O2","N.4-Si","N.am-N.am","N.am-N.ar","N.am-N.pl3","N.am-O.2","N.am-O.3","N.am-O.co2","N.am-P.3","N.am-S.3","N.am-S.O","N.am-S.O2","N.am-S.o2","N.am-Si","N.ar-N.ar","N.ar-N.pl3","N.ar-O.2","N.ar-O.3","N.ar-O.co2","N.ar-P.3","N.ar-S.2","N.ar-S.3","N.ar-S.O2","N.ar-S.o2","N.ar-Si","N.pl3-N.pl3","N.pl3-O.2","N.pl3-O.3","N.pl3-O.co2","N.pl3-P.3","N.pl3-S.3","N.pl3-S.O","N.pl3-S.O2","N.pl3-S.o","N.pl3-S.o2","N.pl3-Si","O-O","O.2-O.2","O.2-O.3","O.2-P.3","O.2-S.2","O.2-S.3","O.2-S.O","O.2-S.O2","O.2-S.o","O.2-S.o2","O.3-O.3","O.3-P.3","O.3-S.3","O.3-S.O","O.3-S.O2","O.3-S.o","O.3-S.o2","O.3-Si","O.co2-P.3","O.co2-S.3","O.co2-S.O2","O.co2-S.o2","P.3-S.2","P.3-S.3","S.2-S.2","S.3-S.3","S.3-S.O2","S.3-Si","symmetry"]]
#X_test=test_data[["B-C.3","B-C.ar","B-O.3","Br-C.1","Br-C.2","Br-C.3","Br-C.ar","Br-H","Br-N.4","Br-N.am","Br-N.ar","Br-N.pl3","C-C.2","C-C.3","C-C.ar","C-H","C-N.4","C-N.am","C-N.pl3","C.1-C.1","C.1-C.2","C.1-C.3","C.1-C.ar","C.1-Cl","C.1-F","C.1-H","C.1-N.1","C.1-N.2","C.1-N.3","C.1-N.am","C.1-N.ar","C.1-N.pl3","C.1-O.2","C.1-O.3","C.1-P.3","C.1-S.2","C.1-S.3","C.1-S.O","C.1-S.O2","C.1-Si","C.2-C.2","C.2-C.3","C.2-C.ar","C.2-C.cat","C.2-Cl","C.2-F","C.2-Fe","C.2-H","C.2-I","C.2-N.1","C.2-N.2","C.2-N.3","C.2-N.4","C.2-N.am","C.2-N.ar","C.2-N.pl3","C.2-O.2","C.2-O.3","C.2-O.co2","C.2-P.3","C.2-S.2","C.2-S.3","C.2-S.O","C.2-S.O2","C.2-S.o","C.2-S.o2","C.2-Si","C.3-C.3","C.3-C.ar","C.3-C.cat","C.3-Cl","C.3-F","C.3-Fe","C.3-H","C.3-I","C.3-N.1","C.3-N.2","C.3-N.3","C.3-N.4","C.3-N.am","C.3-N.ar","C.3-N.pl3","C.3-O","C.3-O.2","C.3-O.3","C.3-P.3","C.3-S.3","C.3-S.O","C.3-S.O2","C.3-S.o","C.3-S.o2","C.3-Si","C.ar-C.ar","C.ar-C.cat","C.ar-Cl","C.ar-F","C.ar-H","C.ar-I","C.ar-N.1","C.ar-N.2","C.ar-N.3","C.ar-N.4","C.ar-N.am","C.ar-N.ar","C.ar-N.pl3","C.ar-O.2","C.ar-O.3","C.ar-O.co2","C.ar-P.3","C.ar-S.2","C.ar-S.3","C.ar-S.O","C.ar-S.O2","C.ar-S.o","C.ar-S.o2","C.ar-Si","C.cat-N.2","C.cat-N.am","C.cat-N.pl3","C.cat-O.3","C.cat-S.3","Cl-N.3","Cl-N.am","Cl-N.ar","Cl-N.pl3","Cl-O.3","F-N.3","F-N.am","F-N.ar","F-N.pl3","F-O.3","F-S.3","H-N.2","H-N.3","H-N.4","H-N.am","H-N.ar","H-N.pl3","H-O","H-O.2","H-O.3","H-P.3","H-S.3","H-S.O","H-S.O2","H-Si","I-N.3","I-N.am","I-N.ar","N.1-N.1","N.1-N.2","N.1-N.pl3","N.2-N.2","N.2-N.3","N.2-N.4","N.2-N.am","N.2-N.ar","N.2-N.pl3","N.2-O.2","N.2-O.3","N.2-O.co2","N.2-P.3","N.2-S.3","N.2-S.O","N.2-S.O2","N.2-S.o2","N.3-N.3","N.3-N.4","N.3-N.am","N.3-N.ar","N.3-N.pl3","N.3-O.2","N.3-O.3","N.3-O.co2","N.3-P.3","N.3-S.3","N.3-S.O","N.3-S.O2","N.3-S.o2","N.3-Si","N.4-N.4","N.4-N.am","N.4-N.pl3","N.4-O.3","N.4-O.co2","N.4-P.3","N.4-S.O2","N.4-Si","N.am-N.am","N.am-N.ar","N.am-N.pl3","N.am-O.2","N.am-O.3","N.am-O.co2","N.am-P.3","N.am-S.3","N.am-S.O","N.am-S.O2","N.am-S.o2","N.am-Si","N.ar-N.ar","N.ar-N.pl3","N.ar-O.2","N.ar-O.3","N.ar-O.co2","N.ar-P.3","N.ar-S.2","N.ar-S.3","N.ar-S.O2","N.ar-S.o2","N.ar-Si","N.pl3-N.pl3","N.pl3-O.2","N.pl3-O.3","N.pl3-O.co2","N.pl3-P.3","N.pl3-S.3","N.pl3-S.O","N.pl3-S.O2","N.pl3-S.o","N.pl3-S.o2","N.pl3-Si","O-O","O.2-O.2","O.2-O.3","O.2-P.3","O.2-S.2","O.2-S.3","O.2-S.O","O.2-S.O2","O.2-S.o","O.2-S.o2","O.3-O.3","O.3-P.3","O.3-S.3","O.3-S.O","O.3-S.O2","O.3-S.o","O.3-S.o2","O.3-Si","O.co2-P.3","O.co2-S.3","O.co2-S.O2","O.co2-S.o2","P.3-S.2","P.3-S.3","S.2-S.2","S.3-S.3","S.3-S.O2","S.3-Si"]]
y_test=test_data["entropy"]

from sklearn.svm import SVR
estimator=SVR(kernel='rbf', C=10000, epsilon=0.1)
#estimator.fit(X_train,y_train)

import joblib
#joblib.dump(estimator,"/4T/projects/entropy/SVR_V2/sv_bond.pkl")
estimator=joblib.load("/4T/projects/entropy/SVR_V2/sv_bond.pkl")

data_test=pd.DataFrame(y_test)
y_pre=estimator.predict(X_test)
data_test["Pre"]=y_pre
data_test.to_csv("./sv_bond_results_std6074.csv")

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
r2=r2_score(y_test, y_pre)
mae = mean_absolute_error(y_true=y_test, y_pred=y_pre)
mse = mean_squared_error(y_true=y_test, y_pred=y_pre)
rmse = mean_squared_error(y_true=y_test, y_pred=y_pre,squared=False)
print("R2:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

def mean_relative_error_alt(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.float64)
    if np.any(y_true == 0):
        raise ValueError("y_true contains zero values, cannot compute relative error.")
    return np.mean(abs(y_true - y_pred) / y_true)
MRE=mean_relative_error_alt(y_true=y_test, y_pred=y_pre)
print("MRE:",MRE)

from scipy.stats import pearsonr
corr, _ = pearsonr(y_test, y_pre)
print('Pearsons correlation: %.4f' % corr)


import matplotlib.pyplot as plt
plt.scatter(y_test,y_pre)
plt.show()
