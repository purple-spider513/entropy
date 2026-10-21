import pandas as pd
import numpy as np
test_data=pd.read_csv("./hybrid_pdbbind.csv")
#X_test=test_data[["ring","heavy","H","s_heavy_bond","MW","B","Br","C.1","C.2","C.3","C.ar","C.cat","Cl","F","Fe","I","N.1","N.2","N.3","N.4","N.am","N.ar","N.pl3","O.2","O.3","O.co2","P.3","S.2","S.3","S.O","S.O2","S.o","S.o2","Si","symmetry"]]
X_test=test_data[["H","B","Br","C.1","C.2","C.3","C.ar","C.cat","Cl","F","Fe","I","N.1","N.2","N.3","N.4","N.am","N.ar","N.pl3","O.2","O.3","O.co2","P.3","S.2","S.3","S.O","S.O2","S.o","S.o2","Si"]]
y_test=test_data["entropy"]

from sklearn.svm import SVR
estimator=SVR(kernel='rbf', C=10000, epsilon=0.1)
#estimator.fit(X_train,y_train)

import joblib
#joblib.dump(estimator,"/4T/projects/entropy/SVR_V2/sv_atom.pkl")
estimator=joblib.load("/4T/projects/entropy/SVR_V2/sv_atom.pkl")

data_test=pd.DataFrame(y_test)
y_pre=estimator.predict(X_test)
data_test["Pre"]=y_pre
data_test.to_csv("./sv_atom_results_stdpdbbind.csv")

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
