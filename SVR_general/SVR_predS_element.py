import pandas as pd
import numpy as np
data=pd.read_csv("./hybrid_train_091926.csv")
#X=data[["ring","heavy","s_heavy_bond","MW","B","Br","C","Cl","F","Fe","H","I","N","O","P.3","S","Si","symmetry"]]
X=data[["B","Br","C","Cl","F","Fe","H","I","N","O","P.3","S","Si"]]
y=data["entropy"]

#from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#from sklearn.preprocessing import StandardScaler
#transfer=StandardScaler()
#X_train=transfer.fit_transform(X_train)
#X_test=transfer.fit_transform(X_test)

#estimator=MLPRegressor(hidden_layer_sizes=256,random_state=1, max_iter=1000, tol=0.1)i
#estimator=LinearRegression()

from sklearn.svm import SVR
estimator=SVR(kernel='rbf', C=10000, epsilon=0.1)
estimator.fit(X_train,y_train)

import joblib
joblib.dump(estimator,"/4T/projects/entropy/SVR_V2/sv_element.pkl")
estimator=joblib.load("/4T/projects/entropy/SVR_V2/sv_element.pkl")

data_test=pd.DataFrame(y_test)
y_pre=estimator.predict(X_test)
data_test["Pre"]=y_pre
data_test.to_csv("./sv_element_results.csv")

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_pre)
print(r2)

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_true=y_test, y_pred=y_pre)
mse = mean_squared_error(y_true=y_test, y_pred=y_pre)
rmse = mean_squared_error(y_true=y_test, y_pred=y_pre,squared=False)
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
