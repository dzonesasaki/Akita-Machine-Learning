import numpy as np
from sklearn import linear_model
from  sklearn.ensemble import RandomForestClassifier

x=np.array([1,2,3,4],dtype=float).reshape(-1,1)
y=np.array([0,-1,-2,-3],dtype=float).reshape(-1,1)

modelLa = linear_model.Lasso(alpha=0.1)
modelLa.fit(x,y)
ypLa = modelLa.predict(1.5)

print(ypLa)


modelRg = linear_model.Ridge(alpha=0.5)
modelRg.fit(x,y)
ypRg = modelRg.predict(1.5)

print(ypRg)


modelRf = RandomForestClassifier()
modelRf.fit(x,np.ravel(y))
ypRf = modelRf.predict(1.5)

print(ypRf)
