import numpy as np

x=np.array([1,2,3,4],dtype=float)
y=np.array([0,-1,-2,-3],dtype=float)

ap,bp = np.polyfit(x,y,1)
yp = np.polyval([ap,bp], 1.5)

print([ap,bp])
print(yp)

x1 = np.vstack([ x, [1.0]*4])
xA = np.dot(x1, x1.T)
yA = np.dot(x1, np.matrix(y).T)
xinv = np.linalg.inv(xA)
coefM = np.dot(xinv , yA)

ym = np.dot([1.5,1] , coefM)

print(coefM.T)
print(ym)
