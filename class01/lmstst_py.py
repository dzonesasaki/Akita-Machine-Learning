
x=[1,2,3,4]
y=[0,-1,-2,-3]

a=0.3
b=-0.3

rateA=0.01
rateB=0.01
Nloop=10000
Nsmp=len(x)

for k in range(Nloop):
	errs = a*x[k%Nsmp] + b - y[k%Nsmp]
	a = a - rateA*errs*x[k%Nsmp]
	b = b - rateB*errs

print(a)
print(b)


