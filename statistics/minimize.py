import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.stats as st

def mean(array):
    return array.sum()/array.size

def biased_var(array):
    return np.square(array).sum()/array.size-np.square(mean(array))

def f(x, sample):
    numzero=(sample==0).sum()

    numone = (sample == 1).sum()
    numtwo =(sample == 2).sum()
    numthree=(sample == 3).sum()
    return (((2*x)**numzero)*(x**numone)*((2*(1-x))**numtwo)*((1-x)**numthree))/(3**sample.size)

def f1(x, sample):
    S=0
    for i in sample:
        S=S+(x-i)
    return S
def interval(n, Q, S):
    upperbound=round(n*S/st.chi2.ppf(1-Q, n-1), 3)
    lowerbound=round(n*S/st.chi2.ppf(Q, n-1), 3)
    return (lowerbound, upperbound)

sample1=np.sort(np.loadtxt("r3z1.csv", delimiter=",", dtype=np.str0)[1:].astype("float32"))
print(sample1)


solution = opt.minimize_scalar(lambda x: -f(x, sample1), bounds=[0,1], method='bounded')

print(round(solution.x, 4))
x = np.arange(-100, np.min(sample1), 0.0001)
plt.plot(x, f1(x, sample1))
plt.plot(x[-1], f1(x, sample1)[-1], "*")
plt.show()


sample2=np.sort(np.loadtxt("r3z2.csv", delimiter=",", dtype=np.str0)[1:].astype("float32"))
Q=0.975
print(mean(sample2))
print(sample2.size)
print(round(biased_var(sample2), 2))
r=interval(sample2.size, Q, biased_var(sample2))
print("("+str(r[0])+", "+str(r[1])+")")
print("("+str(round(np.sqrt(r[0]), 2))+", "+str(round(np.sqrt(r[1]), 2))+")")