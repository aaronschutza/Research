import math
import numpy


def c_r(v):
    return 1.0/(10.0**(v/4.0))

crange = [c_r(i) for i in range(7,14)]

gamma = 12
fa = 0.5
crange = [fa/gamma]
fb = 0.01


def n_d(d,r,c):
    def f(x):
        if x < gamma+1:
            return c*x
        else:
            return fb
    def prod(list):
        res = 1.0
        for i in list:
            res = res*i
        return res
    def p_perp(j): 
        return [numpy.power(1.0-f(i),r) for i in range(0,j+1)]
    def prod_p_perp(j):
        return prod(p_perp(j))
    return (1.0-numpy.power(1-f(d),r))*prod_p_perp(d-1)

print(crange)

for cv in crange:
    out2 = open("Fr"+"_"+str(fb)+".out","w")
    fr = []
    rlen = 41
    raxis = numpy.linspace(0.0,1.0,rlen)
    for rv in raxis:
        res = 0.0
        print("working on "+str(rv))
        if rv>0.0:
            i = 0
            done = 0
            oldNew = 0.0
            while (done < 1):
                i = i + 1
                nd = n_d(i,rv,cv)
                new = i * nd
                if (nd < 0.0001):
                    done = 1
                res = res + new
                oldNew = new
            fr.append(1.0/res)
            print(i)
        else:
            fr.append(0.0)
    i = 0
    for rv in raxis:
        if (rv>0.0):
            out2.write(str(rv)+" "+str(fr[i]/(fr[i]+fr[rlen-1-i]))+"\n")
            #out2.write(str(rv)+" "+str(fr[i])+"\n")
        else:
            out2.write(str(rv)+" "+str(0.0)+"\n")
        i = i + 1
    out2.close()

