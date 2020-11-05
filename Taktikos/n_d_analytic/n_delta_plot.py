import math
import numpy


def c_r(v):
    return 1.0/(10.0**(v/4.0))

crange = [c_r(i) for i in range(4,27)]

print(crange)
out = open("n_delta_t.out","w")

for v in crange:
    c = v
    def prod(list):
        res = 1.0
        for i in list:
            res = res*i
        return res
    def p_perp(j): 
        return [(1.0-c*i) for i in range(0,j+1)]
    def prod_p_perp(j):
        return prod(p_perp(j))
    def n_d(d):
        return c*d*prod_p_perp(d-1)
    def n_d_a(d):
        return c*d*numpy.exp((d-1/c)*numpy.log(1-c*d)-d)
    out2 = open("n_delta_analytic_"+str(c)+".out","w")
    res = 0.0
    i = 0
    done = 0
    oldNew = 0.0
    out2.write(str(i)+" "+str(0.0)+"\n")
    while (done < 1):
        i = i + 1
        nd = n_d(i)
        new = i * nd
        if (new < oldNew and new < 0.0001):
            done = 1
        res = res + new
        oldNew = new
        out2.write(str(i)+" "+str(nd)+"\n")
    out2.close()
    out.write(str(v)+" "+str(res)+"\n")
    
out.close()
