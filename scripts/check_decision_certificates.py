"""Independent exact-integer certificates for the new manuscript examples."""
from fractions import Fraction
from itertools import product
import json
from src.routing_geometry import balanced_partitions

N=[[8,11,4],[8,5,10],[10,9,4],[5,11,7],[6,10,7],[11,5,7]]
N0=[[40,24,28],[24,44,24],[32,28,32],[43,37,12],[24,20,48],[37,31,24]]
M=[[2,1,7],[4,4,2],[5,2,3]]

def determinant(x):
    a,b,c=x
    return a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])

def costs(x):
    # Integer pair sums divided exactly by group size; no floating distance code.
    return sorted((sum((Fraction(sum((x[i][k]-x[j][k])**2 for k in range(len(x[0]))),len(g))
        for g in groups for pos,i in enumerate(g) for j in g[pos+1:]),Fraction()),groups)
        for groups in balanced_partitions(len(x),len(x[0])))

def assignments(x):
    k=len(x[0]);m=len(x)//k
    return sorted(((sum(x[i][j] for i,j in enumerate(labels)),labels)
        for labels in product(range(k),repeat=len(x)) if all(labels.count(j)==m for j in range(k))),reverse=True)

def main():
    assert determinant(N[:3])==276 and determinant(N0[:3])!=0 and determinant(M)==-70
    c=assignments(N);assert [s for s,_ in c[:2]]==[60,59]
    assert c[0][1]==(1,2,0,1,2,0)
    for row,j in zip(N,c[0][1]):
        priced=[Fraction(v)-price for v,price in zip(row,[1,Fraction(7,2),0])]
        assert all(priced[j]>priced[t] for t in range(3) if t!=j)
    c0=assignments(N0);assert [s for s,_ in c0[:3]]==[238,238,228]
    assert {labels for _,labels in c0[:2]}=={(0,1,2,1,2,0),(0,1,2,0,2,1)}
    anisotropic=[[4*r[0],r[1],r[2]] for r in N]
    transformed=[[sum(r[k]*M[k][j] for k in range(3)) for j in range(3)] for r in N]
    expected=[(N,[14,31]),(anisotropic,[65,119]),(transformed,[200,212]),(N0,[506,578])]
    certificate={}
    for i,(x,best) in enumerate(expected):
        ranks=costs(x);assert [v for v,_ in ranks[:2]]==best
        certificate[str(i)]={'best':int(ranks[0][0]),'runner_up':int(ranks[1][0]),'partition':ranks[0][1]}
    n4=[[16*v+23 for v in row]+[23] for row in N]+[[23,23,23,391]]*2
    a4=assignments(n4);w4=costs(n4)
    assert a4[0][0]>a4[1][0] and w4[0][0]<w4[1][0]
    assert w4[0][1]==((0,2),(1,5),(3,4),(6,7))
    assert a4[0][1]==(1,2,0,1,2,0,3,3)
    print(json.dumps({'exact_integer_certificates':certificate,'four_basis_extension':'passed','boundary_dual_and_rank_checks':'passed'}))

if __name__=='__main__':main()
