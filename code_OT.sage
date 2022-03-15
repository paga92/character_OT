QQq= QQ['q'].fraction_field()
q = QQq.gens()[0]
Symq = SymmetricFunctions(QQq)
p = Symq.powersum()
s = Symq.schur()
h = Symq.homogeneous()
e = Symq.elementary()

## part of the code is code adapted from https://github.com/benyoung/ot

## charcter of Ind_{Z_n}^{S_n} \zeta_n (usally denoted by l_n)
def Lie(n):
    n = Integer(n)
    return s(sum(moebius(d) * p([d])**(n//d) for d in n.divisors())/n)

## mu=(1^{m_1}, 2^{m_2}, ...) a partition. Return m_i
def multiplicity(mu, i):
    return mu.to_exp()[i-1]

## graded character of H^2*(Conf_n(R^3))
def C(n):
    result=Symq(0)
    for llambda in Partitions(n):
        result+= q**(n-len(llambda)) * prod([h([multiplicity(llambda, i)]).plethysm(Lie(i)) for i in range(1, max(llambda)+1)])
    return result

def decompose_partition_into_rectangles(p):
    return [(i, list(p).count(i)) for i in reversed(range(max(p)+1)) if i in p]       

# M = (m1, m2, ... )
# this probably doesn't need to be its own function but whatever
def iterate_over_partition_tuples(M):
    return xmrange_iter([Partitions(u) for u in M])

## return the hook partition (n-i,1^i) assuming 0<=i<n
def hook(n,i):
    return [n-i]+[1]*i

## used in D
def ch_prime_for_D(mj, j):
    return sum([(-q)**k * s(hook(mj,k)).plethysm(Lie(j))  for k in range(mj)])

## compute the character of D_n as in Remark 3.2 [Pagaria, The Frobenius character of the Orlik-Terao algebra of type A]
def D(n):
    result = Symq(0)
    for llambda in Partitions(n):
        summand=Symq(1)
        for j in range(len(llambda.to_exp())):
            if multiplicity(llambda, j+1)>0:
                summand*= ch_prime_for_D(multiplicity(llambda, j+1), j+1)
        result+=q**(n-len(llambda)) * (1-q)**(len([i for i in llambda.to_exp() if i!=0])-1) * summand
    return result

# return the character of R_n = Sym(V_{n-1,1})
def R(n, degree_bound=9):
    result = 0
    for llambda in Partitions(n):
        coeff = QQq(1)
        for (i, part) in enumerate(llambda):
            coeff *= q**(i * part)
        for (i,j) in llambda.cells():
            coeff *= (1-q**(degree_bound + j - i))
            hook = llambda.hook_length(i,j)
            coeff *= sum(q**(t*hook) for t in range(int(degree_bound/hook + 2)))
        coeff = ((1-q)*coeff).numerator().truncate(degree_bound)  
        result += coeff * s(llambda)
    return result

# assuming that the denominator is 1!
def trunc_coef(c, degree_bound=9):
    l=c.numerator().list()
    return sum( q**i *l[i] for i in range(min(degree_bound,len(l))))
    
def truncate(f, degree_bound=9):
    return sum(trunc_coef(cf, degree_bound)*s(mu) for (mu, cf) in Symq.schur()(f))

# assuming that the denominator is 1!
def extract_coeff(f, i):
    return sum((cf.numerator().list()[i])*s(mu) for (mu, cf) in Symq.schur()(f))

# Dual space of degree N
def dual(f, N):
    if f==0:
        return 0
    else:
        return q**N *sum(cf.subs(q=q**(-1))*s(mu) for (mu, cf) in Symq.schur()(f))

# compute the character of OT as in Corollary 4.6 [Pagaria, The Frobenius character of the Orlik-Terao algebra of type A]
def OT2(n, degree_bound=9):
    result = Symq(0)
    for llambda in Partitions(n):
        summand=Symq(1)
        for j in range(len(llambda.to_exp())):
            if multiplicity(llambda, j+1)>0:
                summand*= h([multiplicity(llambda, j+1)]).plethysm(R(j+1, degree_bound).internal_product(Lie(j+1)))
        result+=q**(n-len(llambda)) * summand
    return truncate(result, degree_bound)

# compute the character of T_n using as definition Theorem 4.5 of [Pagaria, The Frobenius character of the Orlik-Terao algebra of type A]
def T(n, degree_bound=9):
    return truncate(q**(n-1) * R(n, degree_bound).internal_product(Lie(n)), degree_bound)

## computing M_n, M^c_n and OT_n following [Mosely, Proudfoot, Young The Orlik-Terao algebra and the cohomology of configuration space]
## code adapted from https://github.com/benyoung/ot
memoize_M = {
    1: s([1]),
    #2: s([2]),
    #3: s([3]) + q*s([1,1,1]),
    #4: s([4]) + q*s([2,1,1]) + q^2 * s([2,2]),
}

def M(n):
    if n not in memoize_M:
        result = Symq(0)
        for i in range(n-1):
            result += q**i * M_coeff_from_OT(n, i)
        memoize_M[n] = result
#        print(n)
    return memoize_M[n]

def M_compact_supp(n):
    return dual(M(n),2*(n-1))

def M_coeff_from_OT(n, i):
    if n == 1:
        if i==0: 
            return s([1])
        else:
            return Symq(0)
    elif i > n-2:
        return Symq(0)
    else:
        result = extract_coeff(low_order_OT(n, degree_bound=i+1), i)
        for k in range(1, i+1):
            left_piece = M_coeff_from_OT(n, i-k)
            right_piece = extract_coeff(R(n), k)
            result -= left_piece.inner_tensor(right_piece)
        return result

def addendum_lambda_OT(llambda, degree_bound=9):
    result = Symq(0)
    L = decompose_partition_into_rectangles(llambda)
    M = [u[1] for u in L]   
    for partition_list in iterate_over_partition_tuples(M):
        reference_rep = prod([s(mu) for mu in partition_list])
        term = reference_rep.scalar(C(len(llambda)))
        for i in range(len(partition_list)):
            mu = partition_list[i]
            r_i = L[i][0]
            left_piece = M_compact_supp(r_i)
            right_piece = R(r_i, degree_bound)
            term *= s(mu).plethysm(left_piece.inner_tensor(right_piece))
        result += term
    return result

## do NOT return the low degree part of OT_n: it misses the addendum associated with the partition [n]
def low_order_OT(n, degree_bound=9):
    result = Symq(0)
    for llambda in Partitions(n):
        if llambda != Partition([n]):
            result += addendum_lambda_OT(llambda, degree_bound)
    return result

# return the character of OT_n computed recursevely as described in [Mosely, Proudfoot, Young The Orlik-Terao algebra and the cohomology of configuration space]
def OT(n, degree_bound=9):
    return truncate(low_order_OT(n, degree_bound)+addendum_lambda_OT(Partition([n]), degree_bound), degree_bound)
#    return truncate(R(n, degree_bound).internal_product(M(n)),degree_bound)

#some checking functions
def check_OT_OT2(n, degree_bound=9):
    return OT(n,degree_bound)==OT2(n,degree_bound)

def check_C_D(n):
    return C(n)== D(n).internal_product(s([n])+q *s([n-1,1]))

def check_D_M(n):
    return M(n)== D(n)

print(check_OT_OT2(4, degree_bound=13))
print(check_C_D(5))
print(check_D_M(4))
print(D(10))
print(OT2(8,degree_bound=20))
print(T(8,degree_bound=20))