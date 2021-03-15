---

layout: post

title:  "Inequality Solving with CVP"

date: 2021-03-12

author: rkm0959

tags: [ctf, cryptography]

---


# 서론

이 글은 제가 CTF 대회에서 자주 사용하는 도구인 https://github.com/rkm0959/Inequality_Solving_with_CVP 를 위한 설명서입니다. 이미 README에 많은 설명이 있지만, 설명서라기에는 부실한 점이 많아 이를 보강합니다.

제가 이 글을 작성하는 목적은

- 이 repository의 존재를 더 많은 사람들에게 홍보하기 위해서
- 이 repository의 기본적인 아이디어를 CTF를 하지 않는 사람들에게도 알리기 위해서
- 이 repository가 더 많은 사람들에게 사용되는 바람이 있기 때문에
- 이 repository의 README에 부족한 점을 보충 및 repository 자체 강화

입니다. 이 점을 참고하시고 글을 읽어주시면 감사하겠습니다.

이 글에서 모든 나눗셈은 floor function을 거친 결과로 생각하시기 바랍니다.

# 본론 


## CVP란 무엇인가

우선 CVP가 무엇인지 알아야 이 도구를 이해할 수 있습니다. 이는 rbtree님의 글 http://www.secmem.org/blog/2020/10/23/SVP-and-CVP/ 에서 잘 설명되어 있지만, 다시 한 번 설명합니다.

선형독립인 $n$차원 벡터 $v_1, \cdots , v_d$가 있다고 합시다. 이때, 

$$L = \left\{ \sum_{i=1}^d a_i v_i | a_i \in \mathbb{Z} \right\}$$

라는 집합을 생각해봅시다. 즉, 각 벡터들의 정수 계수 선형결합을 모두 모아놓은 집합을 생각합니다. 이를 Lattice라고 부릅니다. 

Lattice와 관련된 중요한 문제 중 하나가 바로 CVP, Closest Vector Problem으로, 목표 벡터 $v$가 주어졌을 때 $v$와 가장 가까우면서 $L$에 속한 벡터를 찾는 문제입니다. NP-hard 문제 중 하나지만, 가장 가까운 벡터를 찾는 게 아니라 "꽤 가까운 벡터를 찾는 것", 즉 문제를 근사적으로 해결하는 것은 어렵지 않은 문제입니다.  **Babai's Nearest Plane Algorithm**으로 할 수 있기 때문입니다. 자세한 결과는 생략하겠습니다.

이제부터 Babai의 알고리즘의 강력한 힘을 믿으며 논의를 진행하겠습니다.


## 부등식에서 CVP로 

$n$개의 변수와 $m$개의 부등식이 있다고 가정하겠습니다. 각 변수의 이름을 $x_1, x_2, \cdots, x_n$이라 하고, 부등식의 형태는 모두 

$$lb_j \le \sum_{i=1}^n a_{ij} x_i \le ub_j$$

형태라고 가정하겠습니다. 단, $x_i$, $lb_j, ub_j$ 및 $a_{ij}$ 들은 모두 정수입니다. 

잠시 $a_i \le b_i$가 모든 $i$에 대해서 성립할 경우 $(a_1, \cdots , a_n) \preceq (b_1, \cdots , b_n)$이라고 정의합시다. 그러면 우리가 해결하는 문제는 

$$v_i = (a_{i1}, a_{i2}, \cdots, a_{im})$$

이라 했을 때 

$$lb = (lb_1, \cdots , lb_m) \preceq \sum_{i=1}^n x_i v_i \preceq (ub_1, \cdots , ub_m) = ub$$

라고 쓸 수 있습니다. 이제 $\{v_i\}$들로 생성되는 Lattice

$$L = \left\{ \sum_{i=1}^n a_i v_i | a_i \in \mathbb{Z} \right\}$$

를 생각하면, 결국 우리가 푸는 문제는 $lb$와 $ub$ 사이에 들어오는 $L$의 원소를 찾는 문제가 됩니다. 이러한 문제를 CVP로 바꾸는 가장 직관적인 방법은

- $lb$와 $ub$의 가운데에 있는 벡터 $v_{target} = (lb + ub) / 2$를 생각한 다음
- 이 벡터를 목표 벡터로 하는 CVP를 해결

하는 것입니다. 성공했다면, 적당한 $v_{result} = \sum_{i=1}^n x_i v_i \in L$을 얻을 수 있습니다.

이제 $x_i$를 복구하는 것이 문제인데, 이에 대해서는 다양한 접근이 가능합니다.

- 많은 경우에서 $lb_j \le x_i \le ub_j$ 형태의 부등식이 사용되고, 이때 $v_{result}$의 $j$번째 항이 $x_i$가 됩니다.
- 많은 경우에서 $n \le m$이고 $v_i$들이 선형독립입니다. 이때는 $v_{result} = \sum_{i=1}^n x_i v_i$를 system of linear equations로 보고 직접 풀 수 있습니다.


## 가중치의 사용

위 논의에서 가장 큰 논리의 구멍은 

$$lb \preceq \sum_{i=1}^n x_iv_i \preceq ub$$

라는 요구조건을 

$$ \sum_{i=1}^n x_i v_i \approx v_{target} = (lb+ub)/2$$

로 바꾼 것에 있습니다. 여기에는 사실 허점이 있는데, 아래의 극단적인 예시를 보면 더욱 명확해집니다.

$lb = (0, 0)$, $ub = (10^{300}, 2)$이 있다고 합시다. 우리의 새로운 요구조건은 $(10^{300}/2, 1)$에 가까운 벡터를 만드는 것입니다. 그렇다면, CVP 알고리즘은 두 번째 entry가 $0$ 이상 $2$ 이하여야 한다는 기존의 요구조건을 무시하고, 첫 번째 entry가 $10^{300}/2$에 최대한 가까워지도록 노력할 것입니다. CVP는 각 entry의 대소를 중요하게 보는 것이 아니라, 두 벡터의 차이의 크기만을 보기 때문에 이러한 문제가 발생하게 되는 것으로 해석할 수 있습니다.

다르게 생각하면, 이는 $lb, ub$ 벡터들의 entry의 스케일이 다르기 때문에 발생하는 문제라고도 볼 수 있습니다. $ub_i - lb_i$의 값이 전체적으로 비슷하다면 위와 같은 문제가 발생할 가능성이 줄어듭니다. 그러니 이 값이 비슷하도록 **가중치**를 부여합시다.

$$M = \max_{1 \le j \le m}\{ub_j - lb_j\}$$

라고 정의합시다. 이제 $j$번째 부등식이 $lb_j \neq ub_j$를 만족한다면, 그에 대한 가중치를 $w_j = M/(ub_j - lb_j)$로 둡시다. 즉,

$$lb_j \le \sum_{i=1}^n a_{ij} x_i \le ub_j \implies w_j \cdot lb_j \le \sum_{i=1}^n (a_{ij} w_j) \cdot x_i \le w_j \cdot ub_j$$

와 같이 식을 변형해줍시다. 만약 $lb_j = ub_j$라면 어떻게 할까요? 이 경우에는 $w_j$를 매우 큰 정수로 잡아주면 됩니다.  

특히, $lb_j = ub_j$인 경우에 큰 가중치를 잡으면 된다는 것은 직접적으로 설명할 수 있습니다. 큰 가중치 $w_j$를 걸어

$$ w_j \cdot lb_j \le \sum_{i=1}^n (a_{ij} w_j) \cdot x_i \le w_j \cdot ub_j $$

를 얻었다고 합시다. 만약 $\sum_{i=1}^n (a_{ij}w_j) \cdot x_i$가 $w_j \cdot lb_j =  w_j \cdot ub_j$와 달랐다면, 그 차이는

$$ \left| \sum_{i=1}^n (a_{ij} w_j) x_i - w_j (lb_j + ub_j) / 2 \right| \ge w_j $$

가 됩니다. 즉, **CVP 결과가 $j$번째 부등식을 만족시키지 못한다면 차이 벡터의 크기가 $w_j$ 이상**이라는 결론을 얻습니다. CVP는 차이 벡터의 크기를 최소한으로 만들고자 하는데, $j$번째 부등식을 만족시키지 못한다면 매우 큰 크기의 벡터가 나오는 게 강제되므로, CVP는 $j$번째 부등식을 만족시킬 수 밖에 없습니다. 더욱 정확한 논의를 위해서는 CVP 문제와 determinant의 관계 등 더 많은 정보가 필요하지만, 직관적으로는 이미 충분히 납득할 수 있을 것이라고 생각합니다. 이렇게 가중치까지 걸어주면, 모든 준비가 끝납니다.

## 휴리스틱 

CTF와 암호 체계를 깨는 context에서 중요한 또 다른 점은, 주어진 식을 만족하는 $x_i$들이 유일성입니다. 이를 증명하는 것은 어려운 일이지만, 휴리스틱을 통해서 해의 개수를 근사하는 것은 어렵지 않습니다. $n = m$과 $v_i$들의 선형독립성을 가정하겠습니다. CTF 문제풀이에서는 많은 경우에서 성립합니다.

$$lb \preceq \sum_{i=1}^n x_i v_i \preceq ub$$

에서 시작합시다. $lb$와 $ub$ 사이에 있는 벡터들이 이루는 영역의 부피는 

$$ \prod_{i=1}^n (ub_i - lb_i)$$

라고 쓸 수 있습니다. 또한, $\sum_{i=1}^n x_i v_i$ 형태의 선형결합에 대응되는 부피는 $B$를 $[v_1 | v_2 | \cdots , | v_n]$이라 했을 때 $|\det(B)|$입니다. 그러니, 답이 되는 $(x_1, \cdots, x_n)$의 개수는 대강

$$ \prod_{i=1}^n (ub_i - lb_i) \cdot \frac{1}{|\det(B)|}$$

와 같다고 볼 수 있겠습니다. 이 값이 지나치게 크다면 다른 접근을 시도하는 것이 현명합니다.

$n \neq m$인 경우에도 비슷한 분석을 시도할 수는 있으나, 제가 실제로 이 도구를 사용한 경우에는 항상 $n = m$이었으므로 생략하겠습니다.

## 구현

SageMath 구현체는 다음과 같습니다. rbtree님의 LLL repository를 참고하였습니다.

```python
from sage.modules.free_module_integer import IntegerLattice

# Directly taken from rbtree's LLL repository
# From https://oddcoder.com/LOL-34c3/, https://hackmd.io/@hakatashi/B1OM7HFVI
def Babai_CVP(mat, target):
	M = IntegerLattice(mat, lll_reduce=True).reduced_basis
	G = M.gram_schmidt()[0]
	diff = target
	for i in reversed(range(G.nrows())):
		diff -=  M[i] * ((diff * G[i]) / (G[i] * G[i])).round()
	return target - diff


def solve(mat, lb, ub, weight = None):
	num_var  = mat.nrows()
	num_ineq = mat.ncols()

	max_element = 0 
	for i in range(num_var):
		for j in range(num_ineq):
			max_element = max(max_element, abs(mat[i, j]))

	if weight == None:
		weight = num_ineq * max_element

    # sanity checker
	if len(lb) != num_ineq:
		print("Fail: len(lb) != num_ineq")
		return

	if len(ub) != num_ineq:
		print("Fail: len(ub) != num_ineq")
		return

	for i in range(num_ineq):
		if lb[i] > ub[i]:
			print("Fail: lb[i] > ub[i] at index", i)
			return

    # heuristic for number of solutions
	DET = 0

	if num_var == num_ineq:
		DET = abs(mat.det())
		num_sol = 1
		for i in range(num_ineq):
			num_sol *= (ub[i] - lb[i])
		if DET == 0:
			print("Zero Determinant")
		else:
			num_sol //= DET
			# + 1 added in for the sake of not making it zero...
			print("Expected Number of Solutions : ", num_sol + 1)

	# scaling process begins
	max_diff = max([ub[i] - lb[i] for i in range(num_ineq)])
	applied_weights = []

	for i in range(num_ineq):
		ineq_weight = weight if lb[i] == ub[i] else max_diff // (ub[i] - lb[i])
		applied_weights.append(ineq_weight)
		for j in range(num_var):
			mat[j, i] *= ineq_weight
		lb[i] *= ineq_weight
		ub[i] *= ineq_weight

	# Solve CVP
	target = vector([(lb[i] + ub[i]) // 2 for i in range(num_ineq)])
	result = Babai_CVP(mat, target)

	for i in range(num_ineq):
		if (lb[i] <= result[i] <= ub[i]) == False:
			print("Fail : inequality does not hold after solving")
			break
    
    # recover x
	fin = None

	if DET != 0:
		mat = mat.transpose()
		fin = mat.solve_right(result)
	
	## recover your result
	return result, applied_weights, fin

```

위 코드는 행렬 `mat`, 벡터 `lb, ub`와 자연수 `weight`를 받습니다.

- `mat`은 부등식의 $a_{ij}$에 해당하는 $n \times m$ 행렬입니다.
- `lb, ub`는 말 그대로 $lb, ub$에 해당하는 벡터입니다.
- `weight`는 필요한 경우에만 넘기면 되는 인자로, $lb_i = ub_i$일 때 줄 가중치의 값입니다.

sanity check를 통과하지 못한 경우, 그 이유를 말해줍니다.

- 전달받은 데이터의 차원이 맞지 않는다던가
- $lb \preceq ub$ 조차 성립하지 않는다던가

그 후, $n = m$이면 휴리스틱을 통해 예상되는 해의 개수를 출력합니다.

- 만약 determinant가 0이라면, 이를 알려줍니다.
- 아니라면, 휴리스틱의 공식을 통해서 예상되는 해의 개수를 알려줍니다.
- 이 예상되는 해의 개수가 지나치게 큰 경우, 다른 접근을 시도하는 게 현명합니다.

최종 출력 결과는 세 벡터 `result`, `applied_weights`, `fin`입니다.

- `result`는 CVP의 결과 벡터 $v_{result}$입니다.
- `applied_weights`는 말 그대로 적용된 가중치입니다. 특히, `result`는 가중치가 적용된 벡터이므로, 이에 주의하여 답을 복구해야 합니다.
- `fin`은 $x_i$들의 값을 갖고 있는 벡터로, $x_i$들을 복구하기 위해서 선형방정식을 푼 결과입니다.

## 예시 문제 - SECCON CTF 2020 sharsable

```python
from Crypto.Util.number import getPrime, GCD
from flag import FLAG
import random
 
def egcd(a, b):
    r0, r1 = a, b
    s0, s1 = 1, 0
    t0, t1 = 0, 1
    while r1 > 0:
        q = r0 // r1
        r0, r1 = r1, r0 % r1
        s0, s1 = s1, s0 - q * s1
        t0, t1 = t1, t0 - q * t1
    return s0, t0
 
def generateKey():
    p = getPrime(512)
    q = getPrime(512)
    n = p * q
    phi = (p-1)*(q-1)
 
    while True:
        d1 = getPrime(int(n.bit_length()*0.16))
        e1 = random.randint(1, phi)
        ed1 = e1 * d1 % phi
 
        d2 = getPrime(int(n.bit_length()*0.16))
        e2, k = egcd(d2, phi)
        e2 = e2 * (phi + 1 - ed1) % phi
        ed2 = e2 * d2 % phi
 
        if GCD(e1, e2) > 10:
            break
 
    assert((ed1 + ed2) % phi == 1)
 
    return (n, (e1, d1), (e2, d2))
 
n, A, B = generateKey()
M = int.from_bytes(FLAG, 'big')
C1 = pow(M, A[0], n)
C2 = pow(M, B[0], n)
assert(pow(C1, A[1], n) * pow(C2, B[1], n) % n == M)
 
import json
print(json.dumps({
    "n": n,
    "A": (A[0], C1),
    "B": (B[0], C2),
    #"d": (A[1], B[1]), # for debug
    }))

```

문제를 해결한 당시의 writeup은 https://rkm0959.tistory.com/165 를 참고합시다.

문제에서 주어진 조건은

- $d_1, d_2$는 대략 $n^{0.16}$ 이하로 작은 편
- $e_1d_1 + e_2d_2 \equiv 1 \pmod{\phi(n)}$
- $gcd(e_1, e_2) > 10$이므로 common modulus attack 사용 불가능

문제에서 우리의 목표는 $d_1, d_2$를 복원하는 것입니다. 풀이를 위해서는 먼저

$$ e_1d_1 + e_2d_2 = k \phi(n) + 1 = k(n-p-q+1)+1 \equiv -k(p+q-1)+1 \pmod{n}$$

이 성립함을 확인합시다. 여기서 $k$는 정수이고, $e_1, e_2 < n$이므로 $d_1, d_2 < n^{0.16}$이므로 $k < 2 \cdot n^{0.16}$ 역시 성립합니다. 또한, $p + q < 3\sqrt{n/2}$인 점을 이용하면, 대략

$$ -3 \sqrt{2} n^{0.66} \le (e_1 d_1 + e_2d_2 \pmod{n}) \le 0$$

을 얻습니다. 이는 새로운 변수 $d_3$을 하나 잡으면,

$$ -3 \sqrt{2}n^{0.66} \le e_1d_1 + e_2d_2 + nd_3 \le 0 $$

으로 쓸 수 있습니다. 이 부등식과 이미 아는 부등식

$$ 0 \le d_1, d_2 < n^{0.16}$$

을 합치면, 변수 3개, 부등식 3개를 얻습니다. 이제 위 도구를 사용하여 $d_1, d_2$를 복원하면 됩니다.

```python
# Example Challenge 2 in https://github.com/rkm0959/Inequality_Solving_with_CVP

# find actual data in the repository :)
n  = 
e_1 = 
C_1 = 
e_2 = 
C_2 = 

# build matrix
M = matrix(ZZ, 3, 3)

# encode d_1
M[0, 0] = 1 

# encode d_2
M[1, 1] = 1 

# encode e_1d_1 + e_2d_2 + nd_3
M[0, 2] = e_1
M[1, 2] = e_2
M[2, 2] = n

# build lb/ub
lb = [0, 0, -int(n ** 0.66 * 3 * sqrt(2))]
ub = [int(n ** 0.16), int(n ** 0.16), 0]

# solve system
res, weights, fin = solve(M, lb, ub)

d_1 = int(fin[0])
d_2 = int(fin[1])

val = (pow(C_1, d_1, n) * pow(C_2, d_2, n)) % n
print(long_to_bytes(val))
```

## 특수 케이스

지금까지 우리는 linear inequality의 system을 CVP로 해결했습니다. 하지만 매우 특수한 경우에서는 더욱 효율적이고 강력한 알고리즘이 존재합니다. 이제부터 다룰 문제는 

$$ L \le Ax + My \le R, \quad S \le x \le E $$

형태의 문제입니다. 이는 사실 조금만 생각해보면

$$ L \le Ax \pmod{M}  \le R, \quad S \le x \le E$$

과 동일한 문제임을 알 수 있습니다. 여기서 주의해야 할 점이 있습니다. 

$$ L \le Ax \pmod{M} \le R$$

이라는 부등식의 의미가 생각하는 것과 다를 수 있습니다. 

$0$부터 시작해서 $M-1$까지 갔다가 다시 $0$으로 돌아오는 시계 같은 원을 하나 생각을 합시다. 여기서 위 부등식의 의미는, $Ax \pmod{M}$이 $L$에서 시작해서 시계방향으로 $R$에서 끝나는 "원호" 위에 존재한다는 것입니다. 예를 들어, 충분히 큰 $M$에 대해서 

$$M-2 \le Ax \pmod{M} \le 1$$

의 의미는 $Ax \pmod{M} = M-2 ,M-1, 0, 1$이라는 뜻입니다. 

이 문제의 특별한 점은, 

$$ L \le Ax \pmod{M} \le R$$

을 만족하는 최소의 음이 아닌 정수 $x$를 찾는 것이 빠르게 된다는 점입니다.

그 원리를 여기에 설명하기에는 너무 길어지니, 제 PS 정수론 가이드 https://rkm0959.tistory.com/188 와 Codeforces Good Bye 2014의 G번, NWRRC 2019의 G번을 참고하시기 바랍니다. 

이제 기존의 문제도 해결할 수 있습니다. 먼저 

$$ L \le Ax \pmod{M}  \le R, \quad S \le x \le E$$

가 있으면, $L' \equiv L - AS \pmod{M}$, $R' \equiv R - AS \pmod{M}$을 생각하고

$$ L' \le Ax' \pmod{M} \le R, \quad 0 \le x' \le E-S$$

으로 식을 바꿀 수 있습니다. 이제 

$$ L' \le At \pmod{M} \le R$$

을 만족하는 최소의 음이 아닌 정수 $t$을 찾아줍시다. 이제 풀어야 하는 문제는

$$L' \le Ax' \pmod{M} \le R', \quad t+1 \le x' \le E-S$$

이므로, 다시 $L'' = L' - (t+1)S \pmod{M}$, $R'' = R' - (t+1)S \pmod{M}$이라 하고

$$L'' \le Ax'' \pmod{M} \le R'', \quad 0 \le x'' \le E-S-(t+1)$$

을 만족하는 최소의 음이 아닌 정수를 찾고, 이를 $x$에 대한 허용 구간이 공집합이 될 때까지 반복하면 됩니다.

확률적으로 생각하면, 해의 개수는 대략 

$$ (E-S+1)(R-L+1) / M$$

개 정도 있을 것이라고 생각할 수 있습니다. 


## 구현
python3에서 구현을 했습니다.

```python
from Crypto.Util.number import GCD

def ceil(n, m): # returns ceil(n/m)
	return (n + m - 1) // m

def is_inside(L, R, M, val): # is L <= val <= R in mod M context?
	if L <= R:
		return L <= val <= R
	else:
		R += M
		if L <= val <= R:
			return True
		if L <= val + M <= R:
			return True 
		return False

## some notes : it's good idea to check for gcd(A, M) = 1
def optf(A, M, L, R): # minimum nonnegative x s.t. L <= Ax mod M <= R
	if L == 0:
		return 0
	if 2 * A > M:
		L, R = R, L
		A, L, R = M - A, M - L, M - R
	cc_1 = ceil(L, A)
	if A * cc_1 <= R:
		return cc_1
	cc_2 = optf(A - M % A, A, L % A, R % A)
	return ceil(L + M * cc_2, A)

# check if L <= Ax (mod M) <= R has a solution
def sol_ex(A, M, L, R):
	if L == 0 or L > R:
		return True
	g = GCD(A, M)
	if (L - 1) // g == R // g:
		return False
	return True

## find all solutions for L <= Ax mod M <= R, S <= x <= E:
def solve(A, M, L, R, S, E):
	# this is for estimate only : if very large, might be a bad idea to run this
	print("Expected Number of Solutions : ", ((E - S + 1) * (R - L + 1)) // M + 1)
	if sol_ex(A, M, L, R) == False:
		return []
	cur = S - 1
	ans = []
	num_sol = 0
	while cur <= E:
		NL = (L - A * (cur + 1)) % M
		NR = (R - A * (cur + 1)) % M
		if NL > NR:
			cur += 1
		else:
			val = optf(A, M, NL, NR)
			cur += 1 + val
		if cur <= E:
			ans.append(cur)
			# remove assert for performance if needed
			assert is_inside(L, R, M, (A * cur) % M)
			num_sol += 1
	print("Actual Number of Solutions : ", num_sol)
	return ans
```

solve 함수를 호출하여 원하는 $x$의 list를 얻을 수 있습니다.

만약 예상되는 해의 개수가 지나치게 많다면, 다른 접근을 생각하는 것이 현명합니다.

## 예시 문제 - PBCTF 2020 Special Gift

```python
from Crypto.Util.number import getStrongPrime, inverse, bytes_to_long, GCD as gcd
from Crypto.Random.random import randint
from flag import flag
 
p = getStrongPrime(512)
q = getStrongPrime(512)
N = p * q
phi = (p - 1) * (q - 1)
 
# Hehe, boi
while True:
    d = randint(int(N ** 0.399), int(N ** 0.4))
    if gcd(d, phi) == 1:
        break
 
e = inverse(d, phi)
 
# Here's a special gift. Big.
gift = d >> 120
 
enc = pow(bytes_to_long(flag), e, N)
 
print("N =", N)
print("e =", e)
print("gift =", gift)
print("enc =", enc)
```

첫 예시 문제와 동일한 식

$$ ed =  k \phi(n) + 1 = k(n-p-q+1)+1 \equiv -k(p+q-1)+1 \pmod{n}$$

를 생각합시다. 이번에는 $d < n^{0.4}$이므로 $k < n^{0.4}$이고, $p+q < 3 \sqrt{n/2}$를 이용하면 대략

$$ -3 \cdot n^{0.9} \le ed \pmod{n} \le 0$$

를 얻습니다. 좌변의 상수 $3$은 매우 넉넉하게 잡은 것입니다.

그런데 $gift = \lfloor d / 2^{120} \rfloor$를 알고 있으니, $d = gift \cdot 2^{120} + c$라고 쓸 수 있습니다. 이제

$$ (-3 \cdot n^{0.9} - e \cdot gift \cdot 2^{120}) \pmod{n} \le ec \pmod{n} \le (-e \cdot gift \cdot 2^{120}) \pmod{n}$$

를 얻고, 동시에 $0 \le c < 2^{120}$이니, 위 알고리즘을 적용할 환경이 준비가 됩니다.

휴리스틱을 적용하면 대략 $2^{120} \cdot 3 \cdot n^{-0.1} \approx 6 \cdot 10^5$개의 해가 나올 것을 예상할 수 있습니다.

이를 전부 구하고, 각 $c$에 대하여 decryption을 시도해보면 flag를 얻을 수 있습니다.

```python
from Crypto.Util.number import GCD, long_to_bytes
from tqdm import tqdm

def ceil(n, m): # returns ceil(n/m)
	return (n + m - 1) // m

def is_inside(L, R, M, val): # is L <= val <= R in mod M context?
	if L <= R:
		return L <= val <= R
	else:
		R += M
		if L <= val <= R:
			return True
		if L <= val + M <= R:
			return True 
		return False

## some notes : it's good idea to check for gcd(A, M) = 1
def optf(A, M, L, R): # minimum nonnegative x s.t. L <= Ax mod M <= R
	if L == 0:
		return 0
	if 2 * A > M:
		L, R = R, L
		A, L, R = M - A, M - L, M - R
	cc_1 = ceil(L, A)
	if A * cc_1 <= R:
		return cc_1
	cc_2 = optf(A - M % A, A, L % A, R % A)
	return ceil(L + M * cc_2, A)

# check if L <= Ax (mod M) <= R has a solution
def sol_ex(A, M, L, R):
	if L == 0 or L > R:
		return True
	g = GCD(A, M)
	if (L - 1) // g == R // g:
		return False
	return True

## find all solutions for L <= Ax mod M <= R, S <= x <= E:
def solve(A, M, L, R, S, E):
	# this is for estimate only : if very large, might be a bad idea to run this
	print("Expected Number of Solutions : ", ((E - S + 1) * (R - L + 1)) // M + 1)
	if sol_ex(A, M, L, R) == False:
		return []
	cur = S - 1
	ans = []
	num_sol = 0
	while cur <= E:
		NL = (L - A * (cur + 1)) % M
		NR = (R - A * (cur + 1)) % M
		if NL > NR:
			cur += 1
		else:
			val = optf(A, M, NL, NR)
			cur += 1 + val
		if cur <= E:
			ans.append(cur)
			# remove assert for performance if needed
			assert is_inside(L, R, M, (A * cur) % M)
			num_sol += 1
	print("Actual Number of Solutions : ", num_sol)
	return ans


# find actual data in the repository! 
N = 
e = 
gift = 
enc = 
 
R = (-e * (gift << 120)) % N
L = (-e * (gift << 120) - 3 * (int)(N ** 0.9)) % N

val = solve(e, N, L, R, 0, 1 << 120)
num_sol = len(val)

for i in tqdm(range(num_sol)):
	d = (gift << 120) + val[i]
	s = long_to_bytes(pow(enc, d, N))
	if s[:5] == b"pbctf":
		print(s)
```

# 결론 및 후기

지금까지 제 repository "Inequality Solving with CVP"의 기본 원리를 알아보았고, 실제 CTF에서 출제된 문제들을 풀어보면서 사용법도 알아보았습니다. 정말 많은 Lattice를 활용하는 문제가 이 도구로 해결될 수 있으니, 많이 사용해주시면 정말 기쁠 것 같습니다. 

RSA 관련 문제를 예시로 들었으나, Hidden Number Problem, LCG Recovery 등 선형적인 방정식으로 나타낼 수 있으면 전부 이 방법을 시도할 가치가 있습니다. repository에 예시 문제가 위에서 소개한 것을 포함해서 총 8문제가 있으니, 참고해주시기 바랍니다.

이 repository의 가치는 Lattice 알고리즘 단계를 완전한 black-box로 만들어준다는 것입니다. 부등식의 system이 있다면, 더 생각할 필요없이 한 번 돌려보면 알아서 격자를 만들고, 가중치를 만드는 과정을 대신 맡아줍니다. 

Lattice를 사용하는 대표적인 **암호를 깨는** 알고리즘에는

- Coppersmith's Algorithm을 사용한 polynomial root 계산
- 이 글에서 다루는, Linear System of Inequality를 만들고 해결

이 있습니다. (정보를 더 원한다면, https://eprint.iacr.org/2020/1506.pdf 참고)

첫 알고리즘의 경우에는 dicegang의 defund라는 분이 만든 repository인 https://github.com/defund/coppersmith 가 있습니다. 제 repository는 두 번째 알고리즘에 집중하는 것으로 볼 수 있는데, 제 구현체도 defund의 구현체처럼 많은 사람들이 사용해주었으면 좋겠습니다. 

긴 글 읽어주셔서 감사합니다. 질문은 항상 https://rkm0959.tistory.com 에서 받습니다.