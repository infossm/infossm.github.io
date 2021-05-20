---
layout: post
title: "Data Structure For Range Mode Query"
author: Aeren
date: 2021-05-20
tags: [data-structure, algorithm]

---

<h2 id="table of contents">Table Of Contents</h2>

* [Introduction](#introduction)
* [Hardness Result](#hardness_result)
* [First Method](#first_method)
* [Second Method](#second_method)
* [Third Method](#third_method)
* [Final Method](#final_method)



<h2 id="introduction">Introduction</h2>

안녕하세요, Aeren입니다!

Competitive programming을 해본 적 있는 분이라면 range sum query, range minimum query 등등의 다양한 range query문제를 접해보셨을 것입니다. 많은 range query problem들은 linear memory만으로 sublinear time query를 가능하게 해주는 data structure가 존재합니다. 이번 글에서는 비슷한 맥락의 range mode query 를 해결하는 linear memory data structure에 대해 알아 볼 것입니다.

이 글은 다음 [논문](https://cs.au.dk/~larsen/papers/linear_mode.pdf)을 바탕으로 작성되었습니다.

Array 혹은 multiset $A$가 주어져 있을 때 $A$의 **mode**란 $A$내에서 등장하는 frequency가 가장 높은 원소를 의미하며 $m = mode(A)$이라 표기하겠습니다. 그리고 $\Delta = distinct(A)$를 $A$의 서로 다른 원소의 갯수, $n=\vert A\vert$를 $A$의 크기, $w$를 저희가 가정할 cell-probe model에서의 word size라고 하겠습니다. 또한 임의의 $0\le i\le j\le n$에 대하여 $A[i:j)$를 $i\le k\lt j$인 $k$로 이루어진 $A$의 subarray라고 하겠습니다. 마지막으로, 일반성을 잃지 않고, $A$는 $[0,\Delta)$ 구간의 정수로만 이루어져 있다고 가정하겠습니다. (만약 아니라면 compression을 통해 위와 equivalent한 상태로 바꿔줄 수 있습니다.)

이 글의 개요는 다음과 같습니다.

- 먼저, $\sqrt{n}\times\sqrt{n}$ boolean matrix multiplication으로 부터의 reduction을 통해 range mode query가 대략 얼마나 어려운지 알아 볼 것입니다.
- 그 다음으로 $O(n)$ 의 공간을 요구하는 preprocessing을 통해 query 당 $O(\sqrt{n})$시간 안에 위 문제를 해결하는 법을 알아볼 것입니다.
- 그 후 $m\le\sqrt{nw}$가 성립할 때 $O(n)$의 공간을 요구하는 preprocessing을 통해 query당 $O(\sqrt{n/w})$시간 안에 위 문제를 해결하는 법을 알아볼 것입니다.
- 그리고 나서  $O(n)$ 의 공간을 요구하는 preprocessing을 통해 query당 $O(\Delta)$시간 안에 위 문제를 해결하는 방법을 알아볼 것입니다.
- 마지막으로 위 방법들을 종합하여 $O(n)$의 공간을 요구하는 preprocessing을 통해 query당 $O(\sqrt{n/w})$시간 안에 위 문제를 해결하는 법을 알아볼 것입니다.



<h2 id="hardness_result">Hardness Result</h2>

Commutative semiring $S=(\lbrace0,1\rbrace, OR, AND)$ 에 대하여 두 matrix $M,N\in\mathbb{M}_{\sqrt{n}\times\sqrt{n}}(S)$ 의 곱을 구하는 문제를 생각해봅시다.
$P=M\cdot N$  이라면 $P_{ij}$ 의 값이 1이 될 필요충분조건은 $M_{ik}=N_{kj}=1$ 인 $0\le k\lt\sqrt{n}$ 가 존재하는 것입니다. 다시말해서, $M_i=\lbrace j:M_{ij}=1\rbrace$, $N_j=\lbrace i:N_{ij}=1\rbrace$ 라고 정의하면, $P_{ij}=1$ 이 성립할 필요충분조건은 $M_i$와 $N_j$의 multiset union의 mode가 2인 것입니다.
길이 $n$짜리 배열 $L$과 $R$을 준비하겠습니다. 각 배열은 $\sqrt{n}$ 짜리 block $\sqrt{n}$개로 이루어져 있으며 각 block은 $\lbrace 1,2,\cdots,\sqrt{n}\rbrace$의 permutation입니다. 또한 임의의 $i$에 대하여 $0\le i\lt\sqrt{n}$에 대하여 $L[\sqrt{n}i, \sqrt{n}(i+1)-\vert M_i\vert)$은 $\lbrace 1,2,...,\sqrt{n}\rbrace-M_i$의 임의의 permutation, $L[\sqrt{n}(i+1)-\vert M_i\vert, \sqrt{n}(i+1))$은 $M_i$의 임의의 permutation, 비슷하게 $R[\sqrt{n}i,\sqrt{n}i+\vert N_i\vert)$은 $N_i$의 임의의 permutation, $R[\sqrt{n}i+\vert N_i\vert,\sqrt{n}(i+1))$은 $\lbrace 1,2,\cdots,\sqrt{n}\rbrace-N_i$의 임의의 permutation입니다. 풀어서 설명하자면, 각 $i$에 대하여 $M_i$의 원소들은 $L$의 $i$번째 block의 suffix에, $N_i$의 원소들은 $R$의 $i$번째 block의 prefix에 배치되어 있습니다.
이제 배열 $A$를 $L$과 $R$의 concatenation이라고 정의하면 $M_i$와 $N_j$의 multiset union의 mode가 2일 필요충분조건은 $A[\sqrt{n}(i+1)-\vert M_i\vert,n+\sqrt{n}j+\vert N_j\vert)$의 mode가 $n+1-i+j$인 것입니다. 따라서 $A$에서 $n$번의 range mode query를 통해 $P$의 모든 entry를 알아낼 수 있습니다. 즉, 다음 theorem이 성립합니다.

>***THEOREM***
>
>어떤 range mode query data structure가 preprocessing에 $p(n)$만큼의 시간이 들고, 각 쿼리당 $q(n)$만큼의 시간이 든다면, 두개의 $\sqrt{n}\times\sqrt{n}$ boolean matrix의 곱은 $O(p(n)+n\cdot q(n))$시간 안에 구할 수 있다.

즉, matrix multiplication의 exponent의 greatest lower bound를 $\omega$라고 하면 임의의 range mode query data structure는 $\Omega(n^{\omega/2})$의 preprocessing 시간이 걸리던지 혹은 $\Omega(n^{\omega/2-1})$의 query 시간이 걸립니다.
현재 알려진 $\omega$의 upper bound는 약 $2.3727$이므로 $O(n^{1.18645})$보다 나은 preprocessing time과 $O(n^{0.18635})$보다 나은 query 시간을 기대하기에는 무리가 있습니다.
또한 Strassen's algorithm에서와 같은 algebraic technique을 쓰지 않는다면 현재 알려진 가장 빠른 알고리즘은 $O(n^3)$보다 polylog만큼 낫습니다. 즉, 순수한 combinatorial technique으로는 $O(n^{1.5})$보다 작은 preprocessing 시간과 $O(n^{0.5})$보다 작은 query 시간을 기대하기에는 무리가 있습니다.
이 글에서 다룰 data structure는 bit-packing technique을 통해 위의 bound에서 $1/\sqrt{w}$만큼의 query 시간의 speedup을 달성합니다.



<h2 id="first_method">First Method</h2>

이 파트에서 다룰 내용은 다음 theorem의 증명입니다.

> ***THEOREM***
>
> 크기 $n$인 배열 $A$와 정수 $1\le s\le n$가 주어질 때, $O(n+s^2)$의 공간을 필요로 하며 range mode query를 $O(n/s)$시간에 답하는 data structure가 존재한다.

$s=\sqrt{n}$으로 놓으면 linear space를 필요로 하면서 $O(\sqrt{n})$시간에 range mode query를 해결하는 data structure가 얻어집니다.



**Precomputation**

각 $x\in[0,n)$에 대하여 배열 $Q_x$를 $A[i]=x$인 $i$들이 오름차순으로 정렬된 배열이라 정의하고 길이 $n$인 배열 $P$를 모든 $0\le i\lt n$에 대하여 $Q_{A[i]}[P[i]]=i$이 성립하도록 잡겠습니다. $Q_x$와 $P$는  $A$를 단순히 스캔하면서 선형시간에 계산할 수 있으며 각각 $O(n)$의 공간을 필요로 합니다.
이제 $A$를 $s$개의 block으로 분할합니다. $t=\lceil n/s\rceil$라고 하면 각 $0\le i\lt s$에 대하여 $i$번째 block은 subarray $A[t\cdot i,\min(n,t\cdot(i+1)))$을 나타냅니다.
$(s+1)\times(s+1)$의 크기를 갖는 2차원 배열 $S$와 $T$도 준비합니다. 각 $0\le i\lt j\le s$에 대하여 $S[i][j]$는 $i$번째 block부터 $j-1$번째 block까지만 고려할때의 mode이고, $T[i][j]$는 mode의 frequency입니다. $S$와 $T$는 각 $i$를 고정한뒤에 $A$를 스캔하면서 $O(s\cdot n)$시간에 계산할 수 있으며 각각 $O(s^2)$의 공간을 필요로 합니다.



**Query**

먼저 다음 lemma가 필요합니다.

> ***LEMMA***
>
> Multiset $A$와 $B$가 주어질 때 $x$가 $A\cup B$의 mode라면 $x\notin A$이거나 $x$가 $B$의 mode이다.

증명은 매우 자명하므로 생략하겠습니다.

$0\le i\lt j\le n$이 주어질 때 $A[i:j)$의 mode를 계산해야 합니다.
일단 $i$와 $j$가 속해있는 block을 찾습니다. 각각 $l$과 $r$이라고 하겠습니다.
먼저 $l\lt r$일 경우를 보겠습니다. $A[i:j)$는 다음 세 subarray로 분할할 수 있습니다: $L=A[i:t(l+1))$, $M=A[t(l+1),t\cdot r)$, $R=A[t\cdot r,j)$. $M$의 mode와 그의 frequency는 $S[l+1][r]$과 $T[l+1][r]$에 저장되어 있으므로 $O(1)$에 구할 수 있습니다. 이제 위 lemma에 의해 $L\cup R$이 $A[i:j)$의 mode를 포함하지 않는다면 $M$의 mode가 $A[i:j)$의 mode이므로 $L\cup R$에 있는 원소들만 고려해주면 충분합니다. $L$과 $R$은 같은 방법으로 처리할 수 있으므로 $L$만 보겠습니다. 일단 $x:=S[l+1][r], freq:=T[l+1][r]$로 초기화 합니다. 이제 $i\le k\lt t(l+1)$인 $k$를 증가하는 순서대로 봅니다.  만약 $i\le Q[A[k]][P[k]-1]$이라면 $A[k]$와 같은 값을 갖는 원소를 이미 처리했으므로 다음 $k$로 넘어갑니다. 아닐 경우 $A[k]$는 같은 값을 갖는 원소들 중에서 최초로 고려되는 원소입니다. 만약 $\vert Q[A[k]]\vert\le P[k]+freq-1$ 이거나  $r\le Q[A[k]][P[k]+freq-1]$이라면 $A[k]]$의 frequency가 현재 답의 후보의 frequency보다 작으므로 역시 다음 $k$로 넘어갑니다. 위 두 조건으로 걸러지지 않았다면 답의 후보를 현재 원소로 갱신합니다. 이 때 현재 답의 후보의 frequency를 구해야 하는데 이는 $Q[A[k]]$의 원소들 중 $[i,j)$구간에 포함되는 원소들을 $P[k]+freq-1$위치부터 linear하게 스캔하면서 세주면 구할 수 있습니다.
$l=r$인 경우도 $x:=\textrm{Undefined}, freq:=0$로 초기화 해준 뒤 같은 알고리즘을 적용하면 해결할 수 있습니다.

> ***THEOREM***
>
> 위 query 알고리즘은 $O(t)=O(n/s)$시간에 동작한다.

***PROOF***

마지막 linear 스캔부분을 제외하면 $O(t)$시간에 동작함은 자명합니다.
$L$의 서로다른 원소들은 정확히 한번 처리되며 그때마다 $freq$의 변화량만큼의 연산이 필요합니다. 그런데 처리 전의 $freq$는 $M$의 mode의 frequency보다 크거나 같으므로 현재 처리되는 원소의 $M$에서의 frequency보다 크거나 같습니다. 따라서 총 변화량은 "frequency in $L$"과 "frequency in $R$"의 합보다 작거나 같습니다. 그러므로 이를 모두 더하면 $\vert L\vert+\vert R\vert$보다 작거나 같으므로 총 연산량은 $O(t)$입니다.

$\blacksquare$



위 data structure의 구현은 다음 [링크](https://github.com/Aeren1564/Algorithms/blob/master/Algorithm_Implementations_Cpp/Data_Structure/Range_Mode_Query/range_mode_query_solver.sublime-snippet)에서 확인하실 수 있습니다.



<h2 id="second_method">Second Method</h2>

두번째로 다음 theorem을 증명하겠습니다.

> ***THEOREM***
>
> 크기 $n$인 배열 $A$와 정수 $m\le s\le n$가 주어질 때, $O(n+s^2/w)$의 공간을 필요로 하며 range mode query를 $O(n/s)$시간에 답하는 data structure가 존재한다.

$s=\sqrt{nw}$으로 놓으면 $m\le \sqrt{nw}$가 성립할 때 linear space를 필요로 하면서 $O(\sqrt{n/w})$시간에 range mode query를 해결하는 data structure가 얻어집니다.



**Precomputation**

다음 lemma가 필요합니다.

> ***LEMMA***
>
> 크기 $n$인 bit string $S$가 주어질 때 $O(n/w)$의 공간을 필요로 하며 다음 두 query를 O(1)시간에 답하는 data structure가 존재한다.
>
> 1. $rank_S(x,i)$: $x\in\lbrace 0,1\rbrace$과 $0\le i\le n$에 대하여 $S$의 길이 $i$인 prefix에서 $x$의 frequency를 찾는다.
> 2. $select_S(x,i)$: $x\in\lbrace 0,1\rbrace$과 $1\le i\le rank_S(x,\vert S\vert)$에 대하여 $i$번째 $x$의 위치를 찾는다.

***PROOF***

[참조](https://courses.csail.mit.edu/6.851/spring07/scribe/lec21.pdf)

$\blacksquare$

First method에서와 마찬가지로 $Q$와 $P$를 계산합니다.
또한 $T$의 각 row는 단조증가하므로 이전 cell으로부터의 증가량 만큼의 $0$과 ending position을 나타내는 $1$로 이루어진 bit string들의 배열 $B$로 저장하겠습니다.
예를들어 $T[i]=[1,2,2,4,5,5,8]$이라면 $B[i]=010110010110001$이 됩니다.
각 $i$에 대하여 $B[i]$의 길이는 $s+mode(A)\in O(s)$로 bound되어 있음을 쉽게 알 수 있습니다.
이제 위 lemma의 data structure를 각 row에 대하여 계산합니다.
$Q$와 $P$에서 $O(n)$의 공간이 필요하며 $B$ 및 lemma의 data structure에서 $O(s^2/w)$의 공간이 필요하므로 총 $O(n+s^2/w)$의 공간이 필요합니다.



**Query**

First method에서와 마찬가지로 $l,r,L,M,R$을 정의하겠습니다. 이제 같은 query algorithm을 저희가 가지고 있는 precomputation 결과만으로 빠르게 시행하고 싶습니다.
먼저 $M$의 mode와 그의 frequency를 찾아야합니다.
frequency는 $freq=rank_{B[l+1]}(0,select_{B[l+1]}(1,r-l-1))$으로 $O(1)$에 찾을 수 있습니다.
mode를 찾기 위해서 일단 mode가 포함된 마지막 block의 index를 $ind=l+1+rank_{B[l+1]}(1,select_{B[l+1]}(0,freq))$로 찾아줍니다.이제 $ind$블럭의 각 원소를 first method의 query에서 $R$을 스캔하는 방법과 동일하게 $M$에서의 frequency가 $freq$와 일치하는지 검사해 줄 수 있습니다.
마지막으로 $L$과 $R$은 first method에서와 같은방법으로 검사해주면 총 $O(n/s)$시간 안에 주어진 쿼리를 답할 수 있습니다.



<h2 id="third_method">Third Method</h2>

세번째로 다음 theorem을 증명하겠습니다.

> ***THEOREM***
>
> 크기 $n$인 배열 $A$가 주어질 때, $O(n)$의 공간을 필요로 하며 range mode query를 $O(\Delta)$시간에 답하는 data structure가 존재한다.

언뜻 보기에는 저희의 목표와 무관계 해 보이지만 나중에 보이듯이 second method에서 $s<m$인 케이스를 해결하는데에 꼭 필요합니다.



**Precomputation**

이번엔 block size $\Delta$가 되도록 $A$를 분할하겠습니다.
각 $0\le i\le \lfloor n/\Delta\rfloor$와 $0\le x\lt \Delta$에 대하여 $C_x[i]$를 $A[0,\Delta\cdot i)$의 mode라 정의합시다. $C$는 $A$를 한 번 스캔함으로써 linear 시간 안에 계산 할 수 있으며 $\lfloor n/\Delta\rfloor\times\Delta\in O(n)$의 공간을 필요로 합니다.



**Query**

어떤 $0\le i\le n$가 주어질 때 모든 $0\le x\lt \Delta$에 대하여 $A[\lfloor i/\Delta\rfloor\cdot \Delta,i)$의 mode 및 그의 frequency는 $A[\lfloor i/\Delta\rfloor\cdot \Delta,i)$를 스캔하면서 $O(\Delta)$에 계산가능합니다. 위 값을 $C_x[\lfloor i/\Delta\rfloor]$와 더해주면 모든 $x$의 $A[0,i)$에서의 frequency가 얻어집니다. $A[0,j)$도 마찬가지로 $O(\Delta)$에 계산가능합니다. 따라서 모든 $x$에 대하여 $A[i,j)$에서의 frequency가 $O(\Delta)$에 계산가능하므로 그 값들을 스캔하면서 $O(\Delta)$에 mode값을 찾아줄 수 있습니다.



<h2 id="final_method">Final Method</h2>

마지막으로 다음 main theorem을 증명하겠습니다.

> ***MAIN THEOREM***
>
> 크기 $n$인 배열 $A$와 정수 $1\le s\le n$이 주어질 때, $O(n+s^2/w)$의 공간을 필요로 하며 range mode query를 $O(n/s)$시간에 답하는 data structure가 존재한다.

$s=\sqrt{nw}$으로 놓으면 linear space를 필요로 하면서 $O(\sqrt{n/w})$시간에 range mode query를 해결하는 data structure가 얻어집니다.

> ***LEMMA***
>
> 크기 $n$인 배열 $A$와 multiset intersection이 비어있는 $A$의 두 ordered partition $B_1$과 $B_2$가 주어질 때 각각 $s_1(n)$과 $s_2(n)$의 공간을 필요로 하며 $t_1(n)$과 $t_2(n)$시간에 $B_1$과 $B_2$에서 range mode query를 답하는 data structure가 존재한다면, $O(n+s_1(n)+s_2(n))$의 공간을 필요로 하며 $O(t_1(n)+t_2(n))$시간에 $A$에서 range mode query를 답하는 data structure가 존재한다.

***PROOF***

각 $id\in\lbrace1,2\rbrace$와 $0\le i\lt n$에 대하여 $B_{id}$에 속하는 최소 $j \ge i$를 나타내는 $I_{id}[i]$를 linear 시간에 계산할 수 있습니다. $B_1$과 $B_2$의 intersection이 비어있으므로 mode는 단순히 $B_1[I_1[i],I_1[j])$와 $B_2[I_2[i],I_2[j])$를 비교함으로서 구할 수 있습니다. Precomputation에서 $I_{id}[i]$를 저장하는데 $O(n)$, 각각의 data structure를 계산하는데 $s_1(n)$과 $s_2(n)$의 시간이 걸리므로 총 $O(n+s_1(n)+s_2(n))$의 공간을 필요로 하며 각 쿼리당 $O(t_1(n)+t_2(n))$의 시간이 필요합니다.

$\blacksquare$

이제 $A$를 $s$보다 작거나 같은 frequency를 갖는 원소들의 배열 $B_1$과 $s$보다 큰 frequency를 갖는 원소들의 배열 $B_2$로 분할합니다. $B_1$에서는 second method를 쓰면 $O(n+s^2/w)$의 공간을 통해 각 range mode query를 $O(n/s)$시간에 해결할 수 있습니다. $B_2$에서는 third method를 쓰면 $O(n)$공간을 통해 각 range mode query를 $O(\Delta)\subseteq O(n/s)$시간에 해결할 수 있습니다. 따라서 위 lemma에 의해 main theorem이 증명됩니다.
