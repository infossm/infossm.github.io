---
layout: post
title: "Skew-binary Lifting"
author: Aeren
date: 2021-03-23
tags: [data-structure, algorithm, tree]

---

<h2 id="table of contents">Table Of Contents</h2>

* [Prerequisite](#prerequisite)
* [Introduction](#introduction)
* [Implementation](#implementation)
* [Performance Analysis](#performance_analysis)
* [Benchmark](#benchmark)



<h2 id="prerequisite">Prerequisite</h2>

* Binary Lifting - [Tutorial on cp-algorithms](https://cp-algorithms.com/graph/lca_binary_lifting.html)



<h2 id="introduction">Introduction</h2>

안녕하세요, Aeren입니다!

이 글에서 소개할 내용은 skew-binary number system을 기반으로 한 skew-binary lifting입니다. 이 글은 [An Applicative Random-Access Stack by Eugene W. MYERS](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.188.9344&rep=rep1&type=pdf)을 기반으로 작성되었습니다.

일반적인 binary lifting과의 차이점 중 하나는 각 node $u$가 $O(\log(\textrm{depth}[u]))$ 대신 $O(1)$ 만큼의 공간을 필요로 한다는 것입니다. 표로 정리하면 다음과 같습니다.



***(Time) / (Additional Space Required)***

| Operation                  | Binary Lifting                                              | Skew-binary Lifting                       |
| -------------------------- | ----------------------------------------------------------- | ----------------------------------------- |
| Add A Leaf                 | $O(\log(\textrm{depth})) / O(\textrm{log}(\textrm{depth}))$ | $O(1) / O(1)$                             |
| Find The K-th Ancestor     | $O(\log(\textrm{depth})) / \textrm{None}$                   | $O(\log(\textrm{depth})) / \textrm{None}$ |
| Find The LCA               | $O(\log(\textrm{depth})) / \textrm{None}$                   | $O(\log(\textrm{depth})) / \textrm{None}$ |
| Binary Search On Ancestors | $O(\log(\textrm{depth})) / \textrm{None}$                   | $O(\log(\textrm{depth})) / \textrm{None}$ |



또한 binary lifting과 마찬가지로 코드가 매우 단순하고 직관적이며 각 node / edge에 weight를 줄 수도 있습니다. 한 가지 단점이 있다면 연산들의 $\log$에 붙는 상수가 더 큽니다. 이는 [Benchmark](#benchmark)에서 자세히 보실 수 있습니다.



<h2 id="implementation">Implementation</h2>

만약 root node가 없다면 임의의 root node를 지정합니다.

이제 각 node $u$에는 다음 세 정보가 부여됩니다.

* $\textrm{parent}[u]$
* $\textrm{depth}[u]$
* $\textrm{lift}[u]$

위 값들은 각 non-root node가 dfs순으로 생성될 때 다음과 같이 정의됩니다.

![](/assets/images/Aeren_images/Skew-binary-Lifting/Add_A_Leaf.PNG)

$\text{parent}$와 $\textrm{depth}$는 일반적인 rooted tree에서의 정의와 같고, $\textrm{lift}$는 ancestor에 대한 "큰 점프"라고 생각하시면 됩니다.

![](/assets/images/Aeren_images/Skew-binary-Lifting/figure.PNG)

위의 figure에서 각 node에 적힌 숫자는 $\textrm{depth}$를, 파란색 arc는 $\textrm{parent}$를, 그리고 붉은색 arc는 $\textrm{lift}$를 나타냅니다. (root에 대한 정보는 생략하였습니다.)

Find The K-th Ancestor, Find The LCA, 그리고 Binary Search On Ancestors의 구현은 매우 단순합니다. $\textrm{lift}$를 타고 갈 수 있다면 $\textrm{lift}$로, 아니라면 $\textrm{parent}$를 타고 올라가면 됩니다.

![](/assets/images/Aeren_images/Skew-binary-Lifting/Find_The_K-th_Ancestor.PNG)

![](/assets/images/Aeren_images/Skew-binary-Lifting/Find_The_LCA.PNG)

![](/assets/images/Aeren_images/Skew-binary-Lifting/Binary_Search_On_Ancestors.PNG)

다음은 Skew-binary lifting을 활용한 [Baekjoon Online Judge 20931번 - 혹 떼러 갔다 혹 붙여 온다](https://www.acmicpc.net/problem/20931) 의 C++ 예시 코드입니다.

```cpp
#include <bits/stdc++.h>
using namespace std;



int main(){
	cin.tie(0)->sync_with_stdio(0);
	cin.exceptions(ios::badbit | ios::failbit);
	int qn;
	cin >> qn;
	int m = 1;
	vector<int> par{0}, depth{0}, lift{0};
	const long long inf = numeric_limits<long long>::max() / 3;
	vector<long long> len{inf}, lift_len{inf};
	auto append = [&](int u, long long l){
		int v = m ++;
		par.push_back(u);
		depth.push_back(depth[u] + 1);
		len.push_back(l);
		if(depth[lift[u]] - depth[u] == depth[lift[lift[u]]] - depth[lift[u]]){
			lift.push_back(lift[lift[u]]);
			lift_len.push_back(min(inf, l + lift_len[u] + lift_len[lift[u]]));
		}
		else{
			lift.push_back(u);
			lift_len.push_back(l);
		}
	};
	auto trace_up = [&](int u, long long l){
		assert(u < m);
		while(true){
			if(lift_len[u] <= l){
				l -= lift_len[u];
				u = lift[u];
			}
			else if(len[u] <= l){
				l -= len[u];
				u = par[u];
			}
			else{
				return u;
			}
		}
	};
	for(auto qi = 0; qi < qn; ++ qi){
		string type;
		cin >> type;
		static int last_ans = 0;
		if(type[0] == 'a'){
			int u;
			long long l;
			cin >> u >> l, u = (u + last_ans) % m;
			append(u, l);
		}
		else{
			int u;
			long long l;
			cin >> u >> l;
			cout << (last_ans = trace_up(u, l)) << "\n";
		}
	}
	return 0;
}
```



<h2 id="performance_analysis">Performance Analysis</h2>

$\textrm{lift}$값의 정의만 알고있다면 구현 자체는 매우 자명합니다. 자명하지 않은 사실은 위 세 연산이 모두 $\log(\textrm{depth})$에 동작한다는 것이죠. 여기선 이 사실을 증명하고자 합니다.

> ***Theorem***
>
> $\textrm{Find_The_K-th_Ancestor}(u,k)$, $\textrm{Find_The_LCA}(u,v)$, $\textrm{Binary_Search_On_Ancestors}(u,P)$의 시간복잡도는 각각 $O(\log(\textrm{depth}[u]))$, $O(\log(\max(\textrm{depth}[u],\textrm{depth}[v])))$, $O(\log(\textrm{depth}[u]))$이다.

일단, $\textrm{Find_The_K-th_Ancestor}(u,k)$의 시간복잡도가 $O(\log(\textrm{depth}[u]))$라고 가정합시다.

이제 node $u$와 $\textrm{Binary_Search_On_Ancestors}$ 의 조건을 만족하는 binary predicate $P$에 대하여,  $\textrm{Binary_Search_On_Ancestors}(u,P)$가 node $v$를 찾는다면, $\textrm{Binary_Search_On_Ancestors}(u,P)$ 내부에서 $u$에 저장되는 node들이 $\textrm{Find_The_K-th_Ancestor}(u, \textrm{depth}[u]-\textrm{depth}[v])$의 내부에서 $u$에 저장되는 node들과 같다는 것을 알 수 있습니다. 즉, $\textrm{Binary_Search_On_Ancestors}(u,P)$의 시간복잡도 역시 $O(\log(\textrm{depth}[u]))$입니다.

또한 node $u$와 $v$에 대해서 $\textrm{Find_The_LCA}(u,v)$의 시간복잡도 역시 $\textrm{Find_The_K-th_Ancestor}$ 세 번으로 표현 가능함으로 $O(\log(\max(\textrm{depth}[u],\textrm{depth}[v])))$입니다.

따라서 $\textrm{Find_The_K-th_Ancestor}(u,k)$의 시간복잡도가 $O(\log(\textrm{depth}[u]))$임을 보인다면 충분합니다. 이를 위해 skew-binary number system을 소개하겠습니다.



<h3 id="skew-binary number system">Skew-binary Number System</h3>

**Skew-binary number**란 $0,1,2$로 이루어진 sequence중 0이 아닌 항이 유한개 있는 sequence를 뜻합니다. 모든 항이 0인 skew-binary number을 $\bar{0}$라 표기하겠습니다. $\bar{0}$이 아닌 skew-binary number $a$의 **least significant digit**을 $a_i\ne0$이 성립하는 가장 작은 $i$로 정의하고 $LSD(a)$라 표기하고, **most significant digit**을 $a_i\ne0$이 성립하는 가장 큰 $i$로 정의하고 $MSD(a)$라 표기하겠습니다.

**Skew-binary number system**은 skew-binary number $a_n(n=1, 2, ...)$을 음이 아닌 정수 $\sum_{n=1}^\infty a_n(2^n-1)$로 보내는 mapping $\mathcal{S}:CSB\rightarrow\mathbb{Z}_{\ge0}$입니다. 이 number system의 문제점은 $7$ $=\mathcal{S}(0,0,1,...)$ $=\mathcal{S}(1,2,...)$처럼 하나의 음이 아닌 정수를 표현하는 skew-binary number가 여러개인 경우가 있다는 것입니다. 이 문제를 해결하는 것이 canonical skew-binary number입니다.

**Canonical skew-binary number**란 모든 항이 0이거나 $LSD(a)$를 제외한 모든 digit이 0 또는 1인 skew-binary number를 의미합니다. canonical skew-binary number의 집합을 $CSB$라고 표기하겠습니다. 또한 $CSB_i= \{ a\in CSB:(a=\bar{0})\,\vee\,((a\ne\bar{0})\wedge(MSD(a)\le i)) \} $라 정의하겠습니다.



> ***Lemma***
>
> 모든 $a\in CSB_i$에 대하여 $\mathcal{S}(a)\le 2^{i+1}-2$이다.

***Proof***

$a=\bar{0}$이면 자명하게 성립합니다.

아니라면, $j=LSD(a)\le i$일 때, $\mathcal{S}(a)\le2^{j+1}-2+\sum_{k=j+1}^i(2^k-1)=2^{i+1}-(i-j)-2\le2^{i+1}-2$이므로 역시 성립합니다.

$\blacksquare$



위 Lemma에 의해 함수 $a\mapsto\mathcal{S}(a):CSB_i\rightarrow\{n\in\mathbb{Z}:0\le n\le2^{i+1}-2\}$에 대하여 논할 수 있습니다.



> ***Theorem***
>
> $\mathcal{S}\vert _ {CSB}:CSB\rightarrow\mathbb{Z} _ {\ge0}$은 one-to-one correspondence이다.

***Proof***

$a\mapsto\mathcal{S}(a):CSB_i\rightarrow\{n\in\mathbb{Z}:0\le n\le2^{i+1}-2\}$가 one-to-one correspondence임을 보이면 충분합니다. 그런데 $\vert CSB_i\vert =1+\sum_{j=1}^i2^j=2^{i+1}-1=\vert \{n\in\mathbb{Z}:0\le n\le2^{i+1}-2\}\vert <\infty$이므로 위 함수가 surjective함을 보이면 충분합니다. 이는 귀납법으로 쉽게 보일 수 있습니다.

$i=0$일 땐 codomain의 크기가 1이므로 surjective합니다.

어떤 $k$에 대하여 위 명제가 참이라고 가정합시다.

$\{n\in\mathbb{Z}:0\le n\le2^{k+2}-2\}$에 속하는 원소들 중 $\{n\in\mathbb{Z}:0\le n\le2^{k+1}-2\}$에 속하는 원소들은 가정에 의해 어떤 $a\in CSB_k\subseteq CSB_{k+1}$의 image로서 나타납니다.

그렇지 않은 원소들 중 구간 $[2^{k+1}-1,2^{k+2}-3]$에 속하는 원소들은 가정에 의해 $k+1$ 자리가 $1$인 어떤 $a\in CSB_{k+1}$의 image로서 나타납니다.

마지막으로 $2^{k+2}-2$는 $k+1$자리가 $2$, 나머지 자리가 $0$인 $a$의 함수값으로서 나타납니다.
따라서 위 함수는 one-to-one correspondence입니다.

$\blacksquare$



편의상 $\mathcal{S}\vert_{CSB}$를 $\mathcal{S}$라 쓰도록 하겠습니다. $\mathcal{S}$가 one-to-one correspondence이니 이제 inverse mapping $\mathcal{S}^{-1}:\mathbb{Z}_{\ge0}\rightarrow CSB$에 대해 얘기할 수 있습니다. 다음 표에는 각 음이 아닌 정수를 나타내는 canonical skew-binary number가 나와있습니다.



| $\mathbb{Z}_{\ge0}$ | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   | 12   | 13   |
| ------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| $CSB$               | 0    | 1    | 2    | 01   | 11   | 21   | 02   | 001  | 101  | 201  | 011  | 111  | 211  | 021  |



함수 $J:\mathbb{Z}_ {\ge0}\rightarrow\mathbb{Z}_ {\ge0}$를 다음과 같이 정의하겠습니다.

* $J(0)=0$
* Let $n\in\mathbb{Z}_{>0}$ and $a=\mathcal{S}^{-1}(n)$.  
  $J(n)$ $=\mathcal{S}(0,0,...,0,a _ {LSD(a)}-1,a _ {LSD(a)+1},a _ {LSD(a)+2},a _ {LSD(a)+3},...)$

즉, $J$는 양의 정수 $n$에 대하여, $n$의 canonical skew-binary representation의 $LSD$위치에서 1을 빼준 canonical skew-binary representation이 나타내는 정수를 찾아주는 함수입니다.

다음 theorem은 skew-binary lifting의 핵심인 $\textrm{lift}$의 정체를 밝혀줍니다.



> ***Theorem***
>
> 임의의 node $u$에 대하여 $\textrm{depth}[\textrm{lift}[u]]=J(\textrm{depth}[u])$가 성립한다.

풀어서 설명하자면, $\textrm{lift}[u]$는 $u$의 ancestor중에서 $\textrm{depth}$값이 $J(\textrm{depth}[u])$와 같아지는 node입니다.

***Proof***

$\textrm{depth}$에 대한 귀납법으로 보이겠습니다.

$u$가 root라면 자명하게 성립합니다.

이제 어떤 $n\in\mathbb{Z}_{\ge0}$에 대하여 $\textrm{depth}[u]\le n$인 모든 $u$에 대하여 본 theorem이 참이라고 가정하겠습니다.

$\textrm{depth}[u]=n+1$인 $u$를 고정하고, $a=\mathcal{S}^{-1}(n)$이라 합시다.
또한 predicate $p(u)$를

$[\textrm{depth}[\textrm{parent}[u]]-\textrm{depth}[\textrm{lift}[\textrm{parent}[u]]]=\textrm{depth}[\textrm{lift}[\textrm{parent}[u]]]-\textrm{depth}[\textrm{lift}[\textrm{lift}[\textrm{parent}[u]]]]]$

라 정의합시다. ($[f]$는 $f$가 참이면 $\textrm{True}$, 아니라면 $\textrm{False}$라는 의미입니다.)

1. $a=\bar{0}$인 경우, $p(u)=[0-0=0-0]=\textrm{True}$이므로 $\text{lift}[u]=\text{lift}[\text{lift}[\text{parent}[u]]]=\textrm{root}$입니다.
   따라서 $\textrm{depth}[\textrm{lift}[u]]=0=J(1)=J(\textrm{depth}[u])$이므로 본 theorem은 성립합니다.
2. $a$의 $0$이 아닌 항이 정확히 한 개이며, 그 값이 $1$일 경우, $p(u)=[n-0=0-0]=\textrm{False}$이므로 $\textrm{lift}[u]=\textrm{parent}[u]$입니다.
   따라서 $\textrm{depth}[\textrm{lift}[u]]=n=J(n+1)=J(\textrm{depth}[u])$이므로 본 theorem은 성립합니다.
3. $a$의 $0$이 아닌 항이 두 개 이상이며, 그 값이 모두 $1$일 경우, $i$를 $LSD(a)$, $j$를 $i$보다 크면서 $a_j=1$이 성립하는 가장 작은 수라고 정의하면, $p(u)$ $=[n-(n-(2^i-1))$ $=(n-(2^i-1))-(n-(2^i-1)-(2^j-1))]$ $=[2^i-1=2^j-1]$ $=\textrm{False}$이므로 $\textrm{lift}[u]=\textrm{parent}[u]$입니다.
   따라서 $\textrm{depth}[\textrm{lift}[u]]$ $=n$ $=J(n+1)$ $=J(\textrm{depth}[u])$이므로 본 theorem은 성립합니다.
4. 마지막으로, $a\ne\bar{0}$이며 $a_{LSD(a)}=2$인 경우, $i=LSD(a)$라 하면, $p(u)$ $=[n-(n-(2^i-1))$ $=(n-(2^i-1))-(n-2(2^i-1))]$ $=[2^i-1=2^i-1]$ $=\textrm{True}$이므로 $\textrm{lift}[u]=\textrm{lift}[\textrm{lift}[\textrm{parent}[u]]]$입니다.
   따라서 $\textrm{depth}[\textrm{lift}[u]]$ $=n-2(2^i-1)=J(n-2(2^i-1)+(2^{i+1}-1))$ $=J(\textrm{depth}[u])$이므로 본 theorem은 성립합니다.

따라서, 모든 $u$에 대해서 본 theorem이 성립합니다.

$\blacksquare$



이제 main theorem을 증명하겠습니다.

> ***Lemma***
>
> $0<k<\textrm{depth}[u]$, $a=\mathcal{S}^{-1}(\textrm{depth}[u])$, $b=\mathcal{S}^{-1}(\textrm{depth}[u]-k)$라고 하자.
>
> $L=LSD(b),\,U= \max\{ i:a_i\ne b_i\}$
>
> 일 때, $\textrm{Find_The_K-th_Ancestor}(u,k)$의 While문은 정확히
>
> $\sum_{i=1}^{U-1}a_i+(a_U-b_U)+\sum_{i=L}^{U-1}(2-b_i)$
>
> 번 반복된다.

***Proof***

While문이 $t$번 반복 된 후의 $u$를 $u_t$라 정의하고, $n_t=\textrm{depth}[u_t]$라 합시다.
또한 $A_j=\sum_{i=1}^ja_i\,\,(0\le j<U)$, $B_j=\sum_{i=1}^{U-1}a_i+(a_U-b_U)+\sum_{i=j}^{U-1}(2-b_i)\,\,(L\le j<U)$라고 정의합시다.

1. 모든 $0\le i<U$에 대해서, $\mathcal{S}^{-1}(n_{A_i})=(0,...,0,a_{i+1},a_{i+2},...)$이 성립합니다. 이는 귀납법으로 어렵지 않게 보일 수 있습니다.
2. $\mathcal{S}^{-1}(n_{A_{U-1}})=(0,...,0,a_U,a_{U+1},...)$로 부터  
   $\mathcal{S}^{-1}(n_{A_{U-1}+(A_U-B_U)-1})$ $=(0,...,0,b_U+1,a_{U+1},...)$ $=(0,...,0,b_U+1,b_{U+1},...)$이며  
   $\mathcal{S}^{-1}(n_{A_{U-1}+(A_U-B_U)})=(0,...,0,2,b_U,b_{U+1},...)$가 얻어집니다.
3. 다시 귀납법으로, 모든 $U\ge i>L$에 대해서, $\mathcal{S}^{-1}(n_{A_{U-1}+(A_U-B_U)+B_i})=(0,...,0,2,b_i,b_{U+1},...)$임을 보일 수 있습니다.
4. 마지막으로, $\mathcal{S}^{-1}(n_{A_{U-1}+(A_U-B_U)+B_{L+1}})$ $=(0,...,0,2,b_{L+1},b_{U+1},...)$로 부터 $\mathcal{S}^{-1}(n_{A_{U-1}+(A_U-B_U)+B_L})$ $=(0,...,0,b_L,b_{L+1},b_{U+1},...)$ $=(b_1,...,b_{L-1},b_L,b_{L+1},b_{U+1},...)$가 얻어지고 While문이 종료됩니다.

따라서, While문은 정확히 $\sum_{i=1}^{U-1}a_i+(a_U-b_U)+\sum_{i=L}^{U-1}(2-b_i)$번 시행됩니다.

$\blacksquare$



> ***Theorem***
>
> $\textrm{Find_The_K-th_Ancestor}(u,k)$, $\textrm{Find_The_LCA}(u,v)$, $\textrm{Binary_Search_On_Ancestors}(u,P)$의 시간복잡도는 각각 $O(\log(\textrm{depth}[u]))$, $O(\log(\max(\textrm{depth}[u],\textrm{depth}[v])))$, $O(\log(\textrm{depth}[u]))$이다.

***Proof***

위에서 언급했듯이 $\textrm{Find_The_K-th_Ancestor}(u,k)$의 시간복잡도만 보이면 충분합니다.

$k=0$인 경우, While문이 한 번도 반복되지 않으므로 성립합니다.

$k=\textrm{depth}[u]$인 경우, While문이 정확히 $\mathcal{S}^{-1}(\textrm{depth}[u])$의 자리수들의 합만큼 반복되므로 ($\textrm{lift}$를 타고 올라갈 때마다 자리수의 합이 1이 줄어듭니다.) 역시 성립합니다.

$0<k<\textrm{depth}[u]$인 경우, 위 Lemma에서,

$\sum_{i=1}^{U-1}a_i+(a_U-b_U)+\sum_{i=L}^{U-1}(2-b_i)$  

$\le MSD(a)+1+2MSD(a)-3=3MSD(a)-2$  

$\le3\lfloor\log(\textrm{depth}[u]+1)\rfloor-2$  

이므로 마찬가지로 성립합니다.

따라서 $\textrm{Find_The_K-th_Ancestor}(u,k)$의 시간복잡도는 $O(\log(\textrm{depth}[u]))$입니다.

$\blacksquare$



위의 $3\lfloor\log(\textrm{depth}[u]+1)\rfloor-2$ bound는 $a=(2,1,1,...,1,0,...),b=(1,0,0,...)$일 때 달성가능하므로 tight하다는 것을 추가로 알 수 있습니다.



<h2 id="benchmark">Benchmark</h2>

다음은 $\vert V\vert =10^4,10^5,10^6,10^7$인 line graph에서 $Q=10^4,10^5,10^6,10^7$회 uniformly random한 $u$와 $k$를 잡아 $\textrm{Find_The_K-th_Ancestor}(u,k)$를 호출 할 때 binary lifting과 skew-binary lift의 실행시간을 비교한 표입니다.



***(Binary Lifting) / (Skew-binary Lifting) runtimes (in seconds). Bolded indicates faster.***

| $Q\,\backslash \,\vert V\vert $ | $10^4$              | $10^5$              | $10^6$              | $10^7$               |
| ------------------------------- | ------------------- | ------------------- | ------------------- | -------------------- |
| $10^4$                          | 0.00118/**0.00105** | 0.02021/**0.00327** | 0.19027/**0.01536** | 2.09941/**0.09259**  |
| $10^5$                          | **0.00374**/0.03054 | **0.01322**/0.02884 | 0.26477/**0.08301** | 2.35131/**0.46748**  |
| $10^6$                          | **0.03564**/0.09584 | **0.04277**/0.27755 | **0.13318**/0.71671 | 2.21926/**2.19224**  |
| $10^7$                          | **0.31048**/0.94788 | **0.31032**/2.31801 | **0.40462**/6.84995 | **2.46005**/22.15703 |



다음은 위 데이터를 얻는데 사용한 C++ 코드입니다.



```cpp
#include <bits/stdc++.h>
using namespace std;
using namespace chrono;
mt19937 rng(high_resolution_clock::now().time_since_epoch().count());
mt19937_64 rngll(high_resolution_clock::now().time_since_epoch().count());



int main(){
	cin.tie(0)->sync_with_stdio(0);
	cin.exceptions(ios::badbit | ios::failbit);
	cout << fixed << setprecision(5);
	for(int Q: {1e4, 1e5, 1e6, 1e7}){
		for(int V: {1e4, 1e5, 1e6, 1e7}){
			auto pivot = high_resolution_clock::now();
			vector<array<int, 2>> queries(Q);
			for(auto &[u, k]: queries){
				u = rng() % V, k = rng() % (u + 1);
			}
			{ // Binary Lifting

				// Construction
				vector<int> par(V), depth(V);
				vector<vector<int>> lift(30, vector<int>(V));
				for(auto u = 1; u < V; ++ u){
					par[u] = u - 1;
					depth[u] = depth[par[u]] + 1;
					lift[0][u] = par[u];
					for(auto bit = 1; bit < 30; ++ bit){
						lift[bit][u] = lift[bit - 1][lift[bit - 1][u]];
					}
				}

				// Query
				auto Find_The_K_th_Ancestor = [&](int u, int k){
					assert(depth[u] >= k);
					if(!k) return u;
					for(auto bit = 29; bit >= 0; -- bit){
						if(k & 1 << bit){
							u = lift[bit][u];
						}
					}
					return u;
				};
				for(auto [u, k]: queries){
					int res = Find_The_K_th_Ancestor(u, k);
				}

			}
			cout << duration<double>(high_resolution_clock::now() - pivot).count() << " / ";
			pivot = high_resolution_clock::now();
			{ // Skew-binary Lifting

				// Construction
				vector<int> par(V), depth(V), lift(V);
				for(auto u = 1; u < V; ++ u){
					par[u] = u - 1;
					depth[u] = depth[par[u]] + 1;
					if(depth[par[u]] - depth[lift[par[u]]] == depth[lift[par[u]]] - depth[lift[lift[par[u]]]]){
						lift[u] = lift[lift[par[u]]];
					}
					else{
						lift[u] = par[u];
					}
				}

				// Query
				auto Find_The_K_th_Ancestor = [&](int u, int k){
					assert(depth[u] >= k);
					k = depth[u] - k;
					while(depth[u] != k){
						if(depth[lift[u]] > k){
							u = lift[u];
						}
						else{
							u = par[u];
						}
					}
					return u;
				};
				for(auto [u, k]: queries){
					int res = Find_The_K_th_Ancestor(u, k);
				}

			}
			cout << duration<double>(high_resolution_clock::now() - pivot).count() << "  ";
		}
		cout << endl;
	}
	return 0;
}
```

