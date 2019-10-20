---
layout: post
title:  "Prüfer sequence"
date: 2019-10-20
author: junodeveloper
tags: [combinatorics,algorithm]
---



## 소개

안녕하세요. 이번 글에서는 labeled tree를 unique한 수열로 나타내는 Prüfer sequence에 대해 소개해드리려고 합니다. 사실 문제 풀이에 많이 활용되는 개념은 아니지만, 이해하기 쉬우면서도 이를 적용해 풀 수 있는 몇 가지 재밌는(?) 문제가 있어 정리해 보았습니다.



## Prüfer Sequence

Prüfer sequence는 $n$개의 정점을 가진 labeled tree를 다음의 알고리즘에 따라 길이 $n-2$의 수열로 나타낸 것입니다. 트리를 수열로 encode한다는 의미에서 encoding 알고리즘이라고 합니다.

```pseudocode
function Tree_to_Prüfer(T=(V,E))
	a <- an empty array
	while |V| > 2:
		u <- leaf node with the smallest label in V
		lb <- label of the node adjacent to u
		append lb to a
		remove u from V
	return a
```

트리에서 label이 가장 작은 리프 노드를 찾아 제거하고, 그와 인접해 있던 노드의 label을 수열에 추가하고, 다시 리프를 찾아 제거하고, ...의 과정을 남은 노드가 두 개일 때까지 반복합니다. 이렇게 얻어진 수열의 길이는 $n-2$가 됨이 자명합니다.

반대로, Prüfer sequence가 주어지면 이를 원래의 트리로 복원(decoding)하는 것도 가능합니다. 즉, 길이 $n$인 Prüfer sequence가 주어지면 정점의 개수가 $n+2$인 트리로 복원하는 것입니다. 아이디어는 다음과 같습니다:

우선 Prüfer sequence에 나타나지 않은 정점들은 원래 트리에서 리프에 해당함을 알 수 있습니다. (내부 정점들은 이웃한 다른 정점에 의해 최소 한 번 이상 Prüfer sequence에 삽입되기 때문) 그리고 이 정점들 중 label이 최소인 정점은 encoding 알고리즘에 따라 이웃한 정점을 가장 먼저 수열에 삽입하게 되므로, Prüfer sequence의 첫 번째 정점과 이웃하다는 것을 어렵지 않게 알 수 있습니다. 따라서 리프를 $u$, 수열의 첫 번째 정점을 $v$라 했을 때 간선 $(u,v)$는 반드시 $E$에 속해야 합니다.

지금 단계를 encoding 알고리즘 관점에서 본다면 첫 번째 리프인 $u$를 제거한 상황으로 볼 수 있습니다. 그러면 $u$와 인접한 $v$는 $u$를 제거함에 따라 새로운 리프가 될 수도 있습니다. 이를 확인하는 방법은 간단합니다: 만약 $v$가 더 이상 Prüfer sequence에 나타나지 않는다면 $v$는 새로운 리프가 됨을 알 수 있습니다.

이렇게 리프 정점들을 유지하면서, Prüfer sequence의 앞에서부터 차례대로 label이 최소인 리프와 연결해 나가면 $n$개의 간선을 얻을 수 있습니다. 그런데 아직 한 개의 간선이 모자라죠? (정점이 $n+2$개, 간선은 $n+1$개) 위 과정을 수행하고 나면 항상 두 개의 리프 정점이 남게 되는데, 이 둘을 연결하면 됩니다. 그러면 트리의 모든 연결 관계가 복원됩니다.

아래는 앞서 설명한 방법대로 decoding을 수행하는 알고리즘입니다. 리프 정점을 관리하는 집합(L)을 priority queue 등으로 구현하면 총 $O(nlogn)$의 시간복잡도로 구현할 수 있습니다.

```pseudocode
function Prüfer_to_Tree(a[1..n])
	V <- {1, 2, ..., n+1, n+2}
	E <- an empty set
	L <- an empty set
	degree[1..n+2] <- an array with all elements initialized to 1
	
	for i=1 to n:
		degree[a[i]] <- degree[a[i]] + 1
		
	for i=1 to n+2:
		if degree[i] = 1 then
			insert i into L
			
	for i=1 to n:
		u <- smallest number in L
		degree[u] <- degree[u] - 1
		degree[a[i]] <- degree[a[i]] - 1
		insert (u, a[i]) into E
		remove u from L
		if degree[a[i]] = 1 then
			insert a[i] into L
	
	u, v <- remaining two numbers in L
	insert (u, v) into E
	
	return T=(V,E)
```



## 성질

다음은 Prüfer sequence과 관련된 몇 가지 성질을 정리한 것입니다.

1. degree가 $x$인 정점은 Prüfer sequence에서 $x-1$번 나타난다. (역으로, Prüfer sequence에서 $x$번 나타난 정점의 degree는 $x+1$이다 .)
2. 하나의 labeled tree는 하나의 Prüfer sequence를 갖는다. (역으로, 하나의 Prüfer sequence는 하나의 labeled tree를 나타낸다.)
3. 길이 $n$인 Prüfer sequence로 만들 수 있는 서로 다른 labeled tree의 개수는 $(n+2)^n$이다.

1번 - Encoding 알고리즘에서 degree가 $x$인 정점은 인접한 $x-1$개의 정점을 모두 제거한 뒤에 제거됩니다. 즉, 각각의 이웃한 정점이 제거될 때마다 Prüfer sequence에 한 번씩 나타나게 되므로 총 $x-1$번 나타남을 알 수 있습니다. 반대의 경우도 Decoding 알고리즘을 생각해보면 쉽게 알 수 있습니다.

2번 - Encoding과 Decoding모두 deterministic하게 정의되는 알고리즘이기 때문에 labeled tree와 Prüfer sequence가 일대일 대응 관계임을 알 수 있습니다.

3번 - 우선 Prüfer sequence의 원소는 label의 범위, 즉 1 이상 $n+2$ 이하를 만족해야 하므로, 길이 $n$인 모든 가능한 sequence의 개수는 $(n+2)^n$입니다. Cayley's formula에 의하면 정점의 개수가 $n$인 서로 다른 labeled tree의 개수는 $n^{n-2}$인데, $n+2$를 대입하면 $(n+2)^n$개, 즉 Prüfer sequence의 개수와 같습니다. 모든 labeled tree는 unique한 Prüfer sequence를 갖기 때문에, labeled tree와 Prüfer sequence는 bijection 관계임을 알 수 있습니다. 따라서 어떤 sequence를 생각하든 항상 대응되는 labeled tree가 존재한다는 것을 알 수 있습니다.



## 문제 1 - Road Network 2 (boj.kr/8286)

$n$개의 정점에 대한 degree 정보가 주어질 때, 원래 트리로 복원하는 문제입니다. 사실상 앞서 설명한 Decoding 알고리즘을 그대로 사용하면 됩니다. 단, Prüfer sequence가 특정되지 않았기 때문에 주어진 degree 정보와 일치하는 임의의 sequence를 아무거나 가정하고 트리를 복원하면 됩니다. 설명한 방법으로 쉽게 풀 수 있으므로 자세한 설명은 생략하겠습니다.



## 문제 2 - 트리 (boj.kr/13185)

$n$개의 정점이 있고, $m$개의 간선 정보가 주어졌을 때 나머지 $n-1-m$개의 간선을 추가로 연결해서 만들 수 있는 서로 다른 labeled tree의 개수를 세는 문제입니다.

우선 주어진 간선 정보를 바탕으로 컴포넌트로 묶습니다. 이제 각 컴포넌트를 하나의 정점으로 생각하여 적절히 카운팅할 겁니다.

위에서 언급했듯이, 트리상의 어떤 정점이 $d$의 degree를 갖는다면, Prüfer sequence에서는 $d-1$번 등장합니다. 이 성질을 이용하면 각 정점의 degree가 $d_1, d_2, ..., d_n$인 labeled tree의 개수를 셀 수 있습니다. Sequence 상에서 각 정점이 $d_1 - 1, d_2 - 1, ..., d_n - 1$번 등장하는 경우의 수이므로 $\frac{(n-2)!}{(d_1 -1)!(d_2 -1)!...(d_n -1)!}$ 이라는 식을 쉽게 유도할 수 있습니다.

또한 각 컴포넌트에 대해서, 해당 컴포넌트의 degree가 $d$이고, 포함된 정점의 개수가 $s$ 일 때, 간선을 연결하는 방법은 $s^d$ 가지가 있습니다.

따라서 우리가 구할 답은 컴포넌트의 수를 $c$라고 했을 때,  $\sum{\frac{(c-2)!}{(d_1 -1)!...(d_m-1)!}s_1^{d_1}s_2^{d_2}...s_c^{d_c}}$ 입니다.

여기서 $s_1s_2...s_c$항을 빼면 지수가 $d_i-1$이 되어서 분모의 형태와 일치하게 되고, $(d_1-1)+(d_2-1)+...+(d_c-1)=(d_1+...+d_c)-c=(2c-2)-c=c-2$임을 알 수 있습니다. 즉, 위에서 시그마에 해당하는 부분은

$(s_1+s_2+...+s_c)^{c-2}=n^{c-2}$로 간단하게 표현됩니다.

따라서 최종 답은 $s_1s_2...s_c * n^{c-2}$ 입니다. 이제 간단히 코딩만 하면 되므로 더 자세한 설명은 생략하겠습니다.



