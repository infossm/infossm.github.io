Matroid



정의 1. matroid $\mathcal{M} = (S,  \mathcal{I})$ 에서 $S$는 유한집합, $ \mathcal{I} \subset 2^S$ 는 독립집합(independent set)들의 collection이다. 이 때, $I$는 다음 세 가지 조건을 만족하여야 한다.

1. $\phi \in  \mathcal{I}$
2. $Y \subset X, X \in  \mathcal{I} \Rightarrow Y \in  \mathcal{I}$ 
3. $X, Y \in  \mathcal{I}, |X| < |Y|$ 이면 $X + y \in  \mathcal{I}$ 를 만족하는 $y \in Y \setminus X$가 존재



매트로이드는 다양한 집합에서 정의될 수 있다. 그 중 대표적인 예 몇 가지를 살펴보자.

예시 1. Vector matroid

체 $\mathbb{F}$ 상에서 정의된 $m \times n$ 행렬 $A$을 생각하자. $v_i$를 $A$의 $i$번째 column vector라 하면 $v_i$들은 벡터공간 $\mathbb{F}^m$의 벡터이다. $S = \left\{1, 2, .., n \right\}$ ,  $\mathcal{I} = \left\{ I : I \subset S, \left\{v_i\right\}_{i \in I} \: are \: linearly \: independent \right\}$ 로 놓으면 $\mathcal{M} = (S, \mathcal{I})$는 matroid가 됨을 쉽게 알 수 있다. (1, 2번 조건은 자명하고, 3번의 경우 $|X| < |Y|$이고 둘 모두 independent한 vector들의 집합이므로 $X$의 벡터들이 $Y$의 벡터공간 전체를 span할 수 없다)

예시 2. Graphic matroid

$G = (V, E)$가 무향 그래프일 때, $\mathcal{I} = \left\{I : I \subset E, I \: induces \: a \: forest \: in \: G \right\}$로 놓으면 $\mathcal{M} = (E, \mathcal{I})$는 matroid가 된다. 1, 2번 조건은 앞서와 마찬가지로 자명하고, 3번 조건의 경우 $X$에 포함된 edge들을 모두 이었을 때 component의 개수는 $N - |X|$이고, $X + y \in  \mathcal{I}$ 를 만족하는 $y \in Y \setminus X$가 존재하지 않는다면 $Y$의 edge들의 두 끝점이 한 component에 들어가야 하므로 $Y$에 포함된 edge들을 모두 이었을 때 component의 개수는 $N-|X|$ 이하인데 이것은 $N-|Y|$와 같아야 하므로 $|X| <|Y|$에 모순이다. 따라서, 3번 조건 역시 만족한다. Graphic matroid의 경우는 뒤에 다룰 minimum spanning tree를 구하는 kruskal 알고리즘의 증명에 이용된다.

예시 3. Uniform matroid

Uniform matroid는 어쩌면 생각할 수 있는 가장 쉬운 matroid이다. 어떠한 $k$를 정한 후 $S$에서 크기가 $k$ 이하인 모든 부분집합을 independent set으로 놓으면 uniform matroid가 된다.



예시 4. Partition matroid

Partition matroid는 Uniform matroid의 일반화라고 볼 수 있다. $S_1, S_2, ..., S_n$이 $S$의 분할이고, $k_1, k_2, ..., k_n$이 양의 정수일 때, $\mathcal{I} = \left\{ I : I \subset S, | I \cap S_i | \le k_i \: for \: all \: 1 \le i \le n \right\}$  로 정의하면 $\mathcal{M} = (S, \mathcal{I})$는 matroid이다. partition matroid나 uniform matroid 같은 경우는 matroid임이 자명하기 때문에 이것이 매트로이드라는 것이 특별한 것은 아니지만, 다음 포스팅에서 다룰 matroid intersection의 경우 graphic matroid나 vector matroid 등 다른 matroid와의 maximal matroid intersection / maximum weight matroid intersection을 구하는 문제 등에서 사용된다.



예시 5. Transversal matroid

$G = (V, E)$가 bipartition $V_1$과 $V_2 $를 가지는 이분그래프(bipartite graph)일 때, 

$\mathcal{I} = \left\{ I : I \subset V_1, \exist \: a \: matching \: M \: in \: G \: that \: covers \: I \right\}$ 로 두면 $\mathcal{M} = (V_1, \mathcal{I})$ 은 matroid이다. 3번 조건의 경우 $X$ 와 $Y$ 각각에 매칭된 vertex들을 생각하면 자명하다.



예시 6. Matching matroid

무향그래프 $G = (V, E)$에서 $\mathcal{I} = \left\{ I : I \subset V, \exist \: a \: matching \: M \: in \: G \: that \: covers \: I \right\}$ 로 두면 $\mathcal{M} = (V, \mathcal{I})$ 은 matroid이다.



##매트로이드에서 쓰이는 기본 용어 및 성질



정의 2. $\mathcal{M} = (S, \mathcal{I})$ 가 매트로이드일 때, $S$의 부분집합 중 $\mathcal{I}$에 포함되지 않는 것을 $\mathcal{M}$의  dependent set이라 한다.

정의 3. $\mathcal{M}$의 independent set $I$에 대해 $I$를 진부분집합으로 갖는 independent set이 없다면 $I$를 $\mathcal{M}$의 base라고 한다.

성질 1. $\mathcal{M}$의 모든 base들의 크기는 동일하다.

​	이것은 matroid의 3번 조건에 의해 간단하게 증명 가능하다.



정의 4. $\mathcal{M} = (S, \mathcal{I}) 가 matroid이고 $S' \subset S$ 일 때, $\mathcal{I}' = \left\{ I : I \subset S', I \in \mathcal{I} \right\}$ 로 두면 $\mathcal{M}' = (S', \mathcal{I}') $ 역시 matroid이고 이를 $\mathcal{M}$의 $S'$에 대한 restriction이라고 한다.

