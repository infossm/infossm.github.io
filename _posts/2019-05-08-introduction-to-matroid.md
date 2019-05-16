# Matroid



정의 1. matroid $\mathcal{M} = (S,  \mathcal{I})$ 에서 $S$는 유한집합, $ \mathcal{I} \subset 2^S$ 는 독립집합(independent set)들의 collection이다. 이 때, $I$는 다음 세 가지 조건을 만족하여야 한다.

1. $\phi \in  \mathcal{I}$
2. $Y \subset X, X \in  \mathcal{I} \Rightarrow Y \in  \mathcal{I}$ 
3. $X, Y \in  \mathcal{I}, |X| < |Y|$ 이면 $X + y \in  \mathcal{I}$ 를 만족하는 $y \in Y \setminus X$가 존재



매트로이드는 다양한 집합에서 정의될 수 있다. 그 중 대표적인 예 몇 가지를 살펴보자.

예시 1. Vector matroid

체 $\mathbb{F}$ 상에서 정의된 $m \times n$ 행렬 $A$을 생각하자. $v_i$를 $A$의 $i$번째 column vector라 하면 $v_i$들은 벡터공간 $\mathbb{F}^m$의 벡터이다. $S = \left\{1, 2, .., n \right\}$ ,  $\mathcal{I} = \left\{ I : I \subset S, \left\{v_i\right\}_{i \in I} \: are \: linearly \: independent \right\}$ 로 놓으면 $\mathcal{M} = (S, \mathcal{I})$는 matroid가 됨을 쉽게 알 수 있다. (1, 2번 조건은 자명하고, 3번의 경우 $|X| < |Y|$이고 둘 모두 independent한 vector들의 집합이므로 $X$의 벡터들이 $Y$의 벡터공간 전체를 span할 수 없다)

예시 2. Graphic matroid

$G = (V, E)$가 무향 그래프일 때,  $\mathcal{I} = \left\{I : I \subset E, I \: induces \: a \: forest \: in \: G \right\}$로 놓으면 $\mathcal{M} = (E, \mathcal{I})$는 matroid가 된다. 1, 2번 조건은 앞서와 마찬가지로 자명하고, 3번 조건의 경우 $X$에 포함된 edge들을 모두 이었을 때 component의 개수는 $N - |X|$이고, $X + y \in  \mathcal{I}$ 를 만족하는 $y \in Y \setminus X$가 존재하지 않는다면 $Y$의 edge들의 두 끝점이 한 component에 들어가야 하므로 $Y$에 포함된 edge들을 모두 이었을 때 component의 개수는 $N-|X|$ 이하인데 이것은 $N-|Y|$와 같아야 하므로 $|X| <|Y|$에 모순이다. 따라서, 3번 조건 역시 만족한다. Graphic matroid의 경우는 뒤에 다룰 minimum spanning tree를 구하는 kruskal 알고리즘의 증명에 이용된다.



예시 2-1. Graphic matroid의 변형

그래프에서 matroid는 예시 2처럼만 정의할 수 있는 것이 아니다. $G = (V, E)$가 무향 연결 그래프일 때, $\mathcal{I} = \left\{I : I \subset E, E - I \: connects \: all \: vertex \: in \: G \right\}$로 두면 $\mathcal{M} = (E, \mathcal{I})$는 matroid이다. $\mathcal{I}$가 공집합을 포함해야 하므로 $G$가 connected가 아닌 경우에는 matroid가 되지 않음에 주의해야 한다. 또 다른 예시로는 어떤 edge 집합으로 induced되는 그래프에서 각 component의 edge 개수가 vertex 개수를 넘지 않는 경우를 독립집합으로 정의하면 matroid가 된다. (즉, 각 컴포넌트가 트리 형태를 띠거나 하나의 cycle에 tree가 붙어 있는 형태인 경우) 두 가지 matroid 모두 정의 3이 간단히 증명되며, 나중에 소개할 maximum weight independent set algorithm을 이용하면 재미있는 결과를 얻을 수 있다.



예시 3. Uniform matroid

Uniform matroid는 어쩌면 생각할 수 있는 가장 쉬운 matroid이다. 어떠한 $k$를 정한 후 $S$에서 크기가 $k$ 이하인 모든 부분집합을 independent set으로 놓으면 uniform matroid가 된다.



예시 4. Partition matroid

Partition matroid는 Uniform matroid의 일반화라고 볼 수 있다. $S_1, S_2, ..., S_n$이 $S$의 분할이고, $k_1, k_2, ..., k_n$이 양의 정수일 때, $\mathcal{I} = \left\{ I : I \subset S, | I \cap S_i | \le k_i \: for \: all \: 1 \le i \le n \right\}$  로 정의하면 $\mathcal{M} = (S, \mathcal{I})$는 matroid이다. partition matroid나 uniform matroid 같은 경우는 matroid임이 자명하기 때문에 이것이 매트로이드라는 것이 특별한 것은 아니지만, 다음 포스팅에서 다룰 matroid intersection의 경우 graphic matroid나 vector matroid 등 다른 matroid와의 maximal matroid intersection / maximum weight matroid intersection을 구하는 문제 등에서 사용된다.



예시 5. Transversal matroid

$G = (V, E)$가 bipartition $V_1$과 $V_2 $를 가지는 이분그래프(bipartite graph)일 때, 

$\mathcal{I} = \left\{ I : I \subset V_1, \exist \: a \: matching \: M \: in \: G \: that \: covers \: I \right\}$ 로 두면 $\mathcal{M} = (V_1, \mathcal{I})$ 은 matroid이다. 3번 조건의 경우 $X$ 와 $Y$ 각각에 매칭된 vertex들을 생각하면 자명하다.



예시 6. Matching matroid

무향그래프 $G = (V, E)$에서 $\mathcal{I} = \left\{ I : I \subset V, \exist \: a \: matching \: M \: in \: G \: that \: covers \: I \right\}$ 로 두면 $\mathcal{M} = (V, \mathcal{I})$ 은 matroid이다.



## 매트로이드에서 쓰이는 기본 용어 및 성질



정의 2. $\mathcal{M} = (S, \mathcal{I})$가 매트로이드일 때, $S$의 부분집합 중 $\mathcal{I}$에 포함되지 않는 것을 $\mathcal{M}$의  dependent set이라 한다.

정의 3. $\mathcal{M} = (S, \mathcal{I})$가 매트로이드일 때, $S$의 원소 $s$에 대해 $s$ 하나만으로 이루어진 집합이 dependent하다면 $s$를 loop이라고 한다.

loop은 어떠한 independent set에도 포함될 수 없기 때문에 maximal independent set을 구할 때 보통 처음부터 제외시키고 생각한다.

정의 4. $\mathcal{M}$의 independent set $I$에 대해 $I$를 진부분집합으로 갖는 independent set이 없다면 $I$를 $\mathcal{M}$의 base라고 한다.

성질 1. $\mathcal{M}$의 모든 base들의 크기는 동일하다. 또한, base와 크기가 같은 independent set은 base이다.

증명. 이것은 정의 1의 조건 3에 의해 간단히 증명할 수 있다. 두 base의 크기가 다르다면 둘 중 크기가 작은 base에 원소를 하나 추가시켜도 independent하도록 만들 수 있는데 이는 base가 maximal이라는 정의에 모순이기 때문이다. base와 크기가 같은 independent set이 base가 아니라면 maximal이 아닌 것이므로 더 크기가 큰 independent set이 존재해야 하는데 이는 base들의 크기가 동일하다는 것과 대치된다.

성질 2. $\mathcal{M}$의 서로 다른 두 base $B_1$, $B_2$에 대해 $x \in B_1 \setminus B_2$, $y \in B_2 \setminus B_1$이 존재하여 $B_1 - x + y$가 $\mathcal{M}$의 base이다.

증명. $B_1$이 base이므로 $B_1 - x$는 independent set이고, $B_1 - x$와 $B_2$는 서로 다른 independent set이며 모든 base의 크기가 같기 때문에 $B_2$의 크기가 $B_1 - x$보다 크므로 $y \in B_2 \setminus B_1 - x$가 존재하여 $B_1 - x + y$는 independent이다. 이는 $B_1$, $B_2$와 크기가 같은 independent set이므로 base이다. 

정의 5. $\mathcal{M} = (S, \mathcal{I})$ 가 matroid이고 $S' \subset S$ 일 때, $\mathcal{I}' = \left\{ I : I \subset S', I \in \mathcal{I} \right\}$ 로 두면 $\mathcal{M}' = (S', \mathcal{I}') $ 역시 matroid이고 이를 $\mathcal{M}$의 $S'$에 대한 restriction이라고 한다. restriction이 matroid임은 정의에 의해 자명하다.

정의 6. matroid에서 minimal dependent set을 circuit이라 한다. 즉, $C \subset S$가 circuit일 조건은 $C \notin \mathcal{I}$이면서  $\forall x \in C, C-x \in \mathcal{I}$를 만족하는 경우이다.

graphic matroid에서의 circuit은 matroid가 정의된 그래프에서 simple cycle을 이룬다. 또한, 모든 loop은 그 자체로 circuit이다.

성질 3. $C_1$, $C_2$가 서로 다른 두 circuit이고 $x \in C_1 \cap C_2$ 라 하자. $C_1 \cup C_2- x$은 dependent하다. 즉, circuit을 포함한다.

증명. $C_1 \cup C_2- x$가 independent하다고 가정하자. $\mathcal{M}$의 $C_1 \cup C_2$에 대한 restriction을 생각했을 때, $ B = C_1 \cup C_2- x$은 base이다. $C_1$과 $C_2$가 circuit이므로 $I = C_1 \cap C_2$는 independent하다. matroid 정의 중 조건 3을 이용해서 $I$가 $B$와 크기가 같아지기 전까지 $I$에 포함되지 않고 $B$에는 포함되는 원소를 계속 추가하여 independent함을 유지할 수 있다. 그렇게 만들어진 $B$와 크기가 같은 집합을 $B'$라 하자. $|B'| = |C_1 \cup C_2| - 1$ 이고 $C_1 \cap C_2 \subset B'$ 이므로 $C_1 \subset B'$ 또는 $C_2 \subset B'$가 성립한다. 그런데 이는 $B'$는 independent함을 유지하면서 만들어진 집합이라는 것에 모순이다. 따라서, . $C_1 \cup C_2- x$은 dependent하다.

성질 4. $\mathcal{M} = (S, \mathcal{I})$ 가 matroid라 하자. $X \in \mathcal{I}, y \notin X$ 이면 $X+y \in \mathcal{I}$ 이거나 $X+y$가 unique한 circuit $C$를 포함하며, 그 경우 모든 $\widehat{y} \in C$ 에 대해 $X+y-\widehat{y} \in \mathcal{I}$ 가 성립한다.

증명. $X+y \notin \mathcal{I}$이면 $X + y$는 적어도 하나의 circuit을 포함한다. 이를 $C_1$이라 하자. 만약 $X + y$가 다른 circuit $C_2$도 포함한다면 $X$는 independent하기 때문에 $C_1$과 $C_2$는 모두 $y$를 포함한다. $C_1 \cup C_2 - y$는 성질 3에 의해 dependent한데 이는 $C_1 \cup C_2 - y \subset X$에 모순이므로 $X+y$는 단 하나의 circuit만을 가진다. 이를 $C$라 하면 circuit의 정의에 의해 $X+y$의 모든 dependent set은 $C$를  포함해야 한다. 따라서 모든 $\widehat{y} \in C$ 에 대해 $X+y-\widehat{y} \in \mathcal{I}$ 가 성립한다.

### Finding a maximum weight independent set in a matroid



matroid가 주어졌을 때, maximum weight independent set은 매우 빠른 시간에 계산할 수 있다. 또한, 음 아닌 정수 $k$가 주어졌을 때 size가 $k$인 independent set 중 weight가 maximum인 set도 쉽게 구할 수 있다. (minimum은 -1을 곱하면 maximum과 같은 방법으로 구해진다)

정리 1. 다음의 greedy 알고리즘은 $\mathcal{M} = (S, \mathcal{I})$의  maximum weight independent set을 올바르게 구한다.

1. $e \in S $의 weight을  $w(e)$로 표시하자. $e$가 loop(그 자체로 dependent한 원소)이거나 $w(e) \le 0$인 원소이면 무시한다.
2. 1에서 무시한 원소를 제거했을 때 $S = \left\{e_1, ..., e_n\right\}$ 에서 $w(e_1) \ge w(e_2) \ge ... \ge w(e_n)$ 이라 하자.
3. 초기 상태에서 $X \leftarrow \phi$.
4. $i$를 1부터 $n$까지 증가시키면서 $X + e_i$가 $\mathcal{I}$의 원소이면 $X \leftarrow X + e_i$를 대입한다.
5. $X$는 maximum weight independent set이다.

정리 1의 증명.

1번 step을 거치고 나면 $S = \left\{e_1, ..., e_n\right\}$ 에서 $w(e_1) \ge w(e_2) \ge ... \ge w(e_n) > 0$  이라 가정할 수 있다. 여기서 $e_1$을 포함하는 optimal solution(maximum weight independent set)이 항상 존재함을 보이면 $\mathcal{M}$의 $S-e_1$에 대한 restriction 역시 matroid이므로 수학적 귀납법에 의해 위의 greedy한 방법이 항상 optimal solution을 구한다는 것을 쉽게 보일 수 있다. 

claim : $e_1$을 포함하는 maixmum weight independent set이 존재한다.

$I^*$이  maximum weight independent set 중 하나라고 하자. $e_1 \in I^*$인 경우, claim을 만족하였다. $e_1 \notin I^*$인 경우 $I^*+e_1$은 dependent하므로 circuit을 가진다. 성질 4에 의해 $I^* - e + e_1 \in \mathcal{I}$을 만족하는 $e \in I^*$가 존재하고, $w(e) \le w(e_1)$이므로 $I^* - e +e_1$은 $e_1$을 포함하는 maximum weight independent set이 된다. 따라서, claim을 증명하였다. 



위의 알고리즘으로 해결할 수 있는 대표적인 문제로는 graph의 minimum weight spanning tree를 구하는 문제가 있다. 예시 2의 graphic matroid에서 알고리즘을 적용하기만 하면 된다. 이를 Kruskal's algorithm이라 한다. Graphic matroid에서만 이 알고리즘이 성립하는 것은 아니기 때문에 1의 vector matroid에도 이를 적용할 수 있다. 예를 들어, $N$개의 수로 이루어진 집합 $A =  \left\{a_1, a_2, ..., a_N\right\}$ 가 주어졌을 때 $S \subset A$를 골라 $S$의 어떤 공집합이 아닌 부분집합도 원소의 xor값이 0이 되지 않도록 하는 $S$ 중 원소의 합이 가장 큰 $S$를 구하려고 한다고 하자. 각 $a_i$들은 이진법으로 나타내면 체 $GF(2)$ 에서 정의된 벡터로 볼 수 있고, xor이 0이 아닌 것과 각 벡터들이 independent한 것이 동치이므로 매트로이드를 정의할 수 있다. 따라서, 큰 수부터 추가하면서 조건이 유지되는 것만 확인해주면 된다.



### 문제 풀이

Codeforces Round #441 Div. 1 F. Royal Questions

이 문제는 weighted bipartite graph $G = (V,E)$, $V = V_1 + V_2$에서 $V_1$의 모든 vertex $v_1$에 대해 $v_1$의  차수가 2이고 연결된 두 edge의 weight가 같다는 조건을 만족할 때 $G$에서 maximum weighted matching을 구하는 문제이다. 언뜻 보면 이 문제는 weighted bipartite matching을 써야만 해결할 수 있는 문제로 보이지만, $N$ 제한이 20만으로 매우 크기 때문에 weighted bipartite matching을 해결하는 Hungarian method로는 시간 제한 안에 답을 구할 수 없다.

그러나 이 문제는 예시 2-1에서 다룬 graphic matroid의 변형으로 환원할 수 있다. 새로운 그래프 $G' = (V_2, E')$를 정의하자. $G$에서 $v_1$과 연결된 두 edge의 가중치가 $w$, 끝점이 각각 $u_1$과 $u_2$였다면 $G'$에서는 $u_1$과 $u_2$ 사이에 가중치 $w$인 간선을 잇는다. 만약 $G$에서 $ M= \left\{e_1, e_2, ..., e_K\right\}$ 가 matching을 이룬다면, 즉 $M$의 edge들의 양 끝점이 모두 서로 다르다면 $M$의 양 끝점 중 $V_1$에 포함되는 vertex들을 $M_1 = \left\{v_1, v_2, ..., v_K\right\}$ 라 할 때  $M_1$에 포함되는 vertex들 각각에 대해 인접한 $V_2$의 vertex 두 개 중 하나를 잘 선택하면 서로 겹치지 않게 선택할 수 있다는 뜻이다. 이것은 $G'$에서 $M_1$에 의해 만들어지는 edge들만 놓고 보았을 때 edge들의 방향을 잘 주면 $G'$에서 각 vertex의 indegree가 1 이하가 되도록 할 수 있다는 것과 동치이다. 그리고 그것은 각 component에서 vertex의 개수가 edge의 개수보다 같거나 많아야 한다는 것과 동치임을 쉽게 알 수 있다.

따라서, $G$에서 maximum weighted matching을 구하는 문제는 $G'$에서 edge들을 잘 골라서 각 component의 vertex의 개수가 edge의 개수보다 적지 않도록 할 때, 선택된 edge들의 weight 합을 최대화하는 문제가 된다. $\mathcal{M} = (E', \mathcal{I})$는 matroid이므로, 앞서 설명한 알고리즘으로 해결할 수 있다. 즉, $E'$에서 weight가 큰 edge부터 추가하면서 disjoint set union 자료구조로 서로소집합을 관리하면 각 component의 vertex 개수와 edge 개수를 저장할 수 있으므로 $O(M log M)$ 시간에 문제를 해결할 수 있다.

간단한 코드는 아래와 같다.



```c++
#include<cstdio>
#include<algorithm>
using namespace std;
int n, m, SZ[201000], UF[201000];
long long res;
struct point{
    int a, b, c;
    bool operator <(const point &p)const{
        return c<p.c;
    }
}w[201000];
int Find(int a){
    if(a==UF[a])return a;
    return UF[a] = Find(UF[a]);
}
int main(){
    int i, a, b;
    scanf("%d%d",&n,&m); // number of vertices in V2 and V1
    for(i=0;i<m;i++){
        scanf("%d%d%d",&w[i].a,&w[i].b,&w[i].c); // vertex i+1 in V1 is connected to vertex w[i].a and w[i].b with weight w[i].c
    }
    sort(w,w+m);
    for(i=1;i<=n;i++)UF[i]=i,SZ[i]=1; // UF : union-find, SZ : (# of vertices) - (# of edges) in component
    for(i=m-1;i>=0;i--){
        a = Find(w[i].a), b = Find(w[i].b);
        if(SZ[a]+SZ[b]==0)continue; // Add an edge when SZ >= 0 is satisfied after adding the edge
        res += w[i].c;
        if(a==b){
            SZ[a]--;
        }
        else{
            UF[b] = a;
            SZ[a] += SZ[b]-1;
        }
    }
    printf("%lld\n",res);
}
```



위 문제는 예시 2-1에서 소개된 변형된 graph matroid 말고도 Transversal matroid라고 볼 수도 있다. 

$\mathcal{I} = \left\{ I : I \subset V_1, \exist \: a \: matching \: M \: in \: G \: that \: covers \: I \right\}$ 로 두면 $\mathcal{M} = (V_1, \mathcal{I})$는 matroid이고, 각 vertex의 weight는 연결된 두 edge의 weight으로 놓으면 된다. 하지만 각 vertex를 independent set에 추가할 수 있는지 빠르게 확인하기 위해서는 앞서 본 관찰을 하는 것이 필요하기 때문에 이것만으로는 빠른 시간에 문제를 해결하기 부족하다. 문제를 풀 때 matroid 구조를 찾기만 하면 해결되는 문제도 있지만 이처럼 그렇지 않은 경우 역시 존재한다.

## Next article - matroid intersection

다음 편에는 같은 집합에서 정의된 두 매트로이드 $\mathcal{M}_1 = (S, \mathcal{I}_1)$, $\mathcal{M}_2 = (S, \mathcal{I}_2)$에서 모두 independent한 $S$의 부분집합 중 원소의 개수가 가장 많은 (maximum) 집합 또는 가중치가 가장 큰 집합을 구하는 matroid intersection 알고리즘에 대해 다룰 것이다. matroid intersection 알고리즘을 이용하면 다른 방법으로는 절대 풀리지 않을 것처럼 보이는 어려운 문제들도 해결할 수 있다.
