# Centroid of Tree

## Centroid
트리에서 centroid는, 어떤 정점 $v$가 크기 $n$짜리 트리 $T$에서 삭제했을 경우 생기는 subtree들이 모두 각각 크기가 $n/2$ 이하가 되는 경우, $v$를 $T$의 centroid라고 합니다. 

![](/assets/images/juney/centroid/1.png)

위 그림에서는 1번 노드가 centroid 입니다. 1을 삭제했을 때 각각 크기가 3, 4, 4인 서브트리가 생기고 전체 트리의 크기는 11이므로 위에 말한 조건을 만족하기 때문입니다.

Centroid는 트리에서 문제를 해결하는 경우에 중요한 역할을 하는 경우가 많습니다.

먼저, centroid는 어떤 트리에서 존재할까요?

**Theorem 1.** 임의의 트리 $T$에는 항상 centroid가 하나 이상 존재한다

**Proof.** 크기 $n$짜리 트리 $T$에서 다음과 같은 알고리즘을 생각해 봅시다.

1. T에서 임의의 정점 $v$를 고른다.
2. $v$를 삭제했을때 생기는 subtree들이 모두 $n/2$ 이하이면 $v$가 centroid이므로 알고리즘을 마친다.
3. 아닌경우 $n/2$ 보다 큰 서브트리 하나가 존재할 것이다. (두 개 이상 존재하면 $T$의 크기가 $n$이라는 조건에 모순 발생) 그 서브트리의 루트를 $u$라고 할때, $v$를 $u$로 바꿔주고 2. 로 돌아간다.

이때, 같은 정점은 다시 방문되지 않을 것입니다. 그 이유는, 만약 $v$에서 $u$로 이동했다는 것은, $u$의 크기가 $n/2$ 보다 크다는 것이고, $u$ 기준에서 $v$를 루트로 하는 서브트리는 $n/2$이하일 것이기 때문에 이 알고리즘에서 $v$를 다시 방문할 일은 없을 것이기 때문입니다.

따라서 정점의 개수는 $n$으로 정해져 있기 때문에, $n$번 이하의 반복으로 $T$에서 centroid를 항상 찾을 수 있을 것입니다.

Theorem 1. 증명과 동시에 임의의 트리에서 centroid를 찾는 알고리즘을 알게 되었습니다. 이것을 직접 코드로 구현해보면 다음과 같습니다.

```cpp
vector<int> G[MAXN]; // 그래프의 인접리스트
int S[MAXN]; // S[x]: x를 루트로 하는 서브트리의 크기

// init(cur, par): S 배열의 값을 계산해주는 함수
int init(int cur, int par) {
    S[cur] = 1;
    for(int nxt : G[cur]) if(nxt != par) {
        S[cur] += init(nxt, cur);
    }
    return S[cur];
}

// centroid(cur, par, sz): init 후 트리의 centroid를 반환하는 함수
int centroid(int cur, int par, int sz) {
    for(int nxt : G[cur]) if(nxt != par) {
        if(sz / 2 < S[nxt]) {
            return centroid(nxt, cur, sz);
        }
    }
    return cur;
}
```

이제 centroid는 항상 존재하다는 것은 확인했습니다. 그렇다면 centroid는 최대 몇 개까지 존재할 수 있을까요?

**Theroem 2.** 임의의 트리 $T$에서 centroid는 최대 2개 까지 은재할 수 있다.

**Proof.** 크기 $n$짜리 임의의 트리 $T$에서 centroid가 3개 이상 존재한다고 가정해 봅시다. 첫 번째 centroid를 $x$라고 했을때, $x$를 기준으로 한 subtree들이 모두 $n/2$보다 작으면 $x$를 제외한 모든 정점은 $x$를 포함하는 서브트리가 $n/2$보다 클 것이므로 centroid가 될 수 없고 이는 가정에 모순입니다. 따라서 정확히 $n/2$인 subtree는 항상 하나 존재할 것입니다. 그 subtree의 루트를 y라고 합시다. 이때, $y$를 기준으로 x를 루트로하는 서브트리는 크기가 $n/2$일 것이고, 나머지 subtree는 $n/2$ 보다 작을 것입니다. 따라서 $y$는 두 번째 centroid 입니다. 하지만, 방금 언급한 것 처럼 나머지 subtree는 $n/2$ 보다 작을 것이므로, 그 subtree들에서는 위에 말한 이유에서 centroid인 정점이 존재할 수 없습니다. 따라서 $x$와 $y$가 유일한 centroid가 될 수 밖에 없고 이는 가정에 모순이므로, Theorem 2가 증명됩니다.

Theorem 2.를 증명하면서 우리는 자동적으로 한 가지 사실을 알게 되었습니다. 바로 centroid가 2개 존재한다면 그 centroid들은 서로 하나의 간선으로 이어져 있다는 사실입니다.

**Tip 1.** Centroid가 2개인 경우 그 2개는 서로 엣지로 이어져 있습니다. 만약 centroid를 이용하는 문제에서 centroid가 2개 나오게 된다면, 그 둘을 잇는 간선에 edge division을 하여 새로운 트리를 만든다면 centroid가 새로 생긴 정점 하나로 정해지기 때문에, 관련 문제를 코드로 해결하는 경우에 가끔씩 편리한 코딩에 도움이 됩니다.


지금 까지는 centroid와 centroid를 구하는 방법, 그리고 centroid의 몇 가지 특징들을 알아 보았습니다. 다음은 centroid를 이용해서 트리 관련 문제를 해결할 때 사용할 수 있는 유명한 방법들에 대해서 알아보겠습니다.

## Centroid Decomposition
알고리즘 문제들에 익숙하다면, 어떠한 수열에서 분할정복을 하는 문제를 자주 보았을 것입니다. 대표적으로 정렬 알고리즘인 *Merge Sort* 알고리즘이 수열에서 분할정복을 이용합니다.

수열에서 분할정복은 일반적으로 다음과 같은 단계로 이루어집니다.

1. 수열의 중간 지점 $m$을 기준으로 수열을 왼쪽 오른쪽 두 개의 구간으로 나눈다.
2. 왼쪽 구간과 오른쪽 구간에 대하야 부분 문제를 해결한다.
3. 두 구간에서 구한 결과를 바탕으로 전체 문제를 해결한다.

*Merge Sort*의 경우 중간 지점 $m$을 기준으로 왼쪽 구간 오른쪽 구간에 대해서 정렬하고 그것을 바탕으로 전체 구간을 합치는 과정을 재귀 적으로 반복하여 $O(NlogN)$ 이라는 빠른 시간복잡도 안에 정렬을 완료합니다.

트리에서도 수열처럼 분할정복을 하고 싶으면 어떡할까요? 먼저 1.에 해당하는 과정을 봅시다. 수열에서는 중간 지점 $m$ 을 단순히 현재 해결하고 있는 구간 $[l, r]$에 대하여 $m = (l + r) / 2$와 같이 구할 수 있습니다. 트리에서는 어떨까요? 맞습니다. 바로 centroid를 중간 지점으로 잡는 것입니다. 이렇게 하면 수열에서 구간을 반으로 나누는 효과와 비슷한 효과를 낼 수 있기 때문입니다.

나머지도 거의 비슷합니다. 2.에 해당하는 과정의 경우 왼쪽 구간, 오른쪽 구간 대신, centroid를 기준으로 나뉘어진 subtree들에 대해서 문제를 하고, 3.에 해당하는 과정은 subtree들의 결과를 바탕으로 원래 트리에서의 문제를 해결하면 됩니다.

**Tip 2.** 수열을 일직선인 트리(또는 *chain*)이라고 생각하면 수열에서 분할정복 하는 것과 트리에서 분할정복하는 것은 원리가 같다는 것을 이해하는데 편합니다.

이 과정의 기본적인 구조를 코드로 나타내면 다음과 같습니다.
```cpp
vector<int> G[MAXN]; // 그래프의 인접리스트
int S[MAXN]; // S[x]: x를 루트로 하는 서브트리의 크기
int del[MAXN]; // del[x]: 이미 한번 decompse에서 centroid로 지목된 경우 1, 아니면 0

// init(cur, par): S 배열의 값을 계산해주는 함수
int init(int cur, int par) {
    S[cur] = 1;
    for(int nxt : G[cur]) if(nxt != par $$ !del[nxt]) {
        S[cur] += init(nxt, cur);
    }
    return S[cur];
}

// centroid(cur, par, sz): init 후 트리의 centroid를 반환하는 함수
int centroid(int cur, int par, int sz) {
    for(int nxt : G[cur]) if(nxt != par && !del[nxt]) {
        if(sz / 2 < S[nxt]) {
            return centroid(nxt, cur, sz);
        }
    }
    return cur;
}

// decompose(root): root를 루트로하는 subtree에서 분할정복을 통해 subtree의 문제를 해결하는 함수
void decompose(int root) {
    int sz = init(root, -1); // root를 기준으로하는 새로운 트리이기 때문에 새로 init
    int cen = centroid(root, -1, sz); // 현재 트리에서 centroid
    del[cen] = 1; // 다시 방문하지 않을 것이기 때문에 표시
    
    // do something

    for(int nxt : G[cen]) if(!del[nxt]) {
        decompose(i, cen); // cen을 기준으로하는 서브트리들에서 문제 해결
        // do something
    }

    // do something
}
```
