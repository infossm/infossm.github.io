---
layout: post
title: "압축 트라이 (Compressed Trie)"
author: cheetose
date: 2021-02-20
tags: [string, data-structure]
---

## 서론

본 글에서는 트라이를 압축하여 트라이의 깊이를 $O(\sqrt (\sum \vert S \vert))$로 만드는 기법에 대해 설명하고자 합니다. 해당 방법은 트라이와 라빈-카프 알고리즘에 대한 선행 개념이 필요합니다. 해당 개념을 모르는 사람들을 위해 아래에 간략하게 설명하겠습니다.

**트라이**는 접두사 트리로, 어떤 문자열 집합의 prefix를 관리하는 자료구조입니다. 예를 들어 문자열 집합이 {"baby", "bank", "be", "bed", "box", "dad", "dance"}라면 이 집합을 표현한 트라이는 아래 그림과 같습니다. 이 때 붉은색 테두리는 여기서 끝나는 문자열이 존재한다는 것을 의미합니다. 이를 앞으로 valid한 노드라고 표현하겠습니다.

![trie](/assets/images/cheetose-post/1/trie.png)

위 그림에서 볼 수 있듯이 모든 노드는 집합 내의 어떠한 문자열의 접두사를 가리키고 있습니다. 예를 들어 baby라는 문자열에 도달하기 위해서는 루트->b->a->b->y 순으로 이동해야하는데 각 노드까지 방문하는 동안의 모든 문자를 모은 문자열을 나열해보면 ""->"b"->"ba"->"bab"->"baby"로, 모두 "baby"의 접두사입니다.

**라빈-카프 알고리즘**는 어떠한 문자열을 하나의 정수로 표현하여 두 문자열이 같은지 $O(1)$에 확인할 수 있는 해시 기반 알고리즘입니다. 문자열 $S$에 대해서 해시값을 정해보겠습니다. 어떤 적당한 수 $p$를 정합시다. 그리고 $S[1 .. i]$의 해시값을 $H[i] = H[i-1] \times p + S[i]$ 로 설정합니다.

그렇다면 $S[l .. r]$의 해시값은 어떻게 구할 수 있을까요? $H[r]$을 풀어서 쓰면 $S[1] \times p^{r-1} + S[2] \times p^{r-2} + ... + S[r-1] \times p + S[r]$로 표현이 가능하고 $H[l-1]$을 풀어서 쓰면 $S[1] \times p^{l-2} + S[2] \times p^{l-3} + ... + S[l-2] \times p + S[l-1]$로 표현할 수 있습니다. $H[l-1]$에 $p^{r-l+1}$을 곱하면 $H[l-1] \times p^{r-l+1} = S[1] \times p^{r-1} + S[2] \times p^{r-2} + ... + S[l-2] \times p^{r-l+2} + S[l-1] \times p^{r-l+1}$이 되고 이 값을 $H[r]$에서 빼면 $S[l] \times p^{r-l} + S[l+1] \times p^{r-l-1} + ... + S[r-1] \times p + S[r]$이 됩니다. 이러한 과정을 통해 $S[l .. r]$의 해시값을 구할 수 있습니다.

정수의 범위가 무한하진 않으니 실제로는 적당히 큰 $M$으로 나눈 modulo로 각 해시값을 설정해야하고, 해시 충돌의 위험이 있기 때문에 이중/삼중 해시를 이용해서 계산하면 더욱 안전하게 이용할 수 있습니다. 다만 본글에서는 편의를 위해 단일 해시를 이용하여 설명하겠습니다.

## 본론

이번 글의 목표는 해당 방법을 통해서 [Separate String](https://www.acmicpc.net/problem/15525) 문제를 풀어보는 것입니다.

문제를 간단하게 요약하자면 여러 단어들로 이루어진 사전이 존재하고, 10만자 이하의 문자열 $T$가 주어질 때 사전에 있는 단어들만 이용하여 $T$를 분리하는 방법의 수를 구하는 문제입니다. 이 때 사전에 존재하는 모든 단어의 길이의 합은 20만 이하입니다. 앞으로 $T$의 길이를 $N$, 사전에 있는 문자열들의 길이의 합($\sum \vert S \vert$)을 $M$으로 표현하겠습니다.

먼저 트라이를 압축시키지 않은 상태에서의 풀이를 설명하겠습니다.

해당 문제를 풀기 위해 아래와 같은 점화식을 생각해봅시다.

"$d[i]= T[i .. N]$을 사전에 있는 문자열들로 분리시키는 경우의 수"

만약 $T[i .. j]$가 사전에 있는 문자열이라면 $T[i .. j]$와 $T[j+1 .. N]$으로 나눌 수 있고, $d[i]$에는 나머지 $T[j+1 .. N]$를 분리시키는 경우의 수, 즉 $d[j+1]$을 더해주면 됩니다. 트라이의 루트로부터 시작해서 현재 $T[j]$에 해당하는 문자를 보고있고, 해당 트라이에서 $T[i .. j]$에 해당하는 노드가 존재함과 동시에 valid하다면 $d[i]$에 $d[j+1]$을 더하고 다음 문자로 이동시키며 계속해서 업데이트를 해주면 됩니다.

위의 알고리즘을 코드로 작성하면 아래와 같습니다.

```cpp
#include <bits/stdc++.h>
using namespace std;

struct node {
	bool valid; // 해당 노드가 valid한지 나타내는 변수
	int child[26]; // 해당 노드의 각 자식 노드가 가리키는 인덱스
	node() {
		valid = false;
		memset(child, -1, sizeof(child)); // 노드를 생성할 때 child 배열을 -1로 초기화를 해줍니다.
	}
};
vector<node> trie;

int init() {
	node x;
	trie.push_back(x);
	return (int)trie.size() - 1;
}

void add(int n, string &s, int i) {
	if (i == s.length()) { // s의 모든 문자를 추가했으면 valid 값을 true로 바꾸고 종료시킵니다.
		trie[n].valid = true;
		return;
	}
	int c = s[i] - 'a';
	if (trie[n].child[c] == -1) { // c로 향하는 자식이 없으면 새로운 노드를 만들어 연결시킵니다.
		int next = init();
		trie[n].child[c] = next;
	}
	add(trie[n].child[c], s, i + 1);
}

string t;
int d[100000];
const int MOD = 1000000007;
int go(int N) {
	if (N == t.length()) return 1;
	int &ret = d[N];
	if (~ret) return ret;
	ret = 0;
	int n = 0; // 트라이에서의 현재 위치한 정점 번호
	while(1) {
		if (N == t.length()) break;
		int c = t[N] - 'a';
		if (trie[n].child[c] == -1) break; // t[N]에 해당하는 자식이 없으면 반복문을 탈출합니다.
		n = trie[n].child[c]; // 있으면 해당 노드로 이동합니다.
		if (trie[n].valid) ret = (ret + go(N+1)) % MOD; // 그 노드가 valid 하다면 d[N+1]을 더해줍니다.
													   // 이 때의 N이 우리가 지금 채우고 있는 d[N]의 N과 다른 값이라는 점에 주의합니다.
		N++; // 다음 문자로 넘어갑니다.
	}
	return ret;
}
int main() {
	ios_base::sync_with_stdio(0);
	cin.tie(0);
	memset(d, -1, sizeof(d));

	init(); // 트라이의 루트를 만들어줍니다. 루트의 번호는 0입니다.
	int n;
	cin >> n;
	for(int i = 0; i < n; i++) {
		string s;
		cin >> s;
		add(0, s, 0); // s를 트라이에 추가합니다.
	}
	cin >> t;
	cout << go(0) << '\n';
	return 0;
}
```

$i$번째 문자에서 $i+1$번째 문자로 넘어갈 때 트라이에서도 어떤 위치에서 다음으로 이동할 수 있는 곳이 1개뿐이므로 각 $d[i]$를 채우는 데 걸리는 시간은 트라이의 깊이에 비례하게 됩니다. 따라서 해당 알고리즘의 시간복잡도는 $O(NM)$이 됩니다. $S$가 "aaa...aaa" (a가 20만개), $T$가 "aaa..aaa" (a가 10만개)인 경우에는 실제로 굉장히 오래 걸리게 됩니다.

이제 본격적으로 트라이의 깊이를 $O(\sqrt M)$으로 만들어봅시다.

서론에서 예시로 들었던 트라이를 잘 살펴봅시다. 저기서 실제로 의미가 있는 노드는 뭐가 있을까요? 사실 valid한 노드 혹인 자식이 2개 이상인 노드가 아니면 자식 노드와 합칠 수가 있습니다. 왜냐하면 valic하지 않고 자식이 1개뿐인 노드는 다음에 갈 수 있는 곳이 단 하나로 정해져있고, valid하지 않기 때문에 dp 값에 영향을 주지도 않기 때문입니다. 따라서 이러한 노드들만 트라이에서 남겨놓게 된다면 트라이에서 탐색할 때 껑충껑충 뛰어다니며(?) 더 효율적으로 탐색할 수 있습니다. 위의 트라이를 압축한다면 아래 그림처럼 만들 수 있을 것입니다. 마찬가지로 붉은색 테두리는 valid한 노드를 의미합니다.

![compressed trie](/assets/images/cheetose-post/1/compressed trie.png)

위에서 언급했듯이 모든 노드는 valid하거나 자식이 2개 이상입니다. 보시다시피 "dance"에 도달하기 위해서 기존에는 5번 이동해야했지만 압축된 트라이에서는 단 2번에 도달할 수 있음을 볼 수 있습니다.

과연 이러한 행위가 깊이를 얼마나 줄일 수 있을까요?

압축된 트라이에서 정점이 존재한다는 것은 해당 정점이 valid하거나 '루트부터 그 정점까지 이어붙인 문자열을 접두사로 하는 문자열'이 사전에 2개 이상 존재한다는 것을 의미합니다. 이 때 트라이의 깊이를 최대한 깊게 만드는 건 각 정점에 해당하는 문자열이 1글자이고, 모든 정점이 valid한 경우입니다. 즉, 사전에 "a", "aa", "aaa", ... 꼴로 들어있을 때 트라이의 깊이가 가장 깊게 됩니다. 트라이의 깊이를 $x$라 하면 $\sum_{i=1}^x i = M$이므로 $x$는 약 $\sqrt M$임을 알 수 있습니다.

이제 본격적으로 트라이를 압축해봅시다.

![ex1](/assets/images/cheetose-post/1/ex1.png)

현재 위 트라이에서 'a'에 위치해있다고 생각합시다. 다음으로 확인해야하는 문자가 'n'일 때 'n'과 'c'를 생략하고 바로 'e'로 넘어가는 것이 목표입니다. 결론부터 말하자면 트라이를 압축하는 과정에서 정점의 각 자식 정점에 들어갈 정보는 총 3가지입니다.

- 다음 정점으로 넘어가는 동안 만나는 문자열(의 해시값)
- 그 문자열의 길이
- 해당 정점의 번호

예를 들어 위 그림에서 'a' 정점에서 'n'에 해당하는 자식 정점에는 "nce"에 해당하는 해시값, "nce"의 길이인 3, 그리고 압축 전 트라이에서 'e'에 해당하는 노드 번호를 저장합니다. 자세한 구현 방식은 아래 코드의 주석으로 설명하겠습니다. $p$의 값으로는 37, modulo $M$의 값으로는 1000000021 ($10^9 + 21$)을 사용했습니다.

```cpp
#include <bits/stdc++.h>
typedef long long ll;
using namespace std;

struct node {
	bool valid;
	int child[26], child_cnt; // 추가로 자식의 개수를 나타내는 변수를 만들어줍니다. 자식의 개수도 압축의 기준이기 때문에..
	int len, H; // 정점에 해당하는 문자열의 길이와 해시값을 의미합니다.
	node() {
		valid = false;
		child_cnt = 0;
		memset(child, -1, sizeof(child));
	}
};
vector<node> trie;

int init() {
	node x;
	trie.push_back(x);
	return (int)trie.size() - 1;
}

void add(int n, string &s, int i) {
	if (i == s.length()) {
		trie[n].valid = true;
		return;
	}
	int c = s[i] - 'a';
	if (trie[n].child[c] == -1) {
		trie[n].child_cnt++; // 새로운 자식이 생기면 child_cnt를 1 증가시킵니다.
		int next = init();
		trie[n].child[c] = next;
	}
	add(trie[n].child[c], s, i + 1);
}

string t;
ll h[100001], p[100001];
int d[100000];
const int MOD = 1000000021;

tuple<int,int,int> dfs2(int n, int d, int h) { // (정점번호, 문자열 길이, 문자열의 해시값)을 반환하는 함수
	if(trie[n].valid || trie[n].child_cnt > 1)return {n, d, h}; // 정점이 valid하거나 자식이 2개 이상 있으면 여기까지만 압축을 진행하고, 현재까지의 정보들(정점 번호, 문자열 길이, 해시값)을 반환합니다.
	for(int i = 0; i < 26; i++) {
		if(trie[n].child[i] != -1)
			return dfs2(trie[n].child[i], d + 1, (1LL * h * 37 + i + 1) % MOD); // h[i] = (h[i-1]*p) % MOD 를 여기에도 적용시켜 인자로 넘깁니다. 이 때 자식의 개수가 1개임이 보장되어있으므로 바로 return을 해줘도 됩니다.
	}
	return {};
}
void dfs(int n) {
	for(int i = 0; i < 26; i++) {
		if(trie[n].child[i] != -1) {
			auto [nx, L, H] = dfs2(trie[n].child[i], 1, i + 1); // nx, L, H는 각각 도착 정점, 문자열 길이, 해시값을 의미합니다.
			trie[n].child[i] = nx, trie[nx].len = L, trie[nx].H = H; // 현재 정점 n의 i에 해당하는 자식에 nx를 연결시키고, nx에 해당하는 문자열 정보도 수정합니다.
			dfs(nx); // 후에 다시 nx부터 압축을 진행합니다.
		}
	}
}

ll Hash(int l,int r) { // t[l .. r]에 해당하는 해시값을 구하는 함수
	l++, r++;
	int t = r - l + 1;
	ll res = (h[r] - h[l - 1] * p[t]) % MOD;
	if (res < 0) res += MOD;
	return res;
}

int go(int N){
	if(N==t.length()) return 1;
	int &ret = d[N];
	if(~ret) return ret;
	ret = 0;
	int n = 0;
	while(1) {
		if(N == t.length()) break;
		int nx = trie[n].child[t[N] - 'a']; // t[N]에 해당하는 자식의 정점 번호
		if(nx==-1) break; // t[N]에 해당하는 자식이 없다면 break 해줍니다.
		int L = trie[nx].len;
		if(N + L > t.length()) break; // N+L이 t의 길이보다 큰지, 즉 이번에 확인할 문자열의 범위가 t의 범위를 벗어나는지 확인합니다.
		if(trie[nx].H == Hash(N, N + L - 1)) { // nx에 저장되어있는 해시값과 T[N .. N+L-1]의 해시값이 같으면 두 문자열이 일치하여 해당 정점으로 갈 수 있음을 의미합니다.
			if(trie[nx].valid) ret = (ret + go(N + L)) % 1000000007; // 해당 노드가 valid하면 d[N+L]을 더해줍니다. T[N ..]을 T[N .. N+L-1]과 T[N+L ..]로 나눈다고 생각하면 됩니다.
			n = nx;
			N += L;
		} else break; // 해시값이 다르면 둘은 다른 문자열이므로 반복문을 탈출해줍니다.
	}
	return ret;
}

int main() {
	ios_base::sync_with_stdio(0);
	cin.tie(0);
	memset(d, -1, sizeof(d));
	p[0] = 1;
	for(int i = 1; i <= 100000; i++) p[i] = p[i - 1] * 37 % MOD; // p[i]=37^i

	init();
	int n;
	cin >> n;
	for(int i = 0; i < n; i++) {
		string s;
		cin >> s;
		add(0, s, 0);
	}
	dfs(0); // 트라이 압축
	cin >> t;
	for(int i = 0; i < (int)t.length(); i++) h[i + 1] = (h[i] * 37 + t[i] - 'a' + 1) % MOD; // t의 해시값을 저장합니다.
	cout << go(0) << '\n';
}
```

트라이를 압축하지 않은 상태에서의 풀이와 상당히 비슷하지만 트라이의 최대 깊이가 $\sqrt M$이 되었기 때문에 시간복잡도는 $O(N \sqrt M)$이 되어 문제를 해결할 수 있습니다. 이 문제에서는 다행히 단일 해시로도 문제가 풀렸지만 만약 해시 충돌이 일어나 Wrong Answer를 받는다면 이중/삼중 해시를 통해 해결할 수 있습니다.

## 결론

지금까지 트라이의 깊이를 $O(\sqrt M)$으로 줄이는 방법에 대해서 설명했습니다. 이 문제와 뒤에서 소개할 문제들은 모두 아호코라식 알고리즘을 통해 해결할 수 있습니다. 하지만 해당 알고리즘을 모르더라도 약간의 시간복잡도를 희생해서 문제를 풀 수 있는 좋은 방법이라고 생각합니다.

마지막으로 이 방법을 통해 해결할 수 있는 문제 2개를 소개하고 글을 마치겠습니다.

[메이플스토리 연주회](http://www.jungol.co.kr/bbs/board.php?bo_table=pbank&wr_id=3849)(채점불가) - 본문의 문제는 모든 문자열이 다르지만, 이 문제는 같은 문자열이 나올 수 있습니다.

[The ABCD Murderer](https://www.acmicpc.net/problem/16694) - 트라이 압축을 잘하면 ~~아슬아슬하게~~ 시간 내에 통과할 수 있습니다.
