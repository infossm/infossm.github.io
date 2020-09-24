---

layout: post

title: Wavelet Tree

author: junis3

date: 2019-10-20 23:00

tags: [data-structure]

---

## Wavelet Tree란?

Wavelet tree란, 문자열을 저장하는 자료구조입니다. 이진 트리 형태로, 문자열의 길이 $L$에 대해 $L + o(L)$의 메모리를 사용하여(이런 특성이 있는 자료구조를 간결한 자료구조(succinct data structure)라고 합니다) 아래 두 연산을 지원합니다.

- $\mathbf{rank}_c(x)$: 문자열의 첫 인덱스부터 $x$번째 인덱스까지 문자 $c$가 등장하는 횟수를 반환합니다.
- $\mathbf{select}_c(x)$: 문자 $c$가 $x$번째로 등장하는 인덱스를 반환합니다.

Wavelet이라는 이름은 신호를 재귀적으로 저주파수와 고주파수로 나누어 분해하는 wavelet transform의 이름에서 가져왔습니다.

## Wavelet tree의 구조

이 문단에서는 *일반적인 wavelet tree*의 개략적인 구조만을 다룹니다. 구조를 이해하셨다면 다음 문단으로 바로 넘어가셔도 무방합니다. 문자열 $S[1..n]$이 주어질 때, 이 문자열의 wavelet tree는 다음과 같이 건설됩니다.

- 문자열을 이루는 알파벳의 집합 $\Sigma$를 disjoint한 두 부분집합 $\Sigma_1$과 $\Sigma_2$ 둘로 나눕니다(각각 크기가 절반에 가까울수록 좋습니다.) 문자열의 각 문자에 대해서, 이 알파벳이 $\Sigma_1$의 원소라면 0으로, $\Sigma_2$의 원소라면 1로 인코딩해 wavelet tree의 루트에 저장합니다.
- 루트의 왼쪽 서브트리는 $S$ 안에서 $\Sigma_1$에 속하는 알파벳들만 남긴 부분열로 건설한 wavelet tree, 오른쪽 서브트리는 $S$ 안에서 $\Sigma_2$에 속하는 알파벳들만 남긴 부분열로 건설한 wavelet tree입니다.

아래 [그림 1]은 "alabar a la alabarda"라는 긴 문자열을 wavelet tree로 인코딩한 결과를 보여줍니다.

<img src="/assets/images/junis3/1910/1.PNG">

[그림 1] [https://users.dcc.uchile.cl/~gnavarro/ps/cpm12.pdf](https://users.dcc.uchile.cl/~gnavarro/ps/cpm12.pdf)

이 결과 문자열을 이루는 각 알파벳은 0과 1들로 인코딩되고, wavelet tree는 이 인코딩된 결과를 순서를 유지하며 모은 trie와 같습니다. 따라서 질의들도 trie와 유사한 방식으로 구현할 수 있다는 점을 눈치채셨을지도 모릅니다.

- $\mathbf{rank}_c(x)$는 wavelet tree의 루트에서 문자 $c$를 나타내는 이진 표현을 따라 내려가면서 구현할 수 있습니다. 예를 들어, 만약 알파벳 $c$가 루트에서 1로 표현되어 있고, 루트 노드에서 첫 $x$개 문자 중 1의 개수가 $x'$개라면, 오른쪽 자식으로 내려가 첫 $x'$개 문자에 대해 같은 질의를 반복할 수 있습니다. 물론 고속으로 찾기 위해서는 각 노드의 첫 $x$개의 문자 중 1의 개수를 전처리하여 가지고 있어야 할 것입니다.
- $\mathbf{select}_c(x)$는 더 쉽게 구현할 수 있습니다. 만약 각 노드의 각 문자에 대응되는 원래 문자열의 인덱스를 전처리하여 가지고 있다면, 문자 $c$를 나타내는 리프 노드로 내려가서 $c$가 $x$번째로 등장하는 인덱스를 바로 알 수 있습니다.

[https://alexbowe.com/wavelet-trees/](https://alexbowe.com/wavelet-trees/)에서 $\mathbf{rank}$ 질의를 예시를 들어 설명하고 있습니다. 위 설명으로 이해가 잘 안 된다면 참조해 보세요.

## Wavelet Tree를 이용한 문제 해결

위에서 소개한 *일반적인 wavelet tree*는 문자열을 최대한 적은 공간에 표현하려는 노력의 산물이었습니다. 따라서 충분한 메모리 저장공간이 있다면 wavelet tree가 더욱 다양한 기능을 수행하도록 할 수 있습니다.

먼저 우리가 다루던 객체인 문자열의 각 알파벳을 $1$부터 $m$까지의 정수에 대응시켜, 수열로 생각합니다. Wavelet tree는 알파벳의 개수에 대한 로그 시간에 작동했으므로, 정수의 범위가 매우 커져도 수행 시간엔 큰 차이가 없습니다. 아래 [그림 2]는 수열 $S = (3, 3, 9, 1, 2, 1, 7, 6, 4, 8, 9, 4, 3, 7, 5, 9, 2, 7, 3, 5, 1, 3)$을 나타낸 wavelet tree입니다. [그림 1]에서 나타난 것과의 차이점은 [그림 2]는 "수"를 다룬다는 것뿐입니다. 물론 그 덕분에 노드 $S$에 해당하는 알파벳 집합을 둘로 나눌 때 알파벳을 일일이 찾아 나눌 필요 없이 기준이 되는 수 $m_S$를 잡아 왼쪽 자식($l_S$라 쓰겠습니다)에는 $m_S$ 이하인 수들, 오른쪽 자식($r_S$라 쓰겠습니다)에는 $m_S$ 초과인 수들로 나눌 수 있게 되었습니다. 각 노드의 순서열 밑에 기준 $m_S$가 적혀 있습니다.

<img src="/assets/images/junis3/1910/2.PNG">

[그림 2] [https://ioinformatics.org/journal/v10_2016_19_37.pdf](https://ioinformatics.org/journal/v10_2016_19_37.pdf)

각 원소 $x$가 범위 $[1, m]$ 안에 있는 정수열 $a[1..n]$에 대해 다음과 같은 구간 질의를 수행할 수 있습니다.

- Rank; $\mathbf{rank}(root, r, x)$: $r$, $x$가 입력으로 주어질 때, 범위 $i \in [1, r]$에서 $a[i] = x$인 $i$의 개수를 구하라.
- Quantile; $\mathbf{quantile}(root, l, r, k)$: $l$, $r$, $k$가 입력으로 주어질 때, 부분열 $a[l..r]$을 크기 순서대로 정렬했을 때 $k$번째 원소를 구하라. [#](https://www.acmicpc.net/problem/7469)
- Range Counting; $\mathbf{range}(root, l, r, x)$: Rank 질의를 확장한 것이다; $l$, $r$, $x$가 입력으로 주어질 때, 범위 $i \in [l, r]$에서 $a[i] \le x$인 $i$의 개수를 구하라. [#](https://www.acmicpc.net/problem/11660)

를 해결할 수 있게 됩니다. 모두 이미 할 줄 아신다고요? 맞습니다. Rank query와 range counting query는 지속 구간 트리를 이용해서 질의당 $O(\log n)$의 시간에, quantile query는 병합 정렬 트리를 이용해서 질의당 $O(\log ^3 n)$의 시간에 수행할 수 있습니다. 하지만 wavelet tree는 조금 더 빠르고, 제한적 이게나마 업데이트도 할 수 있게 되고, 무엇보다 *새롭습니다*! 업데이트는 나중에 다루어보기로 하고, 먼저 질의를 처리하는 방식을 서술합니다.

질의를 처리하기 위해 전처리가 필요합니다. 먼저 현재 노드 $S$의 첫 $x$개 원소의 0의 개수와 1의 개수(둘 중 하나만 알면 나머지를 압니다)를 계산한 후, 0의 개수를 $0_S(x)$, 1의 개수를 $1_S(x)$로 둡니다. $S[x]$와 $l_S[0_S(x)]$, $r_S[1_S(x)]$가 대응되는 것을 알 수 있습니다. 이제 질의를 다음과 같이 처리할 수 있습니다.

- Rank query: 위에서 했던 것과 같습니다. 루트로부터 재귀적으로 내려가면서 $x$를 찾아 나갑니다. 식으로 표현하면 다음과 같습니다: $x \le m_S$라면, $\mathbf{rank}(S, r, x) = \mathbf{rank}(l_S, 0_S(r), x)$. $x > m_S$라면, $\mathbf{rank}(S, r, x) = \mathbf{rank}(r_S, 1_S(r), x)$.
- Quantile query: 이진 탐색을 이용하지 않습니다. 루트로부터 재귀적으로 내려가면서 $k$번째 원소가 될 수 있는 범위를 좁혀나가, 결국은 어떠한 수로 결정합니다. 만약 어떤 노드 $S$에 대해 부분열 $a[l..r]$ 안에 있는 0의 개수가 $k$ 이상이라면 부분열의 $k$번째 수는 0으로 표현됩니다. 즉 $c = 0_S(r) - 0_S(l-1) \ge k$이면 왼쪽 자식으로 내려갑니다: $\mathbf{quantile}(S, l, r, k) = \mathbf{quantile}(l_S, 0_S(l-1) + 1, 0_S(r), k)$. 반대의 경우에는 오른쪽 자식으로 내려갑니다. 이때에는 오른쪽 자식에서 $k$번째 수가 아니라 $k-c$번째 수를 구해야 함에 유의해야 합니다. 즉, $c < k$이면 $\mathbf{quantile}(S, l, r, k) = \mathbf{quantile}(r_S, 1_S(l-1) + 1, 1_S(r), k-c)$.
- Range counting query: Quantile query와 매우 유사하게 작동합니다. 조건을 수의 개수가 아닌 수의 범위로 바꾸어 처리하면 됩니다. 위 단락과 유사하게 관계식을 써 보세요.

세 질의 모두 $O(\log m)$의 시간에 작동합니다. 특히 quantile query의 시간 복잡도는 $O(\log^3 n)$의 시간이 소요되는 merge sort tree와 비교하면 월등합니다.

Wavelet tree는 제한적인 업데이트도 지원합니다. 아래 세 종류의 업데이트 질의를 빠르게 일관성을 유지하면서 할 수 있습니다.

- 연속한 두 원소를 바꾸기
- 원소를 활성화/비활성화하기
- 수열의 맨 앞이나 맨 뒤에 원소 추가/제거하기 (구현에 따라 맨 앞에 원소를 추가/제거하는 것이 귀찮을 수 있다)

## 결론

소개해 드린 wavelet tree는 획기적인 아이디어를 사용하는 자료구조는 아닙니다만, 기존에 문자열과 수열에 대해 수행하던 쿼리에 대해 새로운 시각을 제공합니다. 만날 길이로만 쪼개보던 수열을 값으로 쪼개보니 조금 달라 보이나요?

## 참고

- [https://en.wikipedia.org/wiki/Wavelet_Tree](https://en.wikipedia.org/wiki/Wavelet_Tree)
- [https://alexbowe.com/wavelet-trees/](https://alexbowe.com/wavelet-trees/)
- [https://ioinformatics.org/journal/v10_2016_19_37.pdf](https://ioinformatics.org/journal/v10_2016_19_37.pdf)
- [https://users.dcc.uchile.cl/~gnavarro/ps/cpm12.pdf](https://users.dcc.uchile.cl/~gnavarro/ps/cpm12.pdf)
