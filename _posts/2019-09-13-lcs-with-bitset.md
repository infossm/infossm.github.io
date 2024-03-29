---
layout: post
title:  "비트 연산을 활용하여 두 문자열의 LCS 빠르게 구하기"
date:   2019-09-12 16:00:00
author: 박수찬
tags: [LCS, bitset]
---

<style type="text/css">
table {
    width: inherit;
}
</style>

이 포스트에서는 두 문자열 $A[1..n]$, $B[1..m]$의 최장 공통 부분 수열(이하 LCS)을 $O(nm/w)$ 시간에 구하는 방법에 대해 서술합니다. 

Lloyd Allison, Trevor I. Dix의 [A bit-string longest-common-subsequence algorithm](https://www.sciencedirect.com/science/article/pii/0020019086900918?via%3Dihub)을 보고 작성한 글입니다.

## 일반적으로 LCS를 구하는 방법

$T[i][j]$를 $A[1..i]$와 $B[1..j]$의 LCS 길이로 정의하면, 아래와 같은 점화식이 성립한다는 사실이 잘 알려져 있습니다.

$$T[i][j] = 
\begin{cases}
    0 & \text{if $i=0$ or $j=0$} \\
    T[i-1][j-1]+1 & \text{if $A[i] = B[j]$} \\
    \max(T[i-1][j],T[i][j-1])& \text{otherwise}
  \end{cases}
$$

$0 \le T[i][j] - T[i][j-1] \le 1$이므로, 새로운 배열 $D[i][j] = T[i][j] - T[i][j-1]$을 정의하면 각 원소는 0 또는 1입니다. 따라서 $D$을 bitset을 통해 관리하도록 하겠습니다. 

## 관찰

예를 들어 $T[i-1]$와 $D[i-1]$이 아래와 같다고 합시다. 이제 $T[i]$와 $D[i]$를 구할 차례입니다.

||1|2|3|4|5|6|7|8|9|10|11|12|13|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|$T[i-1]$|0|1|1|2|2|2|2|3|3|4|5|5|5|
|$D[i-1]$|0|1|0|1|0|0|0|1|0|1|1|0|0|

먼저, $j$를 아래와 같이 $D[i-1]$이 1인 지점이 오른쪽 끝이 되도록 하는 구간들로 나눌 수 있습니다. 편의상 $D[i-1][m+1] = 1$로 가정하여 이러한 구간에 속하지 않는 $j$가 없도록 하였습니다. 

||1|2| |3|4| |5|6|7|8| |9|10| |11||12|13|(14)|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|$T[i-1]$|0|1| |1|2| |2|2|2|3| |3|4| |5| |5|5|x|
|$D[i-1]$|0|1| |0|1| |0|0|0|1| |0|1||1| |0|0|1|

이렇게 나뉜 구간들에서는 $T[i-1][j-1] + 1$ 값 ($T[i][j]$의 상한)이 동일하다는 점에서, 아래와 같은 Lemma를 생각할 수 있습니다.

---

Lemma: 위와 같이 나뉜 모든 구간 $[j_1, j_2]$에 대해, $D[i][j_1..j_2]$에는 1이 정확히 한 번 나타난다.

증명: 조건을 만족하지 않는 가장 왼쪽에 있는 구간을 $[j_1, j_2]$라고 합시다.

- $D[i][j_1..j_2]$가 모두 '0'이라면: $T[i][j_2]$가 $T[i-1][j_2]$보다 작습니다. $D[i][1..j_2]$에서의 1의 개수가 $D[i-1][1..j_2]$에서의 1의 개수보다 1만큼 더 작기 때문입니다. 하지만 정의상 $T[i][j] \ge T[i][j-1]$가 항상 성립하므로 이러한 경우는 없습니다.
- $D[i][j_1..j_2]$에 '1'이 2개 이상 있다면: $T[i][j_2]$가 $T[i-1][j_2 - 1] + 1$ ($=T[i-1][j_2]$보다 더 큽니다. $D[i][1..j_2]$에서의 1의 개수가 $D[i-1][1..j_2]$에서의 1의 개수보다 더 크기 때문입니다. 하지만 $T[i][j] \le T[i-1][j-1] + 1$가 항상 성립하므로 이러한 경우는 없습니다.

두 경우 모두 불가능하므로 모든 구간은 조건을 만족합니다.

----

Lemma에 의해, $D[i]$는 각 구간마다 1을 적당한 위치에 넣음으로써 만들 수 있음을 알 수 있습니다. 1의 위치가 $D[i-1]$과 다르려면 어떤 조건을 만족해야 할까요?  $T[i][j]$ 값이 증가했다는 뜻이므로, $A[i] = B[j]$여야 할 것입니다.

예를 들어 구간 $[5, 8]$에서 '1'의 위치가 $j = 8$에서 $j = 6$으로 움직였다고 하고, $T[i]$와 $D[i]$의 값을 구해 보면 아래와 같습니다.

||1|2| |3|4| |5|6|7|8| |9|10| |11||12|13|(14)|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|$T[i-1]$|0|1| |1|2| |2|2|2|3| |3|4| |5| |5|5|x|
|$T[i]$|1|1| |1|2| |2|<B>3</B>|<B>3</B>|3| |3|4| |5| |6|6|x|
|$D[i-1]$|0|1| |0|1| |0|0|0|1| |0|1||1| |0|0|1|
|$D[i]$|1|0| |0|1| |0|<b>1</b>|0|0| |0|1||1| |1|0|0|

$A[i] = B[6]$이 성립한다는 것은 바로 알 수 있습니다. 여기에, 추가적으로 $A[i] = B[7]$ 또는 $A[i] = B[8]$이 성립하더라도 $T[i]$가 위의 표와 똑같이 갱신된다는 것도 알 수 있습니다. (구간 내에서 $T[i-1][j-1] + 1$의 값이 모두 동일하기 때문입니다.)

즉, $D[i]$의 각 구간에서 '1'이 있는 위치는


- $D[i-1]$에서 '1'이 있던 위치
- $A[i] = B[j]$인 모든 $j$


중 가장 왼쪽이라는 것을 파악할 수 있고, 이를 이용하여 $D[i]$를 구할 수 있습니다.

||1|2| |3|4| |5|6|7|8| |9|10| |11||12|13|(14)|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|$D[i-1]$|0|1| |0|1| |0|0|0|1| |0|1||1| |0|0|1|
|$A[i]=B[j]?$|1|0||0|0||0|1|1|0||0|0||0||1|0|-|
|OR|1|1||0|1||0|1|1|1||0|1||1||1|0|1|
|$D[i]$|1|0| |0|1| |0|<b>1</b>|0|0| |0|1||1| |1|0|0|

## 비트 연산으로 속도 높이기

먼저 $A[i] = B[j]$인 모든 $j$를 빠르게 알기 위해 아래와 같이 정의된 배열 $S$를 전처리해 둡니다.

$$S[c][j] = 
\begin{cases}
    1 & \text{if $B[j] = c$} \\
    0& \text{otherwise}
  \end{cases}
$$

$c$로 가능한 문자 집합을 $\Sigma$라고 할 때, $O(\|\Sigma\| \cdot m / w)$ 공간 및 $O(m)$ 추가 시간으로 이 배열을 구할 수 있습니다. (모든 $j$에 대해 $S[B[j]][j] \leftarrow 1$ 대입하면 됨)

$x = S[A[i]] \vee D[i-1]$라고 합시다. 이제 $D[i-1]$으로 정의된 각 구간에 대해 최하위 비트 하나씩만 남겨두면 됩니다. 위 섹션과는 달리 지금은 구간이 어떻게 나뉘었는지 명시적으로 구해놓지는 않았습니다.

||13|12||11||10|9||8|7|6|5||4|3||2|1|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|$D[i-1]$|0|0| |1||1|0| |1|0|0|0| |1|0| |1|0|
|$x$|0|1||1||1|0||1|1|1|0||1|0||1|1|

어떤 수 `t`의 최하위 비트를 구할 때 자주 사용하는 성질 중 하나는 `t`가 `...1000000` 꼴이라면 `t - 1`은 `...0111111` 꼴이라는 것입니다. 예를 들어 Fenwick Tree 등을 구현할 때 index에서 최하위 비트를 삭제하고자 한다면 `t &= t-1`과 같은 비트 연산을 합니다.

여기서는 구간마다 최하위 비트를 구하고자 하므로, 구간마다 1을 빼 줘야 합니다. $D[i-1]$에서는 구간의 최상위 비트만 켜져 있으므로, 한 칸씩 옮겨 $(D[i-1] \<\< 1) \| 1$을 구하여 구간의 최하위 비트만 켜져 있도록 할 수 있습니다.

||13|12||11||10|9||8|7|6|5||4|3||2|1|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|$x$|0|1||1||1|0||1|1|1|0||1|0||1|1|
|$D[i-1]$|0|0| |1| |1|0| |1|0|0|0| |1|0| |1|0|
|$(D[i-1]\<\<1)+1$|0|1||1||0|1||0|0|0|1||0|1||0|1|

두 수를 빼 주면,

||13|12||11||10|9||8|7|6|5||4|3||2|1|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|$x$|0|1||1||1|0||1|1|1|0||1|0||1|1|
|$(D[i-1]\<\<1)+1$|0|1||1||0|1||0|0|0|1||0|1||0|1|
|$x-((D[i-1]\<\<1)+1)$|0|0||0||0|1||1|1|0|1||0|1||1|0|

각 구간마다 최하위 비트보다 우선순위가 높은 비트는 변하지 않으므로, $x$와 $x-((D[i-1]\<\<1)+1)$의 다른 부분만 구하면 최하위 비트 이하만 1로 남게 됩니다.

||13|12||11||10|9||8|7|6|5||4|3||2|1|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|$x$|0|1||1||1|0||1|1|1|0||1|0||1|1|
|$x-((D[i-1]\<\<1)+1)$|0|0||0||0|1||1|1|0|1||0|1||1|0|
|$x \oplus (x-((D[i-1]\<\<1)+1))$|0|1||1||1|1||0|0|1|1||1|1||0|1|

최종적으로 $x$와의 공통된 비트만을 구하면 구간마다 최하위 비트만 남게 됩니다.

||13|12||11||10|9||8|7|6|5||4|3||2|1|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|$x$|0|1||1||1|0||1|1|1|0||1|0||1|1|
|$x \oplus (x-((D[i-1]\<\<1)+1))$|0|1||1||1|1||0|0|1|1||1|1||0|1|
|$x \wedge (x \oplus (x-((D[i-1]\<\<1)+1)))$|0|1||1||1|0||0|0|1|0||1|0||0|1|

정리하면, 

$x = S[A[i]] \vee D[i-1]$

$D[i] = x \wedge \bigg( x \oplus \Big( x - \big(\left(D[i-1] << 1\right) | 1\big) \Big) \bigg)$

입니다.

위의 식에서 사용되는 연산은 AND, OR, XOR, left shift, 뺄셈뿐이며, 이 연산들은 $w$개의 비트를 한 정수로 묶어 저장하는 방식으로 처리하면 매우 빠르게 할 수 있습니다.

시간복잡도는 $O(nm/w)$가 됩니다. ($w$는 word size)

## 구현

AND, OR, XOR 연산은 블록 단위로 독립적이고, 뺄셈 연산과 shift 연산은 인접한 두 블록 사이에만 영향을 미친다는 점을 이용해서, 배열 접근 횟수를 최소화할 수 있습니다.

필요한 메모리는 $O(\|\Sigma\|m/w)$입니다.

```c++
#define get(arr, x) (((arr[x >> 6] >> (x & 63)) & 1) == 1)
#define set(arr, x) (arr[x >> 6] |= 1llu << (x & 63))
using ulint = unsigned long long;

int lcs(std::string A, std::string B) {
  int N = A.size(), M = B.size();
  int sz = (M >> 6) + 1;

  std::vector<ulint> S[256];
  for(int c = 0; c < 256; c++) S[c].resize(sz);
  for(int j = 0; j < M; j++) set(S[B[j]], j);
  
  std::vector<ulint> row(sz);
  for(int j = 0; j < M; j++) if(A[0] == B[j]) { set(row, j); break; }

  for(int i = 1; i < N; i++) {
    ulint shl_carry = 1;
    ulint minus_carry = 0;
    
    for(int k = 0; k < sz; k++) {
      // Compute k-th block of x == S[A[i]] OR D[i-1]
      ulint x_k = S[A[i]][k] | row[k];

      // Compute k-th block of "(D[i-1] << 1) | 1"
      ulint term_1 = (row[k] << 1) | shl_carry;
      shl_carry = row[k] >> 63;

      // Compute k-th block of "x - ((D[i-1] << 1) | 1)"
      auto sub_carry = [](ulint &x, ulint y){
        ulint tmp = x;
        return (x = tmp - y) > tmp;
      };
      ulint term_2 = x_k;
      minus_carry = sub_carry(term_2, minus_carry);
      minus_carry += sub_carry(term_2, term_1);
      
      row[k] = x_k & (x_k ^ term_2);
    }

    row[M >> 6] &= (1llu << (M & 63)) - 1;
  }

  int ret = 0;
  for(int j = 0; j < M; j++) if(get(row, j)) ret += 1;
  return ret;
}
```

## 예시 문항

### IZhO 2013 - Round Words

[문제 링크](https://oj.uz/problem/view/IZhO13_rowords)

두 원형 문자열 $A$, $B$가 주어졌을 때 LCS의 길이를 구하는 문제입니다. 문자열이 원형이므로, 어떤 위치에서 시작해서 어떤 방향으로 읽어도 관계가 없는데, 모든 가능한 방법 중 최댓값을 구해야 합니다. 문자열의 길이는 2,000 이하입니다.

가장 직관적인 방법은, $B$를 읽을 수 있는 모든 방법을 다 나열해 보면서 그것과 $A$의 LCS 길이를 구해보는 것입니다. $B$를 읽을 수 있는 방법은 $2m$개 정도($B$의 circular shift와 $B$를 뒤집은 문자열의 circular shift)입니다. 시간복잡도가 $O(n^2 m)$이라 일반적인 방법으로 LCS의 길이를 구하면 시간 초과가 나지만, 위의 `lcs(a, b)` 함수를 사용하면 가볍게 통과됩니다: [https://oj.uz/submission/153078](https://oj.uz/submission/153078)

### 2013-2014 Petrozavodsk Winter Training Camp, Moscow SU Trinity Contest - Total LCS

[문제 링크](https://codeforces.com/gym/101237/problem/G)

두 문자열 $A$와 $B$가 주어졌을 때, $B$의 모든 부분문자열 $B[i..j]$에 대해서 $LCS(A, B[i..j])$의 값을 구하는 문제입니다. 문자열의 길이는 2,000 이하입니다.

`lcs(A, B[i:])`을 호출하면 $D$ 배열의 마지막 행을 보고 $LCS(A, B[i..j])$를 모두 알 수 있습니다. 모든 $i$에 대해 이렇게 호출해 보면 시간복잡도는 $O(nm^2)$로 꽤 큰 편이지만, 통과합니다.
