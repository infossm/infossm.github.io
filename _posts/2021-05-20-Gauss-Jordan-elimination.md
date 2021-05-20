---
layout: post
title: "Gauss-Jordan elimination"
date: 2021-05-20
author: gumgood
tags: [algorithm]
---

# 개요

Gauss-Jordan elimination(가우스 조던 소거법)은 미지수 $x_1$, $x_2$, $...$, $x_m$에 대한 $n$개의 일차방정식으로 구성된 연립일차방정식을 푸는 방법입니다. 해가 존재하는지, 존재한다면 유일한지 판단하고 그 중 하나의 해를 구할 수 있습니다.

# 연립일차방정식과 행렬

다음과 같은 연립 일차방정식을 생각해봅시다.

$$
a_{11}x_1 + a_{12}x_2 + ... + a_{1m}x_m = b_1 \\
a_{21}x_1 + a_{22}x_2 + ... + a_{2m}x_m = b_2 \\
\vdots \\
a_{n1}x_1 + a_{n2}x_2 + ... + a_{nm}x_m = b_n
$$

이 때, 각 일차방정식의 계수들과 미지수, 상수항을 묶어 행렬로 나타내면

$$
A =
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1m} \\
a_{21} & a_{22} & \cdots & a_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n1} & a_{n2} & \cdots & a_{nm}
\end{bmatrix},
\ 
X = 
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix},
\ 
B = 
\begin{bmatrix}
b_1 \\
b_2 \\
\vdots \\
b_n
\end{bmatrix}
$$

연립일차방정식을 다음과 같은 행렬의 곱으로 나타낼 수 있습니다.

$$
AX = B
$$

더 간략하게, 미지수를 나타내는 행렬 $X$는 제외하고 행렬 $A$, $B$를 이어붙인 첨가행렬(augmented matrix)로 연립일차방정식을 나타내도록 하겠습니다.

$$
\begin{bmatrix}
\begin{array}{c|c}
A & B
\end{array}
\end{bmatrix}

=

\begin{bmatrix}
\begin{array}{cccc|c}
a_{11} & a_{12} & \cdots & a_{1m} & b_1 \\
a_{21} & a_{22} & \cdots & a_{2m} & b_2 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
a_{n1} & a_{n2} & \cdots & a_{nm} & b_n
\end{array}
\end{bmatrix}
$$

# Gauss-Jordan elimination

Gauss-Jordan elimination을 직관적으로 이해하기 위해  $n = m$이고 답이 유일한 경우부터 보겠습니다. 주어진 일차방정식을 적절히 더하고 빼서 다음과 같은 형태로 만드는 것이 목표입니다.

$$
x_1 = b_1' \\
x_2 = b_2' \\
\vdots \\
x_n = b_n'
$$

행렬로 나타내면 다음과 같습니다.

$$
\begin{bmatrix}
\begin{array}{c|c}
I & B'
\end{array}
\end{bmatrix}

=

\begin{bmatrix}
\begin{array}{cccc|c}
1 & 0 & \cdots & 0 & b_1' \\
0 & 1 & \cdots & 0 & b_2' \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & \cdots & 1 & b_n'
\end{array}
\end{bmatrix}
$$

즉, 첨가행렬에 적절한 **행 연산**을 통해 행렬 $A$ 부분을 단위행렬 $I$로 만들면, 그 과정에서 행렬 $B$ 부분이 연립일차방정식의 해를 나타내는 $B'$로 바뀌게 됩니다.

### Elementary row operation

행렬에 적용할 수 있는 기본 행 연산(elementary row operation)은 세 가지가 있습니다.

* $P_{ij}$ : $i$번째 행과 $j$번째 행을 바꾼다.

* $D_{i}(k)$ : $i$번째 행에 $0$이 아닌 상수 $k$를 곱한다.

* $E_{ij}(k)$ : $i$번째 행에 $0$이 아닌 상수 $k$를 곱한 값을 $j$번째 행에 더한다.

이 연산들은 방정식의 순서를 바꾸거나 더하고 빼는 과정을 나타낸 것으로 determinant가 $0$이 아닌 $ n \times n$ 행렬로 나타낼 수 있습니다.

다음 예시와 함께 Gauss-Jordan elimination을 진행해보겠습니다.

$$
\begin{bmatrix}
\begin{array}{ccc|c}
\phantom{-}0 & \phantom{-}2 & \phantom{-}1 & -1 \\
\phantom{-}2 & \phantom{-}4 & -2 & \phantom{-}2 \\
\phantom{-}3 & \phantom{-}5 & -5 & \phantom{-}1
\end{array}
\end{bmatrix}
$$

1. $a_{11} = 0$이라면 $a_{i1} \ne 1$인 행과 교환합니다.

$$
\begin{bmatrix}
\begin{array}{ccc|c}
\phantom{-}2 & \phantom{-}4 & -2 & \phantom{-}2 \\
\phantom{-}0 & \phantom{-}2 & \phantom{-}1 & -1 \\
\phantom{-}3 & \phantom{-}5 & -5 & \phantom{-}1
\end{array}
\end{bmatrix}
$$

2. $a_{11}$을 $1$로 만들어 주기 위해, 첫 번째 행에 $\frac{1}{a_{11}}$를 곱합니다.

$$
\begin{bmatrix}
\begin{array}{ccc|c}
\phantom{-}1 & \phantom{-}2 & -1 & \phantom{-}1 \\
\phantom{-}0 & \phantom{-}2 & \phantom{-}1 & -1 \\
\phantom{-}3 & \phantom{-}5 & -5 & \phantom{-}1
\end{array}
\end{bmatrix}
$$

3. 첫 번째 열에서 $a_{11}$를 제외한 성분을 모두 $0$으로 만들기 위해, 첫 번째 행에 $-a_{i1}$을 곱한 값을 나머지 $i$번째 행에 더합니다.

$$
\begin{bmatrix}
\begin{array}{ccc|c}
\phantom{-}1 & \phantom{-}2 & -1 & \phantom{-}1 \\
\phantom{-}0 & \phantom{-}2 & \phantom{-}1 & -1 \\
\phantom{-}0 & -1 & -2 & -2
\end{array}
\end{bmatrix}
$$

4. $a_{ii}$에 대해서 위 과정을 반복합니다.
   
   $a_{22}$를 $1$로 만들고 같은 열에 있는 나머지 성분을 모두 $0$으로 만들어줍니다.

$$
\begin{bmatrix}
\begin{array}{ccc|c}
\phantom{-}1 & \phantom{-}2 & -1 & \phantom{-}1 \\
\phantom{-}0 & \phantom{-}1 & \phantom{-}\frac{1}{2} & -\frac{1}{2} \\
\phantom{-}0 & -1 & -2 & -2
\end{array}
\end{bmatrix}
$$

$$
\begin{bmatrix}
\begin{array}{ccc|c}
\phantom{-}1 & \phantom{-}0 & -2 & \phantom{-}2 \\
\phantom{-}0 & \phantom{-}1 & \phantom{-}\frac{1}{2} & -\frac{1}{2} \\
\phantom{-}0 & \phantom{-}0 & -\frac{3}{2} & -\frac{5}{2}
\end{array}
\end{bmatrix}
$$

    $a_{33}$를 1로 만들고 같은 열에 있는 나머지 성분을 모두 $0$으로 만들어줍니다.

$$
\begin{bmatrix}
\begin{array}{ccc|c}
\phantom{-}1 & \phantom{-}0 & -2 & \phantom{-}2 \\
\phantom{-}0 & \phantom{-}1 & \phantom{-}\frac{1}{2} & -\frac{1}{2} \\
\phantom{-}0 & \phantom{-}0 & \phantom{-}1 & \phantom{-}\frac{5}{3}
\end{array}
\end{bmatrix}
$$

$$
\begin{bmatrix}
\begin{array}{ccc|c}
\phantom{-}1 & \phantom{-}0 & \phantom{-}0 & -\frac{16}{3} \\
\phantom{-}0 & \phantom{-}1 & \phantom{-}0 & -\frac{4}{3} \\
\phantom{-}0 & \phantom{-}0 & \phantom{-}1 & \phantom{-}\frac{5}{3}
\end{array}
\end{bmatrix}
$$

따라서 이 연립일차방정식의 해는 $x_1 = -\frac{16}{3}$, $x_2 = -\frac{4}{3}$, $x_3 = \frac{5}{3}$가 됩니다. 각 단계마다 미지수를 하나식 소거시키기 때문에 elimination(소거법)이라 부르게 됩니다.

### Generalization

 $n = m$이고 해가 유일하게 존재하는 경우는 행렬 $A$의 부분을 단위 행렬 $I$로 만드는게 가능합니다. 그런데  $n \ne m$이거나 해가 유일하지 않다면 단위 행렬 $I$로 바꿀 수 없기 때문에 $I$ 대신 해에 가장 가까운 형태인 **reduced row echelon form**(**RREF**)를 정의합시다.

어떤 행렬이 다음 네 가지 성질을 만족하면 RREF라고 합니다.

1. $0$으로만 이뤄진 행은 맨 아래에 위치한다.

2. 각 행에서 가장 왼쪽에 있는 $0$이 아닌 성분은 $1$이다. (이를 pivot이라 합시다)

3. $i$행과 $(i+1)$행 모두 pivot이 있다면 $(i+1)$행 pivot은 $i$행 pivot보다 오른쪽에 위치한다.

4. pivot을 포함하는 열에서 pivot을 제외한 나머지는 모두 $0$이다.

다음과 같은 행렬이 RREF의 예가 될 수 있습니다.

$$
A = 
\begin{bmatrix}
1 & 0 & c_1 & 0 & c_2 \\
0 & 1 & c_3 & 0 & c_4 \\
0 & 0 & 0 & 1 & c_5
\end{bmatrix}
$$

행렬 $A$의 pivot은 $a_{11}$, $a_{22}$, $a_{34}$입니다. 이 pivot을 포함하지 않는 3번째, 5번째 열과 곱해지는 $x_3$, $x_5$는 free-variable로 어떠한 값을 가져도 됩니다. 행렬 $A$로부터 나오는 해를 정리하면

$$
\begin{aligned}
x_1 + c_1 x_3 + c_2 x_5 &= b_1 \\
x_2 + c_3 x_3 + c_4 x_5 &= b_2 \\
x_4 + c_5x_5 &= b_3
\end{aligned}
$$

가 됩니다. 이 때 $x_3$, $x_5$의 값이 임의의 실수 $s$, $t$라고 하면 해는 $s$와 $t$에 관한 식으로 정리할 수 있습니다($x_1 = b_1 - c_1 s - c_2t$, $x_2 = b_2 - c_3 s - c_4 t$, $x_3 = s$, $x_4 = b_3 - c_5 t$, $x_5 = t$ ). 특수해를 구하고 싶은 경우, free variable에 0을 넣어 간단하게 계산할 수 있습니다.

해가 없는 경우, 이렇게 구한 특수해를 연립일차방정식에 대입했을 때 식이 성립하지 않게 됩니다. 따라서 직접 대입해봄으로써 해의 존재성을 확인할 수 있습니다. 다른 방법으로는 $0$으로만 이뤄진 $i$번째 행이 있을 때, $b_i \ne 0$이라면 해가 존재하지 않는 다는 점을 이용하여 확인할 수도 있습니다.

# Complexity

$n$개의 행과 $m$개의 열을 $O(n+m)$에 순회하면서 pivot을 찾습니다. pivot을 찾을 때마다 나머지 행에 빼는 연산 $O(nm)$에 수행하는데, pivot은 최대 $\min (n,m)$개 있을 수 있습니다. 따라서 전체 시간복잡도는 $O(nm \cdot \min(n,m))$입니다. $n=m$인 경우, 시간복잡도는 $O(n^3)$입니다.

# Application

### Inverse Matrix

Gauss-Jordan elimination을 이용하면 역행렬도 쉽게 계산할 수 있습니다. 역행렬을 구하려는 행렬 $A$에 단위 행렬 $I$를 붙인 첨가행렬을 만들어줍니다.

$$
\begin{bmatrix}
\begin{array}{c|c}
A & I
\end{array}
\end{bmatrix}
$$

여기에 Gauss-Jordan elimination를 적용하면 결과는 다음과 같이 됩니다.

$$
\begin{bmatrix}
\begin{array}{c|c}
I & A^{-1}
\end{array}
\end{bmatrix}
$$

이를 코드로 구현하면 다음과 같습니다.

```cpp
void inverse_matrix(vector<vector<double>> &a){
    int n = a.size();
    int m = n + n;
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < n; ++j)
            a[i].push_back(i==j);

    for(int c = 0, r = 0; c < m && r < n; ++c){
        int p = r; // pivot row
        for(int i = r; i < n; ++i)
            if(a[p][c] < a[i][c])
                p = i;
        if(a[p][c] == 0){ puts("no inverse"); return; }; 

        for(int j = 0; j < m; ++j)
            swap(a[p][j], a[r][j]);

        double t = a[r][c];
        for(int j = 0; j < m; ++j)
            a[r][j] /= t;

        for(int i = 0; i < n; ++i) if(i != r){
            double t = a[i][c];
            for(int j = c; j < m; ++j)
                a[i][j] -= a[r][j] * t;
        }
        ++r;
    }

    for(int i=0;i<n;++i,puts(""))
        for(int j=0;j<n;++j)
            printf("%lf ",a[i][n+j]);
}
```

### Gauss-Jordan elimination modulo p

연립일차방정식의 해는 실수가 될 수 있기 때문에 소수 p에 대한 모듈러 연산을 쓰는 문제들도 자주 등장합니다. 모듈러 곱셈의 역원을 전처리 해두면 실수로 계산하는 것과 동일하게 구현할 수 있습니다. 이를 코드로 구현하면 다음과 같습니다.

```cpp
vector<int> gauss_mod(vector<vector<int>> &a,int mod){
    vector<int> inv(mod); // modulo inverse 전처리
    inv[1] = 1;
    for(int i = 2; i < m; ++i)
        inv[i] = m - (m/i) * inv[mod%i] % mod;

    int n = a.size();
    int m = a[0].size();

    vector<int> w(m, -1); // i번째 열에 있는 pivot이 몇 번째 행에 있는지 저장
    for(int c = 0, r = 0; c < m && r < n; ++c){
        int p = r; // pivot row
        for(int i = r; i < n; ++i)
            if(a[p][c] < a[i][c])
                p = i;
        if(a[p][c] == 0) continue; // free variable

        for(int j = 0; j < m; ++j)
            swap(a[p][j], a[r][j]);
        w[c] = r;

        int t = a[r][c];
        for(int j = 0; j < m; ++j)
            a[r][j] = a[r][j] * inv[t] % mod;

        for(int i = 0; i < n; ++i) if(i != r){
            int t = a[i][c];
            for(int j = c; j < m; ++j)
                a[i][j] = (a[i][j] - a[r][j] * t % mod + mod) % mod;
        }
        ++r;
    }

    for(int i = 0; i < n; ++i) // existence of solution
        if(count(a[i].begin(), --a[i].end(), 0) == m-1 && a[i][m-1])
            return vector<int>(); // no solution

    vector<int> ans(m);
    for(int i = 0; i < m; ++i)
        if(~w[i]) ans[i] = a[w[i]][m-1];
    return ans; // solution exist
}
```

### Gauss-Jordan elimination modulo 2

모듈러 연산 중에서도 p = 2인 경우 bitwise 연산으로 구현할 수 있습니다. 모든 성분이 0 또는 1이기 때문에 나누거나 곱하는 연산을 생략할 수 있고, 덧셈과 뺄셈은 XOR 연산으로 대체할 수 있습니다. C++의 bitset 자료구조를 이용하면 시간복잡도 $O(n^3 / 64)$에 더 효율적인 구현이 가능합니다.

```cpp
const int sz = 500;

bitset<sz> gauss_bit(vector<bitset<sz>> &a){
    int n = a.size();
    int m = a[0].size();

    vector<int> w(m, -1);
    for(int c = 0, r = 0; c < m && r < n; ++c){
        for(int i = r; i < n; ++i)
            if(a[i][c]){
                swap(a[i],a[r]);
                break;
            }
        if(a[r][c] == 0) continue;
        w[c] = r;

        for(int i = 0; i < n; ++i) if(i != r)
            if(a[i][c]) a[i] ^= a[r];
        ++r;
    }
    // .. same
}
```

# 연습문제

1. [9254번: 역행렬](https://www.acmicpc.net/problem/9254)

2. [11191번: Xor Maximization](https://www.acmicpc.net/problem/11191) - RREF를 생각하면 힌트를 얻을 수 있습니다

3. [13296번: Primonimo](https://www.acmicpc.net/problem/13296) - 각 그리드를 $x_i$로 두고 연립일차방정식들을 세울 수 있습니다

4. [16384번: Stoichiometry](https://www.acmicpc.net/problem/16384)

5. [20178번: Switches](https://www.acmicpc.net/problem/20178)

6. [20307번: RREF](https://www.acmicpc.net/problem/20307) - 계산 과정에서 수가 매우 커질 수 있어 약간의 휴리스틱이 필요합니다.

# 결론

선형대수학을 공부하셨다면 행렬의 [Determinant](https://en.wikipedia.org/wiki/Determinant)도 쉽게 계산할 수 있을 것입니다. 이를 이용하는 [Kirchhoff's theorem](https://en.wikipedia.org/wiki/Kirchhoff%27s_theorem)도 같이 공부하시면 좋을 것 같습니다.
