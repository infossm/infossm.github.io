---
layout: post
title: Strassen Algorithm
date: 2021-03-20 16:00:00
author: jeonggyun
tags:
---

안녕하세요?

오늘은 행렬을 효율적으로 곱하는 방법과, 그 방법 중 하나이자 분할 정복의 대표적인 예시인 슈트라센 알고리즘(Strassen Algorithm)과 그 구현에 대해 살펴보겠습니다.

모든 예시 코드는 double 형의 $p \times q$크기의 행렬 A와, $q \times r$크기의 행렬 B를 곱하는 기준으로 작성되었습니다.

# 행렬 곱의 기본 형태

가장 기본적인 행렬 곱의 형태로, 행렬곱은 아래와 같은 코드로 작성됩니다.

```cpp
void matmul(double* a, double* b, double* c, int p, int q, int r) {
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < r; ++j) {
            double sum = c[i * r + j];
            for (int k = 0; k < q; ++k)
                sum += a[i * q + k] * b[k * r + j];
            c[i * r + j] = sum;
        }
    }
}
```

간단해서 딱히 문제가 없을 것 같이 생긴 위 코드에도 문제가 있습니다. 보통 알고리즘 문제를 풀거나 할 때는 많이 신경쓰지 않는 점이지만, 위 코드는 캐시의 효율성이 굉장히 떨어지게 됩니다.

그 이유는 위 코드는 2차원 배열에서 column-major로 배열을 접근하기 때문입니다. 2차원 배열은 같은 row에서 column 번호가 증가하는 순으로, 해당 row가 끝나고 다음 row가 오는 순서로 메모리에 저장됩니다.

따라서 같은 column에서 row가 증가하는 순서로 메모리를 접근하게 될 경우, 실제 메모리상에서는 굉장히 먼 주소를 가지게 됩니다. 따라서 locality가 감소하며, CPU에서 연산을 하기 위해 메모리에 접근할 때 한번에 저장되는 캐시의 효율을 받지 못하게 되어 수행 시간이 느리게 됩니다.

<img src="/assets/images/strassen-algo/fig1.png" width="300px">

Fig 1. column-major 메모리 접근

일반적으로 코드를 작성할 때에도, column-major 순서로 배열을 접근하는 것은 가능하면 줄이는 것이 좋습니다.

# 캐시 효율성을 고려한 곱 형태

## ikj 형태

위 문제점의 가장 쉬운 파훼법은 for문에서 ijk순으로 접근하던 것을, ikj 순서로 바꾸어주기만 하면 해결 가능합니다.

```cpp
void matmul_fast(double* a, double* b, double* c, int p, int q, int r) {
    for (int i = 0; i < p; ++i) {
        for (int k = 0; k < q; ++k) {
            double t = a[i * q + k];
            for (int j = 0; j < r; ++j) {
                c[i * r + j] += t * b[k * r + j];
            }
        }
    }
}
```

## block 사용

행렬곱을 할 때 행렬을 작은 block으로 쪼개서, block끼리 곱을 하면 캐시 사용을 극대화시킬 수 있다고 합니다. [출처](https://csapp.cs.cmu.edu/public/waside/waside-blocking.pdf)

```cpp
void matmul_blocking(double* a, double* b, double* c, int p, int q, int r) {
    const int SM = 8;
    for (int i = 0; i < p; i += SM)
        for (int k = 0; k < q; k += SM)
            for (int j = 0; j < r; j += SM)
                for (int ii = i; ii < min(i + SM, p); ++ii)
                    for (int kk = k; kk < min(k + SM, q); ++kk) {
                        double t = a[ii * q + kk];
                        for (int jj = j; jj < min(j + SM, r); ++jj)
                            c[ii * r + jj] += t * b[kk * r + jj];
                    }
}
```

6중 for문으로 간단히 완성된 코드입니다. 여기서 SM이 블럭 크기이다. 블럭 크기는 보통 터미널에

> getconf LEVEL1_DCACHE_LINESIZE

를 입력해서 나온 값을 각 원소들의 size로 나눈 값 (여기서는 sizeof(double) = 8)을 사용하는 것을 권장하는 듯 합니다.

제 환경에서는 LEVEL1_DCACHE_LINESIZE = 64였기 때문에, SM = 64 / 8 = 8을 사용하였습니다.

# 슈트라센 알고리즘 (Strassen algorithm)

위에서 사용하던 $O(n^3)$의 행렬곱의 시간 복잡도를 조금 줄여줄 수 있는, 분할 정복의 대표적인 예시인 슈트라센 알고리즘입니다.

행렬을 가로세로 반씩 쪼개 4등분해서 곱하면, 단순하게 할 경우 작은 행렬의 곱셈이 8번 필요하므로 복잡도는 그대로입니다.

하지만 이를 조금 더 많은 덧셈을 동원하여, 미묘하게 바꾸어 7번의 곱셈만으로 수행할 수 있다는 것이 슈트라센 알고리즘의 핵심입니다.

행렬 A와 B를 각각 4등분하여, $A_{1,1}, A_{1,2}, A_{2,1}, A_{2,2}$, $B_{1,1}, B_{1,2}, B_{2,1}, B_{2,2}$라고 하겠습니다.

이제 구하려는 결과 행렬 C도 각각 4등분되어, $C_{1,1}, C_{1,2}, C_{2,1}, C_{2,2}$로 정의할 때, 아래 식이 성립합니다.

$C_{1,1} = A_{1,1}B_{1,1} + A_{1,2}B_{2,1}$

$C_{1,2} = A_{1,1}B_{2,1} + A_{1,2}B_{2,2}$

$C_{2,1} = A_{2,1}B_{1,1} + A_{2,2}B_{2,1}$

$C_{2,2} = A_{2,1}B_{2,1} + A_{2,2}B_{2,2}$

확인할 수 있듯 곱셈이 총 8번 이루어지는 것을 알 수 있습니다.

이 때 새로운 행렬 $M_1$\~$M_7$을 아래와 같이 정의하겠습니다.

$M_1 = (A_{1,1} + A_{2,2})(B_{1,1} + B_{2,2})$

$M_2 = (A_{2,1} + A_{2,2})B_{1,1}$

$M_3 = A_{1,1}(B_{1,2} - B_{2,2})$

$M_4 = A_{2,2}(B_{2,1} - B_{1,1})$

$M_5 = (A_{1,1} + A_{1,2})B_{2,2}$

$M_6 = (A_{2,1} - A_{1,1})(B_{1,1} + B_{1,2})$

$M_7 = (A_{1,2} - A_{2,2})(B_{2,1} + B_{2,2})$

각각의 $M_i$ 행렬을 만드는 데에 곱셈이 한 번씩 필요하므로, 총 7번의 곱셈이 필요합니다.

이 때, $C_{1,1}, C_{1,2}, C_{2,1}, C_{2,2}$를 $M_1$\~$M_7$을 이용해 나타내는 것이 가능합니다.

$C_{1,1} = M_1 + M_4 - M_5 + M_7$

$C_{1,2} = M_3 + M_5$

$C_{2,1} = M_2 + M_4$

$C_{2,2} = M_1 - M_2 + M_3 + M_6$

따라서 복잡도의 점화식은 아래와 같이 변하게 됩니다.

$O(n) = 7O(n/2) + O(n^2)$

마스터 정리를 통해 위 점화식의 시간복잡도를 구하면, $O(n^{log_{2}{7}}) = O(n^{2.8064})$ 정도로 줄어들게 됩니다.

아래는 구현된 코드입니다.

```cpp
void strassen(double* a, double* b, double* c, int ma, int mb, int mc, int p, int q, int r) {
    if ((long long)p * q * r <= 36000) {
        for (int i = 0; i < p; ++i) {
            for (int k = 0; k < q; ++k) {
                double t = a[i * ma + k];
                if (t == 0.0) continue;
                for (int j = 0; j < r; ++j) {
                    c[i * mc + j] += t * b[k * mb + j];
                }
            }
        }
        return;
    }
    int pp = p / 2, qq = q / 2, rr = r / 2;
 
    double* m1 = (double*)calloc(pp * rr, sizeof(double));
    double* m2 = (double*)calloc(pp * rr, sizeof(double));
    double* m3 = (double*)calloc(pp * rr, sizeof(double));
    double* m4 = (double*)calloc(pp * rr, sizeof(double));
    double* m5 = (double*)calloc(pp * rr, sizeof(double));
 
    double* at1 = (double*)malloc(sizeof(double) * pp * qq);
    double* at2 = (double*)malloc(sizeof(double) * pp * qq);
    double* at3 = (double*)malloc(sizeof(double) * pp * qq);
 
    double* bt1 = (double*)malloc(sizeof(double) * qq * rr);
    double* bt2 = (double*)malloc(sizeof(double) * qq * rr);
    double* bt3 = (double*)malloc(sizeof(double) * qq * rr);
 
    int i, j;
    double t1, t2, t3, t4, t5;
    for (i=0;i<pp;++i) for (j=0;j<qq;++j) {
        t1 = a[i*ma+j]; t2 = a[(i+pp)*ma+j+qq];
        at1[i*qq+j] = t1 + a[i*ma+j+qq];
        at2[i*qq+j] = t1 + t2;
        at3[i*qq+j] = t2 + a[(i+pp)*ma+j];
    }
 
    for (i=0;i<qq;++i) for (j=0;j<rr;++j) {
        t1 = b[i*mb+j]; t2 = b[(i+qq)*mb+j+rr];
        bt1[i*rr+j] = t1;
        bt2[i*rr+j] = t1 + t2;
        bt3[i*rr+j] = t2;
    }
 
    strassen(at1, bt3, m5, qq, rr, rr, pp, qq, rr);
    strassen(at2, bt2, m1, qq, rr, rr, pp, qq, rr);
    strassen(at3, bt1, m2, qq, rr, rr, pp, qq, rr);
     
    for (i=0;i<qq;++i) for (j=0;j<rr;++j) {
        bt1[i*rr+j] += b[i*mb+j+rr];
        bt3[i*rr+j] += b[(i+qq)*mb+j];
    }
 
    for (i=0;i<pp;++i) for (j=0;j<qq;++j) {
        t1 = at2[i*qq+j];
        at1[i*qq+j] -= t1;
        at3[i*qq+j] -= t1;
    }
 
    strassen(at1, bt3, c, qq, rr, mc, pp, qq, rr);
    strassen(at3, bt1, c + pp * mc + rr, qq, rr, mc, pp, qq, rr);
 
    for (i=0;i<qq;++i) for (j=0;j<rr;++j) {
        t1 = bt2[i*rr+j];
        bt1[i*rr+j] -= t1;
        bt3[i*rr+j] -= t1;
    }
 
    strassen(a, bt1, m3, ma, rr, rr, pp, qq, rr);
    strassen(a + pp * ma + qq, bt3, m4, ma, rr, rr, pp, qq, rr);
 
    for (i=0;i<pp;++i) for (j=0;j<rr;++j) {
        t1 = m1[i*rr+j];
        t2 = m2[i*rr+j];
        t3 = m3[i*rr+j];
        t4 = m4[i*rr+j];
        t5 = m5[i*rr+j];
        c[i*mc+j] += t1 + t4 - t5;
        c[i*mc+j+rr] += t3 + t5;
        c[(i+pp)*mc+j] += t2 + t4;
        c[(i+pp)*mc+j+rr] += t1 - t2 + t3;
    }
 
    free(m1);
    free(m2);
    free(m3);
    free(m4);
    free(m5);
 
    free(at1);
    free(at2);
    free(at3);
 
    free(bt1);
    free(bt2);
    free(bt3);
}
 
void matmul_strassen(double* a, double* b, double*c, int p, int q, int r) {
    int pp = p, qq = q, rr = r;
    int mod = 1;
    while ((long long)pp * qq * rr > 36000) {
        if (pp & 1) pp++;
        pp >>= 1;
        if (qq & 1) qq++;
        qq >>= 1;
        if (rr & 1) rr++;
        rr >>= 1;
        mod <<= 1;
    }
    pp *= mod;
    qq *= mod;
    rr *= mod;
 
    double* a_re = (double*)calloc(pp * qq, sizeof(double));
    double* b_re = (double*)calloc(qq * rr, sizeof(double));
    double* c_re = (double*)calloc(pp * rr, sizeof(double));
 
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < q; ++j) {
            a_re[i * qq + j] = a[i * q + j];
        }
    }
 
    for (int i = 0; i < q; ++i) {
        for (int j = 0; j < r; ++j) {
            b_re[i * rr + j] = b[i * r + j];
        }
    }
 
    strassen(a_re, b_re, c_re, qq, rr, rr, pp, qq, rr);
 
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < r; ++j) {
            c[i * r + j] += c_re[i * rr + j];
        }
    }
 
    free(a_re);
    free(b_re);
    free(c_re);
}
```

## 몇 가지 구현의 특이점

슈트라센 알고리즘을 할 때는 행렬을 가로/세로 절반으로 쪼개기 때문에, 행렬의 가로/세로 길이가 짝수여야 합니다. 따라서 0을 적절히 padding하여 크기를 맞추어주어야 합니다.

위 구현을 할 때, $M_6$과 $M_7$의 경우 한 번만 사용되므로 굳이 메모리 할당을 하지 않는 등의 간단한 코드 최적화를 몇 개 진행하였습니다.

행렬이 0으로 초기화될 필요가 있을 경우 cmalloc을 사용하였고, 그렇지 않을 경우 malloc을 사용하였습니다.

# 더 빠른 곱셈 알고리즘

행렬 곱셈의 시간 복잡도를 줄이려는 시도는 계속 있었고, 실제 슈트라센이 1969년 슈트라센 알고리즘을 제안한 뒤 시간복잡도는 계속 줄어왔습니다.

이 중 시간복잡도가 파격적으로 줄어든 케이스로는,

1981년 Schönhage가 [Partial and total matrix multiplication](https://epubs.siam.org/doi/10.1137/0210032) 논문에서 제안한 $O(n^{2.522})$짜리 방법과,

1990년 Coppersmith–Winograd가 [On the asymptotic complexity of matrix multiplication](https://ieeexplore.ieee.org/document/4568320) 논문에서 제안한 $O(n^{2.376})$ 알고리즘이 있습니다.

위 방법들의 논문을 한 번 살펴보긴 하였지만, tensor 연산이라는 특이한 방법을 사용하여 잘 이해하지는 못하였습니다.