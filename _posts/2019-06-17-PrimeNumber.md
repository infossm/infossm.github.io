---
layout: post
title:  "Prime Number"
date:   2019-06-17 23:55:00
author: shjgkwo
tags: [algorithm, number-theory]
---

# 목차

- [1. 개요](#개요)
- [2. 개념](#개념)
- [3. 구현](#구현)
- [4. 문제풀이](#문제풀이)
- [5. 마무리](#마무리)
- [6. 참고자료](#참고자료)

# 개요

## 이 포스트를 쓰며
 학교 고급 알고리즘 시간에 Miller–Rabin Algorithm 에 대해서 공부하게 되었다. 매우 흥미로운 내용이 많았으며, 다른 사람들과 공유하면 좋겠다고 생각하여 소인수 분해 알고리즘 등의 다양한 알고리즘을 알게 된것을 포함하여 공유하고 싶어졌다. 이번 포스트에서는 그 알고리즘들의 구현과 실제 문제에 대해 어떻게 사용되는지 소개하고자 한다.

## 기본 지식
 일단 최소한 Modulo Operation, 즉, 나머지 연산에 대해서 조금 설명할 필요가 있다. 

> Mod 연산은 나머지 연산자로서 $a = qn + r$ 꼴로 나타낼 수 있는 모든 정수에 대해서(단, q는 0보다 크다.) $0 \leq r < q$ 를 만족 할 때, $r = a \mod q$ 로 나타낸다.

 매우 기본적인 정의이다. 다양한 정의가 있지만, 일단은 후술할 모든 알고리즘들이 위의 정의를 사용한다고 생각하면 된다. 여기서 덧셈과 곱셈에 대해 결합법칙과, 분배법칙, 교환법칙이 모두 성립한다. 단, 나눗셈에 대해서는 성립하지 않으므로 후술할 곱셈의 역원의 개념이 반드시 필요하다. 이것들에 대한 증명은 이 포스트를 읽는 사람에게 맡기기로 하겠다. 일단 나머지 연산을 주로 사용하게 되므로 위에 기본 지식들을 꼭 기억해두자.

# 개념

## 합동
 영어로 congruent modulo이며, 나머지 연산을 설명할 때 매우 중요한 개념이다. 아래의 수식을 살펴보자.
> $a  \equiv r (\mod q)$ 

 이 경우엔 $a$와 $r$이 $q$에 대해 합동이라고 하며, 이는 $a$를 $q$로 나누었을때 나머지 연산의 정의에 의해 나온 값이 $r$과 같음을 의미한다. 예시를 들어 $37 \equiv 2 (\mod 7)$ 이라는 식이 있다고 하자. 이 경우엔 37과 2는 합동이다. 왜냐하면 37을 7로 나눈 나머지가 2 이기 때문이다. 이제부터 더 넓은 범주로 나타내면 다음과 같다.

> $a  \equiv b (\mod q)$

 이때, $a$ 와 $b$ 모두 q에 대한 나머지가 같은경우 둘을 합동으로 본다. 즉, $39 \equiv 44 (\mod 5)$ 라고 한다면, 39나 44나 5에 대한 나머지가 4로 같으므로 둘은 합동인 것이다.

## 최대공약수
 너무 유명한 개념이라 Uclidean Algorithm 만 짧게 소개하고 끝내겠다.

```
def gcd(a, b):
    if(b == 0):
        return a
    elif(a % b == 0):
        return b
    else:
        return gcd(b, a % b)
```

 심플하다, 매우 유명한 알고리즘이니 이정도만 설명하고 넘어가도록 한다. 최대공약수를 구하기 위해 서로의 나머지로 계속 나누다가 나누어 떨어지면 그것이 최대 공약수다 라는게 요지이고 이것의 대한 증명은 생략하도록 하겠다.

## 페르마 소정리
 이제 본격적으로 소수의 성질들에 대해 설명할 것이다. 우선 소수의 성질 중에 매우 심플하면서도 매우 중요한 성질이 있는데 이는 바로 페르마 소정리(Fermat’s little theorem)이다. 이 정리는 다음과 같다. 어떤 소수 $p$에 대해. $np$가 아닌($n$은 모든 정수) 모든 정수 $a$에 대해 다음과 같은 식이 성립한다.

> $a^{p-1} \equiv 1 (\mod p)$

 이 심플하면서도 강력한 정리는 나머지가 소수일때 한정하여 곱셈의 역원을 구할 수 있다. 이는 후술할 것이며 그 이외에도 후술할 Miller-Rabin Algorithm 에도 사용된다. 단, 주의할 점은 **필요조건**이지 **충분조건은 아니기** 때문에, 저것이 성립한다고 항상 소수라고 볼수는 없다.

## 빠른 제곱
 한국어로 적절한 번역이 딱히 없는듯하여 임의로 빠른 제곱이라고 표현했다. 일단 영문 표기상 Modular exponentiation 이며, 나머지 연산이 있을때 제곱 연산을 의미한다. 여기서 $a^{n} \mod p$를 계산할 때

```cpp
int ans = 1
for(int i = 0; i < n; i++) {
    ans *= a;
    ans %= p;
}
printf("%d", ans);
```

 라고 naive 하게 계산하면 곱셈의 시간복잡도를 O(1)이라고 가정하면, $O(n)$ 의 시간복잡도를 가진다. 여기서는 $O(log n)$의 시간복잡도로 해결하는 것을 하도록 한다. 우선 pseudo code는 다음과 같다.

```
def exp(a, n, p):
    if n == 0:
        return 1
    tmp = exp(a, n // 2, p)
    tmp = (tmp * tmp) % p
    if n % 2 == 1
        tmp = (tmp * a) % p
    return tmp
```

 일단 Python base로 써 보았으며, //는 소숫점을 버리는 정수 나눗셈 즉, 3 // 2 = 1 이라고 생각하면 된다. 알고리즘의 시간복잡도의 증명은 간단하다. $n$을 2로 계속 나누다가 0에 다다르면 return 하므로 당연하게도 $O(log n)$이다. 두번째는 정당성인데, 이에 대한 증명 역시 간단하다. 일단 $n$을 일단 계산 되었던 것을 제곱 한 뒤, 이진수로 나타낸 것에 대해 1이 나오면 우선 a를 곱한다. 이것을 반복 하게되는데 이 경우엔 $a^{1}$ 부터 시작하여 $a^{2}$, $a^{4}$ 점점 늘어가게 되는 것은 자명하다. 이 와중에 $a$가 곱해져서 $a^{n}$을 완성하게 되는 것이다. 예시를 들어 자세하게 설명하면 $7^{11} \mod 13$ 이 있다고 하자. $11$은 이진수로 $1011_{2}$로 나타낼 수 있고, 맨 처음에 1이 나오게 되므로 위의 pseudo code에 따라 7을 곱하고 return된 $7^{1}$ 을 제곱하여 $7^{2}$으로 만들고 이번엔 0이므로 그냥 return 한다. 그 다음 $7^{4}$ 으로 만든 뒤, 7을 한번 더 곱하여 $7^{5}$을 return 그리고 $7^{10}$을 만든 뒤, 7을 한번 더 곱하여 $7^{11}$이 완성된다.

 이 빠른 제곱법은 후술할 Miller-Rabin 법과 곱셈의 역원을 구하는데에 유용하게 쓰인다.

## Trivial Solution square root
 이는 소수가 가지는 매우 간단한 성질이다. 일단 이 성질에 대해 설명하면 어떤 소수 $p$에 대해 다음과 명제가 성립한다.

> $x^2 \equiv 1 (\mod p)$ 에 대한 해는 항상 $x \equiv 1 or -1 (\mod p)$ 밖에 없다.

 여기서 $1$, $-1$은 알다시피 합동이며 $1-p$, $1-2p$, $1+p$, ... 등 모두 $1$에 속하며, $p-1$, $2p-1$, $-p-1$, ... 등 모두 $-1$에 속한다. 좀더 자세히 설명하면, $0 < r < p$ 인 $r$ 에 대해 $1$과 $p-1$만이 정답으로서 존재한다는 뜻이다. 즉, 저것이외에 다른 해를 가졌다면, 합성수 일 **수**도 있다는 것이다. 마찬가지로 필요조건이지 충분조건이 아니기 때문이다. 당장에 4만 해도 1과 3 이외엔 없다.


# 구현

## Miller-Rabin Algorithm
 이제 본론인 Miller Rabin Algorithm 이다. 이 알고리즘은 소수를 판별하기 위한 확률적 알고리즘으로서 매우 빠른 시간복잡도를 가지고 있다. 시간 복잡도는 곱셈의 시간복잡도를 $O(log^{2} n)$이라고 한다면, $O(log^{3} n)$ 이라는 엄청나게 빠른 스피드를 자랑한다. 만약 long long 범위 내라 곱셈의 시간복잡도를 $O(1)$로 무시가 가능하다면 당연하게도 $O(log n)$ 그럼 이제 본격적인 pseudo code를 보자.

```
def withness(s, n):
    t = n - 1;
    cnt = 0;
    while(t % 2 == 0):
        cnt = cnt + 1
        t = t // 2
    x = exp(s, n, t)
    pre = x
    for i in range(cnt):
        x = mult(x, x, n)
        
        if(x == 1 && pre != 1 && pre != n - 1):
            return true
        
        pre = x
    }
    
    if(x != 1):
        return true
    
    return false
```

이 작업은 withness, 즉, 증인을 찾는 과정이다. 증인이라는 것은 주어진 수 $n$에 대해서 s가 n이 소수가 아니라는, 즉 합성수라는 증인이 되어주는 것을 찾는 것이다. 원리는 간단한데, $n - 1 = 2^{cnt}t$일 때, 우선 $s^{t}$ 을 구할 것인데, 이것을 제곱 해 가면서 non-trivial solution이 발생하는 지 확인하는 것이다. 즉, $n - 1$, $1$과 합동인 수를 제외한 다른 무언가의 제곱이 1이 되는 경우를 발견하는 것이다. **if(x == 1 && pre != 1 && pre != n - 1)** 이 코드가 그것을 나타내준다. 두번째, 마지막에 x가 1이 아닌 경우, 즉 $s^{n-1}$ 이 1이 아니라면 소수의 필요조건을 만족하지 못하므로 합성수라 볼 수 있다. 이 모든것들을 이용해서 랜덤한 수를 뽑아내어 증인인지 확인해 보는 것이다. 놀랍게도 어떤 합성수에 대해 증인이 아닐 확률은 $\frac{1}{2}$라고 한다. 즉, 베르누이 시행을 사용하여 확실한 랜덤이라고 가정하였을 때 여러번의 시행으로 증인이 한번이라도 나올 확률을 구하면 시행 횟수를 $m$ 이라고 한다면 $1 - (\frac{1}{2})^{m}$이 되고 이 확률은 m이 크면 클수록 더욱 높아진다. 하지만 대부분의 경우 3번의 테스트만으로 보통 소수를 판별할 수 있다고 한다.

```cpp
long long gcd(long long a, long long b) {
    if(b == 0) return a;
    else if(a % b == 0) return b;
    else return gcd(b, a % b);
}

long long exp(long long a, long long n, long long p) {
    if(p == 0) return 1;
    long long x = exp(a, n, p >> 1);
    x = mult(x, x, n);
    
    if(p & 1) x = mult(x, a, n);
    return x;
}

bool withness(long long s, long long n) {
    long long t = n - 1;
    int cnt = 0;
    while(!(t & 1)) {
        cnt++;
        t /= 2;
    }
    long long x = exp(s, n, t);
    long long pre = x;
    for(int i = 0; i < cnt; i++) {
        x = mult(x, x, n);
        
        if(x == 1 && pre != 1 && pre != n - 1) return true;
        
        pre = x;
    }
    
    if(x != 1) return true;
    
    return false;
}
```

위의 코드는 앞에서 나온 모든 pseudo code를 C++로 구현한 것들이다.

```cpp
bool is_prime(long long n) {
    for(int i = 0; i < 10; i++) {
        int flag ;
        long long tmp = rnd() % n; // rnd 랜덤함수를 의미한다.
        while(tmp == 0) tmp = rnd() % n;
        flag = withness(tmp, n);
        if(flag) return false;
    }
    return true;
}
```

이 코드는 10번의 테스트를 해보는 코드이다.

## Polard's rho Algorithm

 폴라드 로 알고리즘은 소인수분해를 빠르게 하는 알고리즘이다. 지금까지 사용했던 모든 알고리즘을 총 동원해서 해결한다.

 우선 factor를 뽑아내는 polard's rho algorithm의 pseudo는 아래와 같다.

```
def polard(n):
    i = 0
    x = random(0, n - 1)
    y = x
    k = 2
    d = n
    while(1) :
        i++;
        x = (x * x - 1) % n
        d = gcd(Abs(y - x), n)
        if(d != 1):
            break
        if(i == k):
            y = x
            k = k * 2
```

폴라드로가 적어도 하나의 factor를 찾는데 걸리는 시간은 $O(n^{\frac{1}{4}})$ 라고 한다. 하지만 소수 판정, 최대공약수 등의 온갖 log 알고리즘이 잔뜩 들어가서 정확한 시간복잡도를 구하는건 상당히 애를먹게 되며, 대충 naive한 알고리즘인 $O(n^{\frac{1}{2}})$보단 빠르다는 것만 알아두면 될것 같다. 구현은 아래와 같다.

```cpp
#include <cstdio>
#include <algorithm>

using namespace std;

long long seed = 1987152371;
long long mod = 1000000009;
long long salt = 113;

inline long long rnd() {  // 랜덤 함수이다.
    seed *= seed;
    seed %= mod;
    seed += salt;
    seed %= mod;
    return seed;
}

inline long long mult(long long x, long long y, long long n) { //128비트 곱셈
    __int128 tmp = x;
    tmp *= y;
    tmp %= n;
    return (long long)tmp;
}

long long gcd(long long a, long long b) {
    if(b == 0) return a;
    else if(a % b == 0) return b;
    else return gcd(b, a % b);
}

inline long long Abs(long long x) {
    return x < 0 ? -x : x;
}

long long exp(long long a, long long n, long long p) {
    if(p == 0) return 1;
    long long x = exp(a, n, p >> 1);
    x = mult(x, x, n);
    
    if(p & 1) x = mult(x, a, n);
    return x;
}

inline bool withness(long long s, long long n) {
    long long t = n - 1;
    int cnt = 0;
    while(!(t & 1)) {
        cnt++;
        t /= 2;
    }
    long long x = exp(s, n, t);
    long long pre = x;
    for(int i = 0; i < cnt; i++) {
        x = mult(x, x, n);
        
        if(x == 1 && pre != 1 && pre != n - 1) return true;
        
        pre = x;
    }
    
    if(x != 1) return true;
    
    return false;
}

inline bool is_prime(long long n) {
    for(int i = 0; i < 10; i++) {
        int flag ;
        long long tmp = rnd() % n;
        while(tmp == 0) tmp = rnd() % n;
        flag = withness(tmp, n);
        if(flag) return false;
    }
    return true;
}

long long ans[100010], sz;

void polard_rho(long long n) {
    if(is_prime(n)) {
        ans[sz++] = n;
        return;
    }
    int i = 0;
    long long x = rnd() % n;
    long long y = x;
    long long k = 2;
    long long d = n;
    while(1) {
        i++;
        x = (mult(x, x, n) - 1 + n) % n;
        d = gcd(Abs(y - x), n);
        if(d != 1) {
            break;
        }
        if(i == k) {
            y = x;
            k *= 2;
        }
    }
    if(d != n) {
        polard_rho(d);
        polard_rho(n / d);
        return;
    }
    if(!(n & 1)) {
        polard_rho(2);
        polard_rho(d / 2);
        return;
    }
    for(long long i = 3; i * i <= n; i += 2) if(n % i == 0) {
        polard_rho(i);
        polard_rho(d / i);
        return;
    }
}

int main() {
    long long n;
    scanf("%lld",&n);
    polard_rho(n);
    sort(ans, ans + sz);
    for(int i = 0; i < sz; i++) printf("%lld\n", ans[i]);
    return 0;
}
```

이 코드중에 인자를 n을 발견했을 경우, 2로 나눠보거나 나눠지는 인자를 찾으러 가는 작업이 조금 섞여있다. 그것을 감안하고 보길 바란다.

## 역원
 소수의 경우 곱셈의 역원을 modulo 연산에서 찾으려면 페르마의 소정리를 활용하여 구할 수 있다.
 $ab \equiv 1 (\mod p)$ 에서 b가 a의 역원이라고 볼 수 있는데, $a^{n-1} \equiv 1 (\mod p)$ 이므로
 $a^{n - 2} \equiv a^{-1} (\mod p)$ 가 된다. 이 성질은 나중에 정수론을 좀더 많이 다루게 될때 다시 언급하겠다.
 코드는 한줄이다. exp(a, n - 2, p) 이거 한줄이면 끝나기에 굳이 언급하지는 않겠다.

# 문제풀이

## 큰 수 소인수 분해
 이 [링크](https://www.acmicpc.net/problem/4149)를 통하여 문제를 볼 수 있다.

 그냥 큰 수 소인수분해 문제이다. 사실상 위 구현을 그대로 가져다 써도 만점을 받을 수 있다. 나중에 최적화나 시드 조절등으로 시간을 줄여보는걸 추천한다. 코드는 이미 위에서 언급했으므로 굳이 추가 첨부를 하지 않겠다.

## 환상의 짝궁
 이 [링크](https://www.acmicpc.net/problem/15711)를 통하여 문제를 볼 수 있다.

 이 문제는 골드바흐의 추측을 이용하여 푸는 문제이다. 일단 홀수면 2를 빼고난 뒤 남은 숫자가 소수인지만 판정하면 되고, 짝수면 1을 출력하면 되는 심플한 문제이다. 하지만 중요한건 홀수일때, 2를 빼고났을때 숫자가 여전히 매우 크기 때문에 Miller Rabin Algorithm을 사용할 수 밖에 없다.
 
 아래는 이 문제를 해결하는 코드이다.

```cpp
#include <cstdio>

long long seed = 1987152371; // as possible as BIG!
long long mod = 1000000007; // random range, PRIME NUMBER, example (1e9 + 7)
long long salt = 101; // Salt, Coprime with seed

long long rnd() {  // (x^2 + salt) % mod
    seed *= seed;
    seed %= mod;
    seed += salt;
    seed %= mod;
    return seed;
}

long long mult(long long x, long long y, long long n) {
    __int128 tmp = x;
    tmp *= y;
    tmp %= n;
    return (long long)tmp;
}

long long exp(long long a, long long n, long long p) {
    if(p == 0) return 1;
    long long x = exp(a, n, p >> 1);
    x = mult(x, x, n);
    
    if(p & 1) x = mult(x, a, n);
    return x;
}

bool withness(long long s, long long n) {
    long long t = n - 1;
    int cnt = 0;
    while(!(t & 1)) {
        cnt++;
        t /= 2;
    }
    long long x = exp(s, n, t);
    long long pre = x;
    for(int i = 0; i < cnt; i++) {
        x = mult(x, x, n);
        
        if(x == 1 && pre != 1 && pre != n - 1) return true;
        
        pre = x;
    }
    
    if(x != 1) return true;
    
    return false;
}

int main() {
    int t;
    scanf("%d",&t);
    while(t--) {
        long long x, y;
        scanf("%lld %lld",&x, &y);
        if(!((x + y) & 1)) {
            if(x + y >= 4) printf("YES\n");
            else printf("NO\n");
            continue;
        }
        
        long long n = x + y - 2;
        if(n < 2) {
            printf("NO\n");
            continue;
        }
        bool flag = false;
        for(int i = 0; i < 3; i++) {
            long long tmp = rnd() % n;
            while(tmp == 0) tmp = rnd() % n;
            flag = withness(tmp, n);
            if(flag) break;
        }
        if(flag) printf("NO\n");
        else printf("YES\n");
    }
    return 0;
}
```

# 마무리
 이번 포스트를 읽는 사람들이 소수의 다양한 성질과 그에 관련된 알고리즘들에 관심을 가지고 활용하는데 도움이 되었으면 좋겠다. 좀 더 다양한 응용문제를 가져오고 싶었지만 실제 사용례가 너무 적어서 매우 지엽적인 알고리즘에 속한다고 볼 수 있다. 하지만 그 특유의 성질이 재미있는 것들이 많으니 관심을 많이 가져주었으면 한다.

# 참고자료
- Introduction to Algorithms; 31 chapter; Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein
