---
layout: post
title: "PS에서 C++ 상속 활용하기 2"
date: 2024-12-29 23:00:00
author: psb0623
tags: []
---

[지난 글](https://infossm.github.io/blog/2024/08/27/problem-solving-with-inheritance/)에 이어, PS에서 상속을 활용할 수 있는 사례 하나를 더 소개해보도록 하겠습니다.

## 행렬 라이브러리

PS 문제를 풀다보면, 가끔 행렬을 이용해서 풀어야 하는 문제들을 만날 수 있습니다. 이러한 문제들을 풀기 위해서는 행렬 곱셈 등의 연산을 구현해야 합니다. 비록 코드가 길지는 않지만, 이러한 행렬 곱셈을 매번 구현하는 것은 귀찮은 일이니, 행렬에 관한 라이브러리를 만들어두고 필요할 때마다 가져다 쓰는 것이 편할 것입니다. (더 빠른 행렬 곱셈 알고리즘이나, 반복문의 반복 순서 조정을 통한 캐시 히트율 개선 등 성능적 측면에 대해서는 일단 고려하지 않도록 하겠습니다.)

그러나 행렬과 행렬 곱셈을 라이브러리로 어떻게 구현할지, 즉 행렬을 어떻게 선언하고 곱할 것인지에 대해서는 여러 디자인과 고민점이 있을 수 있습니다.

### 일반적인 구현

행렬은 기본적으로 $N \times M$ 크기의 2차원 배열입니다. 따라서, 가장 간단하게는 행렬을 `vector<vector<int>>`로 표현하고 이 둘을 곱하는 함수를 작성할 수 있습니다.

```c++
typedef vector<vector<int>> Matrix;

Matrix multiply(Matrix& A, Matrix& B) {
    int N = A.size();
    int M = A[0].size();
    int K = B[0].size();
    Matrix ret = Matrix(N, vector<int>(K, 0));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            for (int k = 0; k < M; k++) {
                ret[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return ret;
}
```

위 함수는 올바르게 행렬 곱셈을 수행하긴 하지만, 이 코드에는 여러 문제가 존재합니다. 무엇인지 아시겠나요?

바로, 올바르지 않은 입력에 대한 예외 처리가 부족하다는 점입니다.

- `A`의 열 개수와 `B`의 행 개수가 동일한지 확인해야 합니다.
- `A[0].size()`, `B[0].size()`에 접근하기 전에 `A.size() > 0`와 `B.size() > 0`인지부터 확인해야 합니다.
- 사실 `A[0].size()`가 `A[1].size()`, `A[2].size()`, $\cdots$와 동일할 것이라는 보장도 없습니다. 원칙적으로는 모든 `0 <= i < A.size()`에 대해 `A[i].size()`가 모두 동일한지도 확인해줘야 합니다.

위처럼 행렬의 크기에 대한 체크를 하지 않으면, 행렬 곱셈 도중 올바르지 않은 값에 접근할 수 있습니다. 특히, `vector`에서의 out-of-index 접근은 런타임 에러를 발생시키지 않고 단순히 이상한 값을 반환하는데 그치는 경우가 많기 때문에, 크기가 잘못된 행렬을 넣더라도 에러를 발생시키지 않아 사용자가 오류를 파악하기 힘들 가능성이 높습니다.

이 문제는 사실 위 함수가 행렬과 행렬이 아닌 것을 구분하지 못한다는 근본적 원인에서 기인한다고도 볼 수 있습니다. 위 코드에서 `Matrix`라는 타입은 단순히 `vector<vector<int>>`의 이명일 뿐인데, `vector<vector<int>>`는 너무 많은 것을 포괄하는 타입이라는 점이 문제입니다. 

`vector<vector<int>>`는 모든 행의 크기가 다를 수도 있고, 심지어는 크기가 자유자재로 바뀔 수 있기까지 합니다. 실수로 어떤 그래프의 인접 리스트를 행렬 곱셈 함수의 인자에 넣어도, 이를 명시적으로 막지 않고 오류조차 발생하지 않을 수 있다는 점은 그다지 좋은 디자인이 아닌 것 같습니다.

이러한 문제들은 사실 사용자가 사용법을 명확히 인지하고, 재사용이 빈번하지 않다면 큰 문제가 되지는 않습니다. 그러나 라이브러리 코드를 작성하여 두고두고 활용할 것이라면, 사용자가 라이브러리를 활용할 때 주의해서 신경써야 하는 부분을 최소화하는 것이 좋은 방향일 것입니다.

물론 행렬 크기에 관련한 예외 처리를 해주기 위해 아래와 같이 함수 시작 부분에 `assert`문을 잔뜩 삽입할 수도 있지만, 코드가 쓸데없이 길어지고 성능이 하락하는 등의 문제가 발생합니다.  

```c++
typedef vector<vector<int>> Matrix;

Matrix multiply(Matrix& A, Matrix& B) {
    assert(A.size() > 0 && B.size() > 0);
	assert(A[0].size() == B.size());
	for(int i = 0; i < A.size(); i++) assert(A[0].size() == A[i].size());
	for(int i = 0; i < B.size(); i++) assert(B[0].size() == B[i].size());

	int N = A.size();
    int M = A[0].size();
    int K = B[0].size();
	// ...
```

더 나은 방법은 없을까요?

### 상속을 활용한 구현

위에서 언급했듯이, 행렬은 기본적으로 $N \times M$ 크기의 2차원 배열입니다. 따라서, 우리는 2차원 배열을 그대로 상속받은 후 우리에게 필요한 몇 가지 연산을 구현해주는 방식을 생각해볼 수 있습니다. 이렇게 하면, 2차원 배열의 성질을 그대로 이어받으면서도 별도의 타입을 가지는 `Matrix` 구조체를 만들 수 있을 것입니다.

그리고 위에서 언급했듯이 `vector<vector<int>> A;` 같은 동적 배열은 행렬의 형태와 크기에 대한 아무런 보장도 해주지 않기 때문에, 동적 배열 대신 `int A[2][2];`와 같은 정적 2차원 배열을 상속받아보도록 합시다.

C나 C++에서 정적 2차원 배열이란 단순히 `int`를 향한 포인터에 불과한데, 이를 상속받는다는 것은 말이 안되는 것처럼 보입니다. 그러나 C++에서는 	[`std::array`](https://en.cppreference.com/w/cpp/container/array)라는, 정적 배열을 감싸는 클래스를 STL로 제공하기 때문에 `std::array`를 상속받는 디자인을 생각할 수 있습니다.

이렇게 `std::array`로 2차원 배열을 구현하고 상속받으면, 2차원 배열의 기능과 메서드(`begin()`, `size()` 등)를 그대로 활용하면서, 2차원 배열 사이의 곱셈 등 필요한 연산들을 연산자 오버로딩을 통해 자유롭게 정의할 수 있습니다. 

아래는 상속으로 구현한 행렬 구조체의 예시입니다.

```c++
using namespace std;

template<int N, int M>
struct Matrix: array<array<int, M>, N> {
	typedef array<array<int, M>, N> super;
	Matrix() { for(int i=0;i<N;i++) super::operator[](i).fill(0); }
	template<int K>
	Matrix<N,K> operator*(Matrix<M,K>& o) {
		Matrix<N,K> ret;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < K; j++) {
				for (int k = 0; k < M; k++) {
					ret[i][j] += super::operator[](i)[k] * o[k][j];
				}
			}
		}
		return ret;
	}
};
```

이 코드는 제가 PS에 활용하는 코드이기도 한데요, 한 줄씩 의미를 설명해보도록 하겠습니다.

---

```c++
template<int N, int M>
struct Matrix: array<array<int, M>, N> {
```

구조체 `Matrix`의 선언부입니다. $N \times M$ 크기의 행렬을 표현하기 위해 템플릿 인자로 `N`과 `M`을 받습니다. 예를 들어, $2 \times 3$ 크기의 행렬 $A$를 표현하려면 `Matrix<2,3> A;`와 같이 선언할 수 있습니다.

행렬의 크기를 템플릿 인자로 받는다는 것은 행렬의 크기 정보를 타입 자체에 포함한다는 것인데, 여기에는 여러 장점이 있습니다. 이에 대해서는 추후 다루도록 하겠습니다.

`Matrix`는 `array<array<T, M>, N>`을 상속받습니다. 따라서, `Matrix`는 기본적으로 $N \times M$ 크기의 2차원 배열 그 자체이며, `A[i][j]`처럼 인덱스를 이용해 해당 위치의 값에 접근할 수 있습니다.

---

```c++
	typedef array<array<int, M>, N> super;
```

`super`는 상속받은 기존 구조체 `array<array<int, M>, N>`의 메서드를 편리하게 호출하기 위해 `array<array<int, M>, N>`에 붙인 별명입니다. 예를 들어, 부모의 인덱스 연산자인 `super::operator[](i)`를 호출하여 2차원 배열의 데이터를 읽을 수 있습니다.

정확히 말하자면, `super::operator[](i)`는 기존 2차원 배열의 `i`번째 행에 해당하는 1차원 배열에 대한 참조(`&array<int, M>`)를 반환하도록 되어 있습니다. 그래서 `int val = super::operator[](i)[j];`와 같이 사용하면 기존 2차원 배열의 `i`번째 행, `j`번째 열에 해당하는 값을 읽을 수 있습니다.

---

```c++
	Matrix() { for(int i=0;i<N;i++) super::operator[](i).fill(0); }
```
구조체 `Matrix`의 생성자입니다. 2차원 배열의 데이터를 초기화하는 역할을 합니다. 위에서 언급한 `super::operator[](i)`을 활용해 `i`번째 행에 접근하는 것을 볼 수 있습니다.

`Matrix`는 2차원 배열을 상속받기에, 2차원 배열의 특성도 그대로 이어받습니다. 예를 들어 `Matrix`가 지역 변수로 할당된다면, 2차원 배열처럼 모든 칸이 0에 아닌 더미 값이 들어있을 수 있습니다. 이는 행렬 라이브러리를 사용하는 입장에서 그닥 반갑지 않은 성질이기 때문에, 생성자에서 명시적으로 `0`으로 초기화해주는 방식을 택했습니다.

---

```c++
	template<int K>
	Matrix<N,K> operator*(Matrix<M,K>& o) {
		Matrix<N,K> ret;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < K; j++) {
				for (int k = 0; k < M; k++) {
					ret[i][j] += super::operator[](i)[k] * o[k][j];
				}
			}
		}
		return ret;
	}
```

실제로 행렬 곱셈을 수행하는 함수입니다. `*` 연산자를 오버로딩하여 사용자 코드에서는 `A * B`와 같이 편리하게 활용할 수 있습니다.

템플릿 인자로 `K`를 받는데, 이는 `Matrix<N, M>`의 오른쪽에 곱해질 수 있는 행렬의 타입이 `Matrix<M, 1>`, `Matrix<M, 2>`, $\cdots$로 무수히 많기 때문에, 모든 경우에 대응할 수 있도록 템플릿 함수로 만들어준 것입니다.

이 때, 오른쪽에 곱해지는 상대 행렬의 타입은 `Matrix<M, K>`, 리턴 타입은 `Matrix<N, K>`으로 명시되어 있는데, 이를 통해 $N \times M$인 현재 행렬의 오른쪽에 곱해지는 상대 행렬의 크기가 $M \times K$임을 강제하고, 곱한 결과 행렬의 크기가 $N \times K$임을 보장할 수 있습니다.

그렇다면 곱할 수 없는 두 행렬, 예를 들어 `Matrix<2, 2>`과 `Matrix<3, 1>`을 곱하려고 하면 무슨 일이 발생할까요? 위에서 보듯이 `Matrix<N, M>`에는 `Matrix<M, K>`와 곱할 수 있는 함수만이 정의되어 있습니다. 따라서 C++ 컴파일러는 `Matrix<2, 2>`과 `Matrix<3, 1>`을 곱하는 함수가 존재하지 않는다고 판단하여 컴파일 에러를 발생시킵니다.

즉, 행렬 크기에 대한 예외 처리 코드를 전혀 작성하지 않고도, 타입 시스템에 의해 자동으로 크기가 맞는 행렬만 곱할 수 있도록 강제할 수 있습니다. 실수로 크기가 다른 행렬을 곱하는 경우에는 컴파일 에러가 발생하므로 문제를 파악하기 매우 쉽습니다.

---


## 문제 풀이 예시

위에서 상속으로 구현한 행렬 라이브러리를 통해 예시 문제를 풀어봅시다.

### [피보나치 수 3 (BOJ 2749)](https://www.acmicpc.net/problem/2749)

이 문제는 피보나치 수열 $\{F_n\}$이 다음과 같이 주어질 때,

$$F_0 = 0, F_1 = 1, F_n = F_{n-1} + F_{n-2}$$

$0 \le n \le 10^{18}$인 $n$에 대해 $F_n$을 $10^6$으로 나눈 나머지를 출력하는 문제입니다.

피보나치 수를 아래와 같이 행렬로 표현할 수 있다는 사실은 널리 알려져 있습니다.

$$
\begin{bmatrix} 
F_{n+1} \\ 
F_n 
\end{bmatrix} = 
\begin{bmatrix} 
1 & 1 \\ 
1 & 0 
\end{bmatrix}
^n 
\begin{bmatrix} 
F_1 \\ 
F_{0} 
\end{bmatrix} = 
\begin{bmatrix} 
1 & 1 \\ 
1 & 0 
\end{bmatrix}
^n 
\begin{bmatrix} 
1 \\ 
0
\end{bmatrix}
$$

따라서 빠른 행렬 거듭제곱을 통해 $F_n$의 값을 빠르게 구할 수 있습니다. 우선 모든 연산에 대해 $10^6$으로 나눈 나머지가 필요하므로, 위의 행렬 라이브러리의 원소 타입을 `long long`으로 변경하고 모든 연산에 나머지를 적용하도록 곱셈 함수를 살짝 수정해줍시다.

```c++
#include<bits/stdc++.h>
#define MOD 1000000
using namespace std;
typedef long long ll;

template<int N, int M>
struct Matrix: array<array<ll, M>, N> {
	typedef array<array<ll, M>, N> super;
	Matrix() { for(int i=0;i<N;i++) super::operator[](i).fill(0); }
	template<int K>
	Matrix<N,K> operator*(Matrix<M,K>& o) {
		Matrix<N,K> ret;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < K; j++) {
				for (int k = 0; k < M; k++) {
					ret[i][j] += (super::operator[](i)[k] * o[k][j]) % MOD;
					ret[i][j] %= MOD; // calculate remainder
				}
			}
		}
		return ret;
	}
};
```

그리고, 문제풀이에 필요한 빠른 거듭제곱 함수를 작성합시다. 어떤 행렬을 거듭제곱하려면, 그 행렬은 $N \times N$ 크기의 정사각행렬이여야 하므로 템플릿 인자로 `N`을 받아 아래처럼 작성해줄 수 있습니다.

```c++
template<int N>
Matrix<N,N> power(Matrix<N,N>& A, ll n) {
	if(n==0) { // returns I, i.e. identity matrix.
		Matrix<N,N> ret;
		for(int i=0;i<N;i++) ret[i][i] = 1;
		return ret;
	}
	if(n==1) return A;
	Matrix<N,N> X = power(A, n/2);
	if(n%2) return X * X * A;
	else return X * X;
}
```

이렇게 행렬 라이브러리와 거듭제곱 함수를 작성했다면, `main()` 함수에서 해야 할 일은 매우 간단합니다. $A$와 $\rm{x}$를 아래와 같이 선언하고,

$$
A = 
\begin{bmatrix} 
1 & 1 \\ 
1 & 0 
\end{bmatrix} ,
\rm{x} = 
\begin{bmatrix} 
1 \\ 
0 
\end{bmatrix}
$$ 

$A^n \rm{x}$를 계산한 뒤 2행 1열에 있는 값을 읽으면 됩니다. 코드로 쓰면 아래와 같이 매우 간결하게 쓸 수 있습니다.

```c++
int main() {
	Matrix<2,2> A; Matrix<2,1> x;
	
	A[0][0] = A[0][1] = A[1][0] = 1;
	x[0][0] = 1;
	
	ll n;
	cin >> n;
	cout << (power(A, n) * x)[1][0];
}
```

위 코드들을 전부 합쳐서 제출하면 이 문제를 풀 수 있습니다. 

## 마치며

C++에서 PS에서 상속을 활용해야 하는 경우는 별로 없지만, 상속을 잘 활용하면 기존 컨테이너의 인터페이스와 메서드를 그대로 활용하면서 여러 기능과 연산을 추가할 수 있다는 장점이 있습니다. 이 글에서 제시한 행렬 라이브러리의 예시가 여러분이 다른 라이브러리 코드를 작성하는 데에도 도움이 되었으면 좋겠습니다.