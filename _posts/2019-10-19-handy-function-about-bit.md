---
layout: post
title:  "알아두면 편리한 비트 연산 몇 가지"
date:   2019-10-19 23:59:59
author: Acka1357
tags: builtin function, submask iteration
---

## Bitwise Operation

Problem Solving을 하다 보면, 혹은 최적화 등의 작업을 하다 보면 bit 단위의 연산을 하게 될 때가 있습니다. 일반적으로 논리/정수 자료형에서 많이 사용되기 때문에 본 포스팅의 설명은 해당 두 자료형에 한하며 이를 묶어 '정수'라고 표현하겠습니다. 

간단한 비트 연산은 언어에서 제공하는 연산자를 통해 두 개의 정수를 ```and```, ```or```, ```xor``` 하거나 하나의 정수를 ```shift```, ```not``` 등으로 변형할 수 있습니다. 이 포스팅에서는 앞에서 나온 기본 단위 연산자에 관한 내용은 따로 다루지 않고, 좀 더 나아가 하라면 하겠는데 구현이 귀찮은 몇몇 비트 연산에 대해 좀 더 쉽고 빠른 방법을 소개하려 합니다.



### Builtin Function

정수를 이진수로 나타내어 봅시다. 이때 각 자리를 하나의 비트로 보고 그 값이 1인 곳을 켜져 있다, 0인 곳을 꺼져 있다고 할 때, 어떤 정수에서 켜져 있는 비트가 몇개인지 프로그램에서 어떻게 구할 수 있을까요?

간단하게 아래와 같이 실제로 이진수를 만들어가면 $$O(logN)$$의 시간복잡도에 1의 개수를 셀 수 있을 것입니다.

```cpp
int count_bit(unsigned int x){
	int cnt = 0;
	while(x > 0){
		cnt += (x & 1);
		cnt >>= 1;
	}
	return cnt;
}
```



또는 비트 개수를 세는 일이 아주 많다면, 전처리를 통해 개수를 세는 부분을 $$O(1)$$로 줄일 수 있습니다. 16개 비트로 나타낼 수 있는 정수에 대해 비트 개수를 미리 구해놓는다면 32bit 정수에 대해 비트마스킹을 통해 두 부분을 쪼개어 더하기만 하면 개수를 구할 수 있겠죠? 혹은 아래와 같이 '조금' 복잡한 비트 연산과 마스킹을 이용하면 전처리 없이도 다양한 방법으로 $$O(1)$$의 시간에 구할 수 있습니다.

```cpp
int count_bit(unsigned int ux) {
	unsigned int cnt = ux - ((ux >> 1) & 033333333333) - ((ux >> 2) & 011111111111);
	return ((cnt + (cnt >> 3)) & 030707070707) % 63;
}
```



하지만 여러분의 환경이 GCC, Clang, EDG 등의 C/C++ 컴파일러를 사용한다면, Builtin Function(내장 함수)을 사용해 훨씬 간단하게 해결할 수 있습니다.  Java에서는 Integer 클래스에 BitCount 메소드가 구현되어 있으며, $$O(1)$$의 시간복잡도로 빠르게 동작합니다. 제 견문이 좁아 모든 언어에 대해 소개해드릴 수 없어 아쉽게도 이 부분에서 소개하는 내용은 C/C++ 언어에 한정됩니다.

**내장 함수**란, include/import 등을 통해 사용할 수 있는 라이브러리 함수와 달리 컴파일러에서 기본으로 제공하는 함수입니다. 일반적으로 인라인으로 삽입되어 함수 호출 오버헤드가 발생하지 않고 특정 인스트럭션을 사용하여 유저가 구현하는 알고리즘보다 빠르게 동작할 수 있습니다.

위의 코드는 한 줄의 내장 함수로 대체할 수 있습니다.

```cpp
int cnt = __builtin_popcount(x);
```

Population Count의 약어인 이 함수는 동일하게 x의 비트 중 1인 것의 개수를 반환합니다. 어떤 헤더도 링킹할 필요 없이, 바로 함수를 사용할 수 있습니다.

추가로 가장 앞에 있는 1의 위치와 가장 뒤에 있는 1의 위치를 구하는 방법도 있습니다. 정확히는 1이 등장하기 전 앞/뒤의 0의 개수를 세어주는 함수입니다.

```cpp
int x = 100; // 00000000 00000000 00000000 01100100
printf("Count of leading zero: %d\n", __builtin_clz(x));  // 25
printf("Count of trailing zero: %d\n", __builtin_ctz(x)); // 2
```

100이라는 숫자의 켜진 비트는 3번째와 6번째, 7번째 비트입니다. 이때 leading zero의 개수는 32비트 정수에서 7번째 비트보다 앞에 있는 25개가 되며, trailing zero의 개수는 3번째 비트보다 뒤에 있는 2개가 됩니다. 만약 가장 작은 비트만 가지는 마스크를 만들고 싶다면 ```1 << __builtin_ctz(x)``` 형태의 코드를 작성 할 수 있겠죠? 참고로 가장 마지막 비트를 뽑는 방법으로는 Fenwick Tree 등에 많이 쓰이는 ```(x & -x)```와 같은 방법도 있습니다.

위에서 소개한 세 가지 함수는 모두 **32bit unsigned integer를 기준**으로 합니다. 만약 인자로 음수를 전달하게 되면,  가장 앞의 signed bit가 1인 것을 포함하여 2의 보수 형태가 그대로 계산됩니다. 또한 0은 1인 비트가 하나도 없기 때문에 clz와 ctz는 모두 32를 반환합니다. 

그렇다면 이 내장함수들은 32bit unsigend integer로 정제해야만 쓸 수 있을까요? [링크](https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html)를 보시면 비트와 관련된 것 말고도 다양한 내장함수들이 있는데, 친절하게도 unsinged int, unsigned long, unsigned long long을 인자로 하는 세가지 함수가 모두 있는 것을 확인할 수 있습니다. unsigned long 타입을 이용할 경우엔 함수명 끝에 ```l```을, unsigned long long 타입을 이용할 경우엔 함수명 끝에 ```ll```을 붙여주면 각 타입에 맞는 함수를 사용할 수 있습니다.  

내장함수에 전달하는 인자에 대해서는 타입이 맞지 않을 때 Warning 등을 주지 않으니 주의하세요.



### Submask Iteration

비트의 나열은 하나의 이진수를 나타내기도 하고, 하나의 집합을 나타낼 수도 있습니다. {a<sub>0</sub>, a<sub>1</sub>, a<sub>2</sub>, a<sub>3</sub>, a<sub>4</sub>, a<sub>5</sub>, a<sub>6</sub>, a<sub>7</sub>, a<sub>8</sub>, a<sub>9</sub>}라는 하나의 집합이 있을 때 우리는 591 = 0101001111<sub>(2)</sub>라는 마스크를 통해 {a<sub>0</sub>, a<sub>1</sub>, a<sub>2</sub>, a<sub>3</sub>, a<sub>6</sub>, a<sub>8</sub>}이라는 부분 집합을 나타낼 수 있습니다.  

어떤 집합을 나타내는 정수가 주어졌을 때, 집합에 존재하는 원소의 번호는 아래와 같이 추출할 수 있습니다.

```cpp
vector<int> extract_elem(unsigned int set){
	vector<int> elems;
	for(; set; set -= (set & -set))
		elems.push_back(__builtin_ctz(set));
	return elems;
}
```

만약 ```__builtin_ctz``` 함수를 사용할 수 없는 상황이라면 각 원소의 마스크인 ```(set & -set)```을 통해 원소의 번호를 구할 수 있을 것입니다. 여기에 더해 주어진 마스크가 의미하는 집합의 모든 부분집합을 나타내는 서브마스크는 어떻게 구할 수 있을까요? 위에서 원소들을 구했으니 아래와 같이 재귀함수를 사용하면 $$O({2}^{popcount(x)})$$만큼의 시간을 들여 모든 서브마스크를 구할 수 있을 것입니다.

```cpp
void make_subset(vector<int> &elems, vector<int> &subset, int idx = 0, int cur_set = 0){
	if(idx == (int)elems.size()){
		subset.push_back(cur_set);
		return ;
	}
	make_subset(elems, subset, idx + 1, cur_set);
	make_subset(elems, subset, idx + 1, cur_set | (1 << elems[idx]));
}
```

하지만 비트 연산을 잘 이용하면 아래와 같은 반복문으로 서브마스크를 차례로 순회할 수 있습니다.

```cpp
vector<int> make_subset(unsigned int set){
	vector<int> subset;
	for (int cur_set = set; cur_set; cur_set = ((cur_set - 1)&set))
		subset.push_back(cur_set);
	return subset;
}
```

둘 모두 시간복잡도는 모든 부분집합의 개수인 $$O({2}^{popcount(x)})$$지만 아무래도 반복문 쪽이 좀 더 빠르게 동작하겠죠? 또한 아래 코드는 마스크를 정수로 나타내었을 때 큰 숫자부터 차례로 구한다는 장점이 있습니다. 덧붙여 위의 코드는 공집합도 부분집합으로 구하는 반면, 아래 코드는 구하지 않습니다. 이 부분에 대해서는 필요 유무에 따라 처리해주세요.



#### 모든 부분집합의 부분집합 순회

조금 더 더해서, 원소가 $$N$$개 있는 집합의 모든 부분집합 $$s$$를 순회하며, 각 $$s$$의 모든 부분집합을 순회하는 코드를 생각해 봅시다. 가아아끔 Bitwise DP 문제 중 이러한 연산을 처리해야 할 때가 있습니다. 이 부분은 딱히 더 효율적인 부분에 대해 설명하려는 것이 아니라 시간복잡도에 대한 설명입니다.

```cpp
for(int s = 0; s < (1 << N); s++){
  printf("cur subset: %d\n", s);
  for(int sub = s; sub; sub = (sub - 1) & s)
    printf("..%d is submask of %d\n", sub, s);
}
```

[0, 2<sup>N</sup> - 1] 범위의 모든 부분집합을 순회하며, 위에서 소개한 서브마스크 순회를 이용하면 모든 부분집합의 부분집합을 순회할 수 있습니다. 이때 위 코드는 얼마의 시간복잡도를 가질까요?

k개의 원소를 가진 집합의 부분집합의 개수는 2<sup>k</sup>입니다. 위에서 언급했듯이 각 s에 대하여 s의 원소 개수가 k개라면 안쪽의 반복문은 $$O({2}^{k})$$에 동작할 것입니다.

그렇다면 k개의 원소를 가진 부분집합 s는 몇 개나 있을까요? N개의 원소 중 s개를 뽑는 경우의 수인 $${N \choose k}$$입니다. 각 k에 대해 $$O({2}^{k})$$가 $${N \choose k}$$번씩 반복되므로 총 시간복잡도는 $$O( \sum_{k=0}^{N} {N \choose k}{2}^{k})$$입니다. $$x=y=1$$인 x, y를 통해 우리는 이를 아래와 같이 계산할 수 있습니다.

$$ \sum_{k=0}^{N} {N \choose k}{2}^{k} = \sum_{k=0}^{N} {N \choose k}{x}^{N-k}{(2y)}^{k} = {(x + 2y)}^{N} = {3}^{N}$$ 

따라서 우리는 모든 부분집합의 부분집합 순회의 총 시간복잡도가 $$O({N}^{3})$$인 것을 알 수 있습니다. 이는 해당 로직이 포함된 알고리즘의 시간복잡도 분석에도 유용합니다. 사실 저도 써먹은 일은 드물지만요.



### 마치며

여기까지 알아두면 편리한 비트 연산을 몇 가지 소개해드렸습니다. 지금은 개수가 적지만 앞으로 생각나거나 공부하며 알게 되면 추가해두겠습니다. 혹시 본 포스팅 내용 중 잘못된 내용이 있거나 추가했으면 하는 내용이 있다면 편하게 Acka1357@gmail.com 으로 말씀해주세요.

읽어주셔서 감사합니다 :)