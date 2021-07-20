---
layout: post
title: Double dabble
date: 2021-07-15 12:00:00
author: jeonggyun
tags:
---

안녕하세요?

이진법의 수를 BCD(binary-coded decimal)로 변환하는 알고리즘인 double dabble에 대해 알아보겠습니다.

십진수의 각 자리를 표시하는 데에는 4 bit가 필요하므로, 보통 이진수 네 자리를 묶어 십진수 한 자리를 표시하는 데에 사용할 수 있습니다. 예를 들어 53을 0101 0011과 같이 표시할 수 있습니다. 이를 BCD라고 합니다.

53을 이진수로 표시하면 110101이므로, 저희의 목표는 이를 BCD로 변환하는(또는 반대로), 다시 말해 110101를 0101 0011로 변환하는 것입니다.

위 작업 자체는 단순히 이진수를 십진수로 변환하기만 하면 되는 간단한 진법 변환이므로 매우 쉬운 작업입니다. Double dabble 알고리즘은 그리 새로운 알고리즘이라고 볼 수는 없지만, 그 수행 과정이 꽤 재미있다고 생각하여 소개를 드리고자 합니다.

# Double dabble

Double dabble은 1997년 Xilinx의 [Serial Code Conversion between BCD and Binary](http://www.ingelec.uns.edu.ar/dclac2558/BCD2BIN.PDF)라는 application note에서 소개된 알고리즘으로, shift-and-add-3 알고리즘이라고도 부릅니다.

Double dabble이 가지는 가장 큰 특징으로는, 이름에서도 추측 가능하듯 shift 연산과 더하기 연산만을 사용하여 변환을 진행한다는 점입니다. 때문에 필요한 논리 gate의 수가 적어 특히 FPGA 등을 매우 간단하게 설계할 수 있다는 장점이 있습니다.

알고리즘의 작동 과정은 아래와 같습니다. 원본 이진수를 오른쪽에, BCD가 왼쪽에 생성된다고 하겠습니다.

원본 이진수의 bit 수 번 만큼 왼쪽으로 shift를 합니다. shift를 진행한 이후에, BCD 중 자리수의 값이 5 이상인 것이 있을 경우 해당 자리에 3을 더해줍니다. 단, 맨 마지막 shift가 일어난 이후에는 덧셈은 생략합니다.

아래는 $922 = 1110011010\_{\(2\)}$를 BCD로 변환하는 과정을 나타낸 표입니다.

| BCD | Binary | 비고 |
| ---- | ----- |
| | 1110011010 | |
| 1 | 110011010 | shift |
| 11 | 10011010 | shift |
| 111 | 0011010 | shift |
| 1010 | 0011010 | shift |
| 1 0100 | 011010 | shift |
| 10 1000 | 11010 | shift |
| 10 1011 | 11010 | 첫째 자리($8 \ge 5$)에 3 더함 |
| 101 0111 | 1010 | shift |
| 101 1010 | 1010 | 첫째 자리($7 \ge 5$)에 3 더함 |
| 1000 1010 | 1010 | shift |
| 1 0001 0101 | 010 | shift |
| 1 0001 1000 | 010 | 첫째 자리($5 \ge 5$)에 3 더함 |
| 10 0011 0000 | 10 | shift |
| 100 0110 0001 | 0 | shift |
| 100 1001 0001 | 0 | 둘째 자리($6 \ge 5$)에 3 더함 |
| 1001 0010 0010 | | shift |

위와 같은 과정을 통해 최종적으로 $1110011010\_{\(2\)} = 922\_{\(10\)} = 1001\~\~0010\~\~0010\_{\(BCD\)}$로의 변환을 완료할 수 있습니다.

위와 같은 간단한 방법이 어떻게 작동할 수 있을까요?

먼저 shift를 살펴봅시다. 전체 자리를 왼쪽으로 1만큼 shift해주는 것은 2를 곱하고, 해당 자리의 bit가 켜져있을 경우 1을 더해주는 작업과 같습니다.

그렇다면 핵심은 5 이상일 때 3을 더해주는 작업에 있습니다. 이 작업을 수행해주는 이유는 받아올림이 일어나기 때문입니다.

5에 2를 곱해줄 때, 왼쪽으로 한 칸 shift를 시켜준다면 101에서 1010이 되어야 하지만, 실제로 표시상으로는 10이 되므로 실제로는 0001 0000이 되어야 합니다. 3을 더해주는 것은 이 차이를 보정하기 위함입니다.

다시 말해, 101에 3을 더해 1000으로 바꾸어준 뒤 shift를 진행해 10000이 되도록 만들어주는 것입니다.

# 임의의 짝수 진법에서 사용

위 방법은 꼭 각 자리수가 4 bit인 십진법이 아닌 임의의 짝수 진법에 대해서도 작동합니다.

p진법 상에서 받아올림이 일어나는 수를 x라 하고 $(x \ge p / 2)$, 한 자리를 표현하는 데에 필요한 비트 수를 t라고 했을 때(예시에서 t = 4) BCD 표기법 상에서 2를 곱했을 때는 실제로는 $2^t + 2 * (x - p / 2)$ 형태를 가지는 이진수가 되어야 합니다. 즉, 따라서, shift를 해주기 전에 $2^{t-1} - p / 2$에 해당하는 수를 더해주어야 합니다.

위에서 진행했던 예시인, 각 자리가 4 bit인 십진법으로 표시를 원할 경우 해당 자리가 5 이상일 때 $2^{4-1} - 10 / 2 = 3$을 더해주면 됩니다.

몇 가지 추가적인 예시로, 각 자리가 4 bit인 12진법으로 표시를 원할 경우 자리의 값이 6 이상일 때 $2^{4-1} - 12 / 2 = 2$를 더해주면 되며, 각 자리가 5 bit인 22진법으로 표시를 원할 경우 자리의 값이 11 이상일 때 $2^{5-1} - 22 / 2 = 5$를 더해주면 됩니다.

# 역변환

이진법으로 표현된 수를 BCD로 변환할 수 있는 것과 마찬가지로, double dabble 알고리즘을 통해 BCD를 이진법으로 변환하는 것 또한 가능합니다.

진행했던 과정을 정확히 반대로 진행하면 됩니다. 따라서 왼쪽으로 한 칸씩 shift하며, 8 이상인 수가 나올 때마다 3을 빼주면 됩니다.

역변환이 가능하기 때문에, 임의의 짝수 진법을 이진법으로 바꾼 뒤 다시 변환하면 임의의 짝수 진법끼리 변환 또한 가능합니다.

# 시간 복잡도

일반적으로 진법 사이의 변환은 곱셈과 나눗셈 연산을 이용하여 수행할 수 있습니다.

Double dabble 알고리즘도 결국 shift 연산, 즉 2로 곱하는 연산을 기본으로 하여 작동하므로 본질적으로 기존의 방법과 크게 다르다고 볼 수는 없으며, 따라서 시간복잡도 또한 동일합니다.

일반적인 방법으로 p진법으로 표기된 수를 q진법으로 변환하기 위해서는 $\log\_{p}{n}$회의 곱셈과 $\log\_{q}{n}$회의 나눗셈을 수행해야 하며, 각각의 연산에 소요되는 시간은 $O(\log\_{2}{n} \log\_{2}{p})$, $O(\log\_{2}{n} \log\_{2}{q})$이므로 총 시간복잡도는 $O((\log\_{2}{n})^2)$입니다.

Double dabble 알고리즘 또한 p진법으로 표기된 수를 q진법으로 변환하기 위해 이진법을 거쳐 변환을 진행하면 shift를 하는 데에 $O(\log\_{2}{n})$, add를 하는 데에 $O(\log\_{q}{n} \log\_{2}{q})$가 소요되므로 총 시간복잡도는 $O((\log\_{2}{n})^2)$로 동일합니다.

# 구현

Double dabble 알고리즘은 보통 verilog 등의 프로그램을 통해 많이 구현되는 편이고, C로 구현하기는 그리 적합하지 않지만, 테스트를 위해 알고리즘을 간단하게 구현해 본 코드입니다.

```cpp
#include <iostream>
#define BUF_SIZE 1000
#define BIT_SIZE sizeof(unsigned long long)
#define BIT_PER_NUM 4
using namespace std;

unsigned long long output[BUF_SIZE];
unsigned long long input[BUF_SIZE];

int main() {
	// input
	string input_string;
	cin >> input_string;

	int n = input_string.size();
	for (int i = 0; i < n; ++i) {
		if (input_string[i] == '1') input[i / BIT_SIZE] |= (1ULL << ((BIT_SIZE - 1) - i % BIT_SIZE));
	}

	int input_bit = n;
	int output_bit = 0;

	while (input_bit) {
		// add
		for (int i = 0; i < (output_bit + BIT_PER_NUM - 1) / BIT_PER_NUM; ++i) {
			int value = 0;
			for (int j = BIT_PER_NUM - 1; j >= 0; --j) {
				int pos = BUF_SIZE * BIT_SIZE - 1 - i * BIT_PER_NUM - j;
				value <<= 1;
				if (output[pos / BIT_SIZE] & (1ULL << ((BIT_SIZE - 1) - pos % BIT_SIZE))) value += 1;
			}
			if (value >= 5) {
				value += 3;
				for (int j = 0; j < BIT_PER_NUM; ++j) {
					int pos = BUF_SIZE * BIT_SIZE - 1 - i * BIT_PER_NUM - j;
					if (value & 1) output[pos / BIT_SIZE] |= (1ULL << ((BIT_SIZE - 1) - pos % BIT_SIZE));
					else output[pos / BIT_SIZE] &= ~(1ULL << ((BIT_SIZE - 1) - pos % BIT_SIZE));
					value >>= 1;
				}
				if (i == (output_bit + BIT_PER_NUM - 1) / BIT_PER_NUM - 1) {
					for (int j = BIT_PER_NUM - 1; j >= 0; --j) {
						int pos = BUF_SIZE * BIT_SIZE - 1 - i * BIT_PER_NUM - j;
						if (output[pos / BIT_SIZE] & (1ULL << ((BIT_SIZE - 1) - pos % BIT_SIZE))) {
							output_bit = i * BIT_PER_NUM + j + 1;
							break;
						}
					}					
				}
			}
		}

		// shift
		for (int j = BUF_SIZE - (output_bit + (BIT_SIZE - 1)) / BIT_SIZE; j < BUF_SIZE; ++j) {
			if (output[j] & (1ULL << (BIT_SIZE - 1))) output[j - 1] |= 1;
			output[j] <<= 1;
		}
		output_bit++;
		for (int j = 0; j < (input_bit + (BIT_SIZE - 1)) / BIT_SIZE; ++j) {
			if (input[j] & (1ULL << (BIT_SIZE - 1))) {
				if (j == 0) output[BUF_SIZE - 1] |= 1;
				input[j - 1] |= 1;
			}
			input[j] <<= 1;
		}
		input_bit--;
	}

	// print
	for (int i = BUF_SIZE * BIT_SIZE - output_bit; i < BUF_SIZE * BIT_SIZE; ++i) {
		if (output[i / BIT_SIZE] & (1ULL << ((BIT_SIZE - 1) - i % BIT_SIZE))) cout << '1';
		else cout << '0';
		if (i % BIT_PER_NUM == BIT_PER_NUM - 1) cout << ' ';
	}
}
```

# Reference

아래는 해당 글을 작성하는 데에 참고한 자료 및 사이트입니다.

[Double dabble](https://en.wikipedia.org/wiki/Double_dabble)

[Serial Code Conversion between BCD and Binary](http://www.ingelec.uns.edu.ar/dclac2558/BCD2BIN.PDF)

[PARALLEL DECIMAL MULTIPLIERS USING BINARY MULTIPLIERS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5483001)