---
layout: post
title: "Advanced Game Search Algorithms (1)"
date: 2025-10-25
author: jinhan814
tags: [algorithm, game-theory, problem-solving]
---

## 1. Introduction

이번 글에서는 게임 에이전트의 가장 단순한 형태인 Random Agent부터 시작하여 Greedy, Minimax, Alpha-Beta Pruning의 핵심 원리를 다룹니다. 이후 이어지는 글에서는 MCTS 등의 현대적인 탐색 기법을 알아보고, NNUE 등의 neural network를 이용한 평가 방법과 여러 search prunning 방법을 살펴보겠습니다.

또한 에이전트의 성능을 객관적으로 평가하기 위해 SPRT(Sequential Probability Ratio Test)라는 평가 기법을 소개합니다. 이를 이용하면 통계적으로 두 에이전트 간의 실력 차이를 엄밀하게 검증할 수 있습니다.

이번 글에서 소개하는 방법론은 $2$인, 제로섬, 턴제, 완전정보, 결정론적 전이를 만족하는 게임에 적용이 가능하며, 구체적인 설명을 위해서 ATAXX를 예시로 각 알고리즘을 구현해보겠습니다.

## 2. ATAXX

게임 에이전트를 구현하는 방법을 알아보기에 앞서, 예시로 들 게임의 규칙을 먼저 소개하겠습니다.

### 2-1. 게임 규칙

ATAXX는 $7 \times 7$ 보드에서 진행되는 $2$인 턴제 게임입니다.

![Fig.1](/assets/images/2025-10-25-advanced-game-search/fig1_v4.png)

선공 플레이어는 $(1, 1), (7, 7)$ 위치에 두 개의 돌을 가지고 있고, 후공 플레이어는 $(1, 7)$, $(7, 1)$ 위치에 두 개의 돌을 가지고 있습니다. 나머지 $45$개의 칸은 모두 빈 칸입니다.

게임은 두 플레이어가 번갈아가며 본인의 돌이 놓여있는 시작 칸 $(x_1, y_1)$과 비어있는 도착 칸 $(x_2, y_2)$를 고르며 진행됩니다.

각 턴마다 게임이 진행되는 방식은 다음과 같습니다.

- 이동 거리 $d = \max(|x_2 - x_1|, |y_2 - y_1|)$는 $1$ 또는 $2$여야 합니다.
- 이동 거리가 $1$인 경우는 시작 칸의 돌이 그대로 남고 도착 칸에만 자신의 돌이 새로 생깁니다.
- 이동 거리가 $2$인 경우는 시작 칸의 돌이 도착 칸으로 이동합니다.
- 도착 칸 $(x_2, y_2)$를 기준으로 $8$방향(상하좌우, 대각선)으로 인접한 칸에 상대의 돌이 있다면, 해당 칸을 자신의 돌로 바꿉니다.
- 자신의 턴에 고를 수 있는 $(x_1, y_1), (x_2, y_2)$가 없다면 `PASS`를 선택하며, 이 경우에는 아무 변화 없이 상대에게로 턴이 넘어갑니다.

정리하면, 각 플레이어는 이동거리가 $1$ 또는 $2$인 $(x_1, y_1), (x_2, y_2)$를 골라 턴을 진행하며, 고를 수 있는 행동이 없다면 `PASS`를 선택해 턴을 넘깁니다.

다음은 선공 플레이어가 $(1, 1)$에서 $(2, 2)$로 이동 거리가 $1$인 행동을 수행하는 예시입니다.

![Fig.2](/assets/images/2025-10-25-advanced-game-search/fig2_v3.png)

다음은 후공 플레이어가 $(2, 2)$에서 $(4, 4)$로 이동 거리가 $2$인 행동을 수행하는 예시입니다.

![Fig.3](/assets/images/2025-10-25-advanced-game-search/fig3_v3.png)

마지막으로 다음은 선공 플레이어가 `PASS`를 선택하는 예시입니다.

![Fig.4](/assets/images/2025-10-25-advanced-game-search/fig4_v3.png)

게임은 다음 조건 중 하나가 만족되면 종료됩니다.

- 어느 한 플레이어의 돌이 보드에서 완전히 사라졌을 때
- 보드에 남은 빈 칸이 하나도 없을 때
- 두 플레이어가 모두 $200$번씩 총 $400$턴을 수행했을 때

종료 시점에 돌이 더 많은 플레이어가 승리하며, 만약 두 플레이어의 돌 개수가 같다면 무승부로 마무리합니다.

여기까지가 ATAXX 게임의 룰입니다. $400$턴을 초과하면 게임을 종료한다는 규칙은 게임이 무한히 길어지는 걸 방지하기 위해 임의로 추가했습니다. 다른 플렛폼에서는 $3$회 동형, $50$수 규칙 등을 채택할 수 있음에 주의해주세요.

해당 게임은 [링크](https://alphano.co.kr/problem/1/play)에서 플레이해볼 수 있습니다.

### 2-2. 입출력 형식

이제 에이전트와 심판 프로그램 간의 상호작용 규칙을 정의하겠습니다.

| 명령어 | 심판→에이전트(입력) | 에이전트→심판 (출력) | 시간 제한(ms) | 설명 |
|:--|:--|:--|:--|:--|
| **READY** | `READY (FIRST \| SECOND)` | `OK` | `3000` | 선공/후공 정보를 알립니다. |
| **TURN** | `TURN my_time opp_time` | `MOVE x1 y1 x2 y2` | `my_time` | 내 남은 시간과 상대의 남은 시간을 알립니다. 이번 턴에 내가 선택한 수의 $(x_1, y_1)$, $(x_2, y_2)$를 출력합니다. `PASS`를 선택한 경우는 `MOVE -1 -1 -1 -1`을 출력합니다. |
| **OPP** | `OPP x1 y1 x2 y2 time` | - | - | 상대가 직전에 둔 수와 사용한 시간을 알립니다. 상대가 `PASS`를 선택한 경우는 `OPP -1 -1 -1 -1 time`이 입력됩니다. |
| **FINISH** | `FINISH` | - | - | 게임 종료를 알립니다. 에이전트는 추가 출력 없이 프로그램을 정상 종료해야 합니다. |

각 에이전트의 제한 시간은 게임 당 `10000ms`로 주어지고, 에이전트는 주어진 시간을 잘 분배해서 사용해야 합니다.

이제 이 형식에 맞춰서 게임 에이전트를 구현해보며 여러 방법론의 장단점을 알아보겠습니다.

## 3. Random Agent

가장 먼저 생각해볼 수 있는 정책은 가능한 행동 중 아무 행동이나 랜덤하게 고르는 것입니다.

랜덤 정책을 위의 입출력 형식에 맞게 구현한 코드는 다음과 같습니다.

```cpp
#include <bits/stdc++.h>
using namespace std;

struct board {
	board() : v{} {
		v[1][1] = v[7][7] = 1;
		v[1][7] = v[7][1] = 2;
	}
	void apply_move(int x1, int y1, int x2, int y2, int turn) {
		if (x1 == -1 && y1 == -1 && x2 == -1 && y2 == -1) return;
		int d = max(abs(x2 - x1), abs(y2 - y1));
		if (d == 2) v[x1][y1] = 0;
		v[x2][y2] = turn;
		for (int x = x2 - 1; x <= x2 + 1; x++) {
			for (int y = y2 - 1; y <= y2 + 1; y++) {
				if (x < 1 || x > 7 || y < 1 || y > 7) continue;
				if (v[x][y] == (turn ^ 3)) v[x][y] = turn;
			}
		}
	}
	int get(int x, int y) const {
		return v[x][y];
	}
private:
	int v[8][8];
};

int gen_rand(int l, int r) {
	static mt19937 rd(42);
	return uniform_int_distribution(l, r)(rd);
}

auto find_move(board game, int turn) {
	tuple ret(-1, -1, -1, -1);
	int cnt = 0;
	for (int x1 = 1; x1 <= 7; x1++) {
		for (int y1 = 1; y1 <= 7; y1++) {
			if (game.get(x1, y1) != turn) continue;
			for (int x2 = x1 - 2; x2 <= x1 + 2; x2++) {
				if (x2 < 1 || x2 > 7) continue;
				for (int y2 = y1 - 2; y2 <= y1 + 2; y2++) {
					if (y2 < 1 || y2 > 7) continue;
					if (x2 == x1 && y2 == y1) continue;
					if (game.get(x2, y2) != 0) continue;
					if (gen_rand(1, ++cnt) == 1) ret = tuple(x1, y1, x2, y2);
				}
			}
		}
	}
	return ret;
}

int main() {
	board game;
	int turn;
	while (1) {
		string s; getline(cin, s);
		istringstream in(s);
		string cmd; in >> cmd;
		if (cmd == "READY") {
			string t; in >> t;
			turn = t == "FIRST" ? 1 : 2;
			cout << "OK" << endl;
		}
		else if (cmd == "TURN") {
			int t1, t2; in >> t1 >> t2;
			auto [x1, y1, x2, y2] = find_move(game, turn);
			game.apply_move(x1, y1, x2, y2, turn);
			cout << "MOVE " << x1 << ' ' << y1 << ' ' << x2 << ' ' << y2 << endl;
		}
		else if (cmd == "OPP") {
			int x1, y1, x2, y2; in >> x1 >> y1 >> x2 >> y2;
			int t2; in >> t2;
			game.apply_move(x1, y1, x2, y2, turn ^ 3);
		}
		else if (cmd == "FINISH") {
			break;
		}
		else {
			assert(0);
		}
	}
}
```

코드에서 `find_move` 함수는 $(x_1, y_1, x_2, y_2)$ 조합을 모두 확인하며 가능한 행동을 균등한 확률로 선택합니다. 만약 가능한 조합이 없다면 `PASS`를 의미하는 $(-1, -1, -1, -1)$을 반환합니다.

`apply_move` 함수는 `turn`에 해당하는 플레이어가 $(x_1, y_1, x_2, y_2)$를 선택했을 때 보드의 변화를 반영하는 역할을 수행합니다.

랜덤 정책은 이후 구현할 정책들의 성능을 비교하기 위한 기준선(Baseline)입니다. 새로운 정책이 효과적이라면 랜덤 정책보다 높은 승률을 보여야 하며, 이를 통해 최소한의 성능을 검증하고 구현상의 오류를 조기에 발견할 수 있습니다.

## 4. Greedy Agent

다음으로 `find_move` 함수에서 평가 함수를 이용해 $(x_1, y_1, x_2, y_2)$ 중 평가값이 최대인 행동을 그리디하게 고르는 정책을 알아보겠습니다.

```cpp
int eval(board game, int turn) {
	int ret = 0;
	for (int i = 1; i <= 7; i++) {
		for (int j = 1; j <= 7; j++) {
			int val = game.get(i, j);
			if (val == turn) ret++;
			if (val == (turn ^ 3)) ret--;
		}
	}
	return ret;
}

auto find_move(board game, int turn) {
	tuple ret(-1, -1, -1, -1);
	int opt = -(1 << 30);
	for (int x1 = 1; x1 <= 7; x1++) {
		for (int y1 = 1; y1 <= 7; y1++) {
			if (game.get(x1, y1) != turn) continue;
			for (int x2 = x1 - 2; x2 <= x1 + 2; x2++) {
				if (x2 < 1 || x2 > 7) continue;
				for (int y2 = y1 - 2; y2 <= y1 + 2; y2++) {
					if (y2 < 1 || y2 > 7) continue;
					if (x2 == x1 && y2 == y1) continue;
					if (game.get(x2, y2) != 0) continue;
					board nxt = game;
					nxt.apply_move(x1, y1, x2, y2, turn);
					int val = eval(nxt, turn);
					if (opt < val) {
						ret = tuple(x1, y1, x2, y2);
						opt = val;
					}
				}
			}
		}
	}
	return ret;
}
```

코드는 랜덤 정책에서 `eval` 함수를 새로 구현한 뒤 `find_move` 함수를 이에 맞춰 수정해주면 구현할 수 있습니다.

`eval` 함수는 `game`과 `turn` 인자를 받아 현재 보드의 상태가 `game`일 때 `turn`에 해당하는 플레이어가 유리한 정도를 나타내는 값을 반환합니다.

이를 구현하는 방법은 여러가지가 있습니다. 먼저 떠올릴 수 있는 방법은 두 플레이어가 최적으로 플레이할 때 결과가 승리라면 $1$, 무승부라면 $0$, 패배라면 $-1$을 반환하도록 하는 것입니다. 이 정의에 맞는 `eval` 함수를 구현할 수 있다면 그리디 정책은 실제로 최적의 수를 구합니다. 하지만 게임의 특성 상 game tree가 너무 커서 이 값을 실제로 구하기는 실질적으로 어렵습니다.

이에 대한 대안으로는 `turn`에 해당하는 돌의 개수에서 `turn ^ 3`에 해당하는 돌의 개수를 뺀 값을 반환하도록 하는 휴리스틱 함수를 생각해볼 수 있습니다. 이는 돌 개수가 더 많다면 이길 가능성이 높다는 가정을 바탕으로 유리한 정도를 표현한 함수로, 실제로는 다음 턴에 상대가 어떤 행동을 고르는지에 따라 승패가 뒤집힐 수 있기에 정확한 모델링이 아니지만 근사적으로 `eval` 함수를 구성할 수 있다는 장점이 있습니다. 여기서는 이 방법을 사용하며, `eval` 함수를 개선하는 방법은 다음 글에서 다루겠습니다.

`find_move` 함수는 `eval` 함수를 이용해 행동을 수행한 뒤의 보드의 평가값을 구하고, 이 값이 최대가 되는 행동을 반환합니다. 만약 평가값이 최대인 행동이 여러개라면 $(x_1, y_1, x_2, y_2)$가 사전순으로 최소인 행동을 반환하도록 했습니다.

그리디 정책은 `eval` 함수가 게임의 유불리를 얼마나 정확히 모델링하는가에 따라 성능이 달라집니다. 하지만 돌 개수의 차이와 같은 간단한 모델링만 이용하더라도 랜덤 정책보다는 성능이 개선됨을 기대할 수 있습니다.

## 5. SPRT(Sequential Probability Ratio Test)

지금까지 랜덤 정책과 그리디 정책을 알아보았습니다. 이번 단락에서는 두 정책의 성능을 비교하는 통계적 기법인 SPRT(Sequential Probability Ratio Test)를 알아보겠습니다.

두 정책의 성능을 비교하는 가장 간단한 방법은 여러 번 두 정책끼리 대결을 시켜보는 것입니다. 예를 들어 $100$번 매칭을 돌렸는데 첫 번째 정책이 $20$번, 두 번째 정책이 $80$번 승리했다면 두 번째 정책이 첫 번째 정책보다 더 우수하다고 판단할 수 있습니다. 하지만 이 방법은 두 정책의 실제 승률을 근사적으로 정확하게 구하기 위해선 많은 시행 횟수가 필요하고, 종료 시점 또한 명확히 정하기 어렵다는 단점이 있습니다.

이를 보완하기 위해 일반적으로 사용하는 방법이 SPRT 기법입니다.

## 6. Minimax Algorithm

~

## 7. Alpha-Beta Prunning

~

## 8. Summary

~

## References

[1] [https://en.wikipedia.org/wiki/Ataxx](https://en.wikipedia.org/wiki/Ataxx)