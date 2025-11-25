---
layout: post
title: "Advanced Game Search Algorithms (2)"
date: 2025-11-24
author: jinhan814
tags: [algorithm, game-theory, problem-solving]
---

## 1. Introduction

지난 [Advanced Game Search Algorithms (1)](https://infossm.github.io/blog/2025/10/25/adv-game-search/) 글에서는 Random Agent, Greedy Agent와 

이번 글에서는 게임 에이전트의 가장 단순한 형태인 Random Agent부터 시작하여 Greedy, Minimax, Alpha-Beta Pruning의 핵심 원리를 다룹니다. 이후 이어지는 글에서는 Minimax 기반 에이전트의 추가적인 search pruning 기법을 알아보고, MCTS 등의 현대적인 탐색 기법과 NNUE와 같은 neural network를 이용한 평가 방법 등을 살펴보겠습니다.

또한 에이전트의 성능을 객관적으로 평가하기 위해 SPRT(Sequential Probability Ratio Test)라는 평가 기법을 소개합니다. 이를 이용하면 통계적으로 두 에이전트 간의 실력 차이를 엄밀하게 검증할 수 있습니다.

이번 글에서 소개하는 방법론은 $2$인, 제로섬, 턴제, 완전정보, 결정론적 전이를 만족하는 게임에 적용이 가능하며, 구체적인 설명을 위해서 ATAXX를 예시로 각 알고리즘을 구현해보겠습니다.

## 2. ATAXX

~

## References

[1] [https://en.wikipedia.org/wiki/Ataxx](https://en.wikipedia.org/wiki/Ataxx)

[2] [https://en.wikipedia.org/wiki/Sequential_probability_ratio_test](https://en.wikipedia.org/wiki/Sequential_probability_ratio_test)

[3] [https://mattlapa.com/sprt/](https://mattlapa.com/sprt/)