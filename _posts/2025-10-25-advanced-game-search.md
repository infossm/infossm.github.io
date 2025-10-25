---
layout: post
title: "Advanced Game Search Algorithms (1)"
date: 2025-10-25
author: jinhan814
tags: [algorithm, game-theory, problem-solving]
---

## 1. Introduction

이 글에서는 게임 에이전트의 가장 단순한 형태인 Random Agent부터 시작하여 Greedy, Minimax, Alpha-Beta Pruning의 핵심 원리를 다룹니다. 이후 이어지는 글에서는 MCTS 등의 현대적인 탐색 기법을 알아보고, NNUE 등의 neural network를 이용한 평가 방법과 여러 search prunning 방법을 살펴보겠습니다.

또한 에이전트의 성능을 객관적으로 평가하기 위해 SPRT(Sequential Probability Ratio Test)라는 평가 기법을 소개합니다. 이를 이용하면 통계적으로 두 에이전트 간의 실력 차이를 엄밀하게 검증할 수 있습니다.

이번 글에서 소개하는 방법론은 2인, 제로섬, 턴제, 완전정보, 결정론적 전이를 만족하는 게임에 적용이 가능하며, 구체적인 설명을 위해서 ATAXX를 예시로 각 알고리즘을 구현해보겠습니다.

## 2. ATAXX

ATAXX는 $7 \times 7$ 보드에서 진행되는 $2$인 턴제 게임입니다.

![Fig.1](/assets/images/2025-10-25-advanced-game-search/fig1_v4.png)

## 3. Random Agent

~

## 4. Greedy Agent

~

## 5. SPRT(Sequential Probability Ratio Test)

~

## 6. Minimax Algorithm

~

## 7. Alpha-Beta Prunning

~

## 8. Summary

~

## References

[1] [~](~)