---
layout: post
title: "Mukjjippa (BOJ 32469)"
date: 2025-01-31
author: parkky
tags: [problem-solving]
---

이 글은 2024 KAIST 14th ICPC Mock Competition에 출제했던 문제 중 하나인 [Mukjjippa (BOJ 32469)](https://www.acmicpc.net/problem/32469)에 대해 다룬다.

# Problem

두 플레이어 A와 B가 **묵찌빠**라는 게임을 한다.

이 게임은 여러 턴으로 이루어진다.

$i$번째 턴($1\le i\le n$)에서:
* 각 플레이어는 $\\{\mathrm R,\mathrm S,\mathrm P\\}$에서 정확히 하나를 선택한다. (각각 묵(rock), 찌(scissors), 빠(paper)를 나타낸다.)
* $X_i$와 $Y_i$를 각각 A와 B의 선택이라 하자.
* 만약 $(X_i, Y_i) \in \\{(\mathrm R,\mathrm S),(\mathrm S,\mathrm P),(\mathrm P,\mathrm R)\\}$라면, A가 $(i+1)$번째 턴의 공격자가 되고 게임은 계속된다.
* 만약 그렇지 않고 $(X_i, Y_i) \in \\{(\mathrm R,\mathrm P),(\mathrm S,\mathrm R),(\mathrm P,\mathrm S)\\}$라면, B가 $(i+1)$번째 턴의 공격자가 되고 게임은 계속된다.
* 만약 그렇지 않고 $i$번째 턴의 공격자가 있다면, 그 공격자가 승자가 되고 게임은 종료된다.
* 만약 그렇지 않다면, $(i+1)$번째 턴의 공격자는 없고 게임은 계속된다.

$1$번째 턴의 공격자는 없다.

만약 게임이 $(n+1)$번째 턴의 시작 전에 종료되지 않는다면, 승자는 없다.

각 선택의 확률 분포가 주어진다. 모든 선택은 독립이다.

A가 승리할 확률을 구하라.

# Solution

모든 $i$ ($1\le i\le n$)에 대해 $X_i$와 $Y_i$가 정의된다고 가정해도 된다.

각 $i$ ($1\le i\le n$)에 대해, 다음 사건 중 정확히 하나가 발생한다:
* $E_{i,1}$: $(X_i, Y_i) \in \\{ (\mathrm R, \mathrm R), (\mathrm S, \mathrm S), (\mathrm P, \mathrm P) \\}$
* $E_{i,2}$: $(X_i, Y_i) \in \\{ (\mathrm R, \mathrm S), (\mathrm S, \mathrm P), (\mathrm P, \mathrm R) \\}$
* $E_{i,3}$: $(X_i, Y_i) \in \\{ (\mathrm R, \mathrm P), (\mathrm S, \mathrm R), (\mathrm P, \mathrm S) \\}$

각 $i$ ($1\le i\le n$)에 대해, 선택들의 독립에 의해 각 사건의 확률은 다음과 같이 계산할 수 있다:
* $\Pr(E_{i,1}) = \frac{r_ir_i'+s_is_i'+p_ip_i'}{(r_i+s_i+p_i)(r_i'+s_i'+p_i')}$
* $\Pr(E_{i,2}) = \frac{r_is_i'+s_ip_i'+p_ir_i'}{(r_i+s_i+p_i)(r_i'+s_i'+p_i')}$
* $\Pr(E_{i,3}) = \frac{r_ip_i'+s_ir_i'+p_is_i'}{(r_i+s_i+p_i)(r_i'+s_i'+p_i')}$

각 $i$ ($1\le i\le n+1$)에 대해, 다음 사건 중 정확히 하나가 발생한다:
* $F_{i,1}$: $i$번째 턴이 시작할 때, 게임이 진행 중이고 공격자가 없다.
* $F_{i,2}$: $i$번째 턴이 시작할 때, 게임이 진행 중이고 A가 공격자이다.
* $F_{i,3}$: $i$번째 턴이 시작할 때, 게임이 진행 중이고 B가 공격자이다.
* $F_{i,4}$: $i$번째 턴이 시작할 때, 게임이 진행 중이 아니고 A가 승자이다.
* $F_{i,5}$: $i$번째 턴이 시작할 때, 게임이 진행 중이 아니고 B가 승자이다.

$F_{1,1}$이 발생한다.

각 $i$ ($1\le i\le n$)에 대해, 다음이 성립한다:
* 만약 $F_{i,1}$와 $E_{i,1}$가 발생한다면, $F_{i+1,1}$가 발생한다.
* 만약 $F_{i,1}$와 $E_{i,2}$가 발생한다면, $F_{i+1,2}$가 발생한다.
* 만약 $F_{i,1}$와 $E_{i,3}$가 발생한다면, $F_{i+1,3}$가 발생한다.
* 만약 $F_{i,2}$와 $E_{i,1}$가 발생한다면, $F_{i+1,4}$가 발생한다.
* 만약 $F_{i,2}$와 $E_{i,2}$가 발생한다면, $F_{i+1,2}$가 발생한다.
* 만약 $F_{i,2}$와 $E_{i,3}$가 발생한다면, $F_{i+1,3}$가 발생한다.
* 만약 $F_{i,3}$와 $E_{i,1}$가 발생한다면, $F_{i+1,5}$가 발생한다.
* 만약 $F_{i,3}$와 $E_{i,2}$가 발생한다면, $F_{i+1,2}$가 발생한다.
* 만약 $F_{i,3}$와 $E_{i,3}$가 발생한다면, $F_{i+1,3}$가 발생한다.
* 만약 $F_{i,4}$가 발생한다면, $F_{i+1,4}$가 발생한다.
* 만약 $F_{i,5}$가 발생한다면, $F_{i+1,5}$가 발생한다.

각 $i,j,j'$ ($1\le i\le n$, $1\le j\le3$, $1\le j'\le5$)에 대해 다음이 성립한다:
* $E_{i,j}$는 $i$번째 선택에만 의존한다.
* $F_{i,j'}$는 $1,2,\dots,(i-1)$번째 선택에만 의존한다.
* 모든 선택들은 독립이므로 $E_{i,j}$와 $F_{i,j'}$는 독립이다.

$\Pr(F_{1,1})=1$, $\Pr(F_{1,2})=\Pr(F_{1,3})=\Pr(F_{1,4})=\Pr(F_{1,5})=0$이다.

각 $i$ ($1\le i\le n$)에 대해, 각 사건의 확률은 다음과 같은 점화식으로 계산할 수 있다:
* $\Pr(F_{i+1,1}) = \Pr(F_{i,1})\cdot\Pr(E_{i,1})$
* $\Pr(F_{i+1,2}) = (\Pr(F_{i,1})+\Pr(F_{i,2})+\Pr(F_{i,3}))\cdot\Pr(E_{i,2})$
* $\Pr(F_{i+1,3}) = (\Pr(F_{i,1})+\Pr(F_{i,2})+\Pr(F_{i,3}))\cdot\Pr(E_{i,3})$
* $\Pr(F_{i+1,4}) = \Pr(F_{i,2})\cdot\Pr(E_{i,1})+\Pr(F_{i,4})$
* $\Pr(F_{i+1,5}) = \Pr(F_{i,3})\cdot\Pr(E_{i,1})+\Pr(F_{i,5})$

답은 $\Pr(F_{n+1,4})$이며, 이 값은 $\mathcal O(n)$ 시간에 계산할 수 있다.
