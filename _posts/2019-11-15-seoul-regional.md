---
layout: post
title:  "2019 ACM-ICPC Seoul Regional 풀이"
author: ho94949
date: 2019-11-14 15:00
tags: [ACMICPC, Regional]
---

# 서론

2019년 2019년 11월 9일 토요일에 ACM-ICPC 서울 리저널이 진행 되었다. 대회에 대한 정보는 [http://icpckorea.org](http://icpckorea.org) 에서 찾아볼 수 있다. (학교 기준으로) 1등은 모든 문제를 해결한 서울대학교의 Cafe Mountain, 2등은 9문제를 패널티 1004분으로 해결한 연세대학교의 Inseop is Korea top, 3등은 9문제를 패널티 1464분으로 해결한 KAIST의 CMD이다.

올해에는 12문제가 출제 되었고, 이 문제들에 대한 풀이를 작성해보려고 한다.

# A - Fire on Field

## 문제

$$A[0] = 1, A[1] = 1$$ 이고, 2 이상의 $$i$$에 대해, $$A[i]$$ 를 모든 가능한 $$k > 0, i - 2k \ge 0$$ 을 만족하는 $$k$$에 대해서, $$A[i], A[i-k], A[i-2k]$$가 등차수열이 되지 않도록 하는 (즉, $$A[i]-A[i-k] \neq A[i-k] - A[i-2k]$$) 최소한의 값이라고 하자.

이 때 주어진 $$n$$에 대해서 $$A[n]$$을 구하여라. ($$0 \le n \le 1000$$)

## 풀이

문제에서 주어진 대로 구현을 하는 것이 기본 아이디어이다. 위의 항을 이항 하면, $$A[i] \neq 2A[i-k] - A[i-2k]$$ 가 된다. 각각 $$i$$에 대해서, 불가능한 $$A[i]$$에 값을 모두 구해 놓고, 1부터 올려가면서 가능한 값이 있는지 확인하면 된다. 이는 배열 접근 혹은 `std::set` 등을 이용하면 매우 쉽게 할 수 있다. 시간복잡도는 $$O(n^2)$$ 혹은 $$O(n^2 \log n)$$ 정도이다.



# B - Gene Tree

TBD

# C - Islands

TBD

# D - Ladder Game

TBD

# E - Network Vulnerability

TBD

# F - Quadrilaterals

TBD

# G - Same Color

TBD

# H - Strike Zone

TBD

# I - Thread Knots

TBD

# J - Triangulation

TBD

# K - Wahser

TBD

# L - What's Mine is Mine