---
layout: post
title:  "LiChao Tree"
author: 김진표
date: 2018-10-01 15:00
tags: []
---

LiChao Tree 는 직선이 실시간으로 추가되는 Convex hull trick 문제를 해결하기 위한 자료구조입니다. 
구현이 비교적 간단하면서 유용한 자료구조인데, 한글로 설명된 자료가 없어 포스트를 작성하게 되었습니다.

Convex hull trick, Segment tree에 대한 내용은 따로 설명하지 않고 진행하겠습니다.

# 문제 소개

다음과 같은 두 쿼리를 처리해야 하는 문제를 생각합시다.

(1) 직선 $$ y=ax+b $$ 를 집합에 추가

(2) 집합에 존재하는 직선들 중, 주어진 $$ x=x_i $$ 위치에서의 최댓값을 출력

# 자료구조 소개

LiChao Tree는 Segment tree의 일종이다.

## (1) 간선 추가 쿼리

## (2) $$x$$ 에서 최댓값 쿼리

Test
$$ f(x) = 1 $$ 