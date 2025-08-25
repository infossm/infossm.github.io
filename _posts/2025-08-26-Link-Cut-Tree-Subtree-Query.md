---
layout: post
title: Link Cut Tree Subtree Query
date: 2025-08-26 04:30
author: jthis
tags:
  - algorithm
  - data-structure
---

안녕하세요, **jthis**입니다.

이 글에서는 **Link-Cut Tree**에서 **Subtree Query**를 효율적으로 수행하는 방법에 대해 소개하겠습니다. 편의상 코드는 **Splay Tree**를 기준으로 설명하겠습니다.

# Link Cut Tree

Link-Cut Tree는 **간선 추가 삭제**와 **경로 쿼리(Path Query)** 를 지원하는 자료구조입니다. 시간복잡도는 *amortized* $O(\log N)$이며, **amortized를 제거하여** $O(\log N)$으로도 가능합니다. 또한, 이 자료구조는 **Subtree 연산**도 지원할 수 있습니다. 따라서 Top Tree로 해결할 수 있는 대부분의 쿼리 문제를 거의 모두 **Link-Cut Tree**로 대체할 수 있습니다.

Link-Cut Tree는 Splay Tree로만 구현할 수 있다고 생각할 수도 있지만, 다른 BBST(Balanced Binary Search Tree)로도 충분히 구현할 수 있습니다. BBST에서 **split**과 **merge** 연산만 빠르게 가능하다면, 어떤 BBST라도 Link-Cut Tree를 구성할 수 있습니다.

Link-Cut Tree는 트리를 **경로(Path)** 단위로 여러 개로 쪼개어 관리하는 자료구조입니다. 각 경로는 BBST로 관리하며, 경로에 포함되지 않은 간선은 **가상 간선**으로 생각하여 BBST와는 별도로 관리합니다.
Link-Cut Tree에서는 **access 연산**과 **makeRoot 연산**만 구현하면 됩니다.

**access 연산**은 트리 상의 루트와 특정 노드 $x$를 하나의 경로로 만드는 연산입니다. 이 연산을 통해 루트와 $x$ 사이에 있는 가상 간선을 실제 간선으로 바꾸고, 그 간선에서 부모 쪽과 연결되어 있던 자식과의 연결을 가상 간선으로 바꿉니다.

노드 $a$에서 access 연산을 수행한 뒤, 노드 $b$에서 access 연산을 수행했을 때, **마지막으로 바뀐 가상 간선, 즉 루트와 가장 가까운 가상 간선의 부모 쪽이 $a$와 $b$의 LCA**가 됩니다.

또한, access $x$ 연산을 수행한 뒤 해당 경로에 **reverse 연산**을 적용하면, 노드 $x$가 트리의 루트가 됩니다.

마지막으로, 간선을 추가하는 **Link** 연산을 위해서는 **자식 쪽을 트리에서 루트로 만들어야 합니다**.

```c++
node *access(node *x) {  
    splay(x);
    x->r = nullptr;
    node *res = x;
    while (x->p) {  
        node *p = x->p;  
        res = p;  
        splay(p);  
        p->r = x;  
        splay(x);  
    }  
    return res;  
}

void makeRoot(node *x) {  
    access(x);  
    splay_reverse(x);  
}
```

Link-Cut Tree는 간단히 말해 **BBST 사이의 가상 간선을 효율적으로 관리하는 자료구조**입니다.

# Link Cut Tree Path Query

두 노드 $a$와 $b$에서 **Path Query**를 수행하기 위해서는 *makeRoot* 연산 없이도 처리할 수 있지만, **makeRoot 연산을 사용하면 훨씬 간단하게 해결할 수 있습니다**.
노드 $a$에서 makeRoot 연산을 수행한 뒤, 노드 $b$에서 access 연산을 실행하면, $a$-$b$ 경로만 BBST에 남게 되어 문제를 간단하게 해결할 수 있습니다.

```c++
long long pathQuery(node *a, node *b) {  
    makeRoot(a);  
    access(b);  
    return b->pathSum;  
}
```
# Link Cut Tree Subtree Query

**Subtree 연산**을 수행하기 위해서는 **가상 간선**을 적절히 관리하면 됩니다. 가상 간선은 **access 연산**에서만 변경됩니다.
Subtree에서의 **min**, **max** 연산은 가상 간선을 BBST로 관리하면 해결할 수 있으며, **sum** 연산이 필요하다면 구조체에 `int` 하나를 추가하면 됩니다.
또한, access 연산에서는 **min**, **max** 값은 삽입과 삭제(insert, delete)만 수행하면 되고, **sum**은 차이만 계산하여 누적하면 됩니다.

```c++
void update(node *x) {  
    x->mx = x->val;  
    if (!x->vs.empty())x->mx = max(x->mx, *prev(x->vs.end()));  
    if (x->l)x->mx = max(x->mx, x->l->mx);  
    if (x->r)x->mx = max(x->mx, x->r->mx);  
}
node *access(node *x) {
    splay(x);  
    if (x->r)x->vs.insert(x->r->mx);  
    x->r = nullptr;  
    node *res = x;  
    while (x->p) {  
        node *p = x->p;  
        res = p;  
        splay(p);  
        if (p->r)p->vs.insert(p->r->mx);  
        p->vs.erase(p->vs.find(x->mx));  
        p->r = x;  
        splay(x);  
    }  
    return res;  
}
```

**Subtree에서 최대값 연산**의 구현입니다. 구현의 편의를 위해 `multiset`을 사용했습니다.
# Link Cut Tree Subtree Update
### Subtree add
```c++
void rotate(node *x) {  
    node *p = x->p;  
    if (x == p->l) {  
        p->l = x->r;  
        if (p->l) {  
            p->l->p = p;  
            p->l->gets = p->added;  
        }  
        x->r = p;  
    } else {  
        p->r = x->l;  
        if (p->r) {  
            p->r->p = p;  
            p->r->gets = p->added;  
        }  
        x->l = p;  
    }  
    x->p = p->p;  
    if (x->p)x->gets = x->p->added;  
    p->p = x;  
    p->gets = x->added;  
    if (x->p) {  
        if (p == x->p->l)x->p->l = x;  
        else if (p == x->p->r)x->p->r = x;  
    }  
}  

void adding(node *x, int diff) {  
    x->sum += diff * x->sz;  
    x->now += diff;  
    x->added += diff;  
    x->vsum += diff * x->vsz;  
}  
  
void lazy_down(node *x) {  
    if (x->p) {  
        adding(x, x->p->added - x->gets);  
        x->gets = x->p->added;  
    }  
}
```

`added`와 `gets`라는 변수를 유지합니다.

* **added**: 해당 Subtree에 더해야 할 값의 누적 합
* **gets**: 지금까지 부모로부터 받은 누적 합

`rotate` 함수는 부모와의 연결을 수정하기 때문에, 이 함수도 함께 수정해야 합니다. 또한 **Subtree 크기(size)** 도 유지해야 합니다.

---

### Subtree 변경 (Subtree Change)

`add` 연산과 유사하게 구현할 수 있습니다. 다만 값 자체를 저장하는 대신 **쿼리 인덱스**를 저장하여, 부모와 내가 다를 경우에만 업데이트하는 방식으로 처리합니다.


# 예시 문제

### **1. [백준 14268 - 회사 문화 2](https://www.acmicpc.net/problem/14268)**

* **요구사항**: Subtree에 대한 `add` 업데이트와 특정 노드의 값 출력
* **예시 코드**: [소스 링크](http://boj.kr/ef9fdd7e02954fedbd7c66c022d9c641)

---

### **2. [백준 13515 - 트리와 쿼리 6](https://www.acmicpc.net/problem/13515)**

* **요구사항**: 색깔(흰색/검정색)에 따라 연결 관리
* **아이디어**:
  * **포레스트 2개**를 만든다 (0번 트리, 1번 트리)
  * 정점 `x`가 흰색이면 `x`와 `x`의 부모는 **트리 0번**에서만 연결
  * 정점 `x`가 검정색이면 `x`와 `x`의 부모는 **트리 1번**에서만 연결
  * **쿼리 1**:
	  * 간선 추가 및 간선 삭제
  * **쿼리 2**:
    * u의 색깔에 맞는 트리에서 탐색
    * 두 가지 경우
      1. **트리 전체가 같은 색** → 제거 없이 `subtree size`
      2. **트리 중 루트만 색 다름** → 루트와의 간선 제거 후 `subtree size`
* **예시 코드**: [소스 링크](https://www.acmicpc.net/source/share/a5dff557dde94d048b88e7e590bcce95)

---

### **3. [백준 13516 - 트리와 쿼리 7](https://www.acmicpc.net/problem/13516)**

* **요구사항**: Subtree에서 최대값 구하기
* **아이디어**:
  * 위 문제(13515)와 거의 동일
  * 차이점 → `subtree size` 대신 `subtree max` 계산
* **예시 코드**: [소스 링크](http://boj.kr/1576e635eac9445c9a6f6bed8f00647e)

---

### **4. [백준 18805 - Tree and Easy Queries](https://www.acmicpc.net/problem/18805)**

* **요구사항**: 지름(Diameter) 구하기
* **아이디어**:
  * 동적 트리에서 지름 유지
* **예시 코드**: [소스 링크](http://boj.kr/4a7037d27f994aafb2599fa91b494097)

---

### **5. [백준 10014 - Traveling Saga Problem](https://www.acmicpc.net/problem/10014)**

* **요구사항**: 지름(Diameter) 구하기
* **아이디어**:
  * 위 문제(18805)와 동일한 방식
* **예시 코드**: [소스 링크](http://boj.kr/6df8dbc3230c472ba3f733c621545e1d)

---

# Dynamic Tree 비교

| 구분   | Fragmented Tree | Euler Tour Tree | Link-Cut Tree | Top Tree    |
|---------------|-----------------|-----------------|--------------|-----------|
| Path Query    | 가능            | 제한적          | 가능          | 가능        |
| Subtree Query | 가능            | 가능            | 가능          | 가능        |
| 시간 복잡도    | O(√N)          | O(log N)       | O(log N)     | O(log N)   |
| 구현 난이도    | 중간            | 비교적 쉬움      | 중간          | 어려움       |

# 응용

Link-Cut Tree는 **경로 쿼리(Path Query)** 와 **Subtree 쿼리**를 사용하여 다양하게 응용할 수 있습니다.
* **트리 지름(Diameter)**
* **Rerooting DP**
* **Tree DP**
* **Dynamic Centroid**
* **Online LCA**
* **Subtree Size 계산**
* 기타 동적 트리 관련 쿼리

---

# 실전 문제

* [백준 17936 - 트리와 쿼리 13](https://www.acmicpc.net/problem/17936)
* [백준 31705 - Kolorowy las](https://www.acmicpc.net/problem/31705)
* [백준 1921 - 트리와 쿼리 20](https://www.acmicpc.net/problem/1921)
* [백준 26408 - Game](https://www.acmicpc.net/problem/26408)
* [Luogu P5610](https://www.luogu.com.cn/problem/P5610)
* [Link-Cut Tree 문제집](https://www.acmicpc.net/workbook/view/7004)
