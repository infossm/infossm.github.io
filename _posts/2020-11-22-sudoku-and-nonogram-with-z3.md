---
layout: post
title:  "Sudoku and Nonogram with z3"
date:   2020-11-22 12:40
author: RBTree
tags: [smt]
---

# 서론

이전에 z3와 Boolector를 비교하는 글을 올린 적이 있습니다. ([Link](http://www.secmem.org/blog/2020/06/19/SMT-Solver-in-CTF/))

그런데 정작 z3에 대해서 글이 잘 없어서, 이번에 z3로 퍼즐을 푸는 글을 작성해보게 되었습니다. 스도쿠는 이미 충분히 유명하고, 노노그램은 네모네모 로직이나 피크로스 등을 통해서 많은 분들이 알고 계신 퍼즐이기 때문에 고르게 되었습니다.

이번 글에서는 스도쿠랑 노노그램의 솔버를 하나씩 설명해가면서 작성해보려고 합니다.

# 본론

## Sudoku

스도쿠는 9x9개의 정수를 맞추는 퍼즐 게임입니다. 정수는 모두 1부터 9로 구성되며, 9x9개의 정수 중 몇 개의 값이 고정되어 있습니다. Wikipedia에 있는 sudoku 보드([Link](https://en.wikipedia.org/wiki/Sudoku))를 바탕으로 이를 표현해봅시다.

```python
from z3 import *

board = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

Ans = [
    [Int("Ans_{}{}".format(i, j)) for j in range(9)]
for i in range(9) ]

solver = Solver()

for i in range(9):
    for j in range(9):
        if board[i][j]:
            solver.add(Ans[i][j] == board[i][j])
        solver.add(And(1 <= Ans[i][j], Ans[i][j] <= 9))

```

이제 각 행, 열, 3x3블록 내에 서로 다른 값들이 있어야 한다는 것을 알고 있습니다. 이것은 `Distinct`라는 연산자를 통해서 구현하는 것이 가능합니다.

```python
for i in range(9):
    # Row
    solver.add( Distinct([Ans[i][j] for j in range(9)]) )
    # Column
    solver.add( Distinct([Ans[j][i] for j in range(9)]) )
    # Block
    solver.add( Distinct([Ans[3 * (i // 3) + j1][3 * (i % 3) + j2] for j1 in range(3) for j2 in range(3)]) )
```

마지막으로, 답이 있는지 `solver.check()`를 통해 확인합니다. 있다면, `solver.model()`을 통해 model을 불러내고 model로부터 값을 읽어올 수 있습니다.

```python
if solver.check() == sat:
    model = solver.model()
    for i in range(9):
        s = ""
        for j in range(9):
            s += "{} ".format(model[Ans[i][j]].as_long())
        print(s)

else:
    print("NO")
```

실행하면 위키피디아의 예시 그림과 같이 답을 구하는 것을 볼 수 있습니다.

```shell
python3 sudoku.py
5 3 4 6 7 8 9 1 2
6 7 2 1 9 5 3 4 8
1 9 8 3 4 2 5 6 7
8 5 9 7 6 1 4 2 3
4 2 6 8 5 3 7 9 1
7 1 3 9 2 4 8 5 6
9 6 1 5 3 7 2 8 4
2 8 7 4 1 9 6 3 5
3 4 5 2 8 6 1 7 9
```

## Nonogram

노노그램은 좀 더 어렵습니다. 각 행과 각 열에 대해서 조건이 달려있고, n x m 보드를 채워야 하는데 이를 표현하는 것이 스도쿠처럼 직관적이지 않기 때문입니다.

이번에도 예시는 Wikipedia에서 갖고 와보도록 하겠습니다. ([Link](https://en.wikipedia.org/wiki/Nonogram))

```python
from z3 import *

row_cond = [
    [8, 7, 5, 7], [5, 4, 3, 3], [3, 3, 2, 3], [4, 3, 2, 2], [3, 3, 2, 2],
    [3, 4, 2, 2], [4, 5, 2], [3, 5, 1], [4, 3, 2], [3, 4, 2], [4, 4, 2],
    [3, 6, 2], [3, 2, 3, 1], [4, 3, 4, 2], [3, 2, 3, 2], [6, 5], [4, 5],
    [3, 3], [3, 3], [1, 1]
]

col_cond = [
    [1], [1], [2], [4], [7], [9], [2, 8], [1, 8], [8], [1, 9], [2, 7],
    [3, 4], [6, 4], [8, 5], [1, 11], [1, 7], [8], [1, 4, 8], [6, 8],
    [4, 7], [2, 4], [1, 4], [5], [1, 4], [1, 5], [7], [5], [3], [1], [1]
]

height, width = len(row_cond), len(col_cond)
```

이번에는 각 condition마다 2개의 변수를 지정해 줄 것입니다. 하나는 시작하는 칸의 index, 하나는 끝나는 칸의 index를 의미합니다. 만약 col_cond의 첫 번째 `[1]`에 대해 나타내자면 `Cstart_1_1`, `Cend_1_1`라는 두 변수가 생성되고, 총 1칸만 칠하니 `Cend_1_1 = Cstart_1_1` 이라는 관계가 성립하겠죠. (만약 n이었다면 `Cend_1_1 = Cstart_1_1 + n - 1`)

또한 이런 방법으로 나타내게 되면, `[2, 8]` 과 같은 조건이 나타났을 때 2칸을 연속하게 칠한 후 한 칸 이상을 띄고 8칸을 연속하게 칠해야 하므로 `Cend_i_j < Cstart_i_j+1` 라는 조건을 달아서 연속한 칸이 서로 떨어져있어야 한다는 단서를 달 수 있습니다. 이를 바탕으로 `row_cond`를 옮겨봅시다.

```python
X = [ [ Int("X_{}_{}".format(i, j)) for j in range(width) ] for i in range(height)]
R = [ [ ( Int("Rstart_{}_{}".format(i, j)), Int("Rend_{}_{}".format(i, j)) ) for j in range(len(row_cond[i])) ] for i in range(height) ]
C = [ [ ( Int("Cstart_{}_{}".format(i, j)), Int("Cend_{}_{}".format(i, j)) ) for j in range(len(col_cond[i])) ] for i in range(width) ]
solver = Solver()

for i in range(height):
    for j in range(len(row_cond[i])):
        solver.add( And(0 <= R[i][j][0], R[i][j][0] < width) )
        solver.add( And(0 <= R[i][j][1], R[i][j][1] < width) )
        solver.add( R[i][j][1] == R[i][j][0] + row_cond[i][j] - 1)
    
    for j in range(len(row_cond[i]) - 1):
        solver.add( R[i][j][1] + 1 < R[i][j + 1][0] )
```

그리고 각 `(i, j)`번째 칸 `X[i][j]`는, 어떤 `k`에 대해 `Rstart_i_k`와 `Rend_i_k+1` 사이에 있으면 1, 아니면 0이라는 식으로 조건을 달아 칠해줍시다.ㄴ,ㄴ

```python
for i in range(height):
    # ...
    for j in range(width):
        zero_cond = True
        for k in range(len(row_cond[i])):
            solver.add( If( And(R[i][k][0] <= j, j <= R[i][k][1]), X[i][j] == 1, True) )
            zero_cond = And(zero_cond, Not(And(R[i][k][0] <= j, j <= R[i][k][1])))
        solver.add( If(zero_cond, X[i][j] == 0, True) )
```

마찬가지로 `col_cond`도 옮겨봅시다.

```python
for i in range(width):
    for j in range(len(col_cond[i])):
        solver.add( And(0 <= C[i][j][0], C[i][j][0] < height) )
        solver.add( And(0 <= C[i][j][1], C[i][j][1] < height) )
        solver.add( C[i][j][1] == C[i][j][0] + col_cond[i][j] - 1)
    
    for j in range(len(col_cond[i]) - 1):
        solver.add( C[i][j][1] + 1 < C[i][j + 1][0] )
    
    for j in range(height):
        zero_cond = True
        for k in range(len(col_cond[i])):
            solver.add( If( And(C[i][k][0] <= j, j <= C[i][k][1]), X[j][i] == 1, True) )
            zero_cond = And(zero_cond, Not(And(C[i][k][0] <= j, j <= C[i][k][1])))
        solver.add( If(zero_cond, X[j][i] == 0, True) )
```

마지막으로 이를 출력해봅시다.

```python
if solver.check() == sat:
    model = solver.model()
    for i in range(height):
        s = ""
        for j in range(width):
            if model[X[i][j]].as_long():
                s += "! "
            else:
                s += "  "
        print(s)
else:
    print("NO")
```

```shell
python3 nonogram.py
! ! ! ! ! ! ! !   ! ! ! ! ! ! !   ! ! ! ! !   ! ! ! ! ! ! !
    ! ! ! ! !       ! ! ! !         ! ! !         ! ! !
      ! ! !           ! ! !         ! !           ! ! !
      ! ! ! !           ! ! !       ! !           ! !
        ! ! !           ! ! !     ! !             ! !
        ! ! !           ! ! ! !   ! !           ! !
        ! ! ! !           ! ! ! ! !             ! !
          ! ! !           ! ! ! ! !             !
          ! ! ! !           ! ! !             ! !
            ! ! !           ! ! ! !           ! !
            ! ! ! !         ! ! ! !         ! !
              ! ! !       ! ! ! ! ! !       ! !
              ! ! !       ! !   ! ! !       !
              ! ! ! !   ! ! !   ! ! ! !   ! !
                ! ! !   ! !       ! ! !   ! !
                ! ! ! ! ! !       ! ! ! ! !
                  ! ! ! !         ! ! ! ! !
                  ! ! !             ! ! !
                  ! ! !             ! ! !
                    !                 !
```

잘 출력되는 것을 확인해볼 수 있습니다.

# 결론

z3로 작성한 Sudoku solver에 대해서는 인터넷에서 찾아봐도 쉽게 나옵니다. 하지만 nonogram solver는 쉽게 찾아보기 어렵습니다.

이번 기회에 z3에 대해서 익숙해지고, 퍼즐을 좋아한다면 해당 퍼즐을 z3로 옮길 방법을 궁리해보는 것도 좋은 경험이 될 수 있을 것이라고 생각합니다.