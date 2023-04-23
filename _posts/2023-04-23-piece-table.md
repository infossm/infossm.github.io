---
layout: post
title: "Piece Table"
author: buttercrab
date: 2023-04-23
tags: [rope, string, algorithm, data-structure]
---

## 서론

텍스트 에디터는 하나의 문자열을 조작하는 것을 매우 빠르게 하는 자료구조가 필요합니다.
길이가 $n$인 문자열 $s$에 대해서 다음과 같은 기능들이 있어야 합니다:

1. $s = s[0..i] + s[j..n], 0 \le i < j \le n$, 즉, $i$ ~ $j$까지의 문자들을 지운다
2. $s = s[0..i] + t + s[i..n]$, 즉, $i$ 위치에 문자열 t를 넣는다
3. $s[i..j]$, 즉, $i$부터 $j$까지의 문자열을 빠르게 구한다

텍스트 에디터에는 문자열 찾기, 줄 번호 등 여러 기능이 있어야 하지만 우선 가장 중요한 기능은 위 세가지라고 할 수 있습니다.

그러면 이 블로그에서 언급된 적 있는 rope는 위 기능을 완벽하게 해냅니다.
각 쿼리에서 바뀌거나 보여지는 문자열의 길이를 $m$이라고 하면 $O(m \log n)$ 만에 모든 쿼리를 수행합니다.
그런데, 실제로 rope가 많이 쓰일까요?

[Visual Studio Code 블로그](https://code.visualstudio.com/blogs/2018/03/23/text-buffer-reimplementation)에 따르면 Visual Studio Code는 piece table을 사용합니다.
그럼 piece table이란 무엇일까요?

## Piece Table이란?

Rope의 단점은 무엇이 있을까요?
그것은 바로 문자 각각 메모리에 다른 위치에 저장된다는 뜻입니다.
시간복잡도 상으로는 좋을 지 몰라도 실제로 사용할 때는 느릴 수 있습니다.
보통 우리가 텍스트 에디터를 사용할 때에는 뭉텅이로 작성하거나 지우는 경우가 많습니다.
또한 우리가 보는 텍스트 에디터는 연속된 문자열을 보여줍니다.

즉, 연속된 문자열을 다루지 않는 rope은 텍스트 에디터에 적합하지 않다는 것입니다.
그래서 rope을 개선한 것이 piece table입니다.

Piece Table은 간단합니다.
현재 문자열을 조각으로 저장합니다.
조각은 연속된 문자열입니다.
그리고 조각들을 배열로 저장합니다.
우리가 문자열을 조작하면 조각들을 조정하면 됩니다.

다음과 같은 코드로 작성할 수 있습니다:

```cpp
struct Piece {
    int start, len;
};

struct PieceTable {
    string str;
    vector<Piece> pieces;
};
```

Piece Table에는 문자열 `str`와 조각들의 배열 `pieces`가 있습니다.
조각은 문자열 `str`의 시작 위치 `start`와 길이 `len`로 구성됩니다.
조각들은 `str`의 연속된 부분 문자열입니다.

문자열 `str`의 중간에 넣는 것이 아니라 append하는 방식으로 문자열을 조작합니다.
다음과 같은 예를 봅시다:

원래 문자열 `str`은 `abcd`라고 합시다.

`s` = "abcd"
`t` = "pp"

이제 `s`에 `t`를 넣어봅시다:

`s` = "abppcd"

그럼 `str`은 어떻게 될까요?

- 원래 `str`은 `abcd`였습니다. 그리고 piece table은 다음과 같습니다:
  | `piece` | `content` |
  | ------- | --------- |
  | [0, 4] | "abcd" |

- `t`를 넣었습니다. `str`은 `abcdpp`가 됩니다. 그리고 piece table은 다음과 같습니다:
    | `piece` | `content` |
    | ------- | --------- |
    | [0, 2] | "ab" |
    | [4, 2] | "pp" |
    | [2, 2] | "cd" |

즉, `str`의 중간에 넣는 것이 아니라 append하는 방식으로 문자열을 조작합니다.
그리고 조각을 배열해서 연속된 문자열을 보여줍니다.

## Piece Table의 성능

이렇게 문자열을 조각으로 저장하면 성능은 어떨까요?
연속된 문자열을 조작하기 때문에 캐시 효율이 좋아져서 상당히 빠릅니다.
또한 연속된 문자열을 다루어서 메모리 효율도 높아지게 됩니다.
하지만 시간복잡도 상으로는 rope보다 느립니다.
Piece의 수를 $k$라고 하면, 문자열을 조각으로 저장하면 $O(k)$만에 문자열을 조작할 수 있습니다.
그래서 수정이 적을 때는 빠르지만, 수정이 많을 때는 rope보다 느립니다.

그러면 이를 보완하기 위해서 어떻게 해야 할까요?

## Piece Table의 개선

Piece Table의 개선은 간단합니다.
Piece Table의 조각들을 rope으로 저장하면 됩니다.
그러면 rope의 성능을 그대로 가져올 수 있습니다.

Rope는 범용적으로 생각하면 dynamic array라고 생각할 수 있습니다.
원하는 인덱스에 element를 넣거나 지울 수 있습니다.
원래의 배열은 선형의 시간복잡도를 가지지만, rope은 로그의 시간복잡도를 가집니다.

그러면 우리는 이걸 이용해서 Piece Table을 개선할 수 있습니다.
각각의 piece를 rope으로 저장하면 됩니다.
그러면 piece table의 장점과 rope의 장점을 모두 가질 수 있습니다.

다음과 같이 구현을 할 수 있습니다:

- [Swift Piece Table](https://github.com/buttercrab/swift-piece-table)
- [Dart Piece Table](https://github.com/buttercrab/dart-piece-table)

README를 보면 벤치마크 결과가 실제로 빠르게 나오는 것을 볼 수 있습니다.

## 마치며

Piece Table을 사용한 Visual Studio Code 블로그에도 Piece Table을 사용하고 이를 트리를 이용해 개선하였다고 합니다.
Visual Studio Code에서 사용할만큼 뛰어난 자료구조라는 것을 알 수 있습니다.