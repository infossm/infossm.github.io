---
layout: post
title:  "How to build WebAssembly app with Rust"
date:   2020-02-19 22:00
author: RBTree
tags: [rust, web, webassembly, wasm]
---

# 서론

작년에 마작에서 현재 들고 있는 패의 점수 기대치를 계산하는 웹 사이트를 제작을 하려고 하고 있었는데, 다음과 같은 문제에 부딪혔습니다.

1. 가능하면 클라이언트 사이드에서 패의 정보를 입력하고, 해당 패의 점수 기대치를 클라이언트에서 계산하게 하고 싶다. 하지만 웹에서 클라이언트 사이드에서 계산하는 선택지는 거의 JavaScript 뿐이다.
2. JavaScript를 사용할 줄은 알지만, 굳이 개발하면서 JS를 쓰고 싶지는 않다. 대안으로 TypeScript가 있지만, 그리고 JS보다 훨씬 낫지만, 역시 마음에 들지는 않는다.

그래서 더 생각해본 결과, 클라이언트 사이드에서 계산하는 것을 포기하고 Python + Django를 통해 간단하게 개발을 시작하게 되었습니다. 하지만 Python으로 코드를 작성할수록 코드가 더욱 마음에 들지 않기 시작했고, 결국 방치된 상태로 그대로 남게 되었습니다.

올해 들어서 이 문제를 다시 곰곰히 생각을 해봤습니다. 1월에 뭘 했나 돌아보니, [Rust](https://www.rust-lang.org/)를 책으로 공부했습니다. (공부할 때 쓴 책은 [러스트 프로그래밍 공식 가이드](https://jpub.tistory.com/980)입니다. 관심 있는 분은 확인해보세요.) 그리고 Rust는 WebAssembly를 지원하는 언어로 유명하다는 것도 기억해냈죠.

WebAssembly(줄여서 Wasm)는 스택 기반의 VM 및 그 언어를 지칭하는 것으로, 웹에서 클라이언트 사이드 앱을 만드는 데에도 사용할 수 있습니다. 현재 C와 C++, 그리고 Rust를 통해서 작성이 가능한 상태이죠.

그래서 이번에는 Python을 버리고 Rust를 통해 위의 마작 계산기를 Wasm 앱의 형태로 만들려고 하고 있습니다. 하지만 계산기를 완벽하게 만들기에는 시간이 무척 없었기에, 목표를 간추려 이 글에서는 N-Queen 문제를 푸는 알고리즘을 Rust를 통해 간단하게 작성한 뒤 Wasm으로 Import 해보고, 더 나아가 JS로 작성한 코드와 속도를 비교해보고자 합니다.

# 본론

## 들어가기에 앞서

Rust의 설치 방법에 대해서는 이 글에서 설명하지 않습니다. 또한 Rust 문법에 대해서도 자세히 설명하지는 않습니다. 대신 코드를 작성하면서 왜 이런 식으로 코드를 작성했는지는 간략히 설명하도록 하겠습니다.

## 설치

Rust가 설치되어있다면, 다음 두 커맨드를 실행할 수 있을 것입니다.

- rustc 
- cargo

Wasm 앱을 작성하기 위해서는 wasm-pack을 설치해야 합니다. 또한 wasm-pack을 사용하기 위해서는 템플릿 프로젝트를 내려받아야 하는데, 이를 위해서는 cargo-generate를 설치해야 합니다.

- wasm-pack([설치 링크](https://rustwasm.github.io/wasm-pack/installer/))
- cargo-generate(`cargo install cargo-generate`로 설치)

Wasm 앱은 JavaScript wrapper가 필요하고, 이를 작성하기 위해서는 npm에 올라가 있는 관련 툴이 필요하므로 다음을 설치해야합니다.

- npm([설치 링크](https://www.npmjs.com/get-npm))

## 프로젝트 세팅

###러스트 프로젝트 세팅하기

우선 다음 커맨드를 통해 wasm-pack template를 받아옵니다.

```
cargo generate --git https://github.com/rustwasm/wasm-pack-template
```

실행하면 cargo-generate에서 프로젝트 이름을 무엇으로 할지 물어보는데, 저는 간단히 wasm-bench로 지었습니다. 프로젝트를 열어보면, 다른 Rust 프로젝트와 큰 차이 없이 `src/lib.rs`를 찾아볼 수 있습니다. 하지만 그 안의 코드는 큰 차이가 있죠.

```rust
mod utils;

use wasm_bindgen::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
extern {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet() {
    alert("Hello, wasm-bench!");
}
```

`wasm_bindgen`이라는 attribute가 눈에 띄는데, JavaScript wrapper와 통신하기 위한 부분입니다. 위와 같이 작성하게 되면, JS의 `window.alert`가 Rust의 `alert`로 불러와져 호출할 수 있게 되며, wrapper에서는 Rust의 `greet` 함수를 호출할 수 있게 되어 이를 호출하면 `window.alert("Hello, wasm-bench!")`와 동일한 행동을 하게 됩니다.

### 빌드해보기

빌드는 `wasm-pack`을 통해 할 수 있습니다. 현재 프로젝트 폴더에서 빌드해봅시다.

```
wasm-pack build
```

빌드 이후에는 `pkg` 디렉토리가 생기는 것을 확인할 수 있습니다.

### 앱 세팅하기

앱을 세팅할 때는 `npm`의 도움을 받습니다. 다음 커맨드를 통해 wasm app 템플릿을 받아옵니다.

```
npm init wasm-app www
```

이 템플릿은 아직 위에서 빌드한 패키지를 사용하고 있지 않습니다. 이를 사용하기 위해서 `package.json`의 `dependencies`를 추가해줍니다.

```json
  ...
  "dependencies": {
    "wasm-bench": "file:../pkg"
  },
  "devDependencies": {
  ...
```

그 뒤 `www` 디렉토리 안에서 디펜던시를 받아옵니다.

```
cd www
npm install
```

마지막으로, 빌드한 패키지를 바로 사용해봅시다. `www/index.js`를 열어보시면 다음과 같이 코드가 작성되어 있는 것을 보실 수 있습니다.

```javascript
import * as wasm from "hello-wasm-pack";

wasm.greet();
```

여기서 import하는 `hello-wasm-pack`을 우리의 `wasm-bench`로 바꿔줍시다.

```javascript
import * as wasm from "wasm-bench";

wasm.greet();
```

### 앱 실행하기

앱을 실행할 때는 `www`디렉토리 안에서 `npm`을 다음과 같이 사용합니다.

```
npm run start
```

실행하면 어느 포트에서 돌아가고 있는지 메시지가 나옵니다. (`Project is running at:`) 해당 위치에 접속해보면 다음과 같은 창을 확인하실 수 있습니다.

![](/assets/images/rbtree/wasm1.png)

## N-Queen 구현

이제 N-Queen을 Rust와 JavaScript에서 구현해보고자 합니다. 두 함수는 모두 NxN board의 사이즈 `N`을 인자로 받아서, NxN board에서 가능한 경우의 수를 계산하는 것을 목표로 합니다.

### Rust

```rust
mod utils;

use wasm_bindgen::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

fn solve(row: usize, mut check: &mut [Vec<bool>], mut cnt: &mut u64) {
    let n: usize = check.len();
    if row == n {
        *cnt += 1;
        return;
    }
    for col in 0..n {
        if !(0..row).any(|i| {
            check[i][col]
                || col + i >= row && check[i][col + i - row]
                || col + row < n + i && check[i][col + row - i]
        }) {
            check[row][col] = true;
            solve(row + 1, &mut check, &mut cnt);
            check[row][col] = false;
        }
    }
}

#[wasm_bindgen]
pub fn calculate(n: usize) -> u64 {
    let mut check: Vec<Vec<bool>> = vec![vec![false; n]; n];
    let mut cnt: u64 = 0;
    solve(0, &mut check, &mut cnt);
    return cnt;
}

```

코드 작성 과정에서 [이 코드](https://rosettacode.org/wiki/N-queens_problem#Rust)를 참조했습니다.

우선 맨 아래의 `calculate()`부터 살펴봅시다. `n`이라는 값을 받을 수 있도록 합니다. `n`은 usize로 지정을 했는데, Rust에서는 배열의 크기와 같은 system-dependent한 값에 대해서는 usize라는 자료형을 사용하게 합니다.

`n`이라는 값의 크기를 모르기 때문에, Rust의 Vec 자료형을 사용해 `check` 배열을 정의합니다. `&mut`은 레퍼런스를 넘기는 것인데, `mut` 키워드는 넘겨진 레퍼런스가 값이 바뀔 수 있음을 의미합니다. 이를 통해 check 배열과 cnt의 값이 `solve()` 함수에 의해 바뀔 수 있도록 합니다. 마지막으로, cnt를 반환합니다.

`solve()` 함수는 N-Queen 문제를 풀어보신 분들이라면 Rust에 대해서 자세히 모르시더라도 쉽게 읽으실 수 있을 것입니다. 유의할 부분은 `any()` 메소드로, `0..row`는 0부터 row - 1까지 iterate하는 iterator이며 `any()` 메소드를 통해 뒤의 체크식이 true인 값이 하나라도 있는지 체크합니다. 이 체크식은 자신의 바로 위나 대각선에 이미 퀸이 놓여있는지를 체크하기 때문에, if문 안의 로직은 그런 퀸이 없을 때만 실행됩니다.

`www/index.js`는 다음과 같이 수정했습니다.

```javascript
import * as wasm from "wasm-bench";

var sum = 0.0;
const num = 10;

for (var siz = 7; siz < 13; siz++) {
  for (var i = 0; i < num; i++) {
    var startTime = new Date();
    wasm.calculate(siz);
    var endTime = new Date();
    var timeDiff = endTime - startTime;
    sum += timeDiff;
  }
  
  var avg = sum / num;
  console.log(`[wasm] Size ${siz}, Count ${num}, Average ${avg}ms`);
}
```

### JavaScript

JS 코드는 [이 링크](https://rosettacode.org/wiki/N-queens_problem#JavaScript)를 참조해 작성했습니다.

```javascript
function queenPuzzle(rows, columns) {
  if (rows <= 0) {
      return [[]];
  } else {
      return addQueen(rows - 1, columns);
  }
}

function addQueen(newRow, columns, prevSolution) {
  var newSolutions = [];
  var prev = queenPuzzle(newRow, columns);
  for (var i = 0; i < prev.length; i++) {
      var solution = prev[i];
      for (var newColumn = 0; newColumn < columns; newColumn++) {
          if (!hasConflict(newRow, newColumn, solution))
              newSolutions.push(solution.concat([newColumn]))
      }
  }
  return newSolutions;
}

function hasConflict(newRow, newColumn, solution) {
  for (var i = 0; i < newRow; i++) {
      if (solution[i]     == newColumn          ||
          solution[i] + i == newColumn + newRow || 
          solution[i] - i == newColumn - newRow) {
              return true;
      }
  }
  return false;
}

for (var siz = 7; siz < 13; siz++) {
  for (var i = 0; i < num; i++) {
    var startTime = new Date();
    queenPuzzle(siz, siz);
    var endTime = new Date();
    var timeDiff = endTime - startTime;
    sum += timeDiff;
  }
  
  var avg = sum / num;
  console.log(`[js] Size ${siz}, Count ${num}, Average ${avg}ms`);
}
```

## 비교

위에서 작성한 코드를 실행해본 결과 다음과 같았습니다.

![](/assets/images/rbtree/wasm2.png)

| 크기 | Wasm    | JavaScript |
| ---- | ------- | ---------- |
| 7    | 0ms     | 260.4ms    |
| 8    | 0.4ms   | 261.3ms    |
| 9    | 2.2ms   | 264.6ms    |
| 10   | 9.7ms   | 280.1ms    |
| 11   | 48.9ms  | 395.2ms    |
| 12   | 259.9ms | 1035.7ms   |

JS 쪽을 살펴보면 기본적으로 걸리는 시간이 있을 뿐더러, 크기에 따른 시간의 증감폭이 Wasm에 비해서 큰 것을 확인할 수 있습니다.

# 결론

Wasm이 JS에 비해서 성능이 잘 나오는 것을 간략하게나마 살펴볼 수 있는 시간이었습니다. 특히 저는 Rust에 대해서 공부를 한 상태이기 때문에, 앞으로 웹 코딩을 할 일이 있다면 Rust + Wasm을 선호하게 되지 않을까 싶습니다.

이 글은 rustwasm의 [Tutorial](https://rustwasm.github.io/docs/book/game-of-life/setup.html)에 기반을 두고 있습니다. 더 상세히 살펴보고 싶은 분이 계시다면 이 쪽을 참고하시는 것을 권장합니다.
