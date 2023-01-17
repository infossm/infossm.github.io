---
layout: post
title: "S/W 멤버십 기술 블로그 고치기"
date: 2020-09-20
author: jh05013
tag: debugging
---

이 블로그에 있는 버그와 문제점을 고쳤습니다.

* 태그 시스템이 제대로 동작하지 않습니다. 거의 모든 태그를 눌러도 404 오류가 나타납니다. 태그 기능을 추가하였습니다.
* 일부 글의 작성자가 표시되지 않습니다. 모두 올바르게 표시되도록 수정하였습니다.
* 일부 글이 표시되지 않습니다. 모두 보이도록 수정하였습니다.
* 검색 기능이 동작하지 않습니다. 정상 동작하도록 수정하였습니다.

사실 저는 알고리즘 문제풀이를 주로 하고, 실제 소프트웨어 등의 코드를 다뤄 본 경험은 별로 없습니다. 그래서 소프트웨어 디버깅의 일종의 체험과 배움을 하는 것에 의의를 뒀습니다. 이 글은 "단순히 어떤 버그가 있었고 어떻게 해결했는지"뿐만 아니라, "어떤 과정을 거쳐 해결 방안을 찾아냈는지"를 같이 서술합니다.

# 태그 고치기

블로그를 둘러보시면 아시겠지만, 글마다 태그가 붙어있고 이걸 클릭할 수 있습니다. 태그를 클릭하면 그 태그가 붙은 모든 글이 표시되어야 하겠지만, 그렇지 않고 "404 File not found" 페이지가 표시되었습니다.

흥미롭게도, 딱 하나의 태그만 정상 작동하였습니다. 바로 [test](http://www.secmem.org/tags/test/) 태그로, "markdown test"라는 테스트용 글만 들어있었습니다. 즉 태그 시스템 자체가 망가져 있는 게 아니었던 겁니다. 태그 시스템은 정상이지만, "test"라는 태그의 정보만 저장되어 있었던 것이죠.

이를 염두에 두고 `infossm.github.io` 폴더를 들여다 보면 `_tags`라는 폴더가 눈에 띕니다. 이 폴더에는 `blog.md`, `membership.md`, 그리고 `test.md` 뿐이었습니다. 즉 태그 버튼이 제대로 동작하지 않는 이유는 이 폴더에 test 태그밖에 없기 때문이었습니다. 모든 태그에 대해 파일을 하나씩 만들면 해결됩니다.

그 김에, 태그를 전체적으로 다듬어 보기로 했습니다. 작성자 분들의 태그 선택을 존중하여 있던 태그를 빼거나 새로운 태그를 넣지는 않고, 일관성이 있도록 태그 이름만 수정하였습니다.

* 대소문자 사용을 일관성 있게 바꿨습니다. `GitHub`처럼 고유명사일 경우 통상적인 대소문자 표기를 따랐고, `FFT`처럼 줄임말일 경우 대문자를 사용하였고, 이외에는 소문자를 사용하였습니다.
* 모든 띄어쓰기는 하이픈(`-`)으로 바꿨습니다. `/`도 폴더 명에서 문제가 생길 것으로 생각하여 `-`로 바꿨습니다.
* `greedy`와 `greedy-algorithm`처럼 뜻이 같은 태그를 하나로 합쳤습니다.

새로운 글을 쓰면서 새로운 태그를 만들고자 하신다면, `_tags` 폴더에 md 파일을 만들어 주시면 되겠습니다.

사용된 태그는 `2019-APC-solutions, AI, AMP, Actor, Aggregate, Ajou-Programming-Contest, Akka, Akka-Typed, Atcoder, Atomics, AudioWorklet, BFS-DFS, BST, Bixby, Bloom-filter, Boost, Boost.Exception, C, C++, C++17, CAM, CNN, CQRS, CTF, Cats, Chrome-extension, ClusterGCN, Codeforces, Composer, DDD, DSA, Delaunay-triangulation, Dijkstra, Docker, ECC, Effective-Modern-C++, Express, FFT, GCC, GCN, GNN, Gaussian-integer, Git, GitHub, Grad-CAM, HPC, HPX, ICPC, IOI, Javascript, KCPC, Keras, Korean-regional, Kotlin, LCS, LD_PRELOAD, LaTeX, Lagrange-interpolation, Linux, MCMC, MTL, MathML, MobX, Modern-C++, Monad, Monad-Transformer, NLP, NP-complete, Node.js, OpenAI, PHP, PageRank, Parsedown, Polygon, Python, RSA, RWST, React, Regional, Rust, SCPC, SHA, SHA256, SMT, Scala, Stern-Brocot-tree, Stirling-number, TailRec, Trampoline, Typescript, UCPC, Wasm, WebWorker, Windows-XP, ZIO, algorithm, android, antivirus, app, attention, audior-rendering, automata, backtracking, bash, big-integer, binary-search, bipartite-matching, bitset, brute-force, builtin-function, cactus, cheatsheet, clustering, combinatorics, competitive-programming, complexity, complexity-theory, computation-geometry, computation-theory, computer-vision, constant, constructive, coroutine, cryptography, data-science, data-structure, debugging, decomposition, deep-Q-learning, deep-learning, degree-sequence, dependency-injection, design-pattern, digital-signature, director, discriminator-rejection-sampling, double, dynamic-programming, editorial, elliptic-curve, error-handling, event-driven-design, event-sourcing, exception, extension, factorial, float, floating-point, front-end, functional-programming, fusion, game-theory, generative-adversarial-networks, geometry, graph-theory, greedy, grid, group-theory, hash, heap, heavy-light-decomposition, hooking, image-translation, introduction, linear-algebra, localization, logic, machine-learning, macros, markdown, mathematics, matrix, matroid, maximum-antichain, maximum-flow, maximum-independent-set, membership-test, meta-programming, microservice, minimum-path-cover, minimum-vertex-cover, monad, multi-armed-bandit, natural-language-processing, neural-network, non-stationary, number-of-divisor, number-theory, open-source, operational-transform, optimization, palindrome, parallel, partition, prime-factorization, priority-queue, problem-solving, proc_macro, procedural-macros, production, programming-contest, pull-request, quantum, random, random-algorithm, rating, regex, reinforcement-learning, rejection-sampling, retrospect, rope, sampling, scalable-Bloom-filter, security, segment-tree, self-attention, sequence-to-sequence, setter, shake-2019, shellcoding, socket, software-design, sorting, sqrt, string, strongly-typing, structure, study, submask-iteration, symbolic-mathematics, tail-recursion, test, test-coverage, test-data, thompson-sampling, time-complexity, tips, topology, tree, university-club, unix, upper-confidence-bound, wavenet, web, web-assembly`입니다.

# 작성자 수정

블로그 상에서 작성자명이 표시되지 않은 글들이 있었습니다. 이는 `author` 란에 표기된 아이디가 `_authors` 폴더에 존재하지 않아서 발생한 문제입니다. 아이디가 적혀있지 않았거나, 해당 파일이 `_authors` 폴더에 없었거나, 파일이 있었으나 그 파일의 `name`과 `title`이 반대로 적혀있었습니다. 세 경우 모두 수정하였고, 이제 모든 글에 작성자명이 제대로 표시됩니다.

# 표시되지 않는 글 표시하기

로컬에서 Jekyll로 이 홈페이지를 돌려 보면 다음 오류가 발생합니다.

![](/assets/images/infossm-github-fix/hidden1.PNG)

놀랍게도 이 글은 [https://secmem.org](https://secmem.org)에서도 찾을 수 없었습니다. 해당 글은 2019년 7월 21일에 작성되었는데, 그 중에 shake! 2019에 대한 글은 없었습니다.

위 스크린샷에는 표시되지 않았지만, "position 39"에서 "invalid byte sequence" 오류가 났습니다. 그런데 39번째 문자는 그냥 제목의 일부로 사용된 띄어쓰기였습니다. 그래서 제목을 아예 "s"로 바꿔 봤지만, position 100을 넘어간 어떤 위치에서 또 "invalid byte sequence" 오류가 났습니다. (정확히 어디였는지는 기억나지 않습니다.)

혹시 특정 문자가 잘못된 게 아니라 파일 자체에 문제가 있었던 게 아닐까요? 그렇게 생각하며 파일의 인코딩을 살펴 봤는데, CP949였습니다. UTF-8로 다시 저장하여 해결했습니다.

이렇게 [shake! 2019 출제 후기 글](https://secmem.org/blog/2019/07/21/shake-2019-review/)이 빛을 보게 되었습니다. 많이 읽어 주세요~

![](/assets/images/infossm-github-fix/hidden2.PNG)

# 검색 기능 고치기

검색 기능이 제대로 동작하지 않았고, 이는 [깃헙 이슈](https://github.com/infossm/infossm.github.io/issues/242)로도 올라와 있었습니다. 고쳐 봅시다.

해당 이슈를 살펴 보면 `Uncaught Error: SimpleJekyllSearch --- failed to get JSON (/search2.json)` 가 뜬다는 정보가 있습니다. 실제로 그렇습니다.

![](/assets/images/infossm-github-fix/search2.PNG)

지난달에 [Karp의 21대 NP-완전 문제](http://www.secmem.org/blog/2020/08/18/karp-21-np-complete/)를 작성할 때는 기다리는 시간을 줄이기 위해 모든 글을 삭제해 놓고 작성했습니다. 이렇게 하면 한 번 고친 다음 결과가 반영되는 데 30초가 걸리는 걸 몇 초 정도로 단축할 수 있기 때문입니다. 흥미롭게도, 이렇게 하면 아무 오류 없이 검색이 잘 됩니다.

![](/assets/images/infossm-github-fix/search1.PNG)

태그 시스템처럼, 검색 시스템 자체가 망가져 있는 게 아니었던 겁니다. 어떤 글이 오류를 일으켰거나, 아니면 글이 너무 많아서 오류가 발생했으리라 짐작했습니다. 하지만 글이 많다고 해서 검색 시스템이 망가질까요? "특정 개수까지만 처리한다"같은 식으로 검색이 구현된 게 아닌 이상, 후자보다는 전자에 더 가까울 것이라고 예상했습니다.

이걸 염두에 두고, 오류를 따라가 봅시다. 콜 스택의 맨 위에 있는 함수는 이것이었습니다.

``` javascript
function throwError(message){ throw new Error('SimpleJekyllSearch --- '+ message)
```

말그대로 오류를 표시하는 메시지니까, 한 단계 내려가 봅시다.

``` javascript
  function initWithURL(url){
    jsonLoader.load(url, function(err,json){
      if( err ){
        throwError('failed to get JSON (' + url + ')')
      }
      initWithJSON(json)
    })
  }
```

`'failed to get JSON (' + url + ')'`은 아까 위에서 본 형태입니다. `jsonLoader.load` 함수가 오류를 내뱉었고, 이걸 에러 핸들링 함수가 잡았음을 알 수 있습니다. 한 단계 더 내려가 봅시다.

``` javascript
function createStateChangeListener(xhr, callback){
  return function(){
    if ( xhr.readyState===4 && xhr.status===200 ){
      try{
        callback(null, JSON.parse(xhr.responseText) )
      }catch(err){
        callback(err, null)
      }
    }
  }
}
```

어떤 글이 오류를 일으켜서 검색 기능이 망가진 거였다면, 파싱 과정에서 문제가 생긴 게 아닐까요? 이 함수를 `infossm.github.io` 폴더에서 검색해 보니, `_site/dest/jekyll-search.js`에 이 함수가 있다고 합니다. 이 함수로 가서 `hi`라는 메시지를 출력한 다음 `JSON.parse(xhr.responseText)`를 try, catch 없이 그냥 실행하게 해보았습니다.

![](/assets/images/infossm-github-fix/search3.PNG)

새로운 오류가 떴습니다. 저 `VM231:1111`을 타고 들어가 보면...

![](/assets/images/infossm-github-fix/search4.PNG)

`$O(N^{1/4 + \epsilon})$ 시간 복잡도에 소인수 분해하기`를 파싱하면서 문제가 생긴 것이었습니다. `\e`가 이상한 글자로 바뀌어 있었습니다. 제목을 유지시키면서 이 오류를 어떻게 없애나 생각하다가, `\epsilon`을 `ε`로 바꿔서 해결했습니다. 이제 검색이 잘 됩니다!

# 후기

아직 몇 가지 오류가 남아있는데, 이들은 다음 기회에 시도해볼 생각입니다.

* 콘솔에 `ERROR '/sw.js' not found.`라는 오류가 표시됩니다.
* `$`로 LaTeX을 작성하면 깨져서 나옵니다. `$`을 사용해야 합니다. 그런데 제 컴퓨터에서 돌려볼 때는 `$`로도 잘 나옵니다.
* 지난달에 제가 작성한 글에서 "Vertex Cover에서 Directed Hamiltonian Circuit으로" 부분이 깨져 있습니다. 아직 원인을 모르겠습니다.

알고리즘 문제를 풀면서 디버깅을 하는 것과, 이미 제시되어 있는 블로그 코드를 디버깅하는 것은 비슷하면서도 다른 느낌이었습니다. 알고리즘 문제에서의 디버깅은 익숙한 언어로 작성한, 내가 잘 알고 있는 단 하나의 코드를 다룹니다. 반면 블로그 디버깅은 여러 파일과 내가 읽어보지 않았던 코드, 그리고 나에게 익숙하지 않은 언어(자바스크립트)를 다룹니다. 하지만 "어디서 문제가 발생한 것인지 추정하면서 거꾸로 거슬러 올라간다"라는 전략은 알고리즘 문제를 풀면서도 늘 하는 일이었고, 블로그를 디버깅할 때에도 비슷하게 쓸 수 있었습니다.

어떤 프로그램을 작성하든, 디버깅은 프로그램 작성 못지 않게 중요한 능력이 아닐까 생각됩니다.
