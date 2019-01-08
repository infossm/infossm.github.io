---
layout: post
title:  "실시간 문서 협업은 어떻게 동작할까"
date:   2019-01-08 16:00:00
author: 배선우
tags: [operational-transform]
---

# 실시간 문서 협업은 어떻게 동작할까

이번 포스트에서는 우리들이 일상 생활에서 자주 유용하게 사용하는 Google docs와 같은 실시간 문서 협업 프로그램이 어떻게 동작하는지 알아보도록 하려 한다.

실시간 문서 협업 프로그램은 Google docs와 같은 일반적인 문서뿐만 아니라, 슬라이드, 스프레드시트, 또는 코드까지도 하나의 문서를 온라인을 통해 여러 사람과 동시에 수정하고 공유할 수 있도록 해준다.

이러한 프로그램을 구현할 수 있는 방법은 여러가지가 있는데, 그 중 가장 보편적으로 사용되고 Google docs에 사용된 기술은 *Operational Transform*이라는 기술이다.

*Operational Transform*은 기본적으로 다수의 클라이언트와 하나의 서버로 구성된다. 클라이언트는 문서를 수정하는 작성자이며, 서버는 여러 클라이언트가 수정한 내용을 동기화하는 중재자이다. 예를 2명의 클라이언트와 하나의 서버가 존재하고, 문서의 초기 상태가 *Hello world* 였다고 하면, 상태는 다음과 같다.

| Client A | Server | Client B |
| ------- | ------ | ------- |
| Hello world  | Hello world   | Hello world |

여기서 Client A 과 Client B 가 각각 다른 내용을 다음과 같이 수정했다고 하면,

| Client A | Server | Client B |
| ------- | ------ | ------- |
| **my** Hello world  | Hello world   | Hello **python** world |

서버는 클라이언트들이 수행한 내용을 받아 다음과 같이 동기화를 수행한다.

| Client A | Server | Client B |
| ------- | ------ | ------- |
| **my** Hello **python** world  | **my** Hello **python** world   | **my** Hello **python** world |

여기서 서버에서 일어나는 동기화의 과정이 바로 *Operational Transform*이다. 이 동작 과정의 핵심은 용어에서도 유추할 수 있는데, 클라이언트에서 *operation*을 서버로 보내면, 서버는 그 *operation*을 받아 해당 *operation*에 대한 *transform*을 적용하는 단순한 절차의 연속이다.
*operation*의 종류는 문서의 종류에 따라 직접 설계할 수 있는데, 기본적으로 일반 문서의 경우 *Insertion*, *Deletion*, *Selection* 등이 포함되고, 스프레드시트와 같은 경우 셀의 *Merge* 또는 *Divide* 등 원하는 명령을 커스텀하게 설계하여 사용하게 된다.


그럼 실제 내부 동작이 어떻게 이루어지는지 그림과 함께 살펴보도록 하자.

![](/assets/images/operational-transform/chart1.png)

최초 클라이언트와 서버의 상태는 모두 **Hello world**로 동일하다. 여기서 Client A은 텍스트의 **0번째** 위치에 **my**라는 텍스트를 *Insert*하고, Client B는 **6번째** 위치에 **python**이라는 텍스트를 *Insert*한다. 즉 각 클라이언트는 하나의 *Insertion Operation*을 생성했다.

두 클라이언트 모두 자신의 *operation*을 서버로 전송할 것이다. 이 때 네트워크 상 Client A의 메세지가 Client B의 메세지보다 먼저 서버에 도착했다고 하자.
서버는 Client A이 보낸 **Insert 0, my**라는 *operation*을 그대로 적용하여, 기존의 서버 상태였던 **Hello world**를 **my Hello world**로 변경한다.

이제 서버는 Client B의 동기화를 위해 Client B 에게 **Insert 0, my**라는 *operation*을 그대로 전송하게 되고, Client B는 자신의 현재 상태인 **Hello python world**에 해당 *operation*을 적용하여 **my Hello python world**라는 기대했던 결과를 얻게 된다.

이제 Client B가 보낸 *operation*이 도착하여, 이를 처리하게 되는데, 문제는 여기서 발생한다.

Client B가 보낸 *Operation*은 **Insert 6, python**이다. 그런데 현재 서버의 상태는 이전에 Client A이 보낸 **Insert 0, my**라는 *operation*에 의해 상태가 **my Hello world**로 변경된 상태이기 떄문에, 여기에 **Insert 6, python**이라는 *operation*을 적용하게 될 경우 **my Helpythonlo world**라는 결과가 나오게 된다.

마찬가지로 Client A의 동기화를 위해 서버는 Client A 에게 **Insert 6, python** 이라는 *operation*을 전송하게 되고, Client A 역시 **my Helpythonlo world** 라는 상태를 가지게 된다.

결과적으로 각 Client와 Server의 상태는 다음과 같게 된다.

| Client A | Server | Client B |
| ------- | ------ | ------- |
| my Helpythonlo world | my Helpythonlo world | my Hello python world |

우리가 의도하고자 하는 결과 상태인 **my Hello python world** 는 Client B만 얻었을 뿐이고, 그 마저도 상태 동기화의 주체인 Server는 **my Helpythonlo world**라는 이상한 상태를 가지게 되었다.

해당 문제를 해결하기 위한 과정이 바로 *Transform* 이다. 서버에서 *Transform* 이 적용되게 되면 상태가 처리되는 과정이 다음과 같이 변하게 된다.

![](/assets/images/operational-transform/chart2.png)

Client B 가 보낸 기존의 파란색 *operation*이 초록색 *operation*으로 변하면서 생긴 변화를 살펴보자. 기존 Client B의 *operation*은 **Insert 6, python** 이었는데, Server가 해당 *operation*을 받아 **Insert 9, python** 으로 *transform* 한 것을 볼 수 있다.

Server가 Client B의 *operation*을 받은 시점에, 이미 Server는 Client A가 보낸 *operation*에 의해 상태가 변경되었는데, Client B는 Server의 상태가 변경되기 전의 상태를 기준으로 *operation*을 보냈기 때문에, 현재 서버의 상태와 Client B의 *operation*의 기준 상태가 일치하지 않는다. 따라서 해당 Client B의 *operation*을 현재 Server의 상태를 기준에 맞도록 *transform* 하게 된다. 

Client B의 *operation*인 **Insert 6, python**의 경우, **Insert 0, my** 라는 *operation*이 적용되지 않은 상태이므로, 해당 *Insertion operation*을 **my** 의 글자 수인 3칸(마지막 공백 한 칸 포함하여)을 밀어주게 되고, **Insert 9, python** 라는 *operation*으로 *transform*되게 된다.

![](/assets/images/operational-transform/chart3.png)

*Transform*은 Server 만 할 수 있는 것이 아니라 Client 에서도 이루어 질 수 있다. 해당 예를 들기 위해, Client B가 보낸 *operation*이 서버를 통해 Client A로 전송되기 전에, Client A가 또 문서를 수정하여, **my unique Hello world** 라는 상태로 바뀌었다고 해보자. 이 경우 해당 *operation*은 **Insert 3, unique**가 된다.

Client A는 상태가 **my unique Hello world** 인 시점에서, Server로 부터 **Insert 9, python** 이라는 *operataion*을 받을 것이다. 이 *operataion*을 그대로 Client A에 적용시킬 경우 이상한 상태가 될 것은 당연하다. 따라서 Client A는 해당 *operation*을 **Insert 16, python**으로 *transform*하여 적용하여, 정상적인 결과인 **my unique Hello python world**를 얻게 된다. 이 처럼 *transform*은 서버뿐만 아니라 클라이언트에서도 자체적으로 일어나게 된다.

그럼 실제 네트워크 상에서 어떻게 패킷이 주고받아지는지 살펴보자.
프로토콜 환경은 패킷의 순서와 전달을 보장하기 위해 TCP를 사용한다. 또한 한 가지 전제로, Client가 Server로 *operation*을 보낼 때 이미 Server로 전송한 *operation*이 존재한다면, 이미 전송한 *operation*에 대한 *acknowledge*, 즉 Server로 부터 해당 *operation*이 적용됐다는 확인을 받은 후에 다음 *operation*을 순차적으로 전송할 수 있다. 이는 Client와 Server간의 동기화를 위해서 필요한 부분이다.

![](/assets/images/operational-transform/diamond1.gif)

이전까지는 Client 2개와 Server 1개로 설명했지만, 실제로 패킷을 주고 받는 상황에서는 Client와 Server가 1:1로 통신하는 상황만을 생각하면 된다. 그림에서 **주황색 실선**은 Client의 상태를, **파란색 실선**은 Server의 상태를 나타내고, **주황색 점선**은 Client의 상태가 서버에 적용되는 상황을, **파란색 점선**은 Server의 상태가 Client에 적용되는 상황을 나타낸다.

위 상황을 순서대로 정리하면,

1. Server와 Client의 최초 상태는 **Hello world**
2. 다른 Client가 전송한 *operation*, **Insert@0, my**에 의해 Server의 상태가 **my Hello world**로 변경
3. Client가 문서를 수정하여, *operation* (**Insert@6, python**) 이 생성되고, 상태는 **Hello python world**로 변경
4. Client의 *operation* (**Insert@6, python**)이 Server로 전송
5. Server에 의해 해당 *operation* 이 *transform* (**Insert@9, python**) 됨
6. Server가 Client로 Client에 적용되지 않은 *operataion*인 **Insert@0, my**를 전송
7. Client는 해당 *operation* (**Insert@0, my**)를 *transform* (그대로 **Insert@0, my**)
8. Server는 Client에게 Client가 전송한 *operation* (**Insert@6, python**)이 적용됐다고 *acknowledge*를 전송

과 같은 과정과 동일하다.

이를 일반화하여 나타내면 다음과 같다.

![](/assets/images/operational-transform/diamond2.gif)

이번에는 조금 더 복잡한 상황에 대한 다이어그램을 살펴보자. 

![](/assets/images/operational-transform/diamond3.gif)

1. Server와 Client의 최초 상태는 **Street coffee bean**
2. 다른 Client가 전송한 *operation*, **Insert@14, red**에 의해 Server의 상태가 **Street coffee redbean**으로 변경
3. Client가 문서를 수정하여, *operation* (**Insert@7, simple**) 이 생성되고, 상태는 **Street simple coffee bean**으로 변경
4. Client의 *operation* (**Insert@7, simple**) 이 Server로 전송됨. Server에서 해당 *operation*에 대한 *transform*이 수행되고, Server 상태에 반영되어 Server의 상태는 **Street simple coffee redbean**으로 변경
5. Client가 문서를 수정하여, *operation* (**Insert@0, Big**) 이 생성되고, 상태는 **Big Street simple coffee bean**으로 변경
6. Client는 아직 이전 *operation* (**Insert@7, simple**) 에 대한 *acknowledge*를 받지 못했으므로 *operation* (**Insert@0, Big**) 은 서버로 전송되지 못하는 상태임에 주의
7. Client는 Server로 부터 다른 Client의 *operation* 인 **Insert@14, red**을 수신
8. 해당 *operation* (**Insert@14, red**) 이  *operation* (**Insert@7, simple**) 를 대상으로 *transform*이 수행되어 **Insert@21, red**로 변경됨
9. 해당 *operation* (**Insert@21, red**) 이  *operation* (**Insert@0, Big**) 을 대상으로 *transform*이 수행되어 **Insert@25, red**로 변경되고, 최종적으로 상태에 반영되어 Clinet의 상태가 **Big Street simple coffee redbean** 으로 변경됨
10. Server는 Client로 *operation* (**Insert@7, simple**) 에 대한 *acknowledge*를 전송
11. Client는 Server로 *operation* (**Insert@0, Big**) 을 전송
12. Server는 Client로 *operation* (**Insert@0, Big**) 에 대한 *acknowledge*를 전송

위와 같이 Server나 Client가 *operation*을 전달받았을 때, 자신의 상태와 전달받은 *operation*의 상태가 일치하지 않는 경우, 이전 상태들에 대한 *operation*에 대해 순차적으로 *transform*을 진행하여 해당 *operation*을 동기화시켜주는 모습을 볼 수 있다.

이런 상황들의 일반화된 다이어그램은 다음과 같이 구성된다.

![](/assets/images/operational-transform/diamond4.gif)

위에서 살펴본 것과 같이 각 Client는 Server와의 상태 동기화를 위해 항상 이전에 보낸 *operation*이 Server로 부터 *acknowledge*를 받았을 경우에만 다음 *operation*을 전송할 수 있다. 이럴 경우 많은 양의 *operation*이 계속 쌓일 경우, 클라이언트와 서버 모두 네트워크 상 무리를 줄 수 있다.

![](/assets/images/operational-transform/diamond5.png)

이러한 상황을 방지하기 위해 Client는 여러 개의 *operation*을 그룹으로 묶어서 서버에 전송하는 방법을 사용한다. 다시 말해, *A, B, C, D, E, F* 라는 *operation*이 Client에 존재하고, *operation C*까지 서버에 전송된 상태라면, *operation D, E, F*와 그 이후에 발생하는 모든 *operation*들은 모두 하나의 *operation*으로 묶여서, 서버로부터 *acknowledge*를 받았을 때 이를 한번에 서버에 전송하게 된다. 
*Operation*을 하나로 묶는 방법은 문서가 어떤 종류냐에 따라 다른데, 일반 Plain text document를 예로 들 경우 다음과 같다.

![](/assets/images/operational-transform/compound.png)

Plain text document가 가지는 *operation*을 생각해보면 텍스트를 삽입하는 *Insert*, 텍스트를 지우는 *Delete* 로 나눌 수 있다. 여기서 *operation*을 grouping하기 위해 *Retain*이라는 *operation*을 추가한다. *Retain*은 단순히 현재 커서가 가르키는 위치를 증가하는 행동으로 생각하면 된다. 예를 들면, 아래 그림과 같이 **Hello world** 라는 텍스트가 존재할 때, 각 상태들에 대한 *operation*들은 다음과 같이 구성된다.

이 외에도 적용된 *operation*을 *undo*하는 과정을 구현하거나, *Google docs*에서 제공하는 것과 같은 *Revision history*를 구현하는 과정도 필요한데, 위와 같은 내용에 대한 좀 더 자세한 정보를 얻고 싶다면 [이곳](http://www.codecommit.com/blog/java/understanding-and-applying-operational-transformation
)에서 확인 할 수 있다. 
또한 Operation Transform을 *Javascript*로 구현한 [Github repository](https://github.com/Operational-Transformation
)나, Google에서 제공하는 [관련 영상](https://www.youtube.com/watch?v=hv14PTbkIs0
), 그리고 해당 문서에 대한 [슬라이드](https://drive.google.com/open?id=0B6urv73TAti0X0VqS2h1bE1sN0U) 도 참고하면 도움이 될 것이다.