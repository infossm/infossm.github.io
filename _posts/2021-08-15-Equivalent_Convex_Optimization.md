---

layout: post

title:  "An Automatic System to Detect Equivalence Between Iterative Algorithms"

date: 2021-08-15

author: rkm0959

tags: [optimization]

---

논문 링크 : https://arxiv.org/abs/2105.04684 

논문 저자 : Shipu Zhao, Laurent Lessard, Madeleine Udell

# Introduction : Optimization Algorithms and Their Equivalence

수학적 최적화에는 문제를 해결하기 위한 다양한 알고리즘이 있습니다. 각 알고리즘은 
- 그 형태를 통해서 우리에게 최적화에 대한 직관을 주기도 하고
- 훌륭한 성능으로 우리가 최적화 문제를 어떻게 해결할 수 있는지 알려줍니다. 

이는 이미 작성했던 여러 글에서도 강조했던 내용입니다. 예를 들어, 단순히 smooth differentiable convex function $f$를 최소화하는 문제에는 여러 알고리즘이 있고, 특히 Accelerated Gradient Method를 기반으로 하는 Acceleration에 대한 여러 연구가 진행되었습니다. 이때,
- Acceleration 현상을 이해하기 위한 알고리즘의 형태에 대한 직관적인 이해에 대한 연구
- Acceleration의 State of the Art를 위한 연구

가 모두 진행되었고, 이들은 모두 중요한 연구들입니다.

이러한 면에서 **동일한**, 즉, 수학적으로 말하면 *isomorphic*한 알고리즘을 여러 형태로 나타내는 것은 전혀 쓸모없는 일이 아닙니다. 같은 알고리즘이어도 어떻게 서술하냐에 따라서 알고리즘의 작동에 대해 생각하는 방법이 달라질 수 있고, 여기서 얻어가는 새로운 아이디어가 있을 수 있기 때문입니다. 

하지만 State of the Art나 "알고리즘 자체의 새로움"을 원하는 입장이라면, 같은 알고리즘을 표현하는 방법이 여러 가지일 수 있다는 점은 굉장히 귀찮은 점입니다. 두 알고리즘이 동치인지 판단하는 것은 보통 수학적 귀납법으로 쉽게 증명 가능한 사실이지만, 새로운 알고리즘을 만드려고 하는 입장에서는 매번 이미 알려진 알고리즘들과 동치인지 확인하는 작업은 상당히 귀찮을 수 밖에 없습니다. 특히, 두 알고리즘이 동치가 되려면 한 알고리즘의 파라미터를 적당히 잡아야 하는 경우도 있어, 동치 여부를 확인하기 위해서 꽤 머리를 써야하는 경우도 있습니다. 

하나의 예시를 들어보겠습니다.

$$f(x) + g(x)$$

를 최소화하기 위해서, PDHG는 

$$x^{k+1} = \text{Prox}_{\alpha f} (x^k - \alpha u^k)$$ 

$$u^{k+1} = \text{Prox}_{\beta g^\star} (u^k + \beta (2x^{k+1} -  x^k))$$

를 선택합니다. 이에 비해서, DRS를 $f$ 먼저 사용하면 

$$x^{k+1/2} = \text{Prox}_{\alpha f}(z^k)$$

$$x^{k+1} = \text{Prox}_{\alpha g} (2x^{k+1/2} - z^k)$$

$$z^{k+1} = z^k + x^{k+1} - x^{k+1/2}$$

가 됩니다. 두 알고리즘이 동치임을 확인하기는 쉽지 않습니다. 실제로는 $\beta = 1/\alpha$를 선택하면 동치가 됩니다. 손으로 충분히 할 수 있지만, 아무래도 굉장히 귀찮은 작업이겠죠? **자동화가 필요해보입니다.**

이번 글에서 공부할 논문은, 이 과정을 자동화해주는 알고리즘입니다. 즉, 두 알고리즘이 주어졌을 때, 두 알고리즘이 동치인 알고리즘인지를 자동으로 판별하는 방법을 제시하고, 이를 구현하여 Linnaeus라는 오픈소스 소프트웨어를 공개했습니다. 물론, 모든 동치 여부를 확인할 수는 없고, 특정 형태만 확인합니다.

# Big Picture : Control Theory and Transfer Function/Matrix

우리가 볼 알고리즘의 종류는 Linear한 알고리즘으로, 그 형태는 Control Theory에서 motivation을 가져와

$$x^{k+1} = Ax^k + Bu^k$$

$$y^k = Cx^k + Du^k$$

$$u^k = \phi(y^k)$$

입니다. 여기서 $\phi$는 non-linear하고, 대표적인 예시로 $\nabla f$가 있겠습니다. 이 점화식을 전개하면

$$y^k = CA^k x^0 + \sum_{j=0}^{k-1} CA^{k-j-1}Bu^j + Du^k$$

를 얻습니다. $y^i$들이 동일하려면, 궁극적으로 위 점화식의 계수들이 동일해야 합니다. 

이를 더욱 간단하게 표현하기 위해서, "생성함수"를 사용하여, 

$$ D + \sum_{k=1}^\infty CA^{k-1}B z^{-k} = C(zI - A)^{-1} B + D$$

를 사용하면, 이게 같으면 동일한 $y^i$ 값을 뽑아냄을 알 수 있습니다. 이를 transfer function이라 부릅니다.

# Equivalence 1 : Oracle Equivalence

알고리즘 A와 알고리즘 B가 oracle equivalent 하다는 것은, 임의의 알고리즘 A의 initialization에 대하여, 알고리즘 B를 적당히 initialize하여 두 알고리즘에서 non-linear oracle $\phi$를 access 하는 위치가 완전히 동일하도록 할 수 있다는 것입니다. 핵심적인 결과는, 두 알고리즘이 한 iteration에서 사용하는 oracle call의 횟수가 동일하다면, 두 알고리즘이 oracle equivalent함은 transfer function이 동일함과 동치입니다. 

논문의 예시를 그대로 가져오겠습니다. Algorithm 1을 

$$x_1^{k+1} = 2x_1^k - x_2^k - \frac{1}{10} \nabla f(2x_1^k - x_2^k)$$

$$x_2^{k+1} = x_1^k$$

라 하고, Algorithm 2를 

$$\xi_1^{k+1} = \xi_1^k - \xi_2^k - \frac{1}{5} \nabla f(\xi_1^k)$$

$$\xi_2^{k+1} = \xi_2^k + \frac{1}{10} \nabla f(\xi_1^k)$$

이라 합시다. 첫 번째 알고리즘을 앞서 다룬 형태로 두면 

$$ \left[ \begin{matrix} x_1^{k+1} \\ x_2^{k+1} \end{matrix} \right] = \left[ \begin{matrix} 2 & -1 \\ 1 & 0\end{matrix} \right] \left[ \begin{matrix} x_1^{k} \\ x_2^{k} \end{matrix} \right]  + \left[ \begin{matrix} -\frac{1}{10} \\ 0 \end{matrix} \right] u^k $$ 

$$ y^k = 2x_1^k - x_2^k = \left[ \begin{matrix} 2 & -1 \end{matrix} \right] \left[ \begin{matrix} x_1^{k} \\ x_2^{k} \end{matrix} \right]$$

$$u^k = \nabla f(y^k)$$

로 쓸 수 있고, 여기서 대응되는 $A, B, C, D$를 뽑을 수 있습니다. 이제 계산하면 

$$C(zI - A)^{-1}B  +D = \frac{-2z+1}{10(z-1)^2}$$

을 얻습니다. 이 작업은 두 번째 알고리즘에서도 비슷하게 

$$ \left[ \begin{matrix} \xi_1^{k+1} \\ \xi_2^{k+1} \end{matrix} \right] = \left[ \begin{matrix} 1 & -1 \\ 0 & 1 \end{matrix} \right] \left[ \begin{matrix} \xi_1^{k} \\ \xi_2^{k} \end{matrix} \right] +\left[ \begin{matrix} -\frac{1}{5} \\ \frac{1}{10} \end{matrix} \right] u^k $$

$$ y^k = \xi_1^k = \left[ \begin{matrix} 1 & 0 \end{matrix} \right] \left[ \begin{matrix} \xi_1^{k} \\ \xi_2^{k} \end{matrix} \right]$$

$$u^k = \nabla f(y^k)$$

로 쓸 수 있고, 같은 방법으로 $A, B, C, D$를 뽑고 계산하면 역시 

$$C(zI - A)^{-1}B  +D = \frac{-2z+1}{10(z-1)^2}$$

을 얻어 두 알고리즘이 동치임을 확인할 수 있습니다. 두 알고리즘이 동치임을 직접 증명해봅시다 :)

# Equivalence 2 : Algorithm Repetition

비슷하게, 한 알고리즘 $B$가 다른 알고리즘 $A$를 단순히 반복해서 얻어지는 알고리즘인지도 확인할 수 있습니다. 

$B$가 $A$를 두 번 반복해서 얻어지는 알고리즘이라 하면, 

$$x_1^k = Ax_B^k + Bu_1^k$$

$$y_1^k = Cx_B^k + D u_1^k$$

$$x_B^{k+1} = Ax_1^k + B u_2^k$$

$$y_2^{k+1} = Cx_1^k + D u_2^k$$

$$u_1^k = \phi(y_1^k), \quad u_2^k = \phi(y_2^k)$$

가 되고, 이를 정리하면 

$$x_B^{k+1} = A^2 x_B^k + AB u_1^k + B u_2^k$$

$$y_1^k = Cx_B^k + D u_1^k$$

$$y_2^k = CA x_B^k + CB u_1^k + Du_2^k$$

가 되어 대응되는 행렬이 

$$\left[ \begin{matrix} A^2 & AB & B \\ C & D & 0 \\ CA & CB & D \end{matrix} \right]$$

가 되고, 이 행렬에 대응되는 transfer function을 구할 수 있습니다. 이를 계산하면, 논문에 의하면 

$$ \left[ \begin{matrix} c(zI- A^2)^{-1} AB + D & C(zI - A^2)^{-1}B \\ CA(zI-A^2)^{-1}AB + CB & CA(zI-A^2)^{-1}B + D \end{matrix} \right] $$

가 됩니다. 역으로, 알고리즘의 transfer function이 위와 같다면 그 알고리즘은 $A$를 두 번 반복한 것과 같습니다. 

물론, $A$를 2번 반복한 것이 아니라 $n$번 반복한 경우도 처리할 수 있습니다! Proposition 7.2를 참고하세요.

# Further Examples : Shift Equivalence, Algorithm Conjugation 

더 복잡하지만, transfer function을 이용해서 확인할 수 있는 equivalence 두 종류를 간단하게 소개하겠습니다. 

첫 번째 종류는 Shift Equivalence입니다. 예를 들어, 알고리즘 $A$가 총 세 개의 update equation (1), (2), (3)으로 이루어져 있다고 합시다. 만약 이 세 개의 update equation의 순서를 뒤섞으면 어떻게 될까요? 막 섞으면 당연히 완전히 새로운 알고리즘이 나오게 될 겁니다. 하지만, 순서가 단순히 기존 순서의 cyclic permutation이라면, 즉 예를 들어 (2), (3), (1) 순서로 섞는다면 근본적으로 알고리즘은 그대로일 겁니다. 

이런 equivalence를 shift equivalence라고 합니다. 논문의 예를 보면, Algorithm 6.1이 

$$x_2^{k+1} = \text{prox}_g (2x_1^{k+1} - x_3^k)$$

$$x_3^{k+1} = x_3^k + x_2^{k+1} - x_1^{k+1}$$

$$x_1^{k+1} = \text{prox}_f(x_3^k)$$

라고 하면, Algorithm 6.2는 그 cyclic permutation

$$x_3^{k+1} = x_3^k + x_2^{k+1} - x_1^{k+1}$$

$$x_1^{k+1} = \text{prox}_f(x_3^k)$$

$$x_2^{k+1} = \text{prox}_g (2x_1^{k+1} - x_3^k)$$

입니다. 이것 역시 transfer function을 활용하여 찾을 수 있습니다. 계산은 Proposition 6.3을 확인해봅시다. 

특히, 이 논문에서는 cyclic permutation 뿐만 아니라, 단순히 update equation 2개의 순서를 당연하게 바꿀 수 있는 경우도 고려합니다. 다시 논문에 있는 예시를 들어보면, Algorithm 6.3은 

$$x_1^{k+1} = x_4^k - t \nabla f(x_4^k)$$

$$x_2^{k+1} = x_1^{k+1} - t \nabla g(x_1^{k+1})$$

$$x_3^{k+1} = x_1^{k+1} - \nabla h(x_1^{k+1})$$

$$x_4^{k+1} = \text{prox}_{tf} (x_2^{k+1} + x_3^{k+1})$$

이고, Algorithm 6.4는 가운데 두 equation을 swap 한 

$$x_1^{k+1} = x_4^k - t \nabla f(x_4^k)$$

$$x_3^{k+1} = x_1^{k+1} - \nabla h(x_1^{k+1})$$

$$x_2^{k+1} = x_1^{k+1} - t \nabla g(x_1^{k+1})$$

$$x_4^{k+1} = \text{prox}_{tf} (x_2^{k+1} + x_3^{k+1})$$

입니다. 두 알고리즘이 사실상 동일하다는 것은 쉽게 파악할 수 있습니다. 

이때, 논문에서는 Control Theory 형태로 알고리즘을 표현했을 때 $D$의 형태에 집중합니다. 생각을 좀 해보면, $D$의 형태가 알고리즘에서 각 nonlinear oracle call에 대한 dependency graph 느낌이 난다는 것을 확인할 수 있습니다. 실제로 위 두 알고리즘은 $D$가 같습니다. 즉, $D$를 활용하면 이러한 면도 detect 할 수 있습니다.

또 다른 equivalence의 종류는 conjugation입니다. 즉, dual 문제를 고려할 때 자주 등장하는 Fenchel conjugate과 이를 활용하는 알고리즘과 관련된 equivalence 역시 transfer function으로 할 수 있습니다. 하지만 유도 과정의 계산이 상당히 복잡합니다. 핵심은, convex function $f$가 있을 때 그 Fenchel conjugate를 

$$f^\star(y) = \sup_x \left( \langle x, y \rangle - f(x) \right)$$

라고 정의하면, 가장 기본적인 relation인 (Moreau's Identity 등)

$$ (\partial f)^{-1} = \partial f^\star, \quad \text{prox}_f(x) + \text{prox}_{f^\star}(x) = x$$

또한, proximal operator의 정의에 의해서 

$$ u = \text{prox}_f(y) \iff y \in u + \partial f(u)$$

가 성립합니다. 그러니 공식들을 gradient/subgradient를 사용하는 알고리즘으로 바꾸어서 생각할 수 있습니다.

계산 과정과 사용되는 notation은 상당히 복잡하여, 여기서는 생략합니다. Proposition 8.1, 8.2를 참고하세요.

# Software : Linnaeus

https://github.com/udellgroup/Linnaeus_software

이론을 알았으니, Linnaeus를 다운받아서 몇가지 예시를 만들어봅시다!

페이지 안에서 다양한 예시들을 확인할 수 있으니, 직접 여러 시도를 해보는 것을 추천합니다. 

아쉽게도, 개인적으로는 아직 프로그램 자체는 사용하기 편한 상태가 아닌 것 같습니다. 

# Conclusion and Further Notes

이 논문에서는 
- Control Theory의 아이디어를 가져와서 
- transfer function이란 개념을 도입하여 

equivalent한 알고리즘을 detect하는 방법을 제시했습니다. 
 
Control Theory의 아이디어를 가져와서 난이도 있는 문제를 해결한 것은 이번이 처음이 아닙니다. 

저자 중 한 명인 Laurent Lessard는 2014년에 Control Theory에서 사용되는 IQC를 가져와서 optimization의 여러 문제를 해결하기도 했습니다. 
이처럼 Control Theory와 optimization을 합치는 논문들이 보이는데, 이후에도 얼마나 많이 나오게 될지 기대가 됩니다. 2014년 논문은 https://arxiv.org/abs/1408.3595 이고, 관심이 있으신 분은 제가 논문을 읽고 공부하면서 짠 코드인 https://github.com/rkm0959/rkm0959_implements 의 "HeavyBall_Counter" 역시 참고하시면 좋을 것 같습니다. 감사합니다!