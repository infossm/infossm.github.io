---
layout: post
title:  "Introduction to Matroid Union"
date:   2019-11-14 20:13:00
author: imeimi2000
tags: [matroid]
---

# Matroid

Matroid에 대한 기본 지식은 다른 글을 참고하는 것이 좋다. 여기서는 Matroid의 정의만 짚고 넘어가자.

**정의 1. 유한집합 $$S$$와 $$S$$의 부분집합족 $$\mathcal{I}$$에 대해 $$M=(S, \mathcal{I})$$가 아래 조건을 만족할 경우 $$M$$을 Matroid라고 한다.**

1. $$\{\}\in\mathcal{I}$$ (Empty set is Independent set)
2. $$B\subset A\in\mathcal{I}\Rightarrow B\in\mathcal{I}$$ (hereditary property)
3. $$A, B\in\mathcal{I}, \mid A\mid>\mid B\mid\Rightarrow\exists x\in A\setminus B$$ $$s.t.$$ $$B+x\in\mathcal{I}$$ (augmentation property)

$$S$$의 부분집합이 $$\mathcal{I}$$에 포함될 경우 독립집합(Independent set)이라고 부른다.

# Matroid Union

Matroid Union은 Matroid의 합집합으로, 다음과 같이 정의된다.

**정의 2. Matroid $$M_1=(S_1, \mathcal{I}_1), M_2=(S_2, \mathcal{I}_2), \cdots, M_k=(S_k, \mathcal{I}_k)$$가 있을 때, Matroid $$k$$개의 Union을 아래와 같이 정의한다.**

$$M_1\lor M_2\lor \cdots \lor M_k=\left(\bigcup\limits_{i=1}^{k}{S_i}, \left\{\bigcup\limits_{i=1}^{k}I_i\mid\forall i\in[1, k], I_i\in\mathcal{I}_i\right\}\right)$$

Matroid의 합집합 또한 Matroid이며, rank를 정의할 수 있다.

**정리 1. Matroid Union $$M=M_1\lor M_2\lor \cdots \lor M_k$$도 Matroid이며, $$M$$의 rank 함수 $$r_M$$은 아래와 같다.**

$$r_M(U)=min_{T\subset U}(\mid U\setminus T\mid+\sum^{k}_{i=1}{r_{M_i}(T\cap S_i)})$$

보조정리 1. 임의의 Matroid $$\hat{M}=(\hat{S}, \hat{\mathcal{I}})$$과 함수 $$f:\hat{S}\rightarrow S$$에 대해 $$M=(S, \{f(\hat{I})\mid\hat{I}\in\hat{\mathcal{I}}\})$$는 Matroid이며, rank 함수 $$r_M$$은 $$r_M(U)=min_{T\subset U}(r_{\hat{M}}(f^{-1}(T))+\mid U\setminus T\mid)$$이다.

$$M$$에서 공집합이 독립집합이고, 독립집합의 부분집합은 모두 독립집합이므로 Matroid의 augmentation property만 증명하면 Matroid임을 보일 수 있다.

모든 $$I\in\mathcal{I}$$에 대해 $$I=f(\hat{I})$$이고 $$\mid I\mid=\mid\hat{I}\mid$$인 $$\hat{I}\in\hat{\mathcal{I}}$$이 존재한다. $$\mathcal{I}$$의 정의에 의해 $$I=f(\hat{I})$$인 $$\hat{I}\in\hat{\mathcal{I}}$$이 존재하고, $$\hat{I}$$에서 $$f(x)$$가 같은 원소들 중 하나씩만 남기면 조건을 만족한다.

두 독립집합 $$\mid I\mid < \mid J\mid$$를 고려하면 Matroid의 정의에 의해 $$\hat{I}+x\in\hat{\mathcal{I}}$$인 $x\in\hat{J}\setminus\hat{I}$이 존재한다. $$\hat{I}$$를 위 조건을 만족하는 것 중 $$\mid\hat{I}\cap\hat{J}\mid$$가 최대인 것으로 선택하자. 이렇게 선택하면 $$f(x)$$가 $$I$$에 포함된다고 가정했을 때 최대성에 의해 $$\hat{I}$$에 $$x$$가 포함되어야 하는데 이는 $$x$$의 정의에 모순이므로 $$f(x)$$가 $$I$$에 포함되지 않는다. 이는 $$y=f(x)\in J\setminus I$$이고 $$I+y\in\mathcal{I}$$임을 뜻한다. 따라서 $$M$$은 augmentation property를 만족하고, Matroid이다.

$$M$$의 rank $$r_M(U)$$를 구하기 위해 Partition Matroid $$M_p=(\hat{S}, \mathcal{I}_p)$$를 다음과 같이 정의하자.

$$\mathcal{I}_p=\left\{I\subset\hat{S}\mid(\forall x\in U, \mid f^{-1}(x)\cap I\mid\leq 1)\land(\forall x\notin U, \mid f^{-1}(x)\cap I\mid=0)\right\}$$

$$M_p$$는 $$f^{-1}(U)$$에 포함되고 $$f(x)=f(y)$$인 서로 다른 두 원소가 없는 집합을 독립집합으로 하는 Matroid이다. $$M_p$$의 rank는 $$r_{M_p}(T)=\mid\left\{x\in U\mid 1\leq\mid f^{-1}(x)\cap T\mid\right\}\mid$$가 된다.

모든 $$M$$의 독립집합 $$I\subset U$$에 대해 $$\hat{I}\subset\hat{U}=f^{-1}(U), I=f(\hat{I})$$이고 $$\hat{M}, M_p$$에서 모두 독립집합인 $$\hat{I}$$를 대응시킬 수 있다. 일대일 대응은 아닐 수 있음에 유의하자. 이 때 대응되는 두 집합에 대해 $$\mid I\mid=\mid\hat{I}\mid$$가 성립한다.

rank의 정의와 $$\mid I\mid=\mid\hat{I}\mid$$에 의해 $$r_M(U)=max(\mid I\mid)=max(\mid\hat{I}\mid)$$이고, $$\hat{I}$$가 $$\hat{M}, M_p$$의 intersection이므로 Min-max Formula에 의해 $$r_M(U)=min(r_{\hat{M}}(\hat{T})+r_{M_p}(\hat{U}\setminus\hat{T}))$$가 된다.
$$\hat{T}$$가 $$f(x)$$ 값이 같은 원소들 중 일부만 포함하고 있다면, 그러한 원소를 모두 제거하여 더 작은 값을 얻을 수 있으므로 $$\hat{T}=f^{-1}(T)$$꼴로 나타나는 경우만 고려하면 된다. 따라서 $$r_M(U)=min(r_{\hat{M}}(f^{-1}(T))+\mid U\setminus T\mid)$$가 된다. $$\blacksquare$$

다시 본 정리로 돌아가서 $$\hat{M}_i=(\hat{S}_i, \mathcal{I}_i)$$으로 정의하자. 여기서 $$\hat{S}_i=\{(i, x)\mid x\in S_i\}$$이다. 이것은 모든 $$\hat{S}_i$$의 원소가 다르도록 설정해준 것으로, 따라서 $$\hat{M}=\hat{M}_1\lor\hat{M}_2\lor\cdots\lor\hat{M}_k$$는 Matroid이다. 그리고 함수 $$f((i, x))=x$$를 정의하자. 보조정리에 의해 $$M$$은 Matroid이고, 아래 식이 성립한다.

$$
\begin{align*}
r_M(U) & =min(r_{\hat{M}}(f^{-1}(T)))+\mid U\setminus T\mid) \\
& =min\left(\sum^{k}_{i=1}{r_{\hat{M}}(f^{-1}(T)\cap\hat{S}_i)+\mid U\setminus T\mid}\right) \\
& =min\left(\sum^{k}_{i=1}{r_{M_i}(T\cap S_i)+\mid U\setminus T\mid}\right)
\end{align*}
$$

위 식은 증명하고자 하는 식과 같다. $$\blacksquare$$

# Matroid Partition

Matroid Partition은 주어진 집합이 Matroid들로 나누어질 수 있는지 판단하고 그 방법을 찾는 문제로, 다르게 말하면 주어진 집합이 Matroid Union의 독립집합인지 판단하는 문제이다. 이러한 문제의 예시로는 무방향 그래프를 스패닝 트리로 분할하는 문제가 있다. 다음 포스팅에서는 Matroid Partition을 찾는 두 가지 방법에 대해서 다룰 것이다.
