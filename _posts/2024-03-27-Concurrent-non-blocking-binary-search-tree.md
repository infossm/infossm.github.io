---
layout: post
title: "Concurrent Non-Blocking Binary Search Tree"
date: 2024-03-27
author: yhunroh
tags: [data-structure, parallel]
---

## Intro

이전의 글에서 Concurrent non-blocking linked list를 알아보았다. 이번에는 Linearlization 및 lock-free-locks의 개념과 함께, 조금 더 복잡하고 활용도가 있는 Binary search tree의 구현에 대해서 알아보자.

## Linearizability

BST는 Linked List와 다른 구조를 가지고 있지만, 원하는 성질 자체는 비슷하다. 여러 개의 프로세서가 동시에 효율적으로 작업할 수 있고, 그 결과들이 모두 옳아야 한다.

그런데, 여기서 '옳음'의 정의가 모호하다. Sequential한 환경에서는 임의의 시점에 하나의 연산만 동작할 수 있지만, Parallel한 환경에서는 동시에 여러 개의 연산이 동작할 수 있기 때문이다.
예를 들어, Set에 1,2,3,5를 insert하는 동시에, 2,3,6을 find하며, 1과 5를 remove하는 경우에, 각 연산의 결과가 옳은지 아닌지는 어떻게 정의할 수 있을까?

일반적으로 concurrent한 자료구조의 correctness는 linearizability로 정의한다. 직관적으로는, 동시에 일어나는 연산들을 적당히 잘 배치해서 sequential한 연산의 결과와 같게 만들 수 있다면 그 구조는 linearizable하다.

예를 들어, 아래 그림처럼 각 연산들의 history를 각 스레드에서의 구간들로 표현할 수 있다. (이때, 각 스레드/프로세서 사이의 시간선은 공유한다고 가정한다.)
각 연산들에 대해, 해당 연산의 구간에서 하나의 linearization point를 지정한다. 모든 연산들을 각각의 linearization point의 순서대로 하나씩 (sequential하게) 실행시켰을 때 나와야 할 결과를 생각할 수 있다. 이 가상의 결과와 실제로 얻은 결과가 동일하도록 linearization point를 항상 지정할 수 있다면, 해당 자료구조는 linearizable하다.

![](/assets/images/yhunroh/bst/2024-0327-01.png)
(Art of multiprocessor programming, Fig 3.13)

![](/assets/images/yhunroh/bst/2024-0327-02.png)
(Art of multiprocessor programming, Fig 3.14)

Linearizability에 대해서 주요한 점들이 몇개 있다.

- Linearization point는 항상 해당 연산의 구간 내에서만 잡아야 한다.
- Linearizability는 composable 하다. 다시 말해, 어떤 자료구조가 linearizable한 자료구조만으로 구성되어 있다면, 그 자료구조 또한 linearizable하다.
- Linearization point는 각 연산 내의 항상 동일한 지점으로 (i.e. 레지스터에 값을 쓰는 것) 정의될 수도 있지만, 그렇지 않을 수도 있다.

직관적으로 생각하면, 애초에 우리가 multiprocessor를 사용함으로서 얻고자 했던 것은 동일한 연산들을 sequential한 것보다 더 빠르고 효율적으로 처리하는 것이다. 따라서, 연산의 결과들이 sequential한 것과 차이가 없을 필요가 있다. 하지만 필연적으로 동시에 여러 연산이 처리되면 순서를 정의하기 어렵기 때문에, 순서를 정의하는 것에 대한 자유도만 두고, 그 순서에 맞는 결과를 요구하는 것이 linearizability이다.

Concurrent한 자료구조의 correctness를 정의하는 다른 방법들도 있지만, linearizability가 단순하고 강력하기 때문에 주로 linearizability를 사용한다.

## Lock-free-locks

이전에 설명했던 Linked List는 non-blocking으로, lock을 전혀 사용하지 않았다. 즉, executor가 각 스레드에게 자원을 어떻게 할당하더라도 언제나 어떤 연산은 진행할 수 있다. 이것을 non-blocking (lock-free)하다고 불렀다.
한편, mutex나 lock를 사용하면, executor가 lock을 기다리고 있는 스레드에게만 자원을 주는 경우에 어떤 연산도 진행하지 않는다. 이 문제는 스레드의 수가 많아질수록 악화되는데, 많은 수의 스레드가 하나의 lock을 공유하면 대기하고 있는 스레드의 수의 비율이 늘어나기 때문에, executor가 lock을 대기하고 있는 스레드에 자원을 할당할 확률이 높아지기 때문이다.

하지만 일반적으로 lock을 사용하여 알고리즘을 고안/구현하는 것이 더 쉽다. 이 문제를 해결하기 위해 lock을 사용해서 구현한 알고리즘을 단순한 치환으로 lock-free하게 변환해주는 방법이 있다. 인터페이스는 lock과 동일하지만 실제로 내부에서는 lock을 사용하지 않고, 대신 lock을 기다리고 있는 스레드들이 모두 lock을 가지고 있는 스레드의 동작을 도와주는 방식으로 구현되어 있다.

핵심은 lock을 사용해서 만든 알고리즘을 어렵지 않게 lock-free하게 변환할 수 있다는 점이다. 따라서 아래의 알고리즘에서는 lock을 사용해서 BST를 구현할 것이다. 자세한 내용은 [해당 논문](https://arxiv.org/pdf/2201.00813.pdf)을 참고하자.

## Key Points & Brainstorming

우리는 Insert(key), Remove(key), Search(key)가 가능한 BST를 만들 것이다. Self-balancing을 넣는 것도 가능하지만, 이 글에서는 다루지 않는다. 또한, 관리의 용이성을 위해 leaf-based tree로 만들 것이다. 즉, 모든 key는 leaf node에만 존재하고, 모든 internal node는 두개의 자식을 가지고 있다.

핵심은 Linked list에서와 비슷하다. Insert/Remove에서 변화하는 노드는 많아야 세개이고, 그렇기 때문에 서로 다른 부분에 이루어지는 연산들은 동시에 진행할 수 있기 때문에 성능 향상을 기대할 수 있다.

각 노드는 key, is_removed 플래그, 그리고 하나의 lock을 가지고 있다. Internal node의 경우에는 두 자식으로 가는 포인터 child[] 또한 들고 있다.
여기서 주의해야 할 점은, is_removed 플래그와 child[] 포인터들은 서로 공유되는 데이터라는 점이다. C++에서는 volatile 키워드를 통해 공유 메모리에 read/write를 outdated cache 문제 없이 할 수 있지만, CAS 연산을 사용해야 한다면 std::atomic 구현체를 사용해야 한다. 이 구현에서는 std::atomic을 사용한다.

Insert와 Remove는 자신이 변화시킬 노드까지 찾아간 후, 그 노드들에 lock을 걸고, lock을 거는 사이에 변화가 있었는지 확인한 후에 원하는 변경사항을 적용한다. 이러한 방식을 일반적으로 optimistic locking이라고 부른다. 만약 이미 추가하려는 키가 있거나, 제거하려는 키가 존재하지 않으면 연산은 실패로 끝난다.

Search는 lock을 전혀 신경쓰지 않고 트리를 순회하여 노드를 찾았는지 여부를 반환한다. lock을 하지 않기 때문에 insert/remove 연산 도중의 트리 구조를 보거나, linearizable하지 않은 결과를 반환할 것 처럼 보이지만, insert/remove 가 실제로 가하는 변화는 1-2개의 단계로 이루어지기 때문에 어렵지 않게 linearizability를 증명할 수 있다.

## Leaf Tree

다양한 concurrent BST 구현들이 있지만, 여기서는 가장 기본적인 형태를 설명한다.
관련된 연구들은 아래에서 확인할 수 있다.

- https://dl.acm.org/doi/10.1145/2555243.2555269
- https://dl.acm.org/doi/10.1145/1835698.1835736
- https://stanford-ppl.github.io/website/papers/ppopp207-bronson.pdf

구현체는 아래 두 버전이 있다. 이 글에서는 위의 버전을 기준으로 설명한다.

- https://github.com/Diuven/pillar/blob/main/structures/leafTree.cpp
- https://github.com/cmuparlay/flock/blob/main/structures/leaftree/set.h

노드는 아래와 같이 정의된다.

```cpp
struct Node;
typedef std::atomic<Node *> Edge;

struct Node
{
	std::atomic<bool> removed;
	std::mutex mtx;

	int key;
	bool is_leaf;
	Node(int k, bool is_leaf) : key(k), is_leaf(is_leaf), removed(false) {}
};

struct LeafNode : Node
{
	int value;
	LeafNode(int k, int v) : Node(k, true), value(v) {}
};

struct InternalNode : Node
{
	Edge child[2]; // [~, key), [key, ~)
	InternalNode(int k, Node *l = nullptr, Node *r = nullptr) : Node(k, false)
	{
		child[0].store(l);
		child[1].store(r);
	}
};
```

`mtx`가 lock이고, `removed`가 is_removed 플래그이다. std::atomic이 아닌 데이터들은 생성 이후에 모두 변경되지 않는다는 점에 주의하자.

이 구현에서는 lock을 사용하기 때문에, CAS등의 복합 연산이 필요하지 않다. 따라서 volatile을 사용해서 정의해도 무관하지만, 이해의 편의를 위해 std::atomic으로 구현하였다.

```cpp
auto find(int key)
{
	InternalNode *gp = nullptr;
	int gp_dir = 1;
	InternalNode *p = root;
	int p_dir = 0;
	Node *l = p->child[p_dir].load();

	while (!l->is_leaf)
	{
		gp = p;
		gp_dir = p_dir;
		p = (InternalNode *)l;
		p_dir = p->key <= key ? 1 : 0;
		l = p->child[p_dir].load(); // LinP for failed insert/delete
	}
	return std::make_tuple(gp, gp_dir, p, p_dir, (LeafNode *)l);
}
```

`find()`는 insert와 delete에서 업데이트할 key를 찾는데 사용하는 subroutine이다. key가 자료구조 안에 있는지를 판별하는 public 연산인 search와는 다르다는 점에 유의하자.

앞서 optimistic locking을 설명한 것 처럼, insert와 delete는 먼저 자신이 작업해야 할 위치를 찾아간 후, lock을 건 후에 찾은 노드들이 조건을 만족하는지를 확인한다. find는 이 중 가장 앞 단계인 작업해야 할 위치를 찾아가는 부분을 맡는다.
노드가 removed되었는지, 혹은 다른 노드가 동시에 작업해서 변화가 일어났는지 여부는 이후에 확인한다.

필요한 부분은 해당 key가 위치할 리프 노드 (l) , 해당 노드의 부모 노드 (p), 그리고 부모 노드의 부모 노드 (gp) 이며, gp - p - l 로 가는 방향이 어느 방향인지 또한 명시하여 총 5개의 값을 반환한다.

```cpp
bool insert(int key, int val)
{
	while (true)
	{
		auto [gp, gp_dir, p, p_dir, leaf] = find(key);
		if (leaf->key == key)
			return false;

		p->mtx.lock();
		Edge *ptr = &(p->child[p_dir]); // desired location
		if (p->removed.load() || ptr->load() != leaf)
		{
			// p updated
			p->mtx.unlock();
			continue;
		}

		LeafNode *new_leaf_node = new LeafNode(key, val);
		InternalNode *new_in_node =
			key < leaf->key
			? new InternalNode(leaf->key, new_leaf_node, leaf)
			: new InternalNode(key, leaf, new_leaf_node);
		ptr->store(new_in_node); // LinP for success

		p->mtx.unlock();
		return true;
	}
}
```

insert는 find로 찾은 노드중 부모 노드의 lock을 바로 걸고, find에서 본 상태 그대로인지, 그리고 removed되었는지를 체크한다. 만약 두 조건 중 하나라도 만족하지 않는 경우에는 재시작한다.
재시작하는 경우는 다른 연산이 그 사이에 성공적으로 진행되었을 경우 뿐이므로, 이 부분에서 livelock으로 빠지진 않는다.

lock을 성공적으로 걸고 나면, 해당 위치의 leaf node를 대체하기 위한 새로운 internal node와 leaf node를 만든다. 전부 만들고 나서 한번의 write 연산으로 포인터를 바꾸는 것을 볼 수 있다.
따라서, insert 연산은 어떠한 중간 단계도 만들지 않고, 항상 트리를 consistent하게 유지한다. 다시 말하면, 성공적인 insert 연산은 해당 write 연산 (ptr->store())을 기준으로 Linearization 할 수 있다는 뜻이다.

한편, 이미 key가 트리에 있는 것으로 보인 경우 insert 연산은 실패한다. 이 경우에는 lock을 얻지 않고 연산이 종료되며, Linearization point는 find() 내의 두번째 마지막 포인터 로드이다. 즉, 반환하는 parent node (leaf node가 아님) 를 찾은 그 read 연산이 해당 insert 연산의 linearization point가 된다.

```cpp
bool remove(int key)
{
	while (true)
	{
		auto [gp, gp_dir, p, p_dir, leaf] = find(key);
		if (leaf->key != key)
			return false; // key not found

		gp->mtx.lock();
		p->mtx.lock();
		Edge *ptr = &(gp->child[gp_dir]);
		if (gp->removed.load() || ptr->load() != p)
		{
			p->mtx.unlock();
			gp->mtx.unlock();
			continue;
		}

		Node *remaining_node = p->child[1 - p_dir].load();
		Node *target_leaf = p->child[p_dir].load();
		if (target_leaf != leaf)
		{
			p->mtx.unlock();
			gp->mtx.unlock();
			continue;
		}

		p->removed.store(true);
		ptr->store(remaining_node); // LinP for success

		// delete p;
		// delete l;

		p->mtx.unlock();
		gp->mtx.unlock();
		return true;
	}
}
```

remove 연산의 경우, insert와 비슷하지만 리프 노드와 그 부모 노드를 제거하여 다른 자식 노드를 한 단계 올리기 때문에 p뿐만 아니라 gp도 변화시켜야 한다.

여기서도 마찬가지로, gp 혹은 p에 변화가 있었다면, 다른 연산이 성공했다는 뜻이다. 따라서 진행은 보장된다.

일반적인 BST처럼, 남아 있을 쪽의 자식 노드를 부모 노드가 있던 자리에 대체하여 포인터를 업데이트한다. 이때 앞으로는 사용되지 않을 부모 노드와 리프 노드의 메모리 공간을 해제해 주어야 하는데, 어떤 연산이 해당 노드들을 참조하고 있을지 모르므로 섣불리 해제해서는 안된다. EBC등의 기법을 사용하여 안전하고 효율적으로 해제할 수 있는데, 이 글에서는 다루지 않는다.

성공한 remove 연산의 linearization point는 gp의 자식 포인터를 업데이트하는 연산이다. 포인터가 바뀐 것을 기점으로 다른 연산들은 모두 새로운 트리를 보게 되고, 그 전의 연산들은 removed 플래그로 인해 재시작하거나 이전 상태의 트리를 보게 된다.
실패한 경우, insert와 마찬가지로 find()에서 반환하는 부모 노드를 찾은 포인터 read 연산이 linearization point이다.

```cpp
bool search(int key)
{
	Node *nd = root->child[0].load();
	while (!nd->is_leaf)
	{
		auto nd_child = ((InternalNode *)nd)->child;
		Edge *ptr = (key < nd->key) ? &(nd_child[0]) : &(nd_child[1]);
		nd = ptr->load(); // LinP
	}

	auto leaf = (LeafNode *)nd;
	return leaf->key == key;
}
```

search 연산은 단순히 노드를 따라 내려가는 것으로 되어 있다. Linearization point는 find연산과 마찬가지로 반환하는 리프 노드의 부모 노드를 찾은 포인터 read 연산이다. Correctness의 설명은 위의 insert/delete와 동일하다.
