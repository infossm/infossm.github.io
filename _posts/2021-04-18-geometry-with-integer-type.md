---
layout: post
title: "정수 자료형으로 기하학 문제 데이터 검증하기"
date: 2021-04-18
author: gumgood
tags: [geometry]
---

# 개요

기하학 문제가 까다로운 이유 중 하나는 실수 자료형을 쓸 때 생기는 부동 소수점 오차 때문입니다. 같은 이유로, 데이터가 문제의 조건에 맞는지 검증하는 과정 역시 쉽지 않습니다. 심지어 실수 자료형으로는 해결할 수 없는 경우도 존재합니다. [BOJ 20939 달고나](https://www.acmicpc.net/problem/20939)를 출제하면서 겪은 문제들과 해결한 과정에 대해 정리하고자 합니다.

# 입력 조건

문제에서 데이터 형식과 범위를 제외하고 검증해야 하는 조건들은 다음과 같습니다.

1. 단순 다각형의 꼭짓점은 반시계방향 순서이다.

2. 선분의 개수와 원의 개수 합이 2,000을 넘지 않는다.

3. 세 개 이상의 선(원 또는 선분)이 한 점을 지나는 경우는 없다.

4. 모든 좌표는 정수다.

여기서 1, 2, 4번은 후술할 방법을 통해 정수 자료형으로 충분히 검증할 수 있습니다. 그러므로 3번 조건을 어떻게 해결할지 생각해봅시다. 한 점을 지나는 세 선이 없다는 것은 선 사이의 모든 교점이 유일하다는 것과 같습니다. 즉, 모든 교점의 유일성을 보장해야 하는데 정수 자료형으로는 쉽지 않아 보입니다.

몇몇 선분 교차와 관련된 문제들에서는 "주어진 점이 epsilon만큼 움직이더라도 답은 변하지 않는다"와 같이 보장하고 있었습니다. 이런 조건을 주면 입력으로 주어지는 좌표에 어느 정도 부동 소수점 오차가 있더라도 선분 간의 교차 여부가 변하지 않아 데이터를 검증하는 validator를 작성하기 쉽습니다. 하지만 본 문제는 모든 좌표가 정수이기도 하고 선분이 원에 접하는 경우 조금만 이동하더라도 교점의 수가 달라질 수 있기 때문에 유효한 방법은 아닙니다.

다른 방법으로는 "선 사이 교점이 epsilon보다 가까운 경우는 주어지지 않는다"와 같이 보장하는 것입니다. 모든 교점을 구해놓은 뒤 분할정복을 통해 가장 가까운 두 점을 찾고 둘 사이 거리가 epsilon보다 큰지 확인해주는 방법입니다. 안타깝게도 이 방법으로 데이터를 보장할 경우, 부동 소수점 오차로 틀려야 할 코드를 저격할 데이터까지 걸러진다는 문제가 있었습니다.

결국 이 문제는 실수 자료형으로는 한계가 있다고 생각했고, 정수 자료형으로 이를 해결하였습니다.

# 교점을 표현하는 방법

교점을 유일하게 표현할 수 있는 경우, map이나 set과 같은 자료구조를 통해 쉽게 중복을 확인할 수 있습니다. 그 표현 방법을 설명하기에 앞서, 교점을 수식화해보겠습니다. 교점이 생기는 경우는 총 세 가지가 있습니다. 편의를 위해 선분은 직선으로 대체하겠습니다.

### 직선-직선 교점

$(x_1,y_1)$과 $(x_2,y_2)$를 지나는 직선과 $(x_3,y_3)$과 $(x_4,y_4)$를 지나는 직선 사이의 교점 $(P_x, P_y)$는 다음과 같습니다.

$$
P_x = \frac{(x_1 y_2 - y_1 x_2)(x_3 - x_4) - (x_1 - x_2)(x_3 y_4 - y_3 x_4)}{(x_1 - x_2)(y_3 - y_4) - (y_1 - y_2)(x_3 - x_4)}
$$

$$
P_y = \frac{(x_1 y_2 - y_1 x_2)(y_3 - y_4) - (y_1 - y_2)(x_3 y_4 - y_3 x_4)}{(x_1 - x_2)(y_3 - y_4) - (y_1 - y_2)(x_3 - x_4)}
$$

### 원-직선 교점

원점이 $(x,y)$이고 반지름이 $r$인 원과 직선의 방정식 Ax + By + C = 0으로 주어지는 직선 사이의 교점 $(P_x,P_y)$는 다음과 같습니다.

$$
P_x = x - \frac{AC}{A^2 + B^2} \pm \frac{B}{A^2 + B^2} \sqrt{(A^2 + B^2)r^2 - C^2}
$$

$$
P_y = y - \frac{BC}{A^2 + B^2} \mp \frac{A}{A^2 + B^2} \sqrt{(A^2 + B^2)r^2 - C^2}
$$

### 원-원 교점

원점이 $(x_1,y_1)$이고 반지름이 $r_1$인 원과 원점이 $(x_2,y_2)$이고 반지름이 $r_2$인 원 사이의 교점 $(P_x,P_y)$는 다음과 같습니다.

$$
d^2 = (x_1 - x_2)^2 +(y_1 - y_2)^2
$$

$$
P_x = x_1 + (x_2 - x_1) \cdot \frac{r_1^2 - r_2^2 + d^2}{2d^2} \pm \frac{y_2-y_1}{2d^2} \sqrt{4r_1^2d^2 - (r_1^2 - r_2^2 + d^2)^2} 
$$

$$
P_y = y_1 + (y_2 - y_1) \cdot \frac{r_1^2 - r_2^2 + d^2}{2d^2} \mp \frac{x_2-x_1}{2d^2} \sqrt{4r_1^2d^2 - (r_1^2 - r_2^2 + d^2)^2} 
$$

이제 교점의 x 좌표, y 좌표를 나타낼 수 있는 자료구조를 정의해봅시다. 각 교점 좌표의 값을 살펴보면 공통으로 $\frac{1}{d}(a + b \sqrt{c})$꼴을 하고 있다는 것을 알 수 있습니다. 유리수 항 하나와 무리수 항 하나로 이뤄져 있고 각 항은 분수 형태를 하고 있습니다. 따라서 이를 잘 이용하면 모든 교점의 좌푯값을 네 개의 정수 자료형으로 표현할 수 있게 됩니다.

간결한 구현과 유일성을 위해 $\frac{p}{q} + \sqrt{\frac{r}{s}}$꼴로 관리하겠습니다. 각 교점은 유일하게 표현되어야 하므로 몇 가지 규칙을 정해야 합니다.

1. $\frac{p}{q}$와 $\frac{r}{s}$는 기약분수다.

2. 각 분수에서 0은 $\frac{0}{1}$로 표현한다.

3. $\sqrt{\frac{r}{s}}$가 유리수인 경우, $\frac{p}{q}$ 항으로 합친다.

4. $\frac{p}{q} - \sqrt{\frac{r}{s}}$꼴인 경우, $p, q, -r, s$로 나타낸다.

$p, q, r, s$에 올 수 있는 최댓값을 $x$라고 하면, 각 항을 기약분수로 나타내기 위해 $O(\log x)$만큼의 시간이 걸립니다. 또한 위의 3번 규칙을 위해 $r, s$ 모두 square number인지 판단하는 과정이 필요하고 여기에 $O(\log x)$만큼의 시간이 걸립니다. 따라서 최대 $O(N^2)$개의 교점에 대해 총 $O(N^2 \log N^2 \cdot \log x)$의 시간복잡도로 유일성을 확인할 수 있습니다.

사실 $\frac{1}{d}(a + b \sqrt{c})$꼴 그대로 관리하지 않은 이유는 앞의 방법보다 느리기 때문입니다. 이런 형태로 관리하는 경우, 유일성을 위해 $c$를 squarefree integer로 만들어야 합니다. 여기에 $O(\sqrt{x})$만큼의 시간이 요구됩니다. 입력의 범위를 고려했을 때, $x$는 대략 $10^{27}$이기 때문에 앞의 방법보다 상당히 느려 적합하지 않습니다.

# 구현

다음과 같이 구현할 수 있습니다. 계산 과정 중에 최대 $10^{27}$까지의 값을 저장해야 해서 `__int128`으로 네 정수를 저장했습니다. 

```cpp
using lint = long long;

struct dtype{
    using bint = __int128;

    bint p, q, r, s;

    dtype(lint a = 0,lint b = 0,lint c = 0,lint d = 1){
        lint sign = (b*d < 0) ? -1 : 1;
        p = a;
        q = d;
        r = b * b * c;
        s = d * d;

        lint sqr = mysqrt(r), sqs = mysqrt(s);
        if(sqr*sqr==r && sqs*sqs==s){
            p = p * sqs + sqr * q * sign;
            q = q * sqs;
            r = 0;
            s = 1;
        }

        lint g;
        g = gcd(p, q); p/=g; q/=g;
        g = gcd(r, s); r/=g; s/=g;

        r = r * sign;
    }

    bint gcd(bint a,bint b){
        return b ? gcd(b, a%b) : a;
    }

    bint mysqrt(bint x){
        bint s = 0, e = bint(1)<<63;
        while(s < e){
            bint m = (s+e) / 2;
            if(m*m < x) s = m+1;
            else e = m;
        }
        return s;
    }
};
```

추가로 이 자료구조에 임의의 비교연산자를 정의하여 stl의 map에 넣을 수 있도록 하였습니다.

```cpp
struct dtype{
    // ...
    bool operator < (const dtype &o) const {
        if(p != o.p) return p < o.p;
        if(q != o.q) return q < o.q;
        if(r != o.r) return r < o.r;
        return s < o.s;
    }
};
```

예를 들어, 원과 원 사이에 생기는 교점을 map에 추가하고 각 위치의 교점 개수가 1을 넘지 않는지 확인해보겠습니다. 우선 원의 교점 개수를 반환하는 함수를 선언했습니다.

```cpp
int cir_cir(pt o1,lint r1,pt o2,lint r2){
    lint l = (r1-r2)*(r1-r2);
    lint u = (r1+r2)*(r1+r2);
    lint d = (o1.x-o2.x)*(o1.x-o2.x) + (o1.y-o2.y)*(o1.y-o2.y);
    if(l>d || d>u) return 0;
    if(l<d && d<u) return 2;
    return 1;
}
```

key값으로 교점의 좌푯값을 가지고 value 값으로 해당 위치에 있는 교점 개수를 넣겠습니다.

```cpp
map<pair<dtype,dtype>, int> inter;
```

모든 원과 원 쌍에 대해 교점이 있다면 해당 위치에 개수를 누적합니다.

```cpp
// check circle-circle intersection point
for(int i=0;i<m;++i)
    for(int j=0;j<i;++j)
        if(int v = cir_cir(center[i],radius[i],center[j],radius[j])){
            lint x1 = center[i].x, y1 = center[i].y, r1 = radius[i];
            lint x2 = center[j].x, y2 = center[j].y, r2 = radius[j];
            lint d2 = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
            
            dtype px((r1*r1 - r2*r2 + d2)*(x2 - x1) + 2ll*d2*x1, +(y2-y1), 4ll*r1*r1*d2 - (r1*r1 - r2*r2 + d2)*(r1*r1 - r2*r2 + d2), 2ll*d2);
            dtype py((r1*r1 - r2*r2 + d2)*(y2 - y1) + 2ll*d2*y1, -(x2-x1), 4ll*r1*r1*d2 - (r1*r1 - r2*r2 + d2)*(r1*r1 - r2*r2 + d2), 2ll*d2);
            dtype qx((r1*r1 - r2*r2 + d2)*(x2 - x1) + 2ll*d2*x1, -(y2-y1), 4ll*r1*r1*d2 - (r1*r1 - r2*r2 + d2)*(r1*r1 - r2*r2 + d2), 2ll*d2);
            dtype qy((r1*r1 - r2*r2 + d2)*(y2 - y1) + 2ll*d2*y1, +(x2-x1), 4ll*r1*r1*d2 - (r1*r1 - r2*r2 + d2)*(r1*r1 - r2*r2 + d2), 2ll*d2);

            if(v == 1){
                inter[make_pair(px, py)]++;
            }else{
                inter[make_pair(px, py)]++;
                inter[make_pair(qx, qy)]++;
            }
      }
```

마지막으로 모든 교점의 좌푯값을 보면서 1보다 크지 않은지 확인해주면 됩니다.

```cpp
// check overlap intersection point
for(auto &it : inter)
    ensuref(it.second <= 1, "overlap intersection point");
```

# 그 외 검증

교점을 제외한 나머지 조건들은 모두 정수 자료형으로 검증이 가능합니다. 잘 알려진 방법들로 해결할 수 있기 때문에 코드와 함께 짧게 설명하겠습니다.

다음과 같이 단순 다각형과 원의 중심, 반지름을 저장합니다. 또한 단순 다각형의 각 선분을 모아 따로 저장했습니다.

```cpp
using pt = pair<lint,lint>;

// Polygons
vector<vector<pt>> polygon;

// Circles
vector<pt> center;
vector<int> radius;

// Lines p1 to p2
vector<pt> p1, p2;
```

벡터의 외적과 ccw 판정은 정수 자료형으로 확인할 수 있습니다.

```cpp
lint crs(pt o, pt p, pt q){
    return (p.x - o.x) * (q.y - o.y) - (p.y - o.y) * (q.x - o.x);
}

int ccw(pt a,pt b,pt c){
    lint r = crs(a, b, c);
    if(r > 0) return 1;
    if(r < 0) return -1;
    return 0;
}
```

이를 이용한 선분의 교차 판정 역시 ccw 판정만 하면 되므로 정수 자료형으로 확인할 수 있습니다.

```cpp
bool seg_seg(pt a,pt b,pt c,pt d){
    int ab = ccw(a,b,c) * ccw(a,b,d);
    int cd = ccw(c,d,a) * ccw(c,d,b);
    if(ab==0 && cd==0){
        if(b<a) swap(a,b);
        if(d<c) swap(c,d);
        return b>=c && a<=d;
    }
    return ab<=0 && cd<=0;
}
```

조건 1에서 각 다각형은 단순 다각형입니다. 연속하지 않은 두 선분이 교차하지 않음을 보임으로써 단순 다각형을 보장할 수 있습니다. 위에서 확인한 선분의 교차 판정 함수를 이용하면 되므로 정수 자료형으로 해결할 수 있습니다.

```cpp
// check simple polygon
for(auto &poly : polygon)
    for(int i=0;i<poly.size();++i)
        for(int k=i+1;k<poly.size();++k){
            int j = (i+1) % poly.size();
            int l = (k+1) % poly.size();
            if(i==l || j==k) continue;
            ensuref(seg_seg(poly[i], poly[j], poly[k], poly[l])==false, "simple polygon");
        }
```

조건 1에서 각 다각형의 꼭짓점은 반시계방향 순서입니다. 단순 다각형에서 모든 이웃한 변에 대한 외적 값을 더하면 절댓값은 다각형의 넓이가 됩니다. 이때, 부호는 꼭짓점 순서에 따라 달라지는데 반시계방향 순이면 +, 시계방향 순이면 -가 됩니다. 이를 이용하여 정수 자료형으로 해결할 수 있습니다.

```cpp
// check counter clock wise order
for(auto &poly : polygon){
    lint area = 0;
    for(int i=0;i<poly.size();++i){
        int j = (i+1) % poly.size();
        area += poly[i].x * poly[j].y - poly[j].x * poly[i].y;
    }
    ensuref(area>0, "counter clock wise order");
}
```

조건 2에서 선분의 개수와 원의 개수 합이 2000이 넘지 않음을 보이면 됩니다.

```cpp
ensuref(p1.size() + center.size() <= 2000, "number of lines <= 2000");
```

조건 3에서 교점을 계산하기 전에 겹쳐져 있는 선이 없는지 확인해야 합니다. 선분이 겹쳐있으면 한 점을 지나는 세 선분이 존재하게 되므로 반드시 확인해야 합니다. 또한 지문에 따로 적지 못했지만 같은 원이 주어지지 않게 확인해줍니다.

```cpp
// check overlap circle
map<pair<pt,lint>,int> circles;
for(int i=0;i<m;++i)
    circles[make_pair(center[i],radius[i])]++;
for(auto &it : circles)
    ensuref(it.second<=1, "overlap circle");
```

# 결론

데이터의 특징을 이용하여 적절한 자료구조를 만들어 정수 자료형으로 데이터 검증을 할 수 있었습니다. 특정한 꼴의 실수임이 보장되는 경우, 그 특징을 이용하면 새로운 자료형을 정의하면 부동소수점 오차 없이 나타낼 수 있습니다. 실수 자료형으로 해결하지 못하는 상황이 생겼을 때 이런 방식으로 접근해보시면 좋겠습니다.

# Reference

1. [Line–line intersection - Wikipedia](https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection)

2. [Circle-Line Intersection - Competitive Programming Algorithms](https://cp-algorithms.com/geometry/circle-line-intersection.html)

3. [Circle-Circle Intersection -- from Wolfram MathWorld](https://mathworld.wolfram.com/Circle-CircleIntersection.html)
