---

layout: post

title:  "Terra Smart Contracts"

date: 2022-1-17

author: rkm0959

tags: [blockchain]

---

# 서론 

이 글에서는 Terra Blockchain에서 Smart Contract를 어떻게 구현하는지 대해서 간단하게 다루어보려고 합니다. 
Terra 블록체인의 가장 대표적인 native 코인은 LUNA인데, 이는 coinmarketcap에서 1월 16일 현재 시가총액 9위 (약 $30B) 입니다. Terra Blockchain에서도 Ethereum과 마찬가지로 Smart Contract들을 올릴 수 있는데, 그 언어와 구조가 Ethereum의 Smart Contract들과 많이 다릅니다. 이 글에서는 Terra 블록체인 위에서의 금융적 흐름보다는 Smart Contract 구조와 구현에 대해서 더 집중하겠습니다. 

글은 기본적으로 Terra Documentation에서 많이 가져왔으나, 조금 더 이해하기 쉽고 찾아볼 reference가 충분하도록 부가설명을 많이 붙였습니다.

# Terra Smart Contract의 구조 

Ethereum의 Smart Contract를 생각해보면, 이를 호출하여 다음 세 가지 종류의 transaction을 할 수 있습니다.

- constructor 호출로 initialization
- blockchain 내부 상태를 바꾸는 (그래서 gas fee를 내야 하는) transaction
- blockchain 내부 상태를 바꾸지 않는 (그래서 gas fee를 낼 필요없는) query transaction

입니다. solidity에서는 constructor(), view 등을 사용하여 이를 표현할 수 있습니다. 

즉, 처음 contract가 deploy 될 때 constructor()가 호출되며, 내부 상태를 바꾸지 않는 function의 경우 view라는 modifier를 사용할 수 있습니다. 

Terra에서는 이 세 종류의 transaction을 조금 더 명확하게 분리합니다.
Terra의 Smart Contract를 짜기 위해서는 다음 세 함수를 구현하면 됩니다.
- instantiate() : initialization을 위한 코드
- execute() : 내부 상태를 바꾸는 transaction을 처리하는 코드
- query() : 내부 상태를 query하는 transaction을 처리하는 코드 

참고로, 매우 중요한 사실인데, Terra의 Smart Contract는 **Rust**로 구현됩니다. 

# Example Terra Smart Contract

예시를 보기 위해서 Terra Documentation https://docs.terra.money/Tutorials/Smart-contracts/Write-smart-contract.html#start-with-a-template 를 확인해봅시다. 

```sh
cargo generate --git https://github.com/CosmWasm/cw-template.git --branch 0.16 --name my-first-contract
cd my-first-contract
```

를 통해서 예시 contract를 받을 수 있습니다.

이 Smart Contract의 목적은 다음과 같습니다.
- count라는 값을 관리합니다. 이 값은 Contract가 생성될 때 초기화할 수 있습니다.
- 누구나 increment를 호출하여 count를 1 증가시킬 수 있습니다
- Contract를 만든 owner는 reset을 호출하여 count라는 값을 새로 정할 수 있습니다
- 누구나 count의 현재 값을 query 할 수 있습니다. 

## State 정의하기 

먼저 Blockchain에서 저장해야 할 값이 무엇인지를 생각해보면
- 당연히 count의 값을 저장해두고 있어야 하며
- 누가 Contract를 만들었는지, 즉 owner를 저장하고 있어야 합니다.

이때 count는 정수이며, owner는 Terra 주소가 될 것입니다. 이를 합치면

```rust
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
pub struct State {
    pub count: i32,
    pub owner: Addr,
}
```

과 같은 State struct를 정의할 수 있습니다. serde를 통해서 Serialization/Deserialization을 지원하고, Clone, Debug, PartialEq, JsonSchema도 지원합니다. 

이제 이 State를 Terra Blockchain이 저장하도록 해야하는데, 이를 위해서는 cw_storage_plus라는 crate를 이용해야 합니다.
- https://docs.rs/cw-storage-plus/latest/cw_storage_plus/ 

이 crate에서는 여러 종류의 저장공간을 정의하는데, 대표적으로는
- Item : 단일 원소를 저장할 때 쓰입니다.
- Map : Solidity의 mapping과 같고, Hashmap을 생각하면 됩니다. 

가 있습니다. 여기서는 Item을 사용하면 됩니다.

```rust
pub const STATE: Item<State> = Item::new("state");
```

여기서 "state"는 Terra Blockchain에서의 storage key입니다.

## Instantiate

Instantiate를 다루기 위해서는 
- instantiate를 하려면 어떤 형태의 data/message를 보내야 하는지 정의하고
- 그러한 data/message를 받았을 때 어떻게 해야하는지 로직을 작성해야 합니다

우선 보내야 하는 메시지는 단순히 초기 count의 값입니다.

```rust
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
pub struct InstantiateMsg {
    pub count: i32,
}
```

이제 instantiate 로직을 정의할 수 있습니다. 

```rust
#[cfg_attr(not(feature = "library"), entry_point)]
pub fn instantiate(
    deps: DepsMut,
    _env: Env,
    info: MessageInfo,
    msg: InstantiateMsg,
) -> Result<Response, ContractError> {
    let state = State {
        count: msg.count,
        owner: info.sender.clone(),
    };
    set_contract_version(deps.storage, CONTRACT_NAME, CONTRACT_VERSION)?;
    STATE.save(deps.storage, &state)?;

    Ok(Response::new()
        .add_attribute("method", "instantiate")
        .add_attribute("owner", info.sender)
        .add_attribute("count", msg.count.to_string()))
}
```

뭔가 많습니다. 천천히 설명하자면 

- DepsMut : storage, api, querier로 구성되어 있습니다.
  - storage는 말 그대로 현재 주소에 저장되어 있는 storage입니다
  - api는 여러 cryptographic helper function을 구성되어 있습니다
  - querier는 말 그대로 query msg를 받아서 이를 처리해주는 struct입니다
    - https://github.com/CosmWasm/cw-plus/blob/main/packages/cw20/src/helpers.rs 참고
- Env : block (BlockInfo)와 contract (ContractInfo)로 구성되어 있습니다
  - block은 height, time, chain_id로 구성되어 있습니다
  - contract는 단순히 해당 주소만 저장하는 struct입니다
- MessageInfo : sender와 funds로 구성되어 있습니다.
  - sender는 누가 이 msg를 보냈는지를 의미하는 Addr입니다.
  - funds는 이 msg를 보낸 사람이 함께 보낸 native 코인들을 의미합니다.
- InstantiateMsg : 앞에서 정의한 struct입니다.

초기 state를 구하고, STATE에 이를 저장합니다. 중간에 contract version을 설정하는데, 이는 

```rust
use cw2::set_contract_version;
```
에서 나옵니다. 예시 contract의 전체 코드를 참고하시고, cw2를 보시면 됩니다.
- https://crates.io/crates/cw2

return 하는 값은 Response인데 (정확히는 Result) 여기에는
- message : 이후 호출할 smart contract execution들
- attributes : logging을 위한 key-value pair들
- events : solidity의 event와 같은 것입니다
- data : binary data (정확히는 ```Option<Binary>```)

가 있습니다. 주어진 코드는 빈 response를 만들고 여기에 add_attribute()를 이용하여 attribute를 추가해주고 있습니다. 이런 형태의 코드는 앞으로도 계속 나올 겁니다.

## Execution

역시 어떤 message를 보내야 execution이 되는지 정의해야 합니다. 그런데 이번에는 가능한 execution 종류가 2개입니다. increment도 가능하고, reset도 가능합니다. 

그렇기 때문에 ExecuteMsg는 enum으로 정의됩니다. 즉, 

```rust
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ExecuteMsg {
    Increment {},
    Reset { count: i32 },
}
```

이제 execution logic을 작성해야 합니다. 작성해야 하는 함수가 2개이니, 먼저 둘 중 어떤 함수를 호출해야 하는지 결정해주는 함수를 만들면 코드가 깔끔해지겠죠? 

```rust

#[cfg_attr(not(feature = "library"), entry_point)]
pub fn execute(
    deps: DepsMut,
    _env: Env,
    info: MessageInfo,
    msg: ExecuteMsg,
) -> Result<Response, ContractError> {
    match msg {
        ExecuteMsg::Increment {} => try_increment(deps),
        ExecuteMsg::Reset { count } => try_reset(deps, info, count),
    }
}
```

match를 이용해서 깔끔하게 코드를 increment와 reset으로 나눠준 코드입니다. 이러한 형태의 구현은 production에서도 사용됩니다. Terraswap의 (Terra의 사실상 첫 DEX) 코드가 github에 공개되어 있으니 참고하시면 좋을 것 같습니다. 이제 increment, reset의 코드를 구현해야 하는데, 먼저 increment를 봅시다. 

```rust
pub fn try_increment(deps: DepsMut) -> Result<Response, ContractError> {
    STATE.update(deps.storage, |mut state| -> Result<_, ContractError> {
        state.count += 1;
        Ok(state)
    })?;

    Ok(Response::new().add_attribute("method", "try_increment"))
}
```

저 update 함수가 중요한데, 정의를 보면 
```rust 
pub fn update<A, E>(&self, store: &mut dyn Storage, action: A) -> Result<T, E>
where
    A: FnOnce(T) -> Result<T, E>,
    E: From<StdError>, 
```

라고 써 있습니다. 즉, 

```rust
A: FnOnce(T) -> Result<T, E>
```
를 준비해주면 이를 Item에 적용시킬 수 있다는 것입니다. 이를 
```rust
|mut state| -> Result<_, ContractError> {
        state.count += 1;
        Ok(state)
    })?;
```
로 구현해주면 원하는 대로 구현이 잘 됩니다.

비슷하게, reset 역시 
```rust
pub fn try_reset(deps: DepsMut, info: MessageInfo, count: i32) -> Result<Response, ContractError> {
    STATE.update(deps.storage, |mut state| -> Result<_, ContractError> {
        if info.sender != state.owner {
            return Err(ContractError::Unauthorized {});
        }
        state.count = count;
        Ok(state)
    })?;
    Ok(Response::new().add_attribute("method", "reset"))
}
```
형태로 구현이 잘 됩니다. 

로직을 보면 sender가 owner와 다르면 error를 발생시킴을 알 수 있습니다.

## Query

우선 쿼리의 종류와 그에 대한 response를 설계해야 합니다.

```rust
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum QueryMsg {
    // GetCount returns the current count as a json-encoded number
    GetCount {},
}

// We define a custom struct for each query response
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
pub struct CountResponse {
    pub count: i32,
}
```

count를 요청하는 query에 담을 argument가 없으니 저렇게 정의하고, response는 count의 값만 알려주면 되니 위와 같이 정의가 됩니다.

이를 구현하는 코드는 다음과 같습니다.
```rust
#[cfg_attr(not(feature = "library"), entry_point)]
pub fn query(deps: Deps, _env: Env, msg: QueryMsg) -> StdResult<Binary> {
    match msg {
        QueryMsg::GetCount {} => to_binary(&query_count(deps)?),
    }
}

fn query_count(deps: Deps) -> StdResult<CountResponse> {
    let state = STATE.load(deps.storage)?;
    Ok(CountResponse { count: state.count })
```

보시는 바와 같이, count의 값을 뽑아내기 위해서 load() 함수를 이용하고, 이를 통해서 얻은 response를 binary로 바꿔서 반환하는 것을 확인할 수 있습니다. 잘 보면 query()의 반환값이 ```StdResult<Binary>```입니다.

실제 Contract의 코드 분리를 보면 (https://github.com/InterWasm/cw-template/tree/main/src)
- msg.rs : 각종 message 및 response struct 정의
- state.rs : blockchain 위에 저장해놓을 state 정의
- error.rs : 각종 error에 대한 정의 
- contract.rs : 메인 로직 작성 

형태로 작성되어 있고, 이는 production에서도 사용하는 방식입니다. 

contract.rs를 보면 unit test들도 작성되어 있는데, 관심 있으시면 읽어보시는 것도 좋습니다. 

mock environment를 구성하기 위한 함수들이 준비되어 있어서, 이를 활용할 수 있습니다.

# 결론 

지금까지 Terra Smart Contract의 구조에 대해서 알아보았습니다. Rust의 강력함을 여러 곳에서 볼 수 있어서 저는 개인적으로 solidity보다 훨씬 더 즐겁게 코드를 읽을 수 있었습니다. Execution으로 가능한 것들을 enum으로 나열하고, match로 잘 분리해주는 과정이 되게 깔끔하다고 느껴졌습니다. 아무래도 Rust니까 solidity보다 더 안전할 것 같기도 하고요. 

하지만 아직 Terra Smart Contract를 deploy 하거나 deploy된 Smart Contract와 직접 interact하는 방법에 대해서는 다루지 않았는데, 이에 대해서는 추후 글에서 다룰 기회가 있을 것으로 생각합니다. 

조금 더 복잡한 Smart Contract 코드를 읽고 싶은 분은
- ERC20에 대응되는 CW20 standard의 implementation 
- Terraswap (DEX) implementation

을 순서대로 읽으면 많은 도움이 됩니다. 참고하시면 좋을 듯 합니다.