---
layout: post
title: "Rust Procedural Macros By Example"
date: 2019-02-10 22:22
author: taeguk
tags: [rust, macros, procedural-macros, proc_macro, meta-programming]
---

안녕하세요. 제가 최근에 Rust 공부를 시작했는데요~
그래서 오늘은 Rust 1.29.0 부터 stable 이 된 Procedural Macros 에 대해서 포스팅해보도록 하겠습니다.
포스팅은 구구절절한 설명보다는 예제 코드 위주가 될 예정입니다!

**[>> 이 글을 좀 더 좋은 가독성으로 읽기 <<](https://taeguk2.blogspot.com/2019/02/rust-procedural-macros-by-example.html)**

## Rust 의 매크로 시스템
Rust 의 매크로 시스템은 매우 강력한데요. 크게 다음과 같이 분류 할 수 있습니다.
1. Declarative Macros
2. Procedural Macros
    * Function-like macros
    * Derive mode macros
	* Attribute macros

첫 번째는 Declarative Macros 는 일반적으로 개발자들이 흔히 알고 있는 "선언적" 형태의 매크로 방식인데요. C/C++ 등의 타언어들과의 차이점은 문자열 전처리기 방식이 아니라 Abstract Syntax Tree 를 직접 건드리는 방식이라는 점입니다.
```rust
// 출처 : https://doc.rust-lang.org/rust-by-example/macros/dsl.html
{% raw %}macro_rules! calculate {
    (eval $e:expr) => {{
        {
            let val: usize = $e; // Force types to be integers
            println!("{} = {}", stringify!{$e}, val);
        }
    }};
}{% endraw %}

fn main() {
    calculate! {
        eval 1 + 2 // hehehe `eval` is _not_ a Rust keyword!
    }
    calculate! {
        eval (1 + 2) * (3 / 4)
    }
}
```
Declarative Macros 는 이번 포스팅의 주제가 아니므로, 더 이상 언급하진 않겠습니다.
두 번째는 이번 포스팅의 주제인 Procedural Macros 인데요. **함수의 실행** 으로서 매크로를 정의하는 방식입니다.
일단 큰 틀에서 쉽게 얘기하자면 아래와 같습니다.
```text
원래의 AST --input--> "함수" --output--> 수정된 AST
```
"함수" 는 인풋으로 원래 소스코드의 AST(Abstract Syntax Tree) 를 받고요. 함수 내부에서  AST 를 수정해서 아웃풋으로 반환합니다. 그러면 이제 실제로 컴파일되는건 수정된 AST 가 되는 것이죠.
여기서 주목해야할 점은 저 "함수" 는 그냥 일반적인 Rust 코드로 작성된다는 점입니다. 즉, 일반적인 Rust 코드로서 Procedural Macro 를 정의할 수 있고, 이 "함수"는 **컴파일 시점에 실행**되게 됩니다. (일종의 컴파일러 플러그인같은 느낌입니다.) 실제 코드로 확인하자면 아래와 같은 식으로 Procedural Macro 를 정의하게 됩니다.
```rust
#[proc_macro]
pub fn some_macro(input: TokenStream) -> TokenStream
{
	// Do something
}
```
실제 코드상에서는 AST 자료구조가 아닌 Token Stream 이 input, output 으로 사용되는데요. AST 자료구조보다는 Token Stream 을 사용하는게 인터페이스상으로 더 안정적이기 때문이라고 합니다. (아무래도 AST 자료구조 인터페이스는 변경될 가능성이 더 클테니까요.) 기초 컴파일러 이론을 잘 모르신다면 Token 이라는 개념을 모르실 수도 있는데요. [위키](https://en.wikipedia.org/wiki/Lexical_analysis)의 "Token" 부분을 읽어보시면 이해가 되실 겁니다.

## Cargo.toml 및 예제 환경
Procedural Macros 를 정의하기 위해서는 Cargo.toml 에 반드시 다음과 같이 명시해줘야만 합니다. 
```toml
[lib]
proc-macro = true
```
그리고, 이 포스팅의 예제에서 사용된 crate dependency 는 다음과 같습니다.
```toml
[dependencies]
proc-macro2 = "0.4"
syn = { version = "0.15", features = ["full", "extra-traits"] }
quote = "0.6"
```
그리고 사용한 툴체인은 `beta-x86_64-pc-windows-msvc - rustc 1.33.0-beta.6 (b203178b6 2019-02-05)` 입니다.

## Function-like macros
이제부터 `Procedural Macros` 의 세부 분류중 하나인 `Function-like macros` 에 대해서 알아보겠습니다. 거두절미하고 간단한 예제를 통해 확인해봅시다.
```rust
//
// library user's code
//
use my_example::make_function;
make_function!();
fn main() {
    // 생성된 함수를 호출합니다.
    generated_function();
}

//
// proc-macro crate's code
//
extern crate proc_macro;
use proc_macro::TokenStream;
#[proc_macro]
pub fn make_function(_: TokenStream) -> TokenStream {
    println!("make_function() called in compile time!");
    // 함수의 정의를 담은 문자열을 파싱해서 TokenStream 으로 만들어 반환합니다.
    "fn generated_function() {
        println!(\"I'm the generated function by procedural macro.\");
    }".parse().unwrap()
}
```
![](https://lh3.googleusercontent.com/ymwVxYiHG34pEcGtOTtv_YjOyLCOly0XBuDAgF8uND8NkX_gg4VSRB_vErd21vg-Zv3HoCcGCQRF)
아주 간단한 예제입니다. 사진에서 보시다시피 `make_function()` 함수는 컴파일타임에 실행됩니다.

그러면 이제부터 이 간단한 예제를 계속 발전시켜나가 볼텐데요.
먼저, 저 매크로에 **생성되는 함수의 이름을 사용자가 설정할 수 있도록 하는 기능**을 추가해보겠습니다~
```rust
//
// library user's code
//
use my_example::make_function;

// fn generated_function() { --snip-- }
make_function!();
// fn foobar() { --snip-- }
make_function!(foobar);

fn main() {
    generated_function();
    foobar();
}

//
// proc-macro crate's code
//
extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;

#[proc_macro]
pub fn make_function(arg: TokenStream) -> TokenStream {
    // syn crate 를 활용해 매크로 인자를 파싱합니다.
    // Identifier 로 파싱이 가능한 경우 인자를 함수명으로 사용하고,
    // 불가능한 경우, "generated_function" 을 함수명으로 사용합니다.
    let func_name: syn::Ident = match syn::parse(arg) {
        Ok(func_name) => func_name,
        Err(..) => syn::Ident::new("generated_function", proc_macro2::Span::call_site())
    };
    // quote! 를 활용하면 소스코드를 TokenStream 으로 변환할 수 있습니다.
    let tokens = quote!{
        fn #func_name () {
            println!("I'm the generated function by procedural macro.");
        }
    };
    tokens.into()
}
```
새로운게 등장했죠? 주석을 통해 기본적으로 설명을 드리긴 했는데요. [syn::Ident](https://docs.rs/syn/0.15.26/syn/struct.Ident.html) 는 Identifier 의 약자로서 키워드나 변수명을 나타내는 구조체입니다. 
```rust
syn::Ident::new("generated_function", proc_macro2::Span::call_site())
```
를 보면, 첫 번째 인자로 Identifier 의 이름이 들어가고 두 번째 인자로 `proc_macro2::Span::call_site()` 가 들어가는데, [proc_macro2::Span](https://docs.rs/proc-macro2/0.4.27/proc_macro2/struct.Span.html) 는 소스코드의 특정 영역을 의미하는 구조체입니다. 여기서 `call_site()` 에 주목해야하는데 이 부분에 대한 설명은 [여기](https://doc.rust-lang.org/beta/proc_macro/struct.Ident.html)를 읽어보시면 됩니다. 읽다보면 **`hygiene`** 이라는 용어가 등장하는데요. 이 부분은 [요기](https://rust-lang.github.io/rustc-guide/macro-expansion.html#hygiene)를 읽어보시면 대략적인 개념을 잡으실 수 있으십니다. 아무튼 그래서 이 예제에서는 `make_function()` 매크로가 생성한 함수를 매크로 외부에서 호출할 수 있어야하기 때문에 `call_site()` 를 사용하였습니다.
그 외에 부족한 설명은 [quote](https://github.com/dtolnay/quote), [syn](https://github.com/dtolnay/syn) 을 참고하시면 될 것 같습니다~

자, 그러면 다음으로 이제 **생성되는 함수의 인자 타입을 사용자가 설정할 수 있도록 하는 기능**을 추가해보겠습니다.
```rust
//
// library user's code
//
use my_example::make_function;

make_function!();
make_function!(foo, u32, f64,);
make_function!(bar, String, u32, f64);
//make_function!(todo, &str); // 지원 안됨.

fn main() {
    generated_function();
    foo(1, 1.23);
    bar(String::from("test"), 1, 1.23);
}

//
// proc-macro crate's code
//
extern crate proc_macro;
use quote::quote;

struct ParsedArguments {
    func_name: proc_macro2::Ident,
    arg_types: Vec<proc_macro2::Ident>,
}

fn parse_arguments(args: proc_macro2::TokenStream) -> ParsedArguments {
    let mut parsed_arg = ParsedArguments {
        func_name: proc_macro2::Ident::new("generated_function", proc_macro2::Span::call_site()),
        arg_types: vec![],
    };

    let mut arg_vec = vec![];
    for arg in args.into_iter() {
        arg_vec.push(arg);
    }

    if arg_vec.len() > 0 {
        // 첫 번째 인자는 함수명을 의미하는 identifier 이여야만 한다.
        let func_name: proc_macro2::Ident = match &arg_vec[0] {
            proc_macro2::TokenTree::Ident(ref func_name) => func_name.clone(),
            _ => panic!("The first token must be an identifier."),
        };
        parsed_arg.func_name = func_name;

        // 그 다음부터는 comma (,) 와 인자 타입을 의미하는 identifier 가 반복해서 와야한다.
        let mut i = 1;
        while i < arg_vec.len() {
            // comma 인지 체크
            match &arg_vec[i] {
                proc_macro2::TokenTree::Punct(ref punct) => {
                    if punct.as_char() != ',' {
                        panic!("The token {} must be a comma.", i + 1);
                    }
                }
                _ => panic!("The token {} must be a comma.", i + 1),
            };

            if i + 1 >= arg_vec.len() {
                break;
            }

            // Identifier 인지 체크
            let type_name = match &arg_vec[i + 1] {
                proc_macro2::TokenTree::Ident(ref type_name) => type_name.clone(),
                _ => panic!("The token {} must be an identifier.", i + 2),
            };
            parsed_arg.arg_types.push(type_name);

            i += 2;
        }
    }

    parsed_arg
}

#[proc_macro]
pub fn make_function(args: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let args = proc_macro2::TokenStream::from(args);
    let parsed_args = parse_arguments(args);
    let func_name = parsed_args.func_name;
    let arg_types = parsed_args.arg_types;

    let tokens = quote!{
        fn #func_name (#(_: #arg_types),*) {
            println!("I'm the generated function by procedural macro.");
        }
    };
    tokens.into()
}
```
다소 길어지긴 했지만, 차근차근 보시면 이해하실 수 있을 것 같습니다. 이전 예제에 비해 달라진 점은 `proc_macro2` crate 를 많이 활용했다는 점인데요. `proc_macro2` 에 대한 설명은 [Github Repo](https://github.com/alexcrichton/proc-macro2) 을 참고하시면 될 것 같습니다. 그리고 `quote!` 쪽에 `#(_: #arg_types),*` 가 보이실텐데요. `quote!` 가 제공하는 기능중 하나인 [Repetition](https://github.com/dtolnay/quote#repetition) 을 활용한 것입니다.

그러면 이번에는 위의 `parse_arguments()` 함수를 [syn::parse](https://docs.rs/syn/0.15.26/syn/parse/index.html) 를 활용해서 리펙토링 해보도록 하겠습니다.
```rust
//
// library user's code
//
use my_example::make_function;

make_function!();
make_function!(foo, u32, f64,);
make_function!(bar, String, u32, f64);
make_function!(error, String, u32,, f64);  // 컴파일 에러
//make_function!(todo, &str); // 지원 안됨.

fn main() {
    generated_function();
    foo(1, 1.23);
    bar(String::from("test"), 1, 1.23);
}

//
// proc-macro crate's code
//
extern crate proc_macro;
use quote::quote;

struct ParsedArguments {
    func_name: syn::Ident,
    arg_types: Vec<syn::Ident>,
}

impl syn::parse::Parse for ParsedArguments {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut parsed_args = ParsedArguments {
            func_name: proc_macro2::Ident::new("generated_function", proc_macro2::Span::call_site()),
            arg_types: vec![],
        };

        if !input.is_empty() {
            // 첫 번째 인자는 함수명을 의미하는 identifier 이여야만 한다.
            parsed_args.func_name = match input.parse() {
                Ok(func_name) => func_name,
                Err(err) => return Err(syn::Error::new(err.span(), format!(
                    "The first token must be an identifier."))),
            };

            // 그 다음부터는 comma (,) 와 인자 타입을 의미하는 identifier 가 반복해서 와야한다.
            let mut i = 1;
            while !input.is_empty() {
                // comma 인지 체크
                match input.parse::<syn::token::Comma>() {
                    Ok(_) => {},
                    Err(err) => return Err(syn::Error::new(err.span(), format!(
                        "The token {} must be a comma.", i + 1))),
                };
                i += 1;

                if input.is_empty() {
                    break;
                }

                // Identifier 인지 체크
                let type_name = match input.parse::<syn::Ident>() {
                    Ok(type_name) => type_name,
                    Err(err) => return Err(syn::Error::new(err.span(), format!(
                        "The token {} must be an identifier.", i + 1))),
                };
                parsed_args.arg_types.push(type_name);
                i += 1;
            }
        }

        Ok(parsed_args)
    }
}

#[proc_macro]
pub fn make_function(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let parsed_args = syn::parse_macro_input!(input as ParsedArguments);
    let func_name = parsed_args.func_name;
    let arg_types = parsed_args.arg_types;

    let tokens = quote!{
        fn #func_name (#(_: #arg_types),*) {
            println!("I'm the generated function by procedural macro.");
        }
    };
    tokens.into()
}
```
`syn` crate 의 [syn::parse::Parse](https://docs.rs/syn/0.15.26/syn/parse/trait.Parse.html), [syn::parse_macro_input](https://docs.rs/syn/0.15.26/syn/macro.parse_macro_input.html) 를 활용해서 리펙토링하고 나니 코드가 더 깔끔해졌습니다. (컴파일에러 메시지를 customize 하기 위해서 코드가 약간 복잡해진 것을 감안해주세요 ㅎㅎ) 뿐만아니라, 컴파일에러시 다음 사진과 같이 구체적으로 어디가 잘못됐는지 사용자에게 알려줄 수 있게 되었습니다.
![](https://lh3.googleusercontent.com/c4rK4ZTAAhV3GqFbTn6ina4C-vyJtzLvITKa-SoYaZU8bJhXIOvqab26S2Y-yVR4Ji_Lw8xp7nAG)

자, 지금까지 차근차근 예제를 발전시켜나가면서 여러가지 내용들을 익혀봤는데요. 여기서 다룬 내용들은 `Function-like Macros` 에만 해당되는 내용은 아니고, `Attribute Macros` 와 `Derive mode macros` 에도 모두 적용가능한 내용들입니다~ 그러면 이제 다른 종류의 매크로들에 대해서도 간단하게 살펴보도록 하겠습니다.

## Attribute Macros
이 것 역시 거두절미하고 바로 간단한 예제부터 보겠습니다.
```rust
//
// library user's code
//
use my_example::foobar;

// attr: ""
// item: "fn func_1() { }"
#[foobar]
fn func_1() {}

// attr: "a , b , c , 1 , 2"
// item: "fn func_2() { }"
#[foobar(a, b, c, 1, 2)]
fn func_2() {}

// attr: "x , y"
// item: "struct Struct;"
#[foobar(x, y)]
struct Struct;

fn main() {
}

//
// proc-macro crate's code
//
extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn foobar(attr: TokenStream, item: TokenStream) -> TokenStream {
    println!("attr: \"{}\"", attr.to_string());
    println!("item: \"{}\"", item.to_string());
    item
}
```
`Attribute Macros` 의 경우에는 파라미터로 2개의 TokenStream 이 들어오는데요. 각각이 무엇을 의미하는지는 위 예제를 보면 알 수 있습니다.

그러면, 이번에는 실제로 쓸만한 걸 만들어 볼까요?? Python 에서의 function decorator 와 비슷한 것을 만들어보도록 하겠습니다.
```rust
//
// library user's code
//
use my_example::decorated;

#[decorated(decorator)]
fn foo(s: &str, x: u32, y: u32) -> u32 {
    println!("foo(\"{}\", {}, {}) -> {}", s, x, y, x + y);
    x + y
}

fn decorator<F>(f: F, s: &str, x: u32, y: u32) -> u32 where F: Fn(&str, u32, u32) -> u32 {
    let ret = f("twice!", x * 2, y * 2);
    println!("decorator<F>(f, \"{}\", {}, {}) -> {}", s, x, y, ret);
    ret
}

fn main() {
    // Output:
    //     foo("twice!", 2, 4) -> 6
    //     decorator<F>(f, "hello", 1, 2) -> 6
    foo("hello", 1, 2);
}

//
// proc-macro crate's code
//
// https://github.com/Manishearth/rust-adorn 를 참고하여 작성하였습니다.
extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;

#[proc_macro_attribute]
pub fn decorated(attr: TokenStream, input: TokenStream) -> TokenStream {
    let decorator_name: syn::Ident = syn::parse(attr).unwrap();
    let mut original_func: syn::ItemFn = syn::parse(input).unwrap();

    // decorator 함수에 전달될 인자들의 이름이 저장되는 벡터
    let mut args_for_decorator = Vec::with_capacity(original_func.decl.inputs.len() + 1);

    let renamed_func_name = syn::Ident::new("_renamed_func", proc_macro2::Span::call_site());
    args_for_decorator.push(quote!(#renamed_func_name));

    // 원본 함수의 이름을 변경한다.
    let original_func_name =  std::mem::replace(&mut original_func.ident, renamed_func_name);

    // 새로 만들어질 함수의 인자들이 저장되는 벡터
    let mut new_args = vec!();
    let mut arg_no = 0;

    // 원본 함수의 인자들을 순회한다.
    for org_arg in original_func.decl.inputs.iter() {
        let arg_name = syn::Ident::new(format!("_arg_{}", arg_no).as_str(),
            proc_macro2::Span::call_site());

        match org_arg {
            // See: https://docs.rs/syn/0.15.26/syn/enum.FnArg.html
            syn::FnArg::Captured(ref cap) => {
                let type_name = &cap.ty;
                new_args.push(quote!(#arg_name: #type_name));
            }
            _ => panic!("Unexpected argument {:?}", org_arg)
        }
        args_for_decorator.push(quote!(#arg_name));
        arg_no += 1;
    }

    let attrs = &original_func.attrs;
    let vis = &original_func.vis;
    let constness = &original_func.constness;
    let unsafety = &original_func.unsafety;
    let abi = &original_func.abi;
    let generics = &original_func.decl.generics;
    let output = &original_func.decl.output;

    // 원본 함수의 이름을 가진 새로운 함수를 생성한다.
    // 새로운 함수는 기존 원본 함수와 똑같은 이름, 속성, constness 등을 가진다.
    // 원본 함수는 이름이 변경된 형태로 새로운 함수의 내부에 정의되게 된다.
    // 새로운 함수는 원본 함수를 첫 번째 인자로, 그리고 자신의 인자들을 그대로 나머지 인자로
    // 해서 decorator 함수를 호출한다.
    let tokens = quote!(
        #(#attrs),*
        #vis #constness #unsafety #abi fn #original_func_name #generics(#(#new_args),*) #output {
            #original_func
            #decorator_name(#(#args_for_decorator),*)
        }
    );

    tokens.into()
}
```
어느정도 그럴듯한 python 스타일의 decorator 가 만들어졌습니다! (위 코드는 [adorn](https://github.com/Manishearth/rust-adorn) 의 코드를 참고하여 Rust 2018 에디션 기반으로 제가 새롭게 rewriting 하였습니다.)

## Derive Mode Macros
위에 제가 작성한 `Function-like Macros` 와 `Attribute Macros` 의 예제들을 이해하셨다면 Derive Mode Macros 를 제가 굳히 설명드리는 건 불필요할 것 같습니다. 관심있으신 분들은 [Rust 레퍼런스](https://doc.rust-lang.org/reference/procedural-macros.html#derive-mode-macros)를 참고하여 직접 코드를 작성해보시면 될 것 같습니다!

## 마무리
Rust 의 매크로 시스템은 정말 엄청 강력한 것 같습니다. 언어의 기능을 확장하고 meta programming 을 하는데 매우 수월한 것 같습니다. C++ 에서도 compile-time meta programming 은 매우 인기있는(?) 주제인데요. C++을 주로 했던 사람 입장에서 Rust 의 `Procedural Macros` 는 그냥 일반 Rust 코드가 compile time 에 실행된다는 게 매우 충격적(?) 이고 부러웠습니다..ㅎㅎ
아, 그리고 이번 포스팅을 하면서 제가 처음으로 20줄이상의 Rust 코드를 짜봤는데요.. Rust 문법도 안 익숙한 상태에서 바로 매크로를 만들어보려니까 다소 힘들었었네요;; ㅎㅎ 아무튼 저의 러스트 hello world 코드라서 뭔가 이상한 부분이 있을 수도 있는데 양해부탁드립니다.. (댓글로 잘못된 점을 지적해주시면 더더욱 감사하겠습니다!)
앞으로 Rust 열심히 공부할께요..!

## 참고자료
* https://doc.rust-lang.org/reference/procedural-macros.html
* https://blog.rust-lang.org/2018/12/21/Procedural-Macros-in-Rust-2018.html
* https://github.com/rust-lang/rust/issues/54727
* https://words.steveklabnik.com/an-overview-of-macros-in-rust
* https://tinkering.xyz/introduction-to-proc-macros/
* https://github.com/Manishearth/rust-adorn
* https://github.com/elichai/log-derive
