---
layout: post
title:  "Judger와 Express를 이용한 채점 서버 구현 후기"
date:   2019-10-21 21:10:00
author: shjgkwo
tags: [C, Node.JS, Express, Docker]
---

# 목차

- [1. 개요](#개요)
- [2. Judger](#Judger)
- [3. Express 서버 구현](#Express-서버-구현)
- [4. 테스트](#테스트)
- [5. 후기](#후기)

# 개요
 채점 서버는 OJ(Online Judge)를 만들때 제일 핵심적인 기능입니다. 학교 졸업 프로젝트의 일환으로, 프로그래밍 학원 선생님한테 유용하게 쓰일법한 웹 어플리케이션을 구상하는 과정에서 OJ의 채점 기능이 반드시 필요하게 되었습니다. 이번에 구현한 채점 서버는 Node.JS의 Express 기반 채점 서버입니다. 클라이언트로부터 소스코드를 받을 때, 시스템 콜 혹은 메모리 오버플로, 버퍼 오버플로등의 문제에서 발생할 수 있는 보안적 이슈를 해결하기 위하여 칭다오 대학에서 만든 OJ에서 SECCOMP기반의 [Judger](https://github.com/QingdaoU/Judger)를 사용하기로 하였습니다. 이번 포스트를 통하여 제가 한 삽질과 제 설계등을 모두에게 공개하여 제 설계의 문제점등을 피드백 받을 수 있었으면 좋겠습니다. 또한 이제 막 Express를 뗀 사람들이, 혹은 OJ를 만들어보고 싶으나 도저히 감이 안잡히는 완전 초보분들에게도 도움이 되었으면 합니다.

# Judger
 제가 맨 처음 서버를 설계할 때 클라이언트로 받은 소스코드를 컴파일하고 실행할 때 발생 할 수 있는 모든 보안적인 문제점들에 대해 매우 고민하였습니다. 클라이언트가 System Call을 사용하여 서버내의 저장되어있는 파일에 접근하여 인풋 데이터를 알아내는 것은 물론이요, 채점 서버에 과부하를 주어 서버를 다운되게 만들게 되면 사실상 그러한 OJ는 더이상 운영하는 것으로는 가치가 없다 판단했습니다. 그에 따라 샌드박스를 반드시 구현해야 하겠다는 필요성을 느꼈습니다. 하지만 어떻게 구현하지, 하면서 긴가민가 하다가 [seccomp](https://en.wikipedia.org/wiki/Seccomp) 이라는 신뢰성이 떨어지는 바이너리 파일을 실행하기 위한 일종의 보안 기술을 찾았습니다. 하지만, 이를 새로이 공부하여 C언어로 샌드박스를 구현하기엔 제가 가진 시간이 너무 한정적이었습니다. 그래서 누군가는 나와 비슷한 생각을 하고 이미 구현해놓은게 없을까 하다가 칭다오 대학에서 오픈소스로 푼 seccomp 기반의 샌드박스를 발견하였습니다. 그것이 바로 지금 언급하고 있는 Judger 입니다. Judger의 코드는 기본적으로 seccomp의 rule을 사용하여 System call 을 탐지하여 프로세스를 고립시키는 것이 목적이며, 동시에 [rusage](https://www.gnu.org/software/libc/manual/html_node/Resource-Usage.html)를 사용하여 자식 프로세스의 리소스 사용량을 모니터링하여 메모리, 스택등의 사용량을 제한하는 작업까지 모두 구현해 놓은 것 입니다. 다행히도 우리학교의 운영체제 시간에서는 저러한 재미난 기술들을 활용하는 것을 잠시나마 배우기 때문에 Judger의 코드를 이해하는데에는 큰 어려움이 없었습니다. 물론, 이 포스트를 읽는 분들 께서는 저러한 기술에 대해서 모르셔도 됩니다. 친절하게도 [Document](https://docs.onlinejudge.me/#/judger/api)를 만들어 놓았기 때문에, 저희는 이러한 기술에 고도의 이해를 필요로 하지 않습니다.

 ## Linux Judger 활용
  그렇다면 Judger 활용은 어떻게 할까요? 일단 운영체제가 Linux 라는 가정하에 먼저 서술하겠습니다.
  
```bash
$ sudo apt-get update
$ sudo apt-get install libseccomp-dev
$ apt-get install gcc
$ apt-get install cmake
$ apt-get install git
```
 제일 먼저 세팅을 해야합니다. seccomp library 를 다운로드 받고 gcc와 cmake를 설치합니다. seccomp을 제외하고는 보통의 리눅스에서 개발하시는 분이라면 대부분 설치 하셨으리라 사료됩니다.

```bash
$ git clone https://github.com/QingdaoU/Judger.git
$ cd ./Judger
$ mkdir build && cd build && cmake .. && make && sudo make install
```

이제 본격적으로 수행하는 작업입니다! 이 스크립트를 쉘에 입력하고 나면 **libjudger.so**가 어디에 설치되었다고 나올것입니다. 이후 cp 명령어를 사용하여 적절한 위치로 옮깁시다. 저 같은 경우 /home/Desktop/tester 디렉토리를 만들어서 그곳으로 옮겨주었습니다.

```bash
$ /home/Desktop/tester/libjudger.so --max_cpu_time=1000 --max_real_time=2000 --max_memory=536870912 --max_process_number=200 --max_output_size=16384 --exe_path="/home/Desktop/tester/test.o" --input_path="/home/Desktop/tester/input.txt" --output_path="/home/Desktop/tester/output.txt" --error_path="/home/Desktop/tester/error.txt" --uid=0 --gid=0 --seccomp_rule_name="c_cpp"
```

조금 지저분해서 죄송합니다. 이런식으로 실행할 수 있다는 예시를 들었습니다. 아마 의도적으로 런타임 에러나, 메모리 초과, 시간 초과를 내지 않았다면 여러분의 스크립트는 다음과 같은 결과를 보게 될것입니다.

```bash
$ {
    "cpu_time": 0,
    "real_time": 8,
    "memory": 1372160,
    "signal": 0,
    "exit_code": 0,
    "error": 0,
    "result": 0
}
```

이런식으로 파싱하기 쉬운 형태가 되어 Js나 Python에서 오브젝트로 만들기 쉬운 형태로 제공해줍니다. 우리는 이 결과값을 잘 활용하면 됩니다.

우리가 주로 쓰게될 것은 **result**입니다. 0은 정상종료, 1은 cpu 시간 초과, 2는 real time 시간초과, 3은 메모리 초과, 4는 런타임 에러, 5는 Judger 자체가 뻗은 경우입니다. 5를 보게되는 경우는 파라미터를 잘못 입력했거나, 진짜로 Judge가 뻗는 흔치 않는 상황이니 파라미터를 제대로 입력했다면 거의 볼일이 없는 숫자입니다. 또한 4번 런타임 에러에서 signal이 25가 나왔다면 출력 버퍼 초과 이므로 Presentaion Error 등을 감지하면 유용할 것 입니다.

## Mac 또는 Window에서 Judger 활용
 [Docker](http://pyrasis.com/private/2014/11/30/publish-docker-for-the-really-impatient-book)를 사용하여 Window와 Mac환경에서도 Judger를 활용하는 것을 보겠습니다. Docker를 이런 용도로 쓰려고 만든건 아니지만 개발 환경에서 테스트 하는것을 보기 위해서 Docker를 사용할것입니다. 위의 Docker 링크에 들어가 원고를 공개해준 작가한테 감사하는 마음을 가지고 Docker를 먼저 설치를 해주세요.
  Docker 설치가 완료되었다면 우선 Judger 부터 만들어야 겠지요? 아마 여러분이 window라면 window 전용 shell, mac 이라면 터미널을 켜봅시다.
 어차피 윈도우나 맥이나 거의 작업은 비슷할것이라 생각되므로 mac을 기준으로 설명하겠습니다.

```bash
$ docker run -i -t --name makeJudger -v /Users/SeoByeongChan/Judger:/home ubuntu
```

참고로 v옵션은 도커 컨테이너와 호스트 사이에 공유할 볼륨을 결정하는 옵션입니다. 저는 /Users/SeoByeongChan/Judger 디렉토리를 공유했으니 여러분은 자유롭게 여러분의 디렉토리를 공유 옵션으로 설정하면 됩니다.

우분투 이미지를 다운로드 받게 되고 나면 makeJudger 라는 이름의 도커 컨테이너가 생성됩니다. 만약 도커 쉘이 실행이 안되었다면,

```bash
$ docker start makeJudger
$ docker attach makeJudger
```

이 스크립트를 실행해봅시다.
이제 부터는 리눅스와 완전히 똑같습니다.

```bash
$ apt-get update
$ apt-get install libseccomp-dev
$ apt-get install git
$ apt-get install gcc
$ apt-get install cmake
```

다만 sudo 명령어가 안될수도 있으니 위의 스크립트로 실행해주세요.

```bash
$ git clone https://github.com/QingdaoU/Judger.git
$ cd ./Judger
$ mkdir build && cd build && cmake .. && make && make install
```

이렇게 하면 docker 컨테이너 /usr/lib/judger 안에 libjudger.so 가 생성이 될것입니다.

```bash
$ cp /usr/lib/judger/libjudger.so /home/libjudger.so
```

이제 호스트와 공유되는 도커 볼륨에 libjudger.so 를 생성했습니다. 
여러분이 mac을 쓰신다면 이런 상황이 되어야 합니다.

![사진1](/assets/images/Judge-Server-shjgkwo/ex1.png)

이제 본론으로 넘어가겠습니다.

# Express 서버 구현
 여러분이 기본적으로 node와 npm을 모두 설치했다는 가정하에 진행하겠습니다. node와 npm설치 가이드는 인터넷에 매우 잘 나와 있으니 천천히 따라오시길 바랍니다.

```bash
$ npm install -g express
$ npm install -g express-generator
$ npm install -g npm-check-updates
$ express --view=hbs Judge-Server
```

익스프레스를 설치한 뒤, Judge-Server를 위한 express 세팅입니다. npm-check-updates 는 여러분이 디펜던시를 관리할 때 유용하게 쓸 수 있는 npm package 입니다.

그 다음은 저는 중요한 패키지들을 깔기 위해서 다음과 같은 패키지들을 깔았습니다.

```bash
$ cd ./Judge-Server
$ npm install mongoose
$ npm install mongoose-auto-increment
$ npm install cors
$ npm install child_process
$ npm install sync-queue
$ npm install fs
```

여기서 테스트 해보려면 npm start를 해봅시다.

```bash
npm start
```

테스트는 여기까지 하고 우리는 bin/www 부터 열어서 몽고 디비 서버부터 연결 할 겁니다. 저는 여러분이 몽고 디비에 어느정도 지식이 있다고 가정하고 글을 쓸 것입니다.

```javascript
/***
 * Server started add
 */

const mongoose = require('mongoose');
const config = require('../config.js');

/* ---------
 mongoose connected */

mongoose.connect(config.mongodbUri, {useUnifiedTopology: true, useNewUrlParser: true, useCreateIndex: true });
mongoose.Promise = global.Promise;

const db = mongoose.connection;
db.on('error', console.error);
db.once('open', ()=>{
    console.log('connected to mongodb server')
});
```

[config.js](https://github.com/Byeong-Chan/Judge-Server/blob/master/judge_server/config.js)에 대한 내용은 이 링크를 클릭해서 살펴볼 수 있습니다. 루트위치에 반드시 만들어줍시다.

그 다음은 model/[model.js](https://github.com/Byeong-Chan/Judge-Server/blob/master/judge_server/model/model.js) 를 만들어 줄 것 인데 이 역시 이 내용을 참고해봅시다. 저는 서버를 만들기 위해 다음과 같은 스키마를 사용했습니다. 메인 서버에서 큐에대한 정보도 핸들링하면 지나치게 메인서버에 부담을 많이 주기 때문에 메인서버는 오직 클라이언트의 코드를 받아서 DB에 저장하는 그 이상의 용도로는 사용하지 않고 핵심기능인 채점 서버에서 그 정보를 토대로 채점하기 위한 스키마입니다.

그 다음 서버를 처음 가동하였을 때, 디비에 쌓인 채점 정보들을 채점큐로 옮겨담는 작업을 할것입니다.

```javascript
const model = require('../model/model');
const syncQueue = require('sync-queue');
const Queue = new syncQueue();

const pushQueue = require('../pushQueue');

model.judgeQueue.find()
    .where('server_number').equals(config.serverNumber)
    .then(result=>{
        for(let i = 0; i < result.length; i++) {
            pushQueue(Queue, result[i]);
        }
    }).catch(err => {
        console.log("database Error!");
        console.log(err);
});


/**
 * Module dependencies.
 */

const app = require('../app');
const debug = require('debug')('judge-server:server');
const http = require('http');

/**
 * Get port from environment and store in Express.
 */

const port = normalizePort(process.env.PORT || '3004');
app.set('port', port);
app.set('judgeQueue', Queue);
```

Queue 에 대한 정보를 app에 넣어줌으로서 라우터에서 Queue를 활용할 수 있게 할 것 입니다.
그럼 이제 [pushQueue.js](https://github.com/Byeong-Chan/Judge-Server/blob/master/judge_server/pushQueue.js) 에 대해 살펴볼까요?

## pushQueue.js
 이제 볼것은 본격적으로 사용하는 pushQueue.js 에 관한 것 입니다. 이 작업이 매우 복잡하므로 천천히 뜯어볼것입니다.

```javascript
const fs = require('fs');
const model = require('./model/model');
const execSync = require('child_process').execSync;
const path = require('path');
```

일단 사용할 패키지 먼저 가져옵니다. 파일 시스템을 자주 활용할 예정이므로 fs를 가져오고, 코드를 실행하기 위해 child_process 패키지를 여기서 사용하게 됩니다.

```javascript
const testerDir = path.join(__dirname, "../tester");
const dockerDir = "/home"; // TODO: 나중에 서버로 옮겼을 때 제거해야합니다.

//TODO: java python compile 구현하세요.

//TODO: 서버를 AWS로 옮겼을 때 script를 docker 없이 사용하는것으로 고쳐주세요.
const Compiler = {
    'c': function(user_code) {
        fs.writeFileSync('../tester/test.c', user_code, 'utf8');
        const script = 'docker run --rm -v' + ' ' + testerDir + ':' + dockerDir + ' ' +
            'gcc gcc /home/test.c -o /home/test.o -O2 -Wall -lm -static -std=c99 -DONLINE_JUDGE -DBOJ';

        let ret = { success: false, stdout: ''};
        try {
            const stdout = execSync(script).toString();
            ret.success = true;
            ret.stdout = stdout;
        } catch(exception) {
            const stdout = exception.stderr.toString();
            ret.success = false;
            ret.stdout = stdout;
        }

        return ret;
    },
    'cpp': function(user_code) {
        fs.writeFileSync('../tester/test.cc', user_code, 'utf8');
        const script = 'docker run --rm -v' + ' ' + testerDir + ':' + dockerDir + ' ' +
            'gcc g++ /home/test.cc -o /home/test.o -O2 -Wall -lm -static -std=gnu++17 -DONLINE_JUDGE -DBOJ';

        let ret = { success: false, stdout: '' };
        try {
            const stdout = execSync(script).toString();
            ret.success = true;
            ret.stdout = stdout;
        } catch(exception) {
            const stdout = exception.stderr.toString();
            ret.success = false;
            ret.stdout = stdout;
        }

        return ret;

    },
    'java': function(user_code) {

    },
    'python': function(user_code) {

    }
};
```

컴파일러 부분입니다. 채점 서버에서 어떤 언어로 채점할것인가에 따라 어떻게 채점할 것인가에 대해 정하고 컴파일 하는 부분입니다. try catch는 컴파일 에러 발생시, 컴파일 에러를 캐치하기 위해 만들었습니다. 중복적인 코드가 존재하는데, 이는 나중에 고칠 예정이며, 이 포스트를 보는 여러분이 한번 아름답게 고쳐보길 바랍니다. 그리고 저는 우선 c, c++ 만 채점할 수 있게 해뒀습니다.

```javascript
const seccomp_rule = function(lang) {
    switch(lang) {
        case 'c':
            return 'c_cpp';
        case 'cpp':
            return 'c_cpp';
        default:
            return 'general';
    }
};
```

이것은 언어에 따른 seccomp_rule을 결정하기 위한 함수입니다.

```javascript
const state_set = function(state_number, pending_number, error_message) {
    model.judge.where('pending_number').equals(pending_number)
        .update({$set: {state: state_number, ErrorMessage: error_message}}).then(result => {
            //TODO: 로깅할것인가?
            console.log(result);
        return model.judgeQueue.where('pending_number').equals(pending_number)
            .deleteOne();
    }).then(result => {
        //TODO: 로깅할것인가?
        console.log(result);
    }).catch(err => {
        //TODO: database error도 로깅할것인가?
        console.log(err);
    });
};
```

이 함수는 채점 결과에 대해 맞았습니다. 틀렸습니다. 컴파일에러 등등.. 을 처리하기 위해 만든 function 입니다. Mongoose Query Builder 에 대해서 공부해두시면 이 코드에 대한 이해가 쉽게 될 것 입니다.

```javascript
const pushQueue = function(Queue, judgeObj) {
    let user_code = "";
    let user_lang = undefined;
    let error_message = "";
    const max_process_number = 200; // TODO: 나중에 고쳐주세요.
    const max_output_size = 2097152; // TODO: 나중에 고쳐주세요.
    Queue.place(function() {
        model.judge.findOne()
            .where('pending_number').equals(judgeObj.pending_number)
            .then(result => {
                if(!result) throw new Error('none-pending');
                user_code = result.code;
                user_lang = result.language;
                return model.problem.findOne()
                    .where('problem_number').equals(result.problem_number);
            }).then(result => {
                if(!result) throw new Error('none-problem');
                if(Compiler[user_lang] === undefined) throw new Error('none-language');

                const lang = user_lang;

                const errCheck = Compiler[lang](user_code);

                error_message = errCheck.stdout;
                if(!errCheck.success) throw new Error('Compile-Error\n' + errCheck.stdout);

                for(let i = 0; i < result.input_list.length; i++) {
                    fs.writeFileSync('../tester/input.txt', result.input_list[i].txt, 'utf8');

                    //TODO: 서버를 AWS로 옮겼을 때 script를 docker 없이 사용하는것으로 고쳐주세요.
                    const script = 'docker run --rm -v' + ' ' + testerDir + ':' + dockerDir + ' ' +
                        lang + ' ' + dockerDir + '/libjudger.so' + ' ' + '--max_cpu_time=' + result.time_limit + ' ' +
                        '--max_real_time=' + (result.time_limit * 5) + ' ' + '--max_memory=' + result.memory_limit + ' ' +
                        '--max_process_number=' + max_process_number + ' ' + '--max_output_size=' + max_output_size + ' ' +
                        '--exe_path=' + '"' + dockerDir + '/test.o"' + ' ' + '--input_path=' + '"' + dockerDir + '/input.txt"' + ' ' +
                        '--output_path=' + '"' + dockerDir + '/output.txt"' + ' ' + '--error_path=' + '"' + dockerDir + '/error.txt"' + ' ' + '--uid=0' + ' ' +
                        '--gid=0' + ' ' + '--seccomp_rule_name=' + seccomp_rule(lang);

                    const stdout = execSync(script).toString();
                    const status = JSON.parse(stdout);
```

실행하기 바로 직전까지의 과정입니다. 설명이 부족했던것 같은데

```bash
$ docker run --rm -v /Users/SeoByeongChan/Desktop/capd/capstone-project/Judge-Server/tester:/home gcc gcc /home/test.c -o /home/test.o -O2 -Wall -lm -static -std=c99 -DONLINE_JUDGE -DBOJ
$ docker run --rm -v /Users/SeoByeongChan/Desktop/capd/capstone-project/Judge-Server/tester:/home gcc /home/libjudger.so --max_cpu_time=1000 --max_real_time=2000 --max_memory=536870912 --max_process_number=200 --max_output_size=16384 --exe_path="/home/test.o" --input_path="/home/input.txt" --output_path="/home/output.txt" --error_path="/home/error.txt" --uid=0 --gid=0 --seccomp_rule_name="c_cpp"
```

를 실행하기 위한 압축 코드입니다. 제가 실수로 원본 깃에서는 dockerDir 에 대한 정의를 안해놓고 /home/을 바로 써버렸는데 나중에 수정할 예정입니다. 위에 js 코드와 Compiler 객체는 String으로 변환한 뒤 child_process 로 실행하는 것 입니다.

이제 나머지 부분입니다.

```javascript
                    // TODO: 개별 채점 해야하는가? 백준식 채점이 좋은가? 개별 채점 구현하려면 스키마를 바꿔야합니다.
                    if(status.result === 4) {
                        if(status.signal === 25) throw new Error('Presentation-Error');
                        else throw new Error('Runtime-Error');
                    }
                    else if(status.result === 1 || status.result === 2) throw new Error('Timelimit-Error');
                    else if(status.result === 3) throw new Error('Memorylimit-Error');
                    else if(status.result !== 0) throw new Error('Server-Error');

                    const user_output = fs.readFileSync('../tester/output.txt', 'utf8').toString();
                    const system_output = result.output_list[i].txt;

                    const user_output_cmp = user_output.split('\n');
                    const system_output_cmp = system_output.split('\n');

                    while(user_output_cmp[user_output_cmp.length - 1].trimEnd() === '') {
                        user_output_cmp.splice(user_output_cmp.length - 1);
                    }

                    while(system_output_cmp[system_output_cmp.length - 1].trimEnd() === '') {
                        system_output_cmp.splice(system_output_cmp.length - 1);
                    }

                    if(system_output_cmp.length !== user_output_cmp.length) throw new Error('Wrong-Answer');

                    for(let j = 0; j < system_output_cmp.length; j++) {
                        if(system_output_cmp[j].trimEnd() !== user_output_cmp[j].trimEnd()) throw new Error('Wrong-Answer');
                    }
                }
                console.log(judgeObj.pending_number);
                state_set(2, judgeObj.pending_number, error_message);
                Queue.next();
            }).catch(err => {
                if(typeof err.message === 'undefined') {
                    console.log('Hello undefined');
                    // TODO: 채점 sever 문제는 중대하므로 로깅으로 나중에 꼭 고칠 것
                    console.log(err);
                }
                else if(typeof err.message !== 'string') {
                    console.log(typeof err.message);
                    // TODO: 채점 sever 문제는 중대하므로 로깅으로 나중에 꼭 고칠 것
                    console.log(err);
                }
                else if(err.message.split('\n')[0] === 'Compile-Error') {
                    /***
                     * 컴파일 에러
                     */
                    state_set(8, judgeObj.pending_number, error_message);
                }
                else if(err.message === 'Wrong-Answer') {
                    /***
                     * 오답
                     */
                    state_set(3, judgeObj.pending_number, error_message);
                }
                else if(err.message === 'Timelimit-Error') {
                    /***
                     * 시간 초과
                     */
                    state_set(4, judgeObj.pending_number, error_message);
                }
                else if(err.message === 'Memorylimit-Error') {
                    /***
                     * 메모리 초과
                     */
                    state_set(5, judgeObj.pending_number, error_message);
                }
                else if(err.message === 'Runtime-Error') {
                    /***
                     * 런타임 에러
                     */
                    state_set(6, judgeObj.pending_number, error_message);
                }
                else if(err.message === 'Presentation-Error') {
                    /***
                     * 출력초과
                     */
                    state_set(7, judgeObj.pending_number, error_message);
                }
                else {
                    //TODO: 채점 server 문제는 중대하므로 로깅으로 나중에 꼭 고칠 것
                    console.log(err);
                }
                Queue.next();
        });
    });
};

module.exports = pushQueue;
```

trimEnd 로 뒤에 붙은 공백만 제거하고 \n으로 split 하여 남는 빈문자열을 trim하는 작업까지 마칩니다.
이후 두개의 배열을 비교하여 일치하면 그 테스트 케이스에 대해 정답, 아니면 오답을 처리하게 됩니다.
또한 실행 과정에서 발생한 시간초과, 메모리 리밋등을 모두 핸들링 합니다.

이제 큐에 집어넣는 과정까지 모두 완료했습니다. 남은건 [app.js](https://github.com/Byeong-Chan/Judge-Server/blob/master/judge_server/app.js) 와 routes/[judgeQueue.js](https://github.com/Byeong-Chan/Judge-Server/blob/master/judge_server/routes/judgeQueue.js)에 관련된 내용인데, 이는 링크로 대체하겠습니다.


# 테스트
 포스트 맨과, mongodb 에서 제공하는 mongo를 이용하여 테스트용 채점 내역, 테스트용 문제, 테스트용 채점 큐를 만들어서 테스트 해본 결과는 다음과 같습니다. npm start 실행중에 정상적으로 동작했으며, mongo 로 출력해본 결과는 다음과 같습니다.

![사진2](/assets/images/Judge-Server-shjgkwo/test1.png)

 맞는 경우에 대한 출력이며 컴파일 에러에 대한 출력은 다음과 같습니다.

![사진3](/assets/images/Judge-Server-shjgkwo/test2.png)

 컴파일 에러에 대한 출력은 다음과 같습니다. 그 밖에 다양한 테스트가 있었지만 일단은 여기 까지만 하고 나머지는 직접 구현해보고자 하는 분들의 숙제로 남겨두겠습니다.

# 후기
 저도 많이 미숙하지만, 완전 초보인 분들에게 어느정도나마 길을 제시할 수 있는 계기가 되었으면 좋겠습니다. 너무 이론보다는 구현 위주로 설명한 것 같습니다. 하지만 제 코드와 설명이 도움이 되었길 바라며 혹시 제 코드에 치명적인 결함이나 피드백을 하시고 싶으시다면 [이슈](https://github.com/Byeong-Chan/Judge-Server/issues)함에 언제든 알려주세요! 만약 해주시면 정말로 감사드립니다.

