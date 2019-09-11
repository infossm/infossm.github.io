---
layout: post
title: "websocket, socket.io를 이용한 양방향 통신"
date: 2019-08-17 21:00
author: zych1751
tags: [web, socket]
---



웹 브라우저에서 양방향 통신을 가능하게 해주는 websocket, socket.io에 대해 공부한 내용에 대하여 정리했습니다.  

  

## 웹 브라우저 에서의 양방향 통신

기존 웹페이지에서 사용하는 http 프로토콜은 요청/응답 패러다임이기에 클라이언트에서 요청을 보내야만 그에 대한 응답을 받습니다. 옛날 홈페이지의 경우에는 동적인 홈페이지가 거의 없어서 큰 문제가 없었으나 동적인 기능이 많이 요구가 됩니다. 그러다 보니 요청을 보내지 않아도 서버에서 클라이언트쪽으로 데이터를 보내야 하는 경우도 종종 생기고는 합니다.  

하지만 http 프로토콜로 통신하는 경우 특성상 연결이 유지되지 않아서 서버에서 먼저 요청을 보내는 것이 불가능합니다. 그래서 Polling, Long Polling, Streaming 방식을 이용하여 비슷한 효과를 구현할 수 있습니다.  

● Polling  

- 클라이언트에서 일정 주기마다 요청을 보내고 서버는 현재 상태를 바로 응답하는 방식입니다. 

  이 방식은 실시간으로 반영되는 것이 중요한 서비스에는 별로 좋지 않고 서버에서 변화가 없더라도 매 요청마다 응답을 내려주기 때문에 불필요한 트래픽이 발생하게 됩니다.  

  ![](/assets/images/websocket-socketio/polling.png)



● Long Polling  

* 클라이언트에서 요청을 보내고 서버에서는 이벤트가 발생했을 때 응답을 내려주고 클라이언트가 응답을 받았을때 다시 다음 응답을 기다리는 요청을 보내는 방식입니다.  

  실시간 반응이 가능하고 polling에 비해서 불필요한 트래픽은 유발하지는 않지만 오히려 이벤트가 잦다면 순간적으로 과부하가 걸리게 됩니다.

  ![](/assets/images/websocket-socketio/long-polling.png)

● Streaming

* 이벤트가 발생했을때 응답을 내려주는데 응답을 완료시키지 않고 계속 연결을 유지하는 방식입니다.  

  Long Polling에 비해 응답마다 다시 요청을 하지 않아도 되므로 효율적이지만, 연결 시간이 길어질수록 연결의 유효성 관리의 부담이 발생합니다.  

  ![](/assets/images/websocket-socketio/streaming.png) 



위와 같은 방법들을 이용하여 양방향 통신을 구현할 수도 있지만 이제는 HTML5에서 소켓 연결을 하는 websocket이 표준으로 등록되어 이를 이용하여 서버-클라이언트간의 양방향 통신을 구현할 수도 있게 되었습니다.



## websocket

websocket이란 웹 서버와 웹 브라우저간 실시간 양방향 통신환경을 제공해주는 실시간 통신 기술입니다.  위의 Polling 방식(요청-응답방식)과 다르게 양방향으로 원할때 요청을 보낼 수 있으며 stateless한 HTTP에 비해 오버헤드가 적으므로 유용하게 사용할 수 있습니다.

![](/assets/images/websocket-socketio/websocket.png)



구현 방식도 간단하게 아래와 같이 구현할 수 있습니다.  

  

● server(node.js)

* ws모듈을 이용하면 쉽게 구현할 수  있습니다.  

  ```javascript
  var WebSocket = require('ws').Server;
  var wss = new WebSocketServer({ port: 3000 });
  
  wss.on('connection', function(ws) {
    ws.on('message', function(message) {
      const sendData = {
        event: 'response',
        data: null
      };
  
      switch(message.event) {
        case 'open':
          console.log(message);
          break;
        case 'request':
          sendData.data = 'some data...';
          ws.send(JSON.stringify(sendData));
          break;
      }
    });
  });
  ```

  

  처음에 웹소켓이 연결될 때, 아래의 callback 함수가 실행됩니다.

  ```javascript
  wss.on('connection', callback)
  ```

  그리고 callback함수의 인자로 클라이언트와 연결된 소켓을 paramter로 넘겨 받는데 이 변수로 메시지를 받았을때의 이벤트를 설정하고(ws.on('message', callback)) 메시지를 보낼수도 있습니다.(ws.send(message))  

  여기서 websocket은 메시지를 보낼 때 문자열로 전송이 되므로 객체의 경우 stringify(서버) -> parse(클라이언트)를 해주어야 합니다.  

  ws모듈의 api문서는 [https://github.com/websockets/ws/blob/HEAD/doc/ws.md](https://github.com/websockets/ws/blob/HEAD/doc/ws.md)에서 확인가능하므로 궁금하신분은 찾아보시기 바랍니다.  

    

● client

* 브라우저에서 WebSocket이 기본 스펙에 있기 때문에 다른 모듈을 사용할 필요 없이 사용할 수 있습니다. (지원여부는 [https://crossbar.io/docs/Browser-Support/](https://crossbar.io/docs/Browser-Support/)에서 확인할 수 있습니다.)

  ```javascript
  var ws = new WebSocket('ws://localhost:3000');
  ws.onopen = (event) => {
    let sendData = { event: 'open' }
    ws.send(JSON.stringify(sendData));
  }
  ws.onmessage = (event) => {
    let recData = JSON.parse(event.data);
    switch (recData.event) {
      case 'response':
        console.log(recData.data);
        break;
    }
  }
  
  function myOnClick() {
    let sendData = {
      event: 'request',
      data: 'some data...'
    };
    ws.send(JSON.stringify(sendData));
  }
  ```

  서버와 비슷하게 연결되었을때 (ws.onopen), 메시지를 받았을때(ws.onmessage), 메시지를 보낼때(ws.send)를 각각 구현함으로써 메시지를 주고 받기가 가능해집니다.  

  마찬가지로 메시지를 받았을때, 보낼때 문자열로만 전송이 가능하므로 stringify, parse를 적절하게 사용하여 데이터를 이용해야 합니다.  

    

## socket.io

socket.io란 위에서 설명한 WebSocket과 같이 클라이언트와 서버의 양방향 통신을 가능하게 해주는 모듈입니다. socket.io는 통신을 시작할 때, 각 브라우저에 대해서 websocket, pooling, streaming, flash socket 등에서 가장 적절한 방법을 찾아 메시지를 보내줍니다. 그 덕분에 socket.io를 통해 개발을 하면 websocket이 지원이 되지 않는 브라우저에서도 메시지를 양방향으로 주고 받을 수 있습니다.

간단하게 채팅방처럼 메시지 데이터를 주고 받는 코드를 작성해봅시다.

● server

* ```javascript
  const server = require('http').createServer();
  const io = require('socket.io')(server);
  
  const clients = [];
  
  io.on('connection', (client) => {
    clients.push(client);
  
    client.on('message', (message) => {
      // handle message...
      client.forEach((c) => c.emit('message', message));
    });
  
    client.on('disconnect', () => {
      console.log('client disconnect...', client.id);
      // handle disconnect..
      clients.filter((c) => c.id !== client.id);
    });
  
    client.on('error', (err) => {
      console.log('received error from client:', client.id);
      // handle error..
    });
  });
  
  server.listen(5000, (err) => {
    if (err) throw err;
    console.log('listening on port 5000');
  });
  ```

  웹소켓에서 구현 했던 것과 비슷하게 구현한 것을 확인할 수 있습니다.  

  io.on('connection', callback) 과 같이 연결되었을 때의 이벤트를 정의하고 각 소켓에 대해서 on('event', callback) 과 같이 원하는 이벤트를 작성할 수 있습니다.  

  연결될 때의 소켓값을 모두 clients에 넣어두고 누군가 message이벤트를 발생시킨다면 각 클라이언트에게 emit함수를 통해 'message' 이벤트를 발생하게 합니다.  


● client

* ```javascript
  const io = require('socket.io-client');
  
  export default function() {
    const socket = io.connect('http://localhost:5000');
  
    function onMessageEvent(callback) {
      socket.on('message', (message) => {
        callback(message);
      });
    }
    
    function messageSend(message) {
      socket.emit('message', message);
    }
      
    function offMessageEvent() {
      socket.off('message');
    }
      
    return {
      onMessageEvent,
      offMessageEvent
    };
  }
  ```

  클라이언트 부분에서는 socket.io-client 모듈을 이용하여 쉽게 서버와 연동할 수 있습니다.  

  그리고 서버에서와 마찬가지로 on, off로 이벤트 리스너를 켜고 끌 수 있으며 emit함수로 서버의 이벤트를 발생시킬 수 있습니다.  

이렇게 구현을 하고나면 client에서 messageSend 함수를 호출하게 되면 접속되어있는 모든 클라이언트에게 메시지가 가게 됩니다. 여기서 다루지 않은 socket.io의 자세한 api가 궁금하다면 서버 api의 경우는 [https://socket.io/docs/server-api/](https://socket.io/docs/server-api/), 클라이언트 api의 경우는 [https://socket.io/docs/client-api/](https://socket.io/docs/client-api/)에서 찾아보시기 바랍니다.  