# Bixby Studio

## Contents

1. 빅스비 스튜디오란?
2. 내가 만든 예제
3. 발전 방향
5. 참고

## 빅스비 스튜디오란?

 우선 빅스비는 삼성전자에서 개발된 음성인식 기반 개인 비서 어플리케이션으로 현재 스마트폰 외에도 여러가지 기기(대부분의 삼성 디바이스)에서 사용되고 있다.  보통 빅스비는 현재 삼성 페이와 연결되어 쇼핑으로 자주 활용된다. 빅스비는 어떤 발화가 주어지면, 그 발화를 해결할 수 있는 캡슐을 검색하게 된다. 예를들어 어떤 장소에 대해 질문을 하는 발화가 주어졌다면, 장소를 찾는 캡슐을 검색하고 그 캡슐안으로 이동해서, 여러 지정된 액션들을 수행하게 된다. 예전에 빅스비가 나왔던 초창기에는 유용하게 사용 가능한 기능들이 별로 없었고 인지도가 많이 적었던 편에 속하지만 현재로서는 성장 가능성이 매우 높다고 생각한다. 

 빅스비 스튜디오는 이런 빅스비의 캡슐들을 개발자들이 만들어 볼 수 있는 공간이다. 앞서 말한 장소 검색 캡슐과 같이, 티켓 구매라는 캡슐을 만들어 줄수도 있다. 티켓구매 캡슐이 완성됬을 경우, 사용자가 "~~ 티켓 예매해줘" 같은 발화를 입력했을 때, 이 발화를 인공지능이 인식해서 티켓구매 캡슐로 이어주게 되고, 티켓 구매 캡슐에서는 액션을 수행하게 된다. 이 액션은 티켓 예매에 대한 구체적인 정보(시간, 자리 위치) 같은 것을 물어볼 수도 있고, 해당 티켓 구매 사이트에 단순히 접속시켜주는 액션을 수행할 수도 있다. 이런 액션들을 지정해주고 여러가지 종류로 바뀔 수 있는 발화들을 학습시킬 수 있는 공간이 빅스비 스튜디오이다. 빅스비 스튜디오는 계속 업데이트 중이며, 이는 아래 제공할 빅스비 디벨로퍼 사이트에서 업데이트 버전을 확인/ 다운로드 해볼 수 있다.

## 내가 만든 예제

 우선 아이디어의 발상은 다음과 같다. 술을 늦게까지 마시다 보면 오늘 지하철 또는 버스의 막차 시간이 궁금할 때가 있다. 만약 빅스비를 이용해서 이런 간단한 발화에 대한 대처가 가능하다면 어떨까 생각해 보았다. 예를들어 xxx역 막차 시간 알려줘 또는 다음 열차 탈 수 있어? 같이 응용된 발화까지 응답이 가능하다면 정말 편리할 거 같고 20~30대 사용자들이 자주 사용하게 될거같다고 생각했다. 따라서 이를 한 번 구현해보려고 시도했다. 이런 작업들을 수행하기 위해서는 여러 API들을 활용해주어야 한다. 찾아본 결과 길찾기에 대한 정보를 받아올 수 있는 api 도 존재했고, 지하철의 막차시간 그리고 지하철 역까지의 거리 같은 정보를 받아 올 수 있는 api 도 존재했다. 

 다음은 빅스비 스튜디오의 구조를 배워야 했다. 빅스비 스튜디오는 앞서 말했듯이 해당 하는 발화가 주어질 경우 연결될 캡슐을 만들어주는 과정이다. 빅스비 스튜디오만의 확장자인 bxb 파일이 사용되고, javascript 파일들을 이용해 실제 api를 수행한다. 빅스비 스튜디오에 들어가게 되면 크게 code, model, resources 로 이루어져 있다. code는 말 그대로 api 연결 또는 여러 자바 스크립트가 들어가게 된다. model의 경우 이 캡슐에서 사용하게될 action과 concept을 지정해주어야 한다. 예를들어 "3+5 는 몇이야?" 같은 사칙연산을 해결해주는 사칙연산캡슐을 제작하게 되었다고 생각해보자. 그러면 연산자에 해당하는 +를 받아올 operator라는 concept과 상수 두개 a,b 가 미리 concept에 선언되어야 하고, action은 이 concept들을 이용해 어떤 action을 수행할지 지정을 해주는 거라고 생각하면된다. code에 새로운 자바스크립트 파일을 만들어 a,b 그리고 operator를 받아 사칙연산을 한 후 return하는 코드를 작성한 후, action에서는 a,b,operator 그리고 코드파일 4가지를 연결해주는 코드를 작성하게 된다. 즉 설계도면을 만들어주는 과정이라고 생각할 수 있다.

resources에는 view 파트와, training 파트가 존재한다. view파트는 이렇게 잘 만들어진 캡슐을 실제 빅스비 화면에 띄워주어야 하는데 이를 지정해준다. 즉 html 같은 느낌이다. view 파트에 대한 공부가 따로 필요하며, 여러가지 view 방식을 연결해서 사용자에게 좀 더 잘 이해될 수 있는 view를 만드는 것이 중요하다.

 training 파트의 경우 만들어진 캡슐이 잘 작동하는지 확인하는 과정을 포함하는 동시에, 빅스비에게 어떤 발화가 주어질 때, 이 캡슐에서 이렇게 작동해라 하는 가이드라인을 제공함으로서, 학습을 시키는 과정이다. 예를 들어 위의 사칙연산 발화가 주어질 경우, 빅스비는 a 가 3이고 b가 5이며, operator가 +라는 것을 인식해야 한다. 이와 같은 것을 지정하고 학습시켜 주는 단계가 training 파트이다. 아무리 코딩을 완벽히 한 캡슐이더라도, training 과정이 부족할 경우 완성된 캡슐이라고 볼 수 없다.

우선 내가 만든 캡슐의 경우 8개의 action을 만들어 주었다. concept에 대한 설명은 각 단어에서 손쉽게 이해가 가능하기 때문에 따로 설명을 하지 않겠다.

`FindBusStation.model.bxb`

```python
action (FindBusStation) {
  type(Search)
  description (이름으로 역을 찾는 함수)
  collect {
    input (stationName) {
      type (StationName)
      min (Required) max (One)
    }
  }
  output (Station)
}
```

stationName 을 입력으로 받아 그 해당하는 역을 return하는 단계이다. 역(Station)이라는 concept으로 주어진 정보에 대해 통일화 시켜야 하므로 꼭 필요한 action이라고 할 수 있다.

`FindNearTrainStation.model.bxb`

```python
action (FindNearTrainStation) {
  description (현재 좌표로 인접 역 이름 찾기)
  type(Search)
  collect {
    input (myPoint) {
      type (MyPoint)
      min (Required) max (One)
      default-init {
        // Note: To enable current location access, in your capsule.bxb,
        // add 'user-profile-access' to capsule.permissions
        intent {
          goal: MyPoint
          route: geo.CurrentLocation
        }
      }
    }
  }
  output (Station){
    throws{
      error(ServerProblem){
        on-catch{
          halt{
            dialog{
              template-macro(unstable)
            }
          }
        }
      }
      unknown-error{
        on-catch{
          halt{
            dialog{
              template-macro(no_result)
            }
          }
        }
      }
    }
  }
}
```

현재 좌표 mypoint를 입력으로 받는다. 이는 geo라는 것을 이용하면 되고, 이를 이용해 Station을 return하는 action이 된다.

`FindRoute.model.bxb`

```python
action (FindRoute) {
  type(Search)
  description (경로 찾아주기)
  collect {
    input (sourceName) {
      type (SourceName)
      min (Optional) max (One)
    }
    input (destinationName){
      type(DestinationName)
      min(Required) max(One)
    }
    input (myPoint) {
      type (MyPoint)
      min (Required) max (One)
      default-init {
        // Note: To enable current location access, in your capsule.bxb,
        // add 'user-profile-access' to capsule.permissions
        intent {
          goal: MyPoint
          route: geo.CurrentLocation
        }
      }
    }
  }
  output (Routes)
}
```

경로를 찾아주는 action이다. sourcename 시작위치(주어질 경우), destinationName 도착위치(반드시 필요), 내 현재 위치 3가지 입력 정보를 바탕으로, Routes concept을 return하게 된다.

`FindTrainStation.model.bxb`

이는 FindBusStation과 동일한 역할을 수행하게 된다.

`GetTime.model.bxb`

```python
action (GetTime) {
  type(Search)
  description (막차,첫차,다음열차)
  collect {
    input (source) {
      type (Source)
      min (Required) max (One)
    }
    input (trainType){
      type(TrainType)
      min (Required) max (One)
    }
  }
  output (Times)
}
```

이는 특정 열차에 대한 막차 첫차 그리고 현재 시간을 기준으로 다음열차 3가지 정보를 return 해주는 action을 지정해 주었다.

`GetTimeTable.model.bxb`

```python
action (GetTimeTable) {
  type(Search)
  description (__DESCRIPTION__)
  collect {
    input (stationName) {
      type (StationName)
      min (Optional) max (One)
    }
  }
  output (TimeTables)
}
```

역 이름이 주어졌을 때 해당하는 역에 대한 timetable을 받아오는 action이다.

`LastFirstNextTrain.model.bxb`

```python
action (LastFirstNextTrain) {
  type(Search)
  description ()
  collect {
    input (stationName) {
      type (StationName)
      min (Required) max (One)
    }
    input (lineNumber){
      type (LineNumber)
      min(Optional) max(One)
    }
    input(trainType){
      type(TrainType)
      min(Required) max(One)
    }
  }
  output (LFNResult)
}
```

위의 GetTime action과 동일하다고 보여지지만 GetTime은 단순히 시간을 값으로 받아오는 작업이라면 이는 그해당하는 열차 자체의 정보를 받아오게 된다.

이렇게 concept과 action을 지정해주고 난 후, 이제 실제 작업을 수행하는 자바 스크립트 파일을 작성하게 된다.

여러 자바스크립트 파일이 존재하는데 그중 FindLFNTrain.js 파일을 살펴 보겠다.

```javascript
var http = require('http')
var console = require('console')
var config = require('config')
module.exports.function = function getTrainTable (stationName, lineNumber, trainType) {
  stationName = stationName.slice(0, stationName.length-1);
  var encodedUrl = encodeURI(config.get('api.timetable') + secret.get('stationKey') + config.get('get.timetable') + stationName);
  var response = http.getUrl(encodedUrl, { format: 'json'});
  var f_nums = response['SearchSTNBySubwayLineInfo']['row'];
  //console.log(f_nums);
  var day = new Date().getDay();
  var dayCode = 1;
  if(day == 0){
    dayCode = 3;
  } else if (day == 6){
    dayCode = 2;
  }
  var tables = [];
  var direct = [];
  for (var i = 0; i < f_nums.length; i++){
    var code = f_nums[i]['STATION_CD'];
    var encodedUrl = encodeURI('http://openAPI.seoul.go.kr:8088/'+secret.get('stationKey')+'/json/SearchSTNTimeTableByIDService/1/300/' + code + '/' + dayCode + '/2');
    var response = http.getUrl(encodedUrl, { format: 'json'});
    var times = [];
    for (var j = 0; j < response['SearchSTNTimeTableByIDService']['row'].length; j++){
      times[j] = response['SearchSTNTimeTableByIDService']['row'][j].ARRIVETIME;
    }
    tables[i] = {times:times, lineNumber:response['SearchSTNTimeTableByIDService']['row'][0].LINE_NUM};
    direct[i] = response['SearchSTNTimeTableByIDService']['row'][0].SUBWAYENAME;
  }
  //console.log(tables);
  var la = 'ang';
  var fi = 'ang';
  var res = [];
  for(var i = 0; i < tables.length; i++){
    if(tables[i].lineNumber == lineNumber || lineNumber == null){
      for(var j = 0; j < tables[i].times.length; j++){
        if(fi == 'ang') fi = tables[i].times[j];
        else if(fi > tables[i].times[j] && tables[i].times[j] != '00:00:00') fi = tables[i].times[j];
        if(la == 'ang') la = tables[i].times[j];
        else if(la < tables[i].times[j]) la = tables[i].times[j];
      }
    
    if(trainType == "막차"){
      res[i] = {
        times:la,
        lineNumber:tables[i].lineNumber,
        trainDirection:direct[i]
      }
    }
    else if(trainType == "첫차"){
      res[i] = {
        times:fi,
        lineNumber:tables[i].lineNumber,
        trainDirection:direct[i]
      }
    }
  }
  }
  //console.log(res);
  
  var ttables = [];
  var ddirect = [];
  for (var i = 0; i < f_nums.length; i++){
    var ccode = f_nums[i]['STATION_CD'];
    var eencodedUrl = encodeURI('http://openAPI.seoul.go.kr:8088/'+secret.get('stationKey')+'/json/SearchSTNTimeTableByIDService/1/300/' + ccode + '/' + dayCode + '/1');
    var rresponse = http.getUrl(eencodedUrl, { format: 'json'});
    var ttimes = [];
    for (var j = 0; j < rresponse['SearchSTNTimeTableByIDService']['row'].length; j++){
      ttimes[j] = rresponse['SearchSTNTimeTableByIDService']['row'][j].ARRIVETIME;
    }
    ttables[i] = {times:ttimes, lineNumber:rresponse['SearchSTNTimeTableByIDService']['row'][0].LINE_NUM};
    ddirect[i] = rresponse['SearchSTNTimeTableByIDService']['row'][0].SUBWAYENAME;
  }
  var lla = 'ang';
  var ffi = 'ang';
  var k = res.length;
  for(var i = 0; i < ttables.length; i++){
    console.log(ttables[i]);
    if(ttables[i].lineNumber == lineNumber || lineNumber == null){
      for(var j = 0; j < ttables[i].times.length; j++){
        //console.log(ffi);
        //console.log(lla);
        if(ffi == 'ang') ffi = ttables[i].times[j];
        else if(ffi > ttables[i].times[j] && ttables[i].times[j] != '00:00:00') ffi = ttables[i].times[j];
        if(lla == 'ang') lla = ttables[i].times[j];
        else if(lla < ttables[i].times[j]) lla = ttables[i].times[j];
      }
    
    if(trainType == "막차"){
      res[k + i] = {
        times:lla,
        lineNumber:ttables[i].lineNumber,
        trainDirection:ddirect[i]
      }
    }
    else if(trainType == "첫차"){
      res[k + i] = {
        times:ffi,
        lineNumber:ttables[i].lineNumber,
        trainDirection:ddirect[i]
      }
    }
    }
  }
  console.log(res);
  var ho ="";
  if(trainType == "첫차"){
   ho = "첫차" ;
  }else{ 
    ho = "막차";
  }
  
  return{
    LFNtrainTable:res,
    stationName:stationName,
    realTrainType:ho
  };
}
```

공공데이터포털의 오픈api를 이용했다. 역이름과 해당 linenumber 그리고 열차의 type 3가지 변수를 다룰 예정이다. stationname을 이용해서, 그 해당 역에서 지나는 linenumber에 해당하는 열차의 타임테이블을 받아오게 된다. 그 후 에는 요청하는 traintype을 이용해서, 타임테이블에서 원하는 값을 지정해주고 return해주면 원하는 작업이 이루어진다.

이번에는 매우 간단한 `FindTrainStation.js` 파일을 살펴보자.

```javascript
var http = require('http')
var console = require('console')
var config = require('config')
var fail = require('fail')
module.exports.function = function findTrainStation (stationName) {
  stationName = stationName.slice(0, stationName.length-1);
  stationName = setTrainStationName(stationName)
  var encodedUrl = encodeURI(config.get('api.subway.url') + secret.get('stationKey') + config.get('find.station') + stationName);
  var response = http.getUrl(encodedUrl, { format: 'json'}); 
  if(response.status == 404 || response.status == 500 || response.status == 502 || response == 504){
    throw fail.checkedError("API Server Error", "ServerProblem");
  }
  var trainList = response.realtimeArrivalList;
  var train = [];
  for (var index = 0; index < trainList.length; index++) {
    console.log(trainList[index]);
    train[index] = {
      remainTime:Math.floor(trainList[index]['barvlDt']/60),
      state:trainList[index]['arvlCd'], 
      direction:trainList[index]['bstatnNm'],
      lineNumber:setTrainLineNumber(String(trainList[index]['subwayList']))
    };
  }
  return {
    stationName:stationName,
    transport:train
  }
}
function setTrainStationName(TrainStationName){
  let result=String(TrainStationName);
  const abbrevStationName = require('./lib/TrainStationName.js');
  for(let i = 0; i<abbrevStationName[0].length; i+=1){
    if(abbrevStationName[0][i] == result){
      result = abbrevStationName[1][i];
      break;
    }
  }
  return result;
}
//api로부터 받은 subwayList값으로부터 lineNumber를 찾는 함수
function setTrainLineNumber(subwayList){
  let result="정보없음";
  const trainLineNumbers = require('./lib/TrainLineNumber.js');
  result = trainLineNumbers[subwayList];
  console.log("LineName: "+result);
  return result;
}
```

이는 역이름을 입력으로 받아서, 그에 해당하는 station 값을 return하는 함수, 그리고 2개의 추가적인 함수를 포함하고 있다. station concept은 역이름과 해당하는 운송수단의 종류 2가지를 구성요소로 갖는다. 

이 외에도 `FindBusStation.js` `FindNearTrainStation.js` `FindRoute.js` `GetTimeTable.js` `GetTrainTable.js` 파일을 만들어 주었다.

## 발전 방향

 직접 실습을 해보면서, 아직 빅스비 스튜디오의 기능이 좀 제한적이고 원하는 기능을 손 쉽게 풀어내지 못하는 아쉬움이 있었다. 또한 원초적인 음성인식에 대한 부분도 약간 개선이 필요하다고 느꼈다.하지만 대체적으로 UI가 정말 배우기 쉬웠고 빅스비에 새로운 기능이 쉽게 추가된다는 것이 놀라웠고 재미있었다. 많이 개선이되어서, 정말 많은 개발자들이 빅스비 스튜디오를 이용해서 빅스비가 할 수있는 일들을 게속 만들어 낸다면 정말 무궁무진한 어플이 될 수 있다고 생각한다.

## 참고

[빅스비 디벨로퍼 사이트]: https://bixby.developer.samsung.com/

