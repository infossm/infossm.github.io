---
layout: post
title: sentiment analysis
date: 2020-07-19 16:03
author: cjmp1
tags: [AI, NLP]
---

# Sentiment Analysis

## Contents

1. 감정 분석이란
2. 자연어처리
3. 구현
4. 마치며
5. 참고자료

## 감정 분석이란

 감정 분석은 일종의 자연어처리에 속하는 인공지능 과제 중 하나입니다. 여기서 자연어처리는 간단하게 사람간의 발화, 또는 이로 이루어진 문장 그리고 지문의 감정을 분석하는 것을 의미합니다. 감정이란 사람이 생각하는 사고와 그에 따른 의견에 의해 느끼고, 나타나는 것을 말합니다. 따라서 감정분석이라는 것은, 텍스트 또는 실제 발화에서 나타나는 그 사람의 태도, 의견, 성향을 분석할 수 있어야 합니다. 

 자연어처리중에서도 감정 분석은 상당히 어려운 분야로 취급됩니다. 이는 같은 의견이라도 사람의 주관에 따라 문장이 달라질 수 있으며, 그 주관이 주입된 작은 변화로 문장이 나타내는 감정이 달라질 수 있기 때문입니다. 따라서 감정분석의 경우 여러가지 감정 (슬픔, 두려움, 즐거움, 기쁨, 행복, 우울, 무서움) 등의 감정을 나누는 것을 시도하기 이전에, 단순히 긍정과 부정 두가지 감정을 분류하는 것을 최우선 과제로 삼았습니다.

 여따 많은 인공지능 분야들이 대부분의 과정 또는 모델들이 모두 논문으로 발표되어 있고, 가져다가 쓰면 되는 수준이지만, 상세히 자연어처리에 대한 내용과 실제 감정 분석이 이루어지는 방식까지  다루어 보도록 하겠습니다.

## 자연어처리

 그러면 자연어처리란 어떻게 이루어지는 것인지 알아보겠습니다. 자연어처리(Naturl Language Processing) 줄여서 NLP 라고 불립니다. 이는 앞에서 간단히 설명했듯이 사람의 음성 또는 문장 즉 다양한 언어현상에 대해서, 컴퓨터가 이를 이해하고 이를 이용해서 어떤 결과를 만들어내는 것을 의미합니다.  결과로는 질문에 대한 답변이 될수도 있고, 챗봇과 같은 대화상대를 해주는 인공지능이 목표가 될 수도 있습니다. 자연어처리와 이미지처리를 합치게 되면, 그림에 대한 질의에 대한 답변과 같은 것도 가능하게 됩니다. 그렇다면 좀 더 세부적으로 정확히 어떻게 컴퓨터가 이를 이해하고 결과를 만들어내는지 알아보겠습니다.

 여러가지 언어현상에 대한 데이터는 다양한 방식으로 존재합니다. 음성 신호로 시그널이 주어지는 경우도 있으며, 아니면 실제 문장 text 데이터로 주어질 수도 있고 그 문장들이 찍힌 이미지 데이터로 존재할 수도 있습니다. 이런 데이터들을 전처리과정을 통해 다듬어야 합니다. 

 이런 전처리과정은 현재로서 꽤나 정형화되어 있습니다. 물론 개인 입맛에 맞게 전처리 부분을 커스터마이징해서 사용하는 것은 중요하지만 기본적인 틀은 동일합니다. 가장 우선적으로 먼저 오류를 제거합니다. 오류에는 오탈자, 띄어쓰기 오류, 처리하고자 하는 언어가 아닌 특수 문자 또는 다른 언어 등을 모두 포함합니다. 그 후에는 문장을 형태소 단위로 나누게 됩니다. 형태소는 어절 보다 더 세부 단계로 최소의 의미를 가지는 단계입니다. 영어의 경우에는 단어가 의미의 최소단계가 될 것입니다. 형태소로 분리가 끝나면, 이제 품사를 태깅 해줍니다. 태깅은 tag 즉 같은 단어이더라도, 다른 형태로 쓰이는 경우가 존재합니다. 가장 대표적인 예로, (나는) 이라는 어절의 경우 나를 의미하는 나(대명사) + 는(조사) 의 경우와, 날다를 의미하는 날(동사) + 는(관형형어미) 처럼 두가지 형태가 존재할 수 있습니다. 따라서, 품사를 태깅해주는 작업은 매우 중요합니다. 품사 태깅 이전에 불용어 제거를 해주는 작업을 진행하기도 합니다. 불용어(stopwords)란 큰 의미가 없는 단어들을 말합니다. 이 경우 정리된 리스트가 있어서 패키지에서 불러와서 사용하는 방법도 있고 사용자가 직접 의미가 없는 데이터를 추가해주기도 합니다. 이 외에도 품사 태깅 이전에 어간 추출이라는 것을 할 수 있습니다. 영어를 예로 들면, play, playing, played 같이 여러 형태가 존재하는 것을 play 로 어간만 남기고 접사를 제거하는 방식을 말합니다. 이렇게 전처리 과정이 모두 끝났다면 다음 단계로 넘어가게 됩니다.

 다음 단계는 임베딩을 진행하는 것입니다. 현재 문장들은 모두 단어로 쪼개어져 품사가 태깅된 채로 존재합니다. 이런 단어들을 빈도순으로 정렬한 후, numbering을 통해 문장을 일종의 숫자 배열(벡터)로 만들게 됩니다. 여기서 에를 든 방식은 countvectorizer로 가장 기초적인 임베딩 기법입니다. 이렇게 임베딩된 숫자 벡터들이 이제 딥러닝 모델에 input으로 들어가게 됩니다. 딥러닝 모델로는 정말 많은 종류의 기법들이 개발이 되었습니다. 저는 이 글에서 가장 기초적인 CNN과, BERT 모델을 대표적으로 구현해보았습니다. input으로 들어간 벡터들은 연산을 통해 마지막 output 에서는 각 감정에 해당하는 확률값을 내보내게 되어서, 가장 높은 확률인 감정을 채택하게 되는 것입니다. 그럼 이제 직접 코드와 함께 살펴보도록 하겠습니다.

## 구현

 데이터의 경우 twitter sentiment analysis 라는 유명한 데이터셋이 존재합니다. 그런데 이 데이터의 경우 많이 쓰였길래 Frineds 드라마 대본 데이터를 이용해 보았습니다. 

```python
import json
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import Text
import collections
from keras.layers.core import Dense, SpatialDropout1D 
from keras.layers.convolutional import Conv1D 
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences 
from keras.utils import np_utils 
from sklearn.model_selection import train_test_split
```

필요한 패키지를 로드합니다. json 파일로 이루어진 데이터를 읽기 위해 json, re 패키지의 경우 문자열에서 의미없는 특수문자등을 제거하기 위해 사용합니다. nltk패키지는 위에서 말했던 불용어 리스트, 품사태깅, 토큰화 등을 자동으로 해주는 패키지입니다. matplotlib 는 학습 경과를 그래프로 살펴보기 위해 필요한 패키지 입니다.  keras에서 각 필요한 모델들을 모두 불러옵니다. sklearn에서는 자동으로 train_test set을 분리해주는 패키지만 사용했습니다. 마지막으로 collections는 임베딩에 필요한 패키지 입니다. 

```python
nltk.download('stopwords')
stops = set(stopwords.words('english'))
stemmer = nltk.stem.SnowballStemmer('english')
```

 불용어 리스트를 미리 다운받아 놓은 후 stops에 저장해 놓습니다. 어간추출기 stemmer도 미리 정의를 해줍니다.

```python
with open('./Friends/friends_train.json') as json_file:
    json_train = json.load(json_file)
with open('./Friends/friends_test.json') as json_file:
    json_test = json.load(json_file)
with open('./Friends/friends_dev.json') as json_file:
    json_dev = json.load(json_file)

def cleaning(str):
    replaceAll = str
    only_english = re.sub('[^a-zA-Z]', ' ', replaceAll)
    no_capitals = only_english.lower().split()
    no_stops = [word for word in no_capitals if not word in stops]
    stemmer_words = [stemmer.stem(word) for word in no_stops]
    return ' '.join(stemmer_words)

i = 0
train_data=[]
for rows in json_train:
    for row in rows:
        train_data.append([cleaning(row['utterance']), row['emotion']])
for rows in json_test:
    for row in rows:
        train_data.append([cleaning(row['utterance']), row['emotion']])
for rows in json_dev:
    for row in rows:
        train_data.append([cleaning(row['utterance']), row['emotion']])
```

 json.load를 사용해서 json 데이터를 읽습니다. json 데이터의 경우 계층적으로 이루어져 있기 때문에, 배열처럼 사용해 줄 수 있습니다. cleaning 이라는 전처리 함수를 선언했습니다. 문장에서 re.sub 를 이용해 영어 글자만을 남긴뒤, 모두 소문자로 통일해줍니다. 그 후 앞서 정의한 stops(불용어 리스트)를 이용해 불용어를 제거해주고, stemming(어간추출)을 진행하게 됩니다. 

```python
cnt = 0
tagged = []
counter = collections.Counter()
for d in train_data:
    cnt = cnt + 1
    if cnt % 1000 == 0:
        print(cnt)
    words = pos_tag(word_tokenize(d[0]))
    for t in words:
        word = "/".join(t)
        tagged.append(word)
        counter[word] += 1
```

이제 품사 태깅을 진행해줍니다. 같은 단어라도 품사에 따라 다른 뜻을 의미하므로, 단어에 품사를 붙인 것을 한 단어로 설정해서, counting을 하게 됩니다. 즉 "apple/noun" 이 한 단어가 되는 것입니다. 

```python
VOCAB_SIZE = 5000
word2index = collections.defaultdict(int)
for wid, word in enumerate(counter.most_common(VOCAB_SIZE)):
    word2index[word[0]] = wid + 1
vocab_sz = len(word2index) + 1
index2word = {v:k for k, v in word2index.items()}
```

 카운팅이 끝났으면 이제 임베딩을 해주어야 합니다. VOCAB_SIZE는 우리가 사용할 코퍼스 뭉치의 크기를 얘기합니다. 수많은 단어들이 나왔을 텐데, 그 중에 빈도수로 현재 정렬이 되어있는 상태입니다. 따라서 이 중에서 사용할 단어의 수를 정의해주어서, 그 단어들만 가지고 임베딩을 진행하게 됩니다. word2index에는 바로 그 숫자 값이 들어가게 되는 것입니다. index2word는 그 역을 정의해준 것 입니다.

```python
def labeltoint(str):
    return {'non-neutral': 0,
             'neutral': 1, 
             'joy': 2,
             'sadness': 3,
             'fear': 4,
             'anger': 5,
             'surprise': 6,
             'disgust': 7}[str]

xs, ys = [], []
cnt = 0
maxlen = 0
for d in train_data:
    cnt = cnt + 1
    ys.append(labeltoint(d[1]))
    if cnt % 1000 == 0:
        print(cnt)
    ang = pos_tag(word_tokenize(d[0]))
    words=[]
    for t in ang:
        words.append("/".join(t))
    if len(words) > maxlen: 
        maxlen = len(words)
    wids = [word2index[word] for word in words]
    xs.append(wids)
```

 드라마 대본 감정 분석의 경우 여러가지 라벨이 존재합니다. 단순히 긍정 부정이 아니라 좀데 새부적이죠, 이 라벨들을 int값으로 바궈주는 함수를 정의하고, 이제 train_data들을 돌며 위에서 정의한 word2index를 이용해서 숫자 벡터로 모두 바꿔주게 됩니다.

```python
X = pad_sequences(xs, maxlen=maxlen) 
Y = np_utils.to_categorical(ys)
 
EMBED_SIZE = 100 
NUM_FILTERS = 256 
NUM_WORDS = 3 
BATCH_SIZE = 64 
NUM_EPOCHS = 20

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=50)
model = Sequential() 
model.add(Embedding(vocab_sz, EMBED_SIZE, input_length=maxlen)) 
model.add(SpatialDropout1D(0.2)) 
#model.add(Conv1D(filters=NUM_FILTERS, kernel_size=NUM_WORDS, activation="relu")) 
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2)) 
model.add(GlobalMaxPooling1D()) 
model.add(Dense(8, activation="softmax")) 
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]) 

history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(x_test, y_test)) 
```

 X,Y 에 최종적인 train data와 train label을 넣어줍니다. 이는 벡터화된 값인 xs를 문장에 포함된 단어 수가 다 다르기 때문에, input 형태를 동일하게 맞춰주기 위해서, 모두 padding을 진행해주게 됩니다. 이 후, train_test_split을 사용해 test data를 나누고, 학습을 진행하는 부분입니다. # 부분을 LSTM 계층대신 사용하면 CNN이됩니다. 여러가지 모델과 하이퍼파라미터들을 시험해볼 수 있는 단계입니다.

```python
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(history.history['accuracy'], 'b', label='train acc')
acc_ax.plot(history.history['val_accuracy'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')

plt.show()
```

 학습의 경과를 출력해봅니다. history 에 모두 저장이 되어있습니다. keras 버전에 따라서 history에서 인자명이 val_acc 를 사용해야되는 등 조금 달라지는데 제 경우에는 val_accuracy를 사용하니 문제없었습니다.

```python
def inttolabel(idx):
    return {0:'non-neutral',
             1:'neutral', 
             2:'joy',
             3:'sadness',
             4:'fear',
             5:'anger',
             6:'surprise',
             7:'disgust'}[idx]

def predict(text): 
    aa = pos_tag(word_tokenize(text))
    pp = []
    for t in aa:
        pp.append("/".join(t))
    wids = [word2index[word] for word in pp]
    x_predict = pad_sequences([wids], maxlen=maxlen) 
    y_predict = model.predict(x_predict) 
    c = 0
    cnt = 0
    for y in y_predict[0]:
        if c < y:
            c = y
            ans = cnt
        cnt += 1
    ans = inttolabel(ans)
    return ans;
```

 이제 실제 데이터에 대해 예측을 진행하는 predict 함수를 만들어줍니다. predict에서 역시 위에서 만들어준 X처럼 품사 태깅을 해주고 벡터화시켜서 패딩까지 진행을 한후 모델이 집어넣게 됩니다. 모델의 output은 model.add(Dense(8, activation="softmax"))  에서 볼 수 있듯이, softmax로 8개의 cell에 각 감정에 대한 확률을 결과로 내놓게 됩니다. 따라서 이 중 max값에 해당하는 cell을 label 로 바꿔주는 inttolabel 함수를 이용해서 결과를 return하게 됩니다.

```python
ans = predict('i love it')
print(ans)
```

 위와 같이 사용해보면 됩니다.

이제 BERT 모델도 소개시켜드리려고 합니다.

전처리 과정은 위에있는 부분과 모두 동일합니다. 추가되는 것은 BertTokenizer가 인지할 수 있도록 문장을 특정형태로 변형해주어야 합니다.

```python
sentences = []
for i in train_data:
  sentences.append(i[0])
sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]
```

 문장앞에 CLS 태그를 붙이고 여러개의 문장이 있는 경우, SEP 태그로 각 문장을 나눠주어야 합니다. 이후 부분은 네이버 영화리뷰 감정분석을 Bert 모델을 이용해 진행한 블로그를 참조했습니다. 진행방식은 CNN과 거의 동일합니다.

```python
maxlen = 0
for i in sentences:
  if maxlen < len(i):
    maxlen = len(i)

MAX_LEN = maxlen + 1
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

attention_masks = []
for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)
```

 3번에서 구해진 token을 숫자 값으로 indexing 하고, maxlen을 이용해 padding 진행, 그리고 attention_masks를 설정 (데이터가 >0 인 단어부분에 attention을 주어서 학습 속도와 성능을 향상시킵니다.)

```python
def labeltoint(str):
    return {'non-neutral': 0,
             'neutral': 1, 
             'joy': 2,
             'sadness': 3,
             'fear': 4,
             'anger': 5,
             'surprise': 6,
             'disgust': 7}[str]

labels = []
for i in train_data:
  labels.append(labeltoint(i[1]))
```

 labeltoint 함수를 생성해 labels에 train_data의 label을 저장, 학습을 위해 torch tensor 형태로 모든 데이터들을 변환해줍니다.

이 과정을 test_data에 대해서도 동일하게 진행해줍니다.

```python
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=8)
model.cuda()

optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # 학습률
                  eps = 1e-8 # 0으로 나누는 것을 방지하기 위한 epsilon 값
                )
epochs = 10
total_steps = len(train_dataloader) * epochs
# 학습률을 조금씩 감소시키는 스케줄러 생성
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)
```

pre_trained된 모델을 불러오게 되며, optimizer를 설정하고, 학습을 진행하게 됩니다. 우리는 구분해야되는 감정이 8가지이므로, num_labels를 8로 수정시켜줍니다. 학습은 pytorch 에서 진행하는 학습방식 고대로 사용하면 됩니다.

epoch 별로 forward를 진행한 후, backward 진행 후, zero_grad를 통해 그래디언트 초기화를 시켜주는 것을 반복해주면 됩니다. 아래에 소개될 링크에 실제 문장을 test하는 코드도 주어져있는데, 

```python
def test_sentences(sentences):
    model.eval()
    inputs, masks = convert_input_data(sentences)

    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device)
            
    with torch.no_grad():     
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    return logits
```

 logits에 각 label 에 해당하는 값이 들어가게 되고, 이를 이용하면, 아래 코드를 이용해서 결과를 산출할 수 있습니다.

```python
def inttolabel(idx):
    return {0:'non-neutral',
             1:'neutral', 
             2:'joy',
             3:'sadness',
             4:'fear',
             5:'anger',
             6:'surprise',
             7:'disgust'}[idx]

logit = test_sentences(['i love it'])

print(inttolabel(np.argmax(logit)))
```

## 마치며

CNN, LSTM 모델에서 train data set 과 validation 을 각각 70% 30% 비율로 나누어 실험한 결과, epoch = 10 실행 결과 86% 정도의 training accuracy와 41% 정도의 validation accuracy를 보여주었습니다. 

 LSTM 모델에서는 epoch = 10 실행결과 85% 의 training accuracy와 45%의 validation accuracy를 보여주었고,

 BERT 모델의 사용결과 epoch = 10 실행 결과 46% 정확도를 보여주었습니다.

긍정/부정을 분석할 경우 정확도가 80%~90% 그 이상까지도 올라가게 되는데 여러가지 감정을 분리해야되다 보니 정확도가 많이 떨어졌던 것 같습니다. (데이터 양의 부재도 존재함)

성능이 긍/부정을 분류할때와 달리 너무 좋지 않았습니다. 학습량이 좀 더 많아진다면 좀 더 나은 성능을 보여줄 수 있을 것 같습니다. 자연어처리 분야를 처음 공부해 보았는데, 흥미있고 재미있는 분야였습니다. SNS 같은 곳에 올라오는 데이터들을 종합해서, 여러가지 분석을 진행해보는 것도 매우 재미있는 시도가 될 것 같다는 생각을 해보았습니다. 기회가 된다면 자연어처리 분야도 꼭 한 번 공부해보시면 좋을 것 같습니다.

## 참고자료

http://aidev.co.kr/chatbotdeeplearning/8709

https://github.com/cjmp1/nlp-sentiment-analysis
