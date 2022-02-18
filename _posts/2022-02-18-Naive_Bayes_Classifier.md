---

layout: post

title: "Analysis of the Naive Bayes Classifier with Spam Filtering and MNIST datasets"

author: antemrdm

date: 2022-02-18

---

# Abstract

Naive Bayes Classifier를 이용하여 MNIST dataset와 email dataset을 학습하고, k-fold cross validation을 이용하여 학습된 모델의 분류성능을 분석해본다.

## Keywords

Naive Bayes Classifier, K-Fold Cross Validation, Spam Filtering, MNIST

# Introduction

## Bayes’ theorem

베이즈 정리는 사전 확률로부터 사후 확률을 계산하는 조건부 확률에 대한 정리이다. 사건 A와 B가 있고, 사건 A가 일어났을 때 사건 B가 일어날 조건부 확률과 A가 일어날 확률, B가 일어날 확률만을 계산할 수 있을 때, 사건 B가 일어났을 때 사건 A가 일어날 조건부 확률은 다음과 같이 구할 수 있다.

$$
P(A|B) = {P(B\cap A)\over P(B)} = {P(A)P(B|A)\over P(B)}
$$

## Naive Bayes Classifier

Naive Bayes Classifier는 간단한 classifier의 일종으로 주로 다른 복잡한 classifier의 대조군으로 사용된다. 베이즈 정리에 기반을 둔 알고리즘이기 때문에 개념상으로 단순한 조건부 확률 모델이다.

## K-Fold Cross Validation

먼저 K-Fold Cross Validation를 위한 fold들을 만들기 위해 스팸 이메일과 스팸이 아닌 이메일을 구분하고 각각을 k개로 나누어 합쳐준다. 이 과정은 균일한 fold를 생성하기 위해 필요한 과정이다. 이후 하나의 fold를 test dataset으로 사용하고 나머지 모든 fold들을 training dataset으로 사용하여 앞서 설계한 분류기에 적용하고 결과를 분석한다. 이러한 교차검증을 사용하는 이유는 데이터의 개수가 적은 dataset에 대하여 정확도를 높일 수 있기 때문이다. 하지만 fold의 개수만큼 알고리즘을 진행해야 하는 만큼 교차검증을 하지 않을 때보다 fold의 수 배만큼 오랜 시간이 걸린다.

## 분류성능평가지표

작성한 모델의 성능 향상을 위해 그 무엇보다 중요한 것이 해당 모델의 성능을 정확하게 평가하는 것이다. 기본적으로 실제 test dataset의 값과 모델이 산출한 값을 비교하여 다음과 같이 4가지의 값을 얻을 수 있다.

| model\real | True | False |
| --- | --- | --- |
| True | True Positive(TP) | False Positive(FP) |
| False | False Negative(FN) | True Negative(TN) |
- True Positive(TP) : 실제 True인 값을 True로 산출한 test case의 수
- False Positive(FP) : 실제 False인 값을 True로 산출한 test case의 수
- False Negative(FN) : 실제 True인 값을 False로 산출한 test case의 수
- True Negative(TN) : 실제 False인 값을 False로 산출한 test case의 수

일반적으로 모델의 성능을 평가할 때, 위 4가지 값을 토대로 하는 정확도와 정밀도, 재현율을 지표로 사용한다.

### 정확도(Accuracy)

$$
(Accuracy) = {TP + TN \over TP + FN + FP + TN}
$$

분류 성능을 가장 직관적으로 나타내는 값이기에 가장 흔하게 사용된다. 하지만 이 값에는 치명적인 단점이 있다. 이는 test dataset이 불균등할 때 나타나는데, 예를 들어 test spam dataset에 모두 임의의 ‘aaa’라는 단어가 들어가 있을 때의 성능을 높인다면 해당 단어가 없는 test spam dataset에 대해서는 성능이 낮아질 수 있다. 따라서 이러한 단점을 보안하기 위해 정밀도와 재현율, f measure를 사용한다.

### 정밀도(Precision)

$$
(Precision) = {TP \over TP + FP}
$$

정밀도는 모델이 True라고 분류한 것 중 실제 True인 것의 비율이다. 임의의 이메일을 스팸이라고 예측했을 때 실제로 스팸인지를 나타내는 지표이다. PPV(Positive Predictive Value)라고 불리기도 한다.

### 재현율(Recall)

$$
(Recall) = {TP \over TP + FN}
$$

재현율은 정밀도와 반대로, 실제로 True인 것 중 모델이 True라고 분류한 것의 비율이다. 임의의 스팸인 이메일을 모델이 스팸이라고 예측하는지를 나타내는 지표이다. Sensitivity라고 불리기도 한다.

### F1 score

$$
(F1 score) = {2 \over {1 \over Precision} + {1 \over Recall}}
$$

정확도를 보안한 정밀도와 재현율에도 단점이 존재한다. 예를 들어 임의의 모델이 어떤 특성에 따라 확실히 스팸이라고 판단할 수 있을 때만 스팸이라고 예측한다면 정밀도는 높아질 것이다. 하지만 이러한 모델이 예측을 잘한다고 말하지는 못한다. 이는 재현율이 낮아지는 것으로도 확인할 수 있다. 따라서 정확하게 모델의 분류성능을 평가하려면 dataset의 특성에 따라 적절히 정밀도와 재현율의 높낮이를 맞추어야 한다. 이를 위한 지표가 F1 score이다.

F1 score는 정밀도와 재현율의 조화평균으로 나타낸다. F1 score는 상술한 정밀도와 재현율을 단점을 보안하기 위해 사용된다. 단순하게 정밀도와 재현율의 산술평균을 사용하지 않고 조화평균을 사용하는 이유는 정밀도와 재현율의 차이가 클 때, 큰 값에 받는 영향을 줄이기 위해서이다.

# Spam filtering

## 데이터 파싱

Spam filtering에 쓰이는 dataset은 [https://www.kaggle.com/abhiroyq1/spam-mail-dataset](https://www.kaggle.com/abhiroyq1/spam-mail-dataset)을 사용했다. 분류기에 적합한 형태로 데이터를 변환하기 위해 먼저 training dataset에 등장하는 모든 단어를 key로 하고 해당 단어가 총 몇 번 사용되었는지를 value로 하는 dictionary를 생성하였다. dictionary에 속한 단어들 중 문자가 아닌 단어나, 한 글자로 이루어진 단어가 있다면 이는 고려 대상으로 적절하지 않기 때문에 제거한다. 남은 단어들 중에서 많이 사용된 상위 3000개의 단어들을 특성(feature)으로 결정한다. 따라서 모든 이메일은 3000차원의 특성을 가지게 된다. 한 이메일에 대해서 특성으로 결정된 3000개의 단어들이 사용된 유무를 나타낸 벡터를 데이터로 활용한다. 본 연구에서는 위 dataset에서 training용으로 제시된 706개의 이메일과 test용으로 제시된 260개의 데이터를 합쳐 총 966개의 이메일 벡터를 통해 K-Fold Cross Validation을 이용하여 Naive Bayes Classifier의 성능을 분석하고자 한다.

### 데이터 예시

```python
# dictionary:
[('order', 1414), ('address', 1299), ('report', 1217), ('mail', 1133), ('language', 1099), ('send', 1080), ('email', 1066), ('program', 1009), …]

# 이메일 벡터:
array[2, 1, 2, ... 1, 0, 0.]
```

## 모델 설계

Spam filtering에서는 임의의 test 이메일 벡터가 스팸인지 아닌지를 판단하기만 하면 되기 때문에 매우 불연속한 특성을 가지고 있다. 따라서 다항 분포를 활용하여 다항 이벤트 모델을 작성하였다.

```python
# -*- coding: utf-8 -*-

import os
from collections import Counter
import numpy as np

TRAIN_DIR = r"spam-mail-dataset/train-mails"
TEST_DIR = r"spam-mail-dataset/test-mails"

def make_Dictionary(root_dir):
    all_words = []
    emails = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
    for mail in emails:
        # Extracting each mail datas
        with open(mail, encoding="latin-1") as m:
            for line in m:
                words = line.split()
                all_words += words
    # creating a dictionary of words alog with number of occurences
    dictionary = Counter(all_words)
    list_to_be_removed = dictionary.keys()
    list_to_be_removed = list(list_to_be_removed)
    for item in list_to_be_removed:
        if item.isalpha() == False:  # 알파벳이 아니면 삭제
            del dictionary[item]
        elif len(item) == 1:  # 한 글자 단어 삭제
            del dictionary[item]
    # Extracting most common 3000 items from the dictionary
    dictionary = dictionary.most_common(3000)
    return dictionary

# Function to extract features from the set corpus

def extract_features(mail_dir):
    files = [os.path.join(mail_dir, fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros(
        (len(files), 3000)
    )  # Creating a Matrix of documents ID vs Word ID
    train_labels = np.zeros(len(files))
    docID = 0
    for fil in files:
        with open(fil, encoding="latin-1") as fi:
            for i, line in enumerate(fi):
                if (
                    i == 2
                ):  # as the Main Text starts in the 3rd line where 1st and 2nd line corresponds to Subject of the mail and a newline respectively
                    words = line.split()
                    for word in words:
                        wordID = 0
                        for i, d in enumerate(dictionary):
                            if d[0] == word:
                                wordID = i
                                features_matrix[docID, wordID] = words.count(word)

        filepathTokens = fil.split("/")
        lastToken = filepathTokens[-1].split("\\")[-1]
        if lastToken.startswith("spmsg"):  # Checks if the file name has "spmsg" in it
            train_labels[docID] = 1
            # Marks the label as 1 if the mail name has "spmsg"
        docID = docID + 1
    return features_matrix, train_labels

dictionary = make_Dictionary(TRAIN_DIR)
train_features_matrix, train_labels = extract_features(TRAIN_DIR)
test_feature_matrix, test_labels = extract_features(TEST_DIR)

print("complete")

def predict_label(train_features_matrix, train_labels, test_doc):
    # doc의 수, spam인 doc의 수, ham인 doc의 수
    num_doc = len(train_labels)
    num_spam = len(np.where(train_labels != 0)[0])
    num_ham = len(np.where(train_labels == 0)[0])
    # 1. p_spam, p_ham 구하기
    p_spam = num_spam / num_doc
    p_ham = num_ham / num_doc
    p_spam_doc = p_spam
    p_ham_doc = p_ham
    for wordID in np.where(test_doc != 0)[0]:  # test_doc에 속한 모든 word에 대해
        # 2. p_word 구하기
        p_word = len(np.where(train_features_matrix[:, wordID] != 0)[0]) / num_doc
        if p_word == 0:  # 만약 p_word가 0이면 나눌 수 없으므로 continue
            continue
        # 3. p_word_spam, p_word_ham 구하기<2>
        # 3-1. num_word_spam, num_word_ham 구하기
        num_word_spam = 0
        num_word_ham = 0
        for spamID in np.where(train_labels != 0)[0]:
            if train_features_matrix[spamID][wordID] != 0:
                num_word_spam += 1
        for hamID in np.where(train_labels == 0)[0]:
            if train_features_matrix[hamID][wordID] != 0:
                num_word_ham += 1
        # 3-2. p_word_spam, p_word_ham 계산
        p_word_spam = (num_word_spam + 1) / (num_spam + len(test_doc))
        p_word_ham = (num_word_ham + 1) / (num_ham + len(test_doc))
        # 4. p_spam_doc 계산
        p_spam_doc *= p_word_spam / p_word
        p_ham_doc *= p_word_ham / p_word
    return 0 if p_ham_doc > p_spam_doc else 1

def naiveBayes(
    model, train_features_matrix, train_labels, test_features_matrix, test_labels
):  # train model, train set, test set
    predicted_labels = np.array(
        [
            model(train_features_matrix, train_labels, test_features_matrix[i])
            for i in range(len(test_labels))
        ]
    )
    accuracy_score = sum(predicted_labels == test_labels) / len(test_labels)
    precision_score = sum(
        [predicted_labels[i] for i in np.where(test_labels == 1)[0]]
    ) / len(np.where(predicted_labels == 1)[0])
    recall_score = sum(
        [predicted_labels[i] for i in np.where(test_labels == 1)[0]]
    ) / len(np.where(test_labels == 1)[0])
    f1_score = (2 * precision_score * recall_score) / (precision_score + recall_score)
    return accuracy_score, precision_score, recall_score, f1_score

def create_fold(features_matrix, labels, k):
    folds = []
    features_matrix_0 = [features_matrix[i] for i in np.where(labels == 0)[0]]
    features_matrix_1 = [features_matrix[i] for i in np.where(labels == 1)[0]]
    fold_size_0 = len(features_matrix_0) // k
    fold_size_1 = len(features_matrix_1) // k
    for i in range(k - 1):
        i_fold = []
        i_labels = []
        i_fold += features_matrix_0[i * fold_size_0 : (i + 1) * fold_size_0]
        i_labels += [0] * fold_size_0
        i_fold += features_matrix_1[i * fold_size_1 : (i + 1) * fold_size_1]
        i_labels += [1] * fold_size_1
        folds.append((np.array(i_fold), np.array(i_labels)))
    i_fold = []
    i_labels = []
    i_fold += features_matrix_0[(k - 1) * fold_size_0 :]
    i_labels += [0] * len(features_matrix_0[(k - 1) * fold_size_0 :])
    i_fold += features_matrix_1[(k - 1) * fold_size_1 :]
    i_labels += [1] * len(features_matrix_1[(k - 1) * fold_size_1 :])
    folds.append((np.array(i_fold), np.array(i_labels)))
    return folds

def k_fold_cross_validation(features_matrix, labels, k):
    global create_fold, predict_label, naiveBayes
    from functools import reduce

    folds = create_fold(features_matrix, labels, k)
    scores = []
    for i in range(k):
        test_feature_matrix, test_labels = folds[i]
        train_features_matrix = reduce(
            lambda x, y: np.vstack((x, y)),
            [folds[foldID][0] for foldID in filter(lambda x: x != i, range(k))],
        )
        train_labels = reduce(
            lambda x, y: np.hstack((x, y)),
            [folds[foldID][1] for foldID in filter(lambda x: x != i, range(k))],
        )
        scores.append(
            list(
                naiveBayes(
                    predict_label,
                    train_features_matrix,
                    train_labels,
                    test_feature_matrix,
                    test_labels,
                )
            )
        )
    return np.array(scores)

features_matrix = np.vstack((train_features_matrix, test_feature_matrix))
labels = np.hstack((train_labels, test_labels))
aa = []
for i in range(4, 11):
    mean_scores = []
    scores = k_fold_cross_validation(features_matrix, labels, i)
    for j in range(4):
        mean_scores.append(np.mean(scores[j]))
    aa.append(mean_scores)
    print(f"[+] K={i}")
for k, scores in enumerate(aa):
    print(
        k,
        "&",
        "%0.5f" % scores[0],
        "&",
        "%0.5f" % scores[2],
        "&",
        "%0.5f" % scores[1],
        "&",
        "%0.5f" % scores[3],
    )
```

## 분류성능분석

앞서 작성한 Naive Bayes Classifier의 분류성능을 분석하기 위해 아래와 같이 K = 4, 5, 6, ... , 10에 대하여 K-Fold Cross Validation을 실행했다.

| K | Accuracy | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- |
| 4 | 0.98756 | 0.98770 | 0.94558 | 0.96401 |
| 5 | 0.99479 | 0.96995 | 0.96515 | 0.98469 |
| 6 | 1.00000 | 0.95287 | 0.95198 | 0.98769 |
| 7 | 1.00000 | 0.95822 | 0.97167 | 0.97857 |
| 8 | 1.00000 | 0.96035 | 0.96721 | 0.96035 |
| 9 | 1.00000 | 0.97265 | 0.97194 | 0.96393 |
| 10 | 1.00000 | 0.96035 | 0.97917 | 0.96991 |

K가 높을수록 정확도는 상승했으며, 다른 값들은 K와 큰 관련이 없었다. K가 6일 때 F1 Score가 0.98769로 가장 높았다. dataset의 특성 때문에 예상보다 높은 성능이 나왔다는 생각이 들어서, 다른 dataset으로도 분석해보면 좋을 것 같다. 실행시간은 평균 11분 32초가 걸렸다.

# MNIST dataset 분석

## 데이터 파싱

MNIST dataset은 [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)에서 확인할 수 있으며, 각 이미지 데이터는 28*28의 행렬로 이루어져 있다. 명도에 따라 0~255 중에서 하나의 값을 가지며 하나의 픽셀이 이미지의 특성을 나타내는 요소가 된다. 즉, 한 이미지당 784차원의 특성을 가지며 각 특성은 명도에 따라 0~255의 값을 가지게 된다. 각 특성이 0과 1로만 존재하는 Spam Filtering dataset과 달리 256가지의 경우가 존재하므로 이는 가우스 분포를 가정하고 그를 이용하여 확률을 계산하는 것이 효율적이다.

### 데이터 예시

![](/assets/images/antemrdm/naive_bayes_classifier/Untitled.png)

## 모델 설계

MNIST dataset 분석을 하기 위해서는 임의의 test 이미지 벡터가 0, 1, 2, ..., 9 중에서 어느 숫자에 가장 가까운지를 판단해야 한다. 또한 각 특성이 가지는 값이 0~255로 다양하기 때문에 각 특성이 가우스 분포를 따른다로 가정하고 확률을 계산한다.

```python
# -*- coding: utf-8 -*-

import numpy as np

from struct import *

fp_train_image = open("train-images-idx3-ubyte", "rb")
fp_train_label = open("train-labels-idx1-ubyte", "rb")

fp_test_image = open("t10k-images-idx3-ubyte", "rb")
fp_test_label = open("t10k-labels-idx1-ubyte", "rb")

train_img = np.zeros((28, 28))
train_lbl = [[], [], [], [], [], [], [], [], [], []]

test_img = np.zeros((28, 28))
test_lbl = [[], [], [], [], [], [], [], [], [], []]

d = 0
l = 0
index = 0
s = fp_train_image.read(16)
l = fp_train_label.read(8)

while True:
    s = fp_train_image.read(784)
    l = fp_train_label.read(1)

    if not s:
        break
    if not l:
        break

    index = int(l[0])

    train_img = np.reshape(unpack(len(s) * "B", s), (28, 28))
    # train_img = np.where(train_img>150, 1, 0)
    train_lbl[index].append(train_img)

d = 0
l = 0
index = 0
s = fp_test_image.read(16)
l = fp_test_label.read(8)

while True:
    s = fp_test_image.read(784)
    l = fp_test_label.read(1)

    if not s:
        break
    if not l:
        break

    index = int(l[0])

    test_img = np.reshape(unpack(len(s) * "B", s), (28, 28))
    # test_img = np.where(test_img>150, 1, 0)
    test_lbl[index].append(test_img)

print("complete")

class NaiveBayes(object):
    def fit(self, train_lbl, smooth=np.math.e):
        self.num_class = len(train_lbl)  # 분류할 class의 수: 10
        num_img_index = [len(train_lbl[i]) for i in range(10)]  # 각 index에 해당하는 img의 개수
        num_all_img = sum(num_img_index)  # 모든 img의 개수
        self.mean_img_index = [
            np.array(train_lbl[index]).mean(axis=0) for index in range(self.num_class)
        ]
        self.var_img_index = [
            np.array(train_lbl[index]).var(axis=0) + smooth
            for index in range(self.num_class)
        ]
        self.log_pi_var = [
            0.5 * np.sum(np.log(self.var_img_index[index]))
            for index in range(self.num_class)
        ]
        self.log_p_index = np.array(
            [
                np.log(num_img_index[index] / num_all_img)
                for index in range(self.num_class)
            ]
        )

    def predict(self, test_img):
        deviation_img_index = [
            test_img - self.mean_img_index[index] for index in range(self.num_class)
        ]
        square_deviation_img_index = [
            deviation_img_index[index].T @ deviation_img_index[index]
            for index in range(self.num_class)
        ]
        square_deviation_img_index = [
            deviation_img_index[index] * deviation_img_index[index]
            for index in range(self.num_class)
        ]
        log_p_img_index = np.array(
            [
                -self.log_pi_var[index]
                - (
                    0.5
                    * np.sum(
                        square_deviation_img_index[index] / self.var_img_index[index]
                    )
                )
                for index in range(self.num_class)
            ]
        )
        log_p_index_img = self.log_p_index + log_p_img_index
        return int(np.argmax(log_p_index_img))

    def score(self, test_lbl):
        predict_labels = [
            [
                self.predict(test_lbl[index][imgID])
                for imgID in range(len(test_lbl[index]))
            ]
            for index in range(self.num_class)
        ]
        score_dic_index = [{} for index in range(self.num_class)]
        global bb
        bb = predict_labels
        num_all_test_img = sum(
            [len(test_lbl[index]) for index in range(self.num_class)]
        )
        for index in range(self.num_class):
            score_dic_index[index]["tp"] = predict_labels[index].count(index)
            score_dic_index[index]["fp"] = (
                sum([predict_labels[i].count(index) for i in range(self.num_class)])
                - score_dic_index[index]["tp"]
            )
            score_dic_index[index]["fn"] = (
                len(predict_labels[index]) - score_dic_index[index]["tp"]
            )
            score_dic_index[index]["tn"] = (
                num_all_test_img
                - score_dic_index[index]["tp"]
                - score_dic_index[index]["fp"]
                - score_dic_index[index]["fn"]
            )
            score_dic_index[index]["accuracy"] = (
                score_dic_index[index]["tp"] + score_dic_index[index]["tn"]
            ) / num_all_test_img
            score_dic_index[index]["precision"] = (score_dic_index[index]["tp"]) / (
                score_dic_index[index]["tp"] + score_dic_index[index]["fp"]
            )
            score_dic_index[index]["recall"] = (score_dic_index[index]["tp"]) / (
                score_dic_index[index]["tp"] + score_dic_index[index]["fn"]
            )
            score_dic_index[index]["f1"] = (
                2
                * score_dic_index[index]["precision"]
                * score_dic_index[index]["recall"]
            ) / (score_dic_index[index]["precision"] + score_dic_index[index]["recall"])
        return score_dic_index

    def k_fold_cross_validation(self, lbl, k):
        folds = []
        num_img_index = [
            [(len(lbl[index]) // k) * (j + 1) for j in range(k)] for index in range(10)
        ]
        for index in range(10):
            num_img_index[index][-1] += len(lbl[index]) % k
            num_img_index[index].insert(0, 0)
        score_dic_index = [
            {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0,
            }
            for index in range(10)
        ]
        for fold in range(k):
            test_lbl = [
                lbl[index][num_img_index[index][fold] : num_img_index[index][fold + 1]]
                for index in range(10)
            ]
            if fold == k - 1:
                train_lbl = [
                    lbl[index][num_img_index[index][0] : num_img_index[index][fold]]
                    for index in range(10)
                ]
            else:
                train_lbl = [
                    lbl[index][num_img_index[index][0] : num_img_index[index][fold]]
                    + lbl[index][num_img_index[index][fold + 1] :]
                    for index in range(10)
                ]
            self.fit(train_lbl)
            score_dic_index_k = self.score(test_lbl)
            for index in range(10):
                for score in score_dic_index_k[index]:
                    score_dic_index[index][score] += score_dic_index_k[index][score]
        for index in range(10):
            for score in score_dic_index[index]:
                score_dic_index[index][score] /= k
        return score_dic_index

k = 4
model = NaiveBayes()
score_dic_index = model.k_fold_cross_validation(train_lbl, k)
print(f"[+] K = {k}")
for index in range(10):
    print(
        index,
        "&",
        "%0.5f" % score_dic_index[index]["accuracy"],
        "&",
        "%0.5f" % score_dic_index[index]["precision"],
        "&",
        "%0.5f" % score_dic_index[index]["recall"],
        "&",
        "%0.5f" % score_dic_index[index]["f1"],
        "\\\\",
    )
```

## 분류성능분석

앞서 작성한 Naive Bayes Classifier의 분류성능을 분석하기 위해 아래와 같이 K = 4에 대하여 K-Fold Cross Validation을 실행했다.

| Index | Accuracy | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- |
| 0 | 0.97548 | 0.86240 | 0.89448 | 0.87813 |
| 1 | 0.96512 | 0.78676 | 0.94601 | 0.85906 |
| 2 | 0.94823 | 0.85194 | 0.57989 | 0.68987 |
| 3 | 0.94843 | 0.80184 | 0.65830 | 0.72267 |
| 4 | 0.93853 | 0.84036 | 0.45549 | 0.59032 |
| 5 | 0.92917 | 0.79323 | 0.29256 | 0.42619 |
| 6 | 0.96383 | 0.75809 | 0.93021 | 0.83537 |
| 7 | 0.95127 | 0.91621 | 0.58676 | 0.71504 |
| 8 | 0.88553 | 0.44296 | 0.67476 | 0.53481 |
| 9 | 0.89307 | 0.47947 | 0.91057 | 0.62810 |
| Average | 0.93987 | 0.75333 | 0.69290 | 0.68795 |

정확도는 모든 숫자에 대해서 높았다. 하지만 정확도는 8과 9에서 비교적 낮았으며, 재현율은 0,1,6,9를 제외하고는 상당히 낮았다. 전체적인 성능을 나타내는 F1 score는 4,5,8,9에서 비교적 낮았고 나머지 숫자들에서는 비교적 높았다. 각 숫자의 특성에 따라 결과가 다르게 나오는 것으로 예상된다. 실행시간은 평균 16초가 걸렸다.

# Conclusion

각 단어들간의 연관성이 많이 없는 이1메일 벡터에서는 각 특성들 사이의 연관성을 무시하는 Naive Bayes Classifier의 F1 Score가 높았지만, 각 픽셀 사이의 연관성이 높은 MINST 이미지 벡터에서는 Naive Bayes Classifier의 F1 Score가 낮았다. 따라서 MNIST dataset을 분류하기 위해서는 각 픽셀간의 연관성을 계산해주는 Bayes Classifier를 사용하는 것이 더 효과적일 것이다.

# References

[1] LazyProgrammer, [https://lazyprogrammer.me/bayes-classifier-and-naive-bayes-tutorial-using/](https://lazyprogrammer.me/bayes-classifier-and-naive-bayes-tutorial-using/), 2015

[2] LazyProgrammer, and bob7783, [https://github.com/lazyprogrammer/machine_learning_examples/blob/master/supervised_class/bayes.py](https://github.com/lazyprogrammer/machine_learning_examples/blob/master/supervised_class/bayes.py), 2018

[3] AtiarRahman, [https://towardsdatascience.com/spam-filtering-using-naive-bayes-98a341224038](https://towardsdatascience.com/spam-filtering-using-naive-bayes-98a341224038), 2019
