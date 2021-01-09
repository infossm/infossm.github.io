---
layout: post
title: Pytorch lightning 튜토리얼
date: 2021-01-07 01:52
author: choyi0521
tags: [pytorch, pytorch-lightning]
---


# 소개
&nbsp;&nbsp;&nbsp;&nbsp;[Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)은 Pytorch에 대한 High level의 인터페이스를 제공하기 위한 라이브러리입니다. 이 라이브러리는 모델 코드와 엔지니어링 코드를 분리해서 코드를 깔끔하게 만들 수 있도록 해주고 16-bit training, 다중 gpu 사용 등을 포함한 다양한 학습 기법을 몇 줄의 코드로 손쉽게 사용할 수 있도록 합니다. 이번 글에서는 pytorch lighting으로 MNIST 데이터 셋에 대한 분류 모델을 구현하면서 전반적인 라이브러리의 사용법에 대해 알아보겠습니다. 글에서 사용된 전체 코드는 [여기](https://github.com/choyi0521/pytorch-lightning-tutorial)에서 확인할 수 있습니다.

# Lightning Module 구현하기

&nbsp;&nbsp;&nbsp;&nbsp;Pytorch lightning에서는 trainer와 모델이 상호작용을 할 수 있도록 pytorch의 nn.Module의 상위 클래스인 lightning module을 구현해야 합니다. ligthning module을 정의하기 위해 LightningModule 클래스를 상속받고 모델, training, validation, test 루프 그리고 optimizer 등을 구현해야 합니다. 오버라이딩할 수 있는 메서드는 많이 있지만 이번 예제에서는 다음과 같이 몇 개의 메서드만 재정의해서 사용해봅시다. 이중 training_step과 configure_optimizers 필수적으로 구현해야 합니다.

```python
class Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        pass
```

&nbsp;&nbsp;&nbsp;&nbsp;forward는 모델의 추론 결과를 제공하고 싶을 때 사용합니다. nn.Module처럼 꼭 정의해야 하는 메서드는 아니지만 self(<입력>)과 같이 사용할 수 있게 만들어주므로 구현해주면 다른 메서드를 구현할 때 편리합니다.

```python
    def forward(self, x):
        return self.model(x)
```

&nbsp;&nbsp;&nbsp;&nbsp;training_step은 학습 루프의 body 부분을 나타냅니다. 이 메소드에서는 argument로 training 데이터로더가 제공하는 batch와 해당 batch의 인덱스가 주어지고 학습 로스를 계산하여 리턴합니다. pytorch lightning은 편리하게도 batch의 텐서를 cpu 혹은 gpu 텐서로 변경하는 코드를 따로 추가하지 않아도 trainer의 설정에 따라 자동으로 적절한 타입으로 변경해줍니다.

&nbsp;&nbsp;&nbsp;&nbsp;예제에서는 모델의 output과 정답 라벨 사이의 cross entropy loss를 구해서 넘겨줍니다.

```python
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss
```

&nbsp;&nbsp;&nbsp;&nbsp;validation_step은 학습 중간에 모델의 성능을 체크하는 용도로 사용합니다. training_step과 마찬가지로 validation 데이터로더에서 제공하는 배치를 가지고 확인하고자 하는 통계량을 기록할 수 있습니다. 하나의 값을 저장할 때는 self.log(<변수 이름>, <값>)과 같이 저장할 수 있고 여러 개의 변수를 저장하고 싶으면 아래 예시와 같이 self.log_dict로 변수 이름, 값 쌍을 가지고 있는 딕셔너리를 저장할 수 있습니다. 각 스탭마다 변수에 저장된 값의 평균이 해당 변수의 최종 값이 됩니다. 특별히 설정을 바꾸지 않으면 변수 중에 'val_loss'가 best 모델을 구하는 기준으로 사용됩니다.

&nbsp;&nbsp;&nbsp;&nbsp;아래 예제에서는 모델의 정확도와 cross entropy loss를 구해서 저장합니다. 여기서 accuracy 함수는 pytorch_lightning.metrics.functional에서 정의되어 있는 함수로 logits에서 최댓값인 라벨이 실제 라벨과 일치하는 비율을 구해줍니다.

```python
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = FM.accuracy(logits, y)
        loss = F.cross_entropy(logits, y)
        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)
```

&nbsp;&nbsp;&nbsp;&nbsp;test_step은 앞의 두 함수와 비슷하게 test 데이터로더에서 제공하는 배치를 가지고 확인하고 싶은 통계량을 기록하는데 사용할 수 있습니다. 예제에서는 정확도와 cross entropy loss를 기록합니다.

```python
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = FM.accuracy(logits, y)
        loss = F.cross_entropy(logits, y)
        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics)
```

&nbsp;&nbsp;&nbsp;&nbsp;configure_optimizers에서는 모델의 최적 파라미터를 찾을 때 사용할 optimizer와 scheduler를 구현합니다. GANs와 같이 여러 모델을 학습하기 위해 여러 optimizer를 사용해야 한다면 리스트로 리턴하면 됩니다. 이 경우에는 training_step에서 optimizer의 인덱스를 추가로 받아서 여러 모델을 번갈아 학습하게 됩니다. 예제에서는 학습해야 할 모델이 하나이므로 하나의 Adam optimzer만 사용하도록 하겠습니다.

```python
    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        return optimizer
```

# 학습 및 테스트

&nbsp;&nbsp;&nbsp;&nbsp;Pytorch lightning에서 학습을 위해 추가로 작성해야 할 코드는 매우 짧습니다. 먼저 코드를 재생산이 가능하도록 만들기 위해 seed_everything으로 모든 랜덤 시드를 고정합니다.

```python
pl.seed_everything(args.seed)
```

&nbsp;&nbsp;&nbsp;&nbsp;학습 및 테스트에 사용할 MNIST 데이터셋을 불러옵니다. 여기서 학습 데이터셋의 일부를 랜덤으로 샘플링해 validation 용도로 사용하겠습니다. 각 나눈 데이터 셋을 가지고 training, validation, test용 데이터로더를 만듭니다.

```python
# dataloaders
dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
train_dataset, val_dataset = random_split(dataset, [55000, 5000])
test_dataset = MNIST('', train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
```

&nbsp;&nbsp;&nbsp;&nbsp;모델을 학습하기 위해서는 학습 로직을 정하는 Trainer를 생성해야 합니다. Pytorch lightning의 Trainer는 굉장히 많은 기능을 제공합니다. 아래 예제에서는 간단히 학습 epoch 수와 gpu 수만 조정할 수 있도록 만들었습니다.

&nbsp;&nbsp;&nbsp;&nbsp;gpus가 0일 때는 cpu를 사용하고 gpus가 1 이상이면 gpu를 사용하여 모델을 학습합니다. gpus가 2 이상이면 자동으로 다중 gpu를 활용해 분산 학습을 진행하게 되는데 기본 설정은 process를 spawn하는 distributed data parallel 방식(ddp_spawn)으로 되어있습니다.

&nbsp;&nbsp;&nbsp;&nbsp;Trainer와 lightning module을 정의하고 난 뒤에 Trainer의 fit 함수로 모델을 학습할 수 있습니다. fit의 파라미터로 모델, training 데이터로더와 validation 데이터로더를 넘겨줍니다.

```python
# model
model = Classifier()

# training
trainer = pl.Trainer(max_epochs=args.n_epochs, gpus=args.n_gpus)
trainer.fit(model, train_loader, val_loader)
```

&nbsp;&nbsp;&nbsp;&nbsp;Trainer의 test 함수에 test 데이터로더를 넘겨주어 모델을 테스트할 수 있습니다. 아래와 같이 따로 어떤 모델을 테스트할지 지정하지 않으면 자동으로 Trainer가 validation을 통해 구한 best 모델을 가지고 테스트를 진행하게 됩니다.

```python
trainer.test(test_dataloaders=test_loader)
```

&nbsp;&nbsp;&nbsp;&nbsp;하나의 gpu로 모델을 30 epochs 동안 학습해서 다음과 같은 결과를 얻을 수 있었습니다.

```
DATALOADER:0 TEST RESULTS
{'test_acc': tensor(0.9702, device='cuda:0'),
 'test_loss': tensor(0.0975, device='cuda:0')}
```

# Checkpoint Callback

&nbsp;&nbsp;&nbsp;&nbsp;Pytorch lightning은 기본적으로 각 버전마다 체크포인트를 저장해줍니다. 하지만, 체크포인트 이름, 저장 주기, 모니터링할 metric 등을 바꾸고 싶으면 체크포인트 callback을 수정해주어야 합니다. 이번 예제에서는 마지막 체크포인트와 validation에서 구한 정확도 순으로 일정 개수의 체크포인트만 저장하는 방법에 대해 알아봅시다.

&nbsp;&nbsp;&nbsp;&nbsp;먼저, 아래와 같이 ModelCheckpoint로 체크포인트 콜백을 생성합니다. 각 파라미터의 의미는 다음과 같습니다.

* filepath: 체크포인트 저장위치와 이름 형식을 지정합니다.
* verbose: 체크포인트 저장 결과를 출력합니다.
* save_last: 마지막 체크포인트를 저장합니다.
* save_top_k: 최대 몇 개의 체크포인트를 저장할지 지정합니다.(save_last에 의해 저장되는 체크포인트는 제외)
* monitor: 어떤 metric을 기준으로 체크포인트를 저장할지 지정합니다.
* mode: 지정한 metric의 어떤 기준(ex. min, max)으로 체크포인트를 저장할지 지정합니다.

&nbsp;&nbsp;&nbsp;&nbsp;예제에서는 checkpoints 폴더에 체크포인트를 epoch=<숫자>.ckpt 형식으로 저장합니다. monitor값을 'val_acc'로 지정하고 mode를 'max'로 지정해서 validation에서 정확도가 높은 순으로 체크포인트를 저장합니다.

```python
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join('checkpoints', '{epoch:d}'),
    verbose=True,
    save_last=True,
    save_top_k=args.save_top_k,
    monitor='val_acc',
    mode='max'
)
```

&nbsp;&nbsp;&nbsp;&nbsp;체크포인트 콜백을 적용하기 위해서 Trainer의 callbacks 파라미터로 체크포인트 콜백을 넘겨줍니다. 저장한 체크포인트를 불러오려면 resume_from_checkpoint로 체크포인트의 위치를 넘겨줍니다. 그러면 자동으로 모델의 가중치, learning rate, epoch 등 모델과 학습 정보를 로딩해서 기존의 학습을 이어가게 됩니다.

```python
trainer_args = {
    'callbacks': [checkpoint_callback],
    'max_epochs': args.n_epochs,
    'gpus': args.n_gpus
}
if args.checkpoint:
    trainer_args['resume_from_checkpoint'] = os.path.join('checkpoints', args.checkpoint)

trainer = pl.Trainer(**trainer_args)
```

# Early Stopping

&nbsp;&nbsp;&nbsp;&nbsp;모델을 학습할 때 적절한 epoch 수를 정하는 것은 정말 어려운 일입니다. 이를 해결하기 위해 한 가지 간단한 대안으로 early stopping을 사용할 수 있습니다. early stopping은 특정 metric 성능이 연속된 일정 epoch 동안 향상되지 않을 경우 학습을 그만하는 방법입니다. Pytorch lightning에서는 Trainer에 콜백을 추가해서 early stopping을 할 수 있습니다.

&nbsp;&nbsp;&nbsp;&nbsp;EarlyStopping 콜백에서 주로 변경해야 하는 파라미터는 다음과 같습니다.

* monitor: 모니터링할 metric을 지정합니다.
* patience: metric 성능이 몇 번의 epoch가 향상 되지않을 때 학습을 멈출건지 지정합니다.
* verbose: 진행 결과를 출력합니다.
* mode: metric을 어떤 기준(ex. min, max)으로 성능을 측정할 지 지정합니다.

&nbsp;&nbsp;&nbsp;&nbsp;아래는 validation에서 구한 정확도를 기준으로 early stopping을 하고 체크포인트를 저장하는 예제입니다.

```python
# callbacks
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join('checkpoints', '{epoch:d}'),
    verbose=True,
    save_last=True,
    save_top_k=args.save_top_k,
    monitor='val_acc',
    mode='max'
)
early_stopping = EarlyStopping(
    monitor='val_acc',
    patience=args.patience,
    verbose=True,
    mode='max'
)

# training
trainer_args = {
    'callbacks': [checkpoint_callback, early_stopping],
    'gpus': args.n_gpus
}
if args.checkpoint:
    trainer_args['resume_from_checkpoint'] = os.path.join('checkpoints', args.checkpoint)

trainer = pl.Trainer(**trainer_args)
```

# 마치며

&nbsp;&nbsp;&nbsp;&nbsp;Pytorch lightning을 사용하면 분산 학습이나 체크포인트 저장 같은 모델 이외에 구현해야할 부분에 큰 시간을 들이지 않아도 돼서 많이 편리했습니다. 꼭 pytorch lightning을 쓰지 않더라도 코드를 깔끔하게 구현하기 위해 라이브러리가 제시하는 코딩 스타일을 한 번쯤 익혀보는 것도 좋을 것 같습니다.

&nbsp;&nbsp;&nbsp;&nbsp;사용하면서 몇 가지 불편한 점도 있었습니다. 예를 들어, 학습에 여러 optimizer와 scheduler가 필요한 경우 training_step 부분의 코드가 가독성이 떨어지고 분산 학습시 validation을 할 때 파일 출력을 메인 gpu에서 밖에 못 하는 등 몇몇 문제를 발견할 수 있었습니다. 또한, 이 글을 쓰고 있는 시점에도 라이브러리가 버젼이나 OS에 따라 문서에 적힌 것과 다르게 동작하거나 구체적으로 설명되지 않은 부분이 있는 것을 확인할 수 있었습니다. 이러한 부분들이 보완되면 대부분의 프로젝트에서 유용하게 쓰일 수 있을 것 같습니다.


# 참고문헌
* [<span style="color:blue">PyTorch Lightning Documentation</span>](https://pytorch-lightning.readthedocs.io/en/latest/)
* [<span style="color:blue">PyTorch Lightning 깃헙 프로젝트</span>](https://github.com/PyTorchLightning/pytorch-lightning)