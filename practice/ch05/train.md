### 1. 첫 번째 훈련

훈련 초기에는 train, val 셋의 loss가 급격히 감소한다.
하지만 두 번째 epoch을 지나면서 두 loss가 벌어지기 시작한다.
이는 모델이 학습은 하나 두 번째 epoch을 지나서는 과대적합된다는 신호이다.
과대적합된다 라는 것은 train set을 통째로 외워버린다는 것이다.
![first_train_result.png](first_train_result.png)

### 2. 온도 스케일링
![temperature_scaling.png](temperature_scaling.png)
