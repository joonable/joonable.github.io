# 인공신경망 원리
@(Perceptron)[AND|OR|XOR]

**인공신경망 (Artificial Neural Network/ANN)** 의 기본이 되는 원리와 구조를 알아보자.  
이글은 **ANN**을 한번도 접해보지 못한 사람에게는 어울리지 않으며, 좀더 *Low-level*에서 **ANN**을 접해보고자 하는 사람에게 추천합니다


## Perceptron의 구조
![입력 노드가 두개인 퍼셉트론 모델](https://joonable.github.io/imgs/deep_learning_images/fig2-1.png)  
 *입력 노드가 두개인 퍼셉트론 모델*

![퍼셉트론 모델 수식](https://joonable.github.io/imgs/deep_learning_images/e2.1.png)  
 *퍼셉트론 모델 수식*

위 모델은 **퍼셉트론(Perceptron)** 모델이고 동작원리를 나타내는 수식이다.  
Perceptron 모델은 ANN 모델이 기본이 되는 모델이며 뇌속에 존재하는 뉴런이라는 세포의 구조와 원리를 본따 만든 모델이다.

여기서 **x1, x2**는 **입력값(inputs)** 을 나타내고 **w1, w2**는 각각의 입력값이 y에 영향을 미치는 정도를 의미하는 **가중치(weights)** 를 나타낸다.  
각각의 입력값과 가중치의 곱들을 더한 값이 **theta(thresold)** 보다 클 때  **y**는 1 혹은 0이라는 **결과값(outputs)** 을 산출한다. 여기서 theta 대신 **bias**를 이용하여 나타낼 수 있고 바뀐 식은 아래와 같다. 
(즉, 위의 식과 아래식은 기호 표기만 다를 뿐 의미는 같다.)
 
![퍼셉트론 모델 수식](https://joonable.github.io/imgs/deep_learning_images/e2.2.png)  
 *퍼셉트론 모델 수식 with bias*
 
이렇게 말로만 하면 안 와닿을 수 있으므로,  
모델이 동작하는 원리를 **AND / OR / XOR 연산**을 통해 알아보도록 하자.  
이때 입력값으로는 이진수인 0과 1만 넣을 수 있다고 가정하여 (0, 0), (0, 1), (1, 0), (1, 1) 이렇게 총 4가지 입력값으로 제한시킨다.  또한 bias는 -0.5로 고정시켜,  **가중치의 역할을 인지하고 이들만 조정해서 해당 연산을 완성시키는 것**이 중요한 key point이다.


-------
***참고***  
*y를 구하는 식을 **선형대수(Linear Algebra)** 를 이용하여 나타내면 다음과 같다.*  
$$	y =  weight * intputs + bias  $$

-------

### AND 연산

| x1 |  x2 |  y   |
|:---:|:---:|:---:|
|  0  |  0  |  0  |
|  0  |  1  |  0  |
|  1  |  0  |  0  |
|  1  |  1  |  1  |

![AND연산 진리표](https://joonable.github.io/imgs/deep_learning_images/fig2-2.png)  
*AND연산 진리표*

 AND 연산은 x1, x2의 값이 모두 1일 때에만 1을 내보나고 아니면 0을 내보내는 연산이다.  
 **(w1, w2) = (0.5, 0.5)**일 경우, 위의 논리 게이트를 만족하고, 이를 자세히 보면 다음과 같다.  
 
|   x1  |  x2   |  x1\*w1 + x2\*w2 - 0.5  |   y   |
|:---:|:---:|:---:|:---:|
|   0   |   0   |   -0.5                |0|
|   0   |   1   |   0                   |0|
|   1   |   0   |  0                    |0|
|   1   |   1   |  0.5                  |1|
  
코드로 나타내면 다음과 같다.
  
```python
def AND(x1, x2)
	w1, w2, bias = 0.5, 0.5, -0.5
	tmp = x1*w1 + x2*w2 - bias
	if tmp <= 0:
		return 0
	elif tmp > 0:
		return 1
```

사실, AND 연산을 만족시키 위한 (w1, w2)는 매우 많다.  
이를 위해 반드시 (w1, w2) = (0.5, 0.5)여야 하는 것은 아니다.  
또한 weights의 역할을 명시적으로 보여주기 위해 bias를 고정했을 뿐,  
bias를 weight처럼 조정하는 것도 모델을 학습하는 한 부분이다.
 
###OR 연산



모델과 가중치를 이용하여 **AND / OR / XOR**을 모두 구현함으로써 **ANN**의 기본이 되는 Perceptron의 원리에 대해 알아보았다.

## 1. ANN의 구조

### 1) 생물학적 구조
> Artificial neural networks (ANNs) or connectionist systems are computing systems vaguely inspired by the biological neural networks that constitute animal brains.  - [Wikipedia](https://en.wikipedia.org/wiki/Artificial_neural_network)

위에 언급 된것 처럼 ANN은 기본적으로 뇌속에 존재하는 뉴런이라는 세포의 구조와 원리를 본따 만든 모델이다. 이부분은
