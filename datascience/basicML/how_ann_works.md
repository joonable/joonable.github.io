
# 인공신경망 원리
@(Perceptron)[AND|OR|XOR]

**인공신경망 (Artificial Neural Network/ANN)**의 기본이 되는 원리와 구조를 알아보자. 이글은 **ANN**을 한번도 접해보지 못한 사람에게는 어울리지 않으며, 좀더 *Low-level*에서 **ANN**을 접해보고자 하는 사람에게 추천합니다

----------

[TOC]


## Perceptron의 구조
$$![입력이 두개인 퍼셉트론 모델]()fig 2-1 ![퍼셉트론 모델 수식]()e 2.1$$

 위 모델은 **퍼셉트론(Perceptron)** 모델이고 동작원리를 나타내는 수식이다.  Perceptron 모델은 ANN 모델이 기본이 되는 모델이며 뇌속에 존재하는 뉴런이라는 세포의 구조와 원리를 본따 만든 모델이다.

 여기서 **x1, x2**는 **입력값(inputs)**을 나타내고 **w1, w2**는 각각의 입력값이 y에 영향을 미치는 정도를 의미하는 **가중치(weights)**를 나타낸다.  각각의 입력값과 가중치의 곱들을 더한 값이 **theta(thresold)**보다 클 때  **y**는 1 혹은 0이라는 **결과값(outpus)**을 산출한다. 여기서 theta 대신 **bias**를 이용하여 나타낼 수 있고 바뀐 식은 아래와 같다. (즉, 위의 식과 아래식은 기호 표기만 다를 뿐 의미는 같다.)
 
$$![퍼셉트론 모델 수식]()e 2.2$$
 
이렇게 말로만 하면 안 와닿을 수 있을으므로, 모델이 동작하는 원리를 **AND / OR / XOR 연산**을 통해 알아보도록 하자. 이때 입력값으로는 이진수인 0과 1만 넣을 수 있다고 가정하여 (0, 0), (0, 1), (1, 0), (1, 1) 이렇게 총 4가지 입력값으로 제한시킨다.  또한 bias는 -0.5로 고정시켜.  가중치의 역할을 인지하고 이들만 조정해서 해당 연산을 완성시키는 것이 중요한 key point이다.

***참고***
*y를 구하는 식을 **선형대수(Linear Algebra)**를 이용하여 나타내면 다음과 같다.*
$$	y =  weight * intpus + bias$$

###AND 연산
| x1 |  x2 |  y   |
|:----:|:----:|:----:
|  0  |  0  |  0  |
|  0  |  1  |  0  |
|  1  |  0  |  0  |
|  1  |  1  |  1  |

$![AND연산 진리표]()fig 2-2  ![입력이 두개인 퍼셉트론 모델]()$ AND 연산은 x1, x2의 값이 모두 1일 때에만 1을 내보나고 아니면 0을 내보내는 연산이다. **(w1, w2) = (0.5, 0.5)**일 경우, 위의 논리 게이트를 만족하고, 이를 자세히 보면 다음과 같다.  
|   x1  |  x2   |  x1*w1 + x2*w2 - 0.5  |   y   |
|:-----:|:-----:|:---------------------:|:-----:|
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

사실, AND 연산을 만족시키 위한 (w1, w2)는 매우 많다. 이를 위해 반드시 (w1, w2) = (0.5, 0.5)여야 하는 것은 아니다. 또한 weights의 역할을 명시적으로 보여주기 위해 bias를 고정했을 뿐 bias를 weight처럼 조정하는 것도 모델을 학습하는 한 부분이다.
 
###OR 연산



모델과 가중치를 이용하여 **AND / OR / XOR**을 모두 구현함으로써 **ANN**의 기본이 되는 Perceptron의 원리에 대해 알아보았다.

## 1. ANN의 구조

### 1) 생물학적 구조
> Artificial neural networks (ANNs) or connectionist systems are computing systems vaguely inspired by the biological neural networks that constitute animal brains.  - [Wikipedia](https://en.wikipedia.org/wiki/Artificial_neural_network)

위에 언급 된것 처럼 ANN은 기본적으로 뇌속에 존재하는 뉴런이라는 세포의 구조와 원리를 본따 만든 모델이다. 이부분은

### 1) 생물학적 구조
``` python
@requires_authorization
def somefunc(param1='', param2=0):
    '''A docstring'''
    if param1 > param2: # interesting
        print 'Greater'
    return (param2 - param1 + 1) or None
class SomeClass:
    pass
>>> message = '''interpreter
... prompt'''
```

### LaTeX expression
$$	x = \dfrac{-b \pm \sqrt{b^2 - 4ac}}{2a} $$

### Table
| Item      |    Value | Qty  |
| :-------- | --------:| :--: |
| Computer  | 1600 USD |  5   |
| Phone     |   12 USD |  12  |
| Pipe      |    1 USD | 234  |

### Diagrams
#### Flow charts
```flow
st=>start: Start
e=>end
op=>operation: My Operation
cond=>condition: Yes or No?

st->op->cond
cond(yes)->e
cond(no)->op
```
#### Sequence diagrams 
```sequence
Alice->Bob: Hello Bob, how are you?
Note right of Bob: Bob thinks
Bob-->Alice: I am good thanks!
```

> **Note:** You can find more information:

> - about **Sequence diagrams** syntax [here][3],
> - about **Flow charts** syntax [here][4].

### Checkbox
You can use `- [ ]` and `- [x]` to create checkboxes, for example:

- [x] Item1
- [ ] Item2
- [ ] Item3

> **Note:** Currently it is only partially supported. You can't toggle checkboxes in Evernote. You can only modify the Markdown in Marxico to do that. Next version will fix this.  


### Dancing with Evernote

#### Notebook & Tags
**Marxico** add `@(Notebook)[tag1|tag2|tag3]` syntax to select notebook and set tags for the note. After typing `@(`, the notebook list would appear, please select one from it.  

#### Title
**Marxico** would adopt the first heading encountered as the note title. For example, in this manual the first line `Welcome to Marxico` is the title.

#### Quick Editing
Note saved by **Marxico** in Evernote would have a red ribbon button on the top-right corner. Click it and it would bring you back to **Marxico** to edit the note. 

> **Note:** Currently **Marxico** is unable to detect and merge any modifications in Evernote by user. Please go back to **Marxico** to edit.

#### Data Synchronization
While saving rich HTML content in Evernote, **Marxico** puts the Markdown text in a hidden area of the note, which makes it possible to get the original text in **Marxico** and edit it again. This is a really brilliant design because:

- it is beyond just one-way exporting HTML which other services do;
- and it avoids privacy and security problems caused by storing content in a intermediate server. 

> **Privacy Statement: All of your notes data are saved in Evernote. Marxico doesn't save any of them.** 

#### Offline Storage
**Marxico** stores your unsynchronized content locally in browser storage, so no worries about network and broswer crash. It also keeps the recent file list you've edited in `Document Management(Ctrl + O)`.

> **Note:** Although browser storage is reliable in the most time, Evernote is born to do that. So please sync the document regularly while writing.

## Shortcuts
Help    `Ctrl + /`
Sync Doc    `Ctrl + S`
Create Doc    `Ctrl + Alt + N`
Maximize Editor    `Ctrl + Enter`
Preview Doc `Ctrl + Alt + Enter`
Doc Management    `Ctrl + O`
Menu    `Ctrl + M`

Bold    `Ctrl + B`
Insert Image    `Ctrl + G`
Insert Link    `Ctrl + L`
Convert Heading    `Ctrl + H`

## About Pro
**Marixo** offers a free trial of 10 days. After that, you need to [purchase](http://marxi.co/purchase.html) the Pro service. Otherwise, you would not be able to sync new notes. Previous notes can be edited and synced all the time.

## Credits
**Marxico** was first built upon [Dillinger][5], and the newest version is almost based on the awesome [StackEdit][6]. Acknowledgments to them and other incredible open source projects!

## Feedback & Bug Report
- Twitter: [@gock2][7]
- Email: <hustgock@gmail.com>

----------
Thank you for reading this manual. Now please press `Ctrl + M` and click `Link with Evernote`. Enjoy your **Marxico** journey!


[^demo]: This is a demo footnote. Read the [MultiMarkdown Syntax Guide](https://github.com/fletcher/MultiMarkdown/wiki/MultiMarkdown-Syntax-Guide#footnotes) to learn more. Note that Evernote disables ID attributes in its notes , so `footnote` and `TOC` are not actually working. 

  [1]: http://marxi.co/client_en
  [2]: https://chrome.google.com/webstore/detail/kidnkfckhbdkfgbicccmdggmpgogehop
  [3]: http://bramp.github.io/js-sequence-diagrams/
  [4]: http://adrai.github.io/flowchart.js/
  [5]: http://dillinger.io
  [6]: http://stackedit.io
  [7]: https://twitter.com/gock2

