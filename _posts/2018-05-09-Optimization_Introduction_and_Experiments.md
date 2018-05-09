---
layout:     post
title:      "学习记录：优化算法介绍与实践"
subtitle:   "BGD, SGD, Mom, Adagrad, RMSProp, Adadelta, Adam"
date:       2018-05-09 22:01:00
author:     "lpt"
header-img: "img/post-bg-2018.jpg"
catalog: true
tags:
    - Optimazation
---

# 优化算法
> 优化算法，查了一些资料，看了沐神的[优化算法](http://zh.gluon.ai/chapter_optimization/index.html)。学习了一番，并做如下记录。转载请注明出处。
## 局部最小值
首先要区分局部最小值和全局最小值，这个应该大家都知道。
![局部最小值与全局最小值](https://upload-images.jianshu.io/upload_images/10171495-7b4fce181aef8b01.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 鞍点
梯度接近0或为0是因为当前解为局部最小值，或者它是个鞍点。它类似于“马鞍”，就如下图中心的那个点，有些优化算法会停滞在鞍点，不会继续往下探索。
![鞍点](https://upload-images.jianshu.io/upload_images/10171495-868b94786af1eb92.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# 经典优化算法
实验用到的是自己生成的数据集，采用的算法是线性回归：
```
# 生成数据集
num_inputs = 4
num_examples =  1000
w_true = nd.array([2.4, 3.6, 1.3, -5.7]).reshape((num_inputs, 1))
b_true = nd.array([2.33])
x = nd.random_normal(shape = (num_examples, num_inputs))
y = nd.dot(x, w_true) + b_true
y += nd.random.normal(scale=1, shape=y.shape)
# print(x, y)
```
![生成的数据](https://upload-images.jianshu.io/upload_images/10171495-a3de1745359f0e6e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 小批量梯度下降 (mini-batch)
算某个batch里梯度的平均值，作为更新的梯度。如果我们只是使用小批量的样本来计算其梯度并且更新：
$$\nabla f_\mathcal{B}(\boldsymbol{x}) = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}}\nabla f_i(\boldsymbol{x})$$ 

小批量随机梯度也是对梯度的无偏估计，即$$\mathbb{E}_\mathcal{B} \nabla f_\mathcal{B}(\boldsymbol{x}) = \nabla f(\boldsymbol{x}).$$

更新$\boldsymbol{x}$：

$$\boldsymbol{x} \leftarrow \boldsymbol{x} - \eta \nabla f_\mathcal{B}(\boldsymbol{x}).$$

在上面的式子中$\mathcal{B}$代表样本批量大小，$\eta$（取正数）称作学习率

## 梯度下降 (Batch gradient descent)
当小批量梯度下降的batch_size为样本总数时，则为梯度下降。

## 随机梯度下降 (Stochastic gradient descent)
小批量梯度下降batch_size取1的情况即为随机梯度下降，即梯度直接加上每个样本的梯度。目标函数为

$$f(\boldsymbol{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\boldsymbol{x}),$$

所以梯度也可以：

$$\nabla f(\boldsymbol{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\boldsymbol{x}),$$

当然，在这里随机梯度是梯度的无偏估计，也就是$$\mathbb{E}_i \nabla f_i(\boldsymbol{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\boldsymbol{x}) = \nabla f(\boldsymbol{x}).$$

然后随机均匀采样$i$并计算$\nabla f_i(\boldsymbol{x})$，更新$\boldsymbol{x}$：

$$\boldsymbol{x} \leftarrow \boldsymbol{x} - \eta \nabla f_i(\boldsymbol{x}).$$

```
# 小批量随机梯度下降
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size
```

实验截图：
![SGD && BGD](https://upload-images.jianshu.io/upload_images/10171495-ebeebc54ffb3176e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![different learning rate for mini-BGD](https://upload-images.jianshu.io/upload_images/10171495-2664baef4f52852d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


## 指数移动加权平均（Exponential Moving Average, EMA或EWMA）

在讲以下几种算法之前，有必要插入一个指数加权平均法的理解，这个方法是真的好用。

考虑这么一个线性组合：
$$y^{(t)} = \gamma y^{(t-1)} + (1-\gamma) x^{(t)}.$$

对它展开：
$$
\begin{align*}
y^{(t)}  &= (1-\gamma) x^{(t)} + \gamma y^{(t-1)}\\
         &= (1-\gamma)x^{(t)} + (1-\gamma) \cdot \gamma x^{(t-1)} + \gamma^2y^{(t-2)}\\
         &= (1-\gamma)x^{(t)} + (1-\gamma) \cdot \gamma x^{(t-1)} + (1-\gamma) \cdot \gamma^2x^{(t-2)} + \gamma^3y^{(t-3)}\\
         &= (1-\gamma)x^{(t)} + (1-\gamma) \cdot \gamma x^{(t-1)} + (1-\gamma) \cdot \gamma^2x^{(t-2)} + (1-\gamma) \cdot \gamma^3x^{(t-3)} + \gamma^4y^{(t-4)}\\
         &\ldots
\end{align*}
$$
由于$ \lim_{n \rightarrow \infty}  (1-\frac{1}{n})^n = \exp(-1) \approx 0.3679,$ 我们把$1-\frac{1}{n}=\gamma$代入，则有：

$ \lim_{\gamma \rightarrow 1}  \gamma^{1/(1-\gamma)} = \exp(-1) \approx 0.3679,$

如果我们把$\exp(-1)$看成很小的项，也就是我们可以近似中忽略所有含$\gamma^{1/(1-\gamma)}$和比$\gamma^{1/(1-\gamma)}$更高阶的系数的项。加入$\gamma=0.99$，那么$\gamma^100$则忽略，也就是看上面那个式子，可以知道当前的y可以看做近100个时刻的x乘以相应系数的和。这就很有用了，意味着我们不仅仅是求开始到现在，而是通过前n个时刻来求现在。

## 动量法
如果使用梯度下降的话，加入是个二维的数据，有一个维度的梯度下降得比较陡，那么x下降的方向则会偏向这个维度，假设如下图，那么它在$x_2$方向会来回震荡。这样明显是会减慢收敛速度的，于是，就思考如何去让它朝着梯度一直同方向的维度跑。如果上下震荡的梯度相互之间有抵消效果的话，则可以完成这一效果。也就是说，如$x_2$方向上的梯度是一段时间内的梯度和，那么产生的震荡效果会抵消；而$x_1$方向因为同一个方向，则叠加产生加速效果。
![梯度下降走势图](https://upload-images.jianshu.io/upload_images/10171495-bdc892283e2af9ec.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


对小批量梯度下降算法做如下修改：

$$
\begin{align*}
\boldsymbol{v} &\leftarrow \gamma \boldsymbol{v} + \eta \nabla f_\mathcal{B}(\boldsymbol{x}),\\
\boldsymbol{x} &\leftarrow \boldsymbol{x} - \boldsymbol{v}.
\end{align*}
$$

对动量法的速度变量做变形：

$$\boldsymbol{v} \leftarrow \gamma \boldsymbol{v} + (1 - \gamma) \frac{\eta \nabla f_\mathcal{B}(\boldsymbol{x})}{1 - \gamma}. $$

由指数加权移动平均可得，速度变量$\boldsymbol{v}$是对$(\eta\nabla f_\mathcal{B}(\boldsymbol{x})) /(1-\gamma)$做了指数加权移动平均。

```
def sgd_momentum(params, vs, lr, mom, batch_size):
    for param, v in zip(params, vs):
        v[:] = mom * v + lr * param.grad / batch_size
        param[:] -= v
```

![动量法](https://upload-images.jianshu.io/upload_images/10171495-3df5172cf3e7236a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


## Adagrad

首先计算小批量随机梯度$\boldsymbol{g}$，然后将该梯度按元素平方后累加到变量$\boldsymbol{s}$：

$$\boldsymbol{s} \leftarrow \boldsymbol{s} + \boldsymbol{g} \odot \boldsymbol{g}. $$

然后，我们将目标函数自变量中每个元素的学习率通过按元素运算重新调整一下：

$$\boldsymbol{g}^\prime \leftarrow \frac{\eta}{\sqrt{\boldsymbol{s} + \epsilon}} \odot \boldsymbol{g},$$

更新：

$$\boldsymbol{x} \leftarrow \boldsymbol{x} - \boldsymbol{g}^\prime.$$

```
def adagrad(params, sqrs, lr, batch_size):
    eps_stable = 1e-7
    for param, sqr in zip(params, sqrs):
        g = param.grad / batch_size
        sqr[:] += g.square()
        param[:] -= lr * g / (sqr + eps_stable).sqrt()
```

## RMSProp

首先计算小批量随机梯度$\boldsymbol{g}$，然后对该梯度按元素平方项$\boldsymbol{g} \odot \boldsymbol{g}$做指数加权移动平均，记为$\boldsymbol{s}$：

$$\boldsymbol{s} \leftarrow \gamma \boldsymbol{s} + (1 - \gamma) \boldsymbol{g} \odot \boldsymbol{g}. $$

然后，和Adagrad一样，将目标函数自变量中每个元素的学习率通过按元素运算重新调整一下：

$$\boldsymbol{g}^\prime \leftarrow \frac{\eta}{\sqrt{\boldsymbol{s} + \epsilon}} \odot \boldsymbol{g}, $$

最后的自变量迭代步骤与小批量随机梯度下降类似：

$$\boldsymbol{x} \leftarrow \boldsymbol{x} - \boldsymbol{g}^\prime $$

需要强调的是，RMSProp只在Adagrad的基础上修改了变量$\boldsymbol{s}$的更新方法：平方项$\boldsymbol{g} \odot \boldsymbol{g}$这里从累加变成了指数加权移动平均（划重点，EMA又来了）。所以变量$\boldsymbol{s}$可以看作是在此之前的$1/(1-\gamma)$个时刻的平方项$\boldsymbol{g} \odot \boldsymbol{g}$的加权平均，因此每个元素的学习率不会像AdaGrad那样，一直降而不升。
```
def rmsprop(params, sqrs, lr, gamma, batch_size):
    eps_stable = 1e-8
    for param, sqr in zip(params, sqrs):
        g = param.grad / batch_size
        sqr[:] = gamma * sqr + (1 - gamma) * g.square()
        param[:] -= lr * g / (sqr + eps_stable).sqrt()
```
![RMSProp](https://upload-images.jianshu.io/upload_images/10171495-7eebdbc0a994fb5b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


## Adadelta

Adadelta算法也像RMSProp一样，使用了小批量随机梯度按元素平方的指数加权移动平均变量$\boldsymbol{s}$，并将其中每个元素初始化为0。

计算小批量随机梯度$\boldsymbol{g}$，然后按元素平方项$\boldsymbol{g} \odot \boldsymbol{g}$（EMA又来啦），记为$\boldsymbol{s}$：

$$\boldsymbol{s} \leftarrow \rho \boldsymbol{s} + (1 - \rho) \boldsymbol{g} \odot \boldsymbol{g}. $$

然后，计算当前需要迭代的目标函数自变量的变化量$\boldsymbol{g}^\prime$：

$$ \boldsymbol{g}^\prime \leftarrow \frac{\sqrt{\Delta\boldsymbol{x} + \epsilon}}{\sqrt{\boldsymbol{s} + \epsilon}}  \odot \boldsymbol{g}, $$

上式中$\Delta\boldsymbol{x}$初始化为零张量，并记录$\boldsymbol{g}^\prime$按元素平方的指数加权移动平均：

$$\Delta\boldsymbol{x} \leftarrow \rho \Delta\boldsymbol{x} + (1 - \rho) \boldsymbol{g}^\prime \odot \boldsymbol{g}^\prime. $$

同样地，最后的自变量迭代步骤与小批量随机梯度下降类似：

$$\boldsymbol{x} \leftarrow \boldsymbol{x} - \boldsymbol{g}^\prime. $$

(Adadelta没有学习率参数)

```
# Adadalta
def adadelta(params, sqrs, deltas, rho, batch_size):
    eps_stable = 1e-5
    for param, sqr, delta in zip(params, sqrs, deltas):
        g = param.grad / batch_size
        sqr[:] = rho * sqr + (1. - rho) * nd.square(g)
        cur_delta = (nd.sqrt(delta + eps_stable)
                    / nd.sqrt(sqr + eps_stable) * g)
        delta[:] = rho * delta + (1. - rho) * cur_delta * cur_delta
        param[:] -= cur_delta
```

## Adam
Adam结合了动量法中的变量 v 和RMSProp中的EMA变量s：

①动量法，给定超参数$\beta_1$且满足$0 \leq \beta_1 < 1$（算法作者建议设为0.9），则有变量$\boldsymbol{v}$:

$$\boldsymbol{v} \leftarrow \beta_1 \boldsymbol{v} + (1 - \beta_1) \boldsymbol{g}. $$

②RMSProp，给定超参数$\beta_2$且满足$0 \leq \beta_2 < 1$（算法作者建议设为0.999），
计算EMA变量$\boldsymbol{s}$：

$$\boldsymbol{s} \leftarrow \beta_2 \boldsymbol{s} + (1 - \beta_2) \boldsymbol{g} \odot \boldsymbol{g}. $$ 

如果$\boldsymbol{v}$和$\boldsymbol{s}$中的元素在迭代初期都初始化为0，那么很有可能产生“冷启动”，也就是说这两个变量的元素在迭代初期都很小。所以Adam的作者采用偏差修正：

$$\hat{\boldsymbol{v}} \leftarrow \frac{\boldsymbol{v}}{1 - \beta_1^t}, $$

$$\hat{\boldsymbol{s}} \leftarrow \frac{\boldsymbol{s}}{1 - \beta_2^t}. $$

也即是，如果$t=1$,$\beta_1=0.9$时，则在$t=1$时刻相当于对变量中的$\boldsymbol{v}$元素放大了10倍；同理，$t=1$,$\beta_2=0.999$时$\boldsymbol{s}$放大1000倍。但当$t$增大，分母逐渐趋于0，对这两个变量的偏差修正作用就微乎其微了。

然后继续进行计算：

$$\boldsymbol{g}^\prime \leftarrow \frac{\eta \hat{\boldsymbol{v}}}{\sqrt{\hat{\boldsymbol{s}} + \epsilon}},$$

$$\boldsymbol{x} \leftarrow \boldsymbol{x} - \boldsymbol{g}^\prime. $$

> $\epsilon$是为了维持数值稳定性而添加的常数，例如$10^{-8}$

emmm... Adam的实现也就是按照公式翻译一下。
```
# Adam。
def adam(params, vs, sqrs, lr, batch_size, t):
    beta1 = 0.9
    beta2 = 0.999
    eps_stable = 1e-8
    for param, v, sqr in zip(params, vs, sqrs):     
        g = param.grad / batch_size
        v[:] = beta1 * v + (1. - beta1) * g
        sqr[:] = beta2 * sqr + (1. - beta2) * nd.square(g)
        v_bias_corr = v / (1. - beta1 ** t)
        sqr_bias_corr = sqr / (1. - beta2 ** t)
        div = lr * v_bias_corr / (nd.sqrt(sqr_bias_corr) + eps_stable)       
        param[:] = param - div
```

> 其他的优化算法，再看吧。
