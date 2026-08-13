---
layout: post
title: "Doing it from Scratch: A Deep Dive into Linear Regression and Backpropagation -- And the Math That Goes With Them"
---

Ordinary Least Squares (OLS) aka Linear Regression is the "Hello World" of machine learning. It is also just a neural network with only 2 parameters! Let me show you what I mean...


At its core, OLS is about finding the best straight line to fit a set of data points. Imagine you have data on house prices based on their size. If you plot these points on a graph, they might roughly form a line moving upwards—as size increases, price increases.

Our goal is to find the equation of that line so we can predict the price of a house we haven't seen before, given just its size.

We all remember the equation of a line right?

$$
y = mx + b
$$

This simple equation contains the two parameters that we will be teaching a machine to learn: $m$ and $b$. 


#### **Example:**
Let's say we have the following two houses: 

* *House 1: 1000 sq ft, $500k*
* *House 2: 3000 sq ft, $800k*

<p align="center">
  <img src="/assets/images/linear_regression_backprop/two_pts.png" width="400" />
</p>

You might remember how to find the slope of a line given two points in algebra class. Then when the teacher asked: *"what is the value of a house with ___ sq ft..."* you could plug in the size and find the value, assuming that the relationship is completely linear. 

But what happens when we observe a third point that does not fit this linear relationship? Let's introduce a 3rd point:

* *House 3: 5000 sq ft, $600k*

<p align="center">
  <img src="/assets/images/linear_regression_backprop/three_pts.png" width="400" />
</p>

8th grade math doesn't cut it any more, it is no longer possible to draw a single line through all 3 points. Unfortunately, very few phenomena in the real world take on a perfectly linear relationship, so we need a systematic way to draw a line that 'best fits' not just 3, but potentially hundreds or thousands of data points. By "best fits", what I really mean is "the line that minimizes the sum of the squared distance from the line to each of the points (i.e. the Mean Squared Error, or MSE)." This is the motivation for Ordinary Least Squares Regression (OLS). We want to teach a computer to measure how "wrong" a specific line is, and find the "least wrong" line.

### The Loss Function: Mean Squared Error (MSE)

We measure error by looking at the difference between what our line predicts and the actual data point. If our line predicts a price of \$300k, but the house actually sold for \$350k, our error is \$50k.

To stop negative and positive errors from cancelling each other out (and to penalize large errors more), we square this difference. We do this for all our data points and take the average.

$$
MSE = \frac{1}{N} \sum_{i=1}^{N} (y_{pred} - y_{actual})^2
$$

Our goal is simply to find the values for $m$ (weight) and $b$ (bias) that result in the **lowest possible MSE**.

Currently, we are thinking about this point-by-point. But imagine we have a whole dataset of $N$ houses where $N$ can be very large. We add a point $(x_1, y_1)$, then another $(x_2, y_2)$...and so on. We don't just have one equation for a line; we have a `system` of equations:

$$
\begin{align}
y_1 &= m x_1 + b + \epsilon_1 \\
y_2 &= m x_2 + b + \epsilon_2 \\
&\vdots \\
y_n &= m x_n + b + \epsilon_n
\end{align}
$$

If you have taken linear algebra, you might recognize this as the classic $Ax = b$.
However, there is a catch. In classic linear algebra, you have the same number of `equations` as `unknowns` -- 2 houses, 2 points, 2 equations, 2 unknown parameters ($m$ and $b$). All good. 

But here, we have way more equations ($N$ data points) than unknowns (2 parameters: $m$ and $b$).


This is called an `overdetermined system`. There is NO line that perfectly passes through every single point (unless your data is perfect, which it never is). Technically speaking, the equation $y = X\beta$ has no solution.

<small>**Disclaimer**: There may be some linear relationships that are pseudo-perfect, where the noise is so negligible compared to the strength of the signal that for all intensive purposes we may consider it perfectly linear. However, the *empirical data* from which these relationships may be derived is almost never going to be absolutely perfectly linear. </small>


So, instead of trying to solve it exactly (which is impossible), we try to find the $\beta$ that gets us "*as close as possible*". There is an intuitive [geometrical explanation](https://graphicallinearalgebra.net/2017/08/09/orthogonality-and-projections/) for this concept, for the inquisitive reader. 

This motivation—solving an unsolvable system by minimizing the error—is what leads us to the following derivation...

### **Ordinary Least Squares (OLS) Matrix Derivation -- Warning! Math!**

The goal of linear regression is to find the vector of coefficients $\hat{\beta}$ that minimizes the sum of squared residuals. To do this, we can start by representing our system of equations in `vector notation`.

$$
y = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}
\quad
X = \begin{bmatrix}
x_1 & 1 \\
x_2 & 1 \\
\vdots & \vdots \\
x_n & 1
\end{bmatrix}
\quad
\beta = \begin{bmatrix} m \\ b \end{bmatrix}
\quad
\epsilon = \begin{bmatrix} \epsilon_1 \\ \epsilon_2 \\ \vdots \\ \epsilon_n \end{bmatrix}
$$

In linear algebra class, you are taught:
*"Take this row of the first matrix, multiply it by the column of the second matrix, sum them up, and that's your first entry."*

Here, we are working backwards from the result we want.

**What We Want:** We want the equation $y_i = m \cdot x_i + b \cdot 1$ for every single data point $i$.

**How to Get It:**

To get $m \cdot x_i$, we need $x_i$ and $m$ to "hit" each other during multiplication.
To get $b \cdot 1$, we need a "dummy" 1 to "hit" $b$ during multiplication.
So if we construct a single row $[x_i, 1]$ and multiply it by the column vector $\begin{bmatrix} m \\ b \end{bmatrix}$:

$$
[x_i, 1] \cdot \begin{bmatrix} m \\ b \end{bmatrix} = (x_i \cdot m) + (1 \cdot b) = mx_i + b
$$

Boom! We recovered our original line equation.
When we stack all these rows on top of each other to make the matrix $X$, doing the matrix multiplication $X\beta$ just performs this "row $\cdot$ column" operation $N$ times in parallel, giving us the full list of predictions for every house at once.
That "column of ones" is the clever trick that lets us treat the bias/intercept ($b$) just like any other weight

### 1. The Model

We start with the linear model in matrix form:

$$y = X\beta + \epsilon$$

* $y$ is an $n \times 1$ vector of observations.
* $X$ is an $n \times k$ matrix of independent variables (the design matrix).
* $\beta$ is a $k \times 1$ vector of parameters to be estimated.
* $\epsilon$ is an $n \times 1$ vector of errors.

### 2. The Objective Function (RSS)

The Residual Sum of Squares (RSS) is the squared norm of the residuals:

$$S(\beta) = \epsilon^T \epsilon$$

$$S(\beta) = (y - X\beta)^T (y - X\beta)$$

### 3. Expanding the Term

Using the linear-algebriac matrix property $(A - B)^T = A^T - B^T$:

$$S(\beta) = (y^T - \beta^T X^T)(y - X\beta)$$

$$S(\beta) = y^T y - y^T X \beta - \beta^T X^T y + \beta^T X^T X \beta$$

Since $y^T X \beta$ is a scalar ($1 \times 1$), it is equal to its own transpose. Note that $(y^T X \beta)^T = \beta^T X^T y$ -- another property of matrix multiplication. We can combine the middle terms:

$$S(\beta) = y^T y - 2\beta^T X^T y + \beta^T X^T X \beta$$

### 4. Taking the Derivative

To find the minimum, we take the partial derivative of $S(\beta)$ with respect to $\beta$:

$$\frac{\partial S}{\partial \beta} = \frac{\partial}{\partial \beta} (y^T y - 2\beta^T X^T y + \beta^T X^T X \beta)$$

Using matrix calculus rules:
1. $\frac{\partial (a^T \beta)}{\partial \beta} = a$
2. $\frac{\partial (\beta^T A \beta)}{\partial \beta} = 2A\beta$

We get:

$$\frac{\partial S}{\partial \beta} = -2X^T y + 2X^T X \beta$$

### 5. Solving for $\hat{\beta}$

Set the gradient to zero to find the minimum:
Rearrange to form the **Normal Equations**:

$$-2X^T y + 2X^T X \hat{\beta} = 0$$

$$X^T X \hat{\beta} = X^T y$$

Provided that $(X^T X)$ is non-singular (invertible), we multiply both sides by $(X^T X)^{-1}$:

$$\hat{\beta} = (X^T X)^{-1} X^T y$$

Python libraries like **`scikit-learn`** and others use a similar linear-algebraic approach to solve for the weights directly. It's fast, exact, and doesn't require any "training loops." Let's see it in action...

### **Finally, Some Code!**
Let's create some synthetic data to play with. We'll generate 100 points that roughly follow a line $y = 3x + 5$, but with some random noise added so it's not perfect.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(23)

n_samples = 100

X = np.random.randn(n_samples, 1)
eps = np.random.randn(n_samples, 1) * 3

# Our true function is y = 3x + 5 + noise
Y = (3 * X + 5) + eps

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

plt.scatter(X, Y)
plt.title("Generated Data")
plt.show()
```

<p align="center">
  <img src="/assets/images/linear_regression_backprop/output_cell3.png" width="500" />
</p>

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, Y_train)

print(f"Coefficient (m): {lr.coef_}, Intercept (b): {lr.intercept_}")
```

```text
Coefficient (m): [[3.07464887]], Intercept (b): [4.83065891]
```

As expected, `scikit-learn` very closely approximates the ground-truth parameters. Let's plot the line and see how it looks...

```python
# plt.scatter(X, Y)
# plt.plot(X, lr.coef_ * X + lr.intercept_, color='red')
# plt.title("Line of Best Fit")
# plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(X, Y, alpha=0.6, label='Data Points')
plt.plot(X, lr.coef_ * X + lr.intercept_, color='red', linewidth=2, label='Prediction')
plt.title("Line of Best Fit", fontsize=14)
plt.xlabel("Input (X)")
plt.ylabel("Target (Y)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

<p align="center">
  <img src="/assets/images/linear_regression_backprop/output_cell6.png" width="500" />
</p>

#### "***What if I told you that Linear Regression is just a neural network with only one neuron***..."

<p align="center">
  <img src="/assets/images/linear_regression_backprop/single_neuron.jpg" width="400" />
</p>

Well, ALMOST... We can see here that besides the sum notation $\Sigma$ and the activation function $f$, our linear equation exactly matches the "cell body".

So, instead of using a closed-form mathematical solution, we can solve our problem using an iterative approach called **Gradient Descent** powered by **Backpropagation**. 


**Forward Pass**: We initialize $m$ and $b$ to be random values sampled from a normal distribution with mean of 0 and variance of 1 and calculate the predictions and the loss.

$$\hat{y_i} = mx_i + b$$

To measure how wrong our random guesses are, we calculate the **Mean Squared Error (MSE)** loss:

$$
L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

where $y_i$ is the true value and $\hat{y}_i$ is our prediction.

```python
#Initialize parameters to random values
m = np.random.randn(1)
b = np.random.randn(1)

print(f"Initial parameters: m={m}, b={b}")

#Calculate predictions
y_pred = m * X_train + b

#Calculate loss
L = np.mean((y_pred - Y_train)**2)

print(f"Initial loss: {L}")
```

```text
Initial parameters: m=[0.43477207], b=[0.16141775]
Initial loss: 41.7152270200187
```

As we can see, with random values for $m$ and $b$, we get a pretty high loss, and our model is not doing anything useful at all. To improve our model, we need to iteratively make small adjustments to the parameters using backpropagation...

#### Step 2: The Backward Pass (Backpropagation & Chain Rule)

To improve the model, we need to know how much a tiny change in $m$ or $b$ affects the total loss $L$.  
This is the **gradient**. We find this by applying the **Chain Rule** to the loss function, working backward from the output to the parameters.

First, the derivative of the loss with respect to our prediction ($\hat{y}_i$):

$$
\frac{\partial L}{\partial \hat{y}_i} = -2(y_i - \hat{y}_i)
$$

Now, we use the **Chain Rule** to find the gradients for $m$ and $b$:

- **For the slope ($m$):**

$$
\frac{\partial L}{\partial m}
= \frac{\partial L}{\partial \hat{y}}
\cdot \frac{\partial \hat{y}}{\partial m}
= \frac{1}{n} \sum -2 x_i (y_i - \hat{y}_i)
$$

- **For the intercept ($b$):**

$$
\frac{\partial L}{\partial b}
= \frac{\partial L}{\partial \hat{y}}
\cdot \frac{\partial \hat{y}}{\partial b}
= \frac{1}{n} \sum -2 (y_i - \hat{y}_i)
$$

```python
dL_dm = (-2 * (X_train * (Y_train - y_pred))).mean(axis=0)
dL_db = (-2 * (Y_train - y_pred)).mean(axis=0)

print(f"Gradient of loss with respect to m: {dL_dm}")
print(f"Gradient of loss with respect to b: {dL_db}")
```

```text
Gradient of loss with respect to m: [-6.52285458]
Gradient of loss with respect to b: [-10.14035916]
```

#### Step 3: The Update (Gradient Descent)

Now that we know the direction of the error (the gradient), we adjust our parameters in the **opposite** direction to minimize the loss.  
We use a **Learning Rate** ($\eta$) to control how large of a step we take.

If the gradient is positive, the parameter is too high; if it's negative, the parameter is too low.

#### The Update Rules:

$$
m_{\text{new}} = m_{\text{old}} - \eta \cdot \frac{\partial L}{\partial m}
$$

$$
b_{\text{new}} = b_{\text{old}} - \eta \cdot \frac{\partial L}{\partial b}
$$

```python
learning_rate = 0.01

m_new = m - learning_rate * dL_dm
b_new = b - learning_rate * dL_db

print(f"Updated parameters: m={m_new}, b={b_new}")
```

```text
Updated parameters: m=[0.50000062], b=[0.26282134]
```

4.  **Repeat**: We iterate through this process and repeat the forward and backward passes for many *epochs*. Eventually, the loss will start to reach an equilibrium, and the gradients become nearly zero.

```python
n_epochs = 200

for i in range(n_epochs):

    #Forward pass
    y_pred = m * X_train + b
    L = np.mean((y_pred - Y_train)**2)

    #Backward pass
    dL_dm = (-2 * (X_train * (Y_train - y_pred))).mean(axis=0)
    dL_db = (-2 * (Y_train - y_pred)).mean(axis=0)

    #Update parameters
    m_new = m - learning_rate * dL_dm
    b_new = b - learning_rate * dL_db

    m = m_new
    b = b_new

    if i % 20 == 0:
        print(f"Epoch {i}, Loss: {L}")


print("--------------------------------")
print(f"Final loss: {L}")
print(f"Final parameters: m={m}, b={b}")
```

```text
Epoch 0, Loss: 41.7152270200187
Epoch 20, Loss: 22.428890389096672
Epoch 40, Loss: 14.677830722081453
Epoch 60, Loss: 11.556057322281418
Epoch 80, Loss: 10.295363104303991
Epoch 100, Loss: 9.78453009391309
Epoch 120, Loss: 9.576674053357408
Epoch 140, Loss: 9.491661853080508
Epoch 160, Loss: 9.456673445047354
Epoch 180, Loss: 9.442164148467723
--------------------------------
Final loss: 9.43628843364173
Final parameters: m=[3.06946794], b=[4.76437659]
```

As you can see, we were able to achieve almost exactly the same slope and intercept with backpropagation as we did with the analytical solution! 
* `(Coefficient (m): [[3.07464887]], Intercept (b): [4.83065891])`

### Pytorch Implementation

In a real-world machine learning scenario, we would not calculate the derivatives for each parameter by hand --in modern Large Language Models, there are hundreds of BILLIONS of parameters. Instead, many machine learning researchers and engineers use libraries like `pytorch`, `tensorflow` and `jax`. These libraries are very powerful and do a lot of stuff for you under the hood (like automatically calculating gradients). Let's see how this is done in `pytorch`...

```python
import torch

#Instead of numpy arrays, the pytorch-native data structure is the tensor.
X_train_pt = torch.tensor(X_train)
X_test_pt = torch.tensor(X_test)
Y_train_pt = torch.tensor(Y_train)
Y_test_pt = torch.tensor(Y_test)
```

Next, we initialize our parameters $m$ (slope) and $b$ (intercept) as we did with the numpy implementation, notice the similarity in `torch.randn()`. We set `requires_grad=True` to tell PyTorch to track operations on these tensors so it can calculate gradients later. Under the hood, pytorch creates a *directed acyclic graph* or *DAG* to automatically solve the chain rule and calculate the gradients for you. As networks get very complex and much deeper, this becomes incredibly useful.

```python
m = torch.randn(1); m.requires_grad = True
b = torch.randn(1); b.requires_grad = True
print(f"Initial random weights: m={m.item():.2f}, b={b.item():.2f}")
```

```text
Initial random weights: m=-1.33, b=-0.78
```

Let's implement the forward pass:

```python
y_pred = m * X_train_pt + b
loss = ((y_pred - Y_train_pt)**2).mean()
print(f"Initial loss: {loss.item():.4f}")
```

```text
Initial loss: 67.1437
```

Now let's implement the backward pass:

```python
loss.backward() #thats it!
```

Finally, let's update the parameters. We previously told pytorch to track and calculate gradients related to $m$ and $b$, but here we want to explicitly tell pytorch to not track this update step in the gradient graph.

```python
learning_rate = 0.01

with torch.no_grad():
    m -= learning_rate * m.grad
    b -= learning_rate * b.grad
print(f"Updated parameters: m={m.item():.2f}, b={b.item():.2f}")
```

```text
Updated parameters: m=-1.23, b=-0.65
```

Now let's put these in a training loop!

```python
n_epochs = 20
learning_rate = 0.01

for i in range(n_epochs):
    # 1. Forward pass
    y_pred = m * X_train_pt + b
    
    # 2. Calculate Loss
    loss = ((y_pred - Y_train_pt)**2).mean()

    # 3. Backward Pass
    loss.backward()

    # 4. Update parameters
    # We wrap in torch.no_grad() because we don't want to track this update step in the gradient graph
    with torch.no_grad():
        m -= learning_rate * m.grad
        b -= learning_rate * b.grad

    print(f"Epoch {i}, Loss: {loss.item():.4f}")

print(f"\nFinal parameters: m={m.item():.2f}, b={b.item():.2f}")
```

```text
Epoch 0, Loss: 64.5527
Epoch 1, Loss: 59.6049
Epoch 2, Loss: 52.7470
Epoch 3, Loss: 44.5986
Epoch 4, Loss: 35.8955
Epoch 5, Loss: 27.4239
Epoch 6, Loss: 19.9487
Epoch 7, Loss: 14.1451
Epoch 8, Loss: 10.5371
Epoch 9, Loss: 9.4504
Epoch 10, Loss: 10.9829
Epoch 11, Loss: 14.9959
Epoch 12, Loss: 21.1267
Epoch 13, Loss: 28.8213
Epoch 14, Loss: 37.3846
Epoch 15, Loss: 46.0429
Epoch 16, Loss: 54.0143
Epoch 17, Loss: 60.5788
Epoch 18, Loss: 65.1436
Epoch 19, Loss: 67.2968

Final parameters: m=7.49, b=10.41
```

Woah, that's weird, look at the loss! If you are not a hardened machine learning engineer, you might not notice that there is a terrible bug in the code. Let me explain...

If we print out the gradients of $m$ and $b$ across each iteration:

```
m.grad: tensor([4.5969]), b.grad: tensor([9.6682])
m.grad: tensor([9.0756]), b.grad: tensor([19.1290])
m.grad: tensor([13.3207]), b.grad: tensor([28.1798])
m.grad: tensor([17.2227]), b.grad: tensor([36.6264])
m.grad: tensor([23.6035]), b.grad: tensor([51.0014])
m.grad: tensor([25.9155]), b.grad: tensor([56.6229])
m.grad: tensor([27.5543]), b.grad: tensor([61.0333])
m.grad: tensor([28.4749]), b.grad: tensor([64.1392])
m.grad: tensor([28.6501]), b.grad: tensor([65.8759])
m.grad: tensor([28.0712]), b.grad: tensor([66.2080])
m.grad: tensor([26.7484]), b.grad: tensor([65.1307])
m.grad: tensor([24.7105]), b.grad: tensor([62.6696])
m.grad: tensor([22.0045]), b.grad: tensor([58.8800])
m.grad: tensor([18.6942]), b.grad: tensor([53.8459])
m.grad: tensor([14.8588]), b.grad: tensor([47.6782])
m.grad: tensor([10.5912]), b.grad: tensor([40.5117])
m.grad: tensor([5.9958]), b.grad: tensor([32.5029])
m.grad: tensor([1.1858]), b.grad: tensor([23.8258])
m.grad: tensor([-3.7196]), b.grad: tensor([14.6685])
```

...we can see that they are very unstable. This is because pytorch does a `+=` to the gradients when `backward()` is called. So the gradients will just continue to grow and grow, and the loss will never stabilize. To fix this, we need to manually set the gradients to zero at each iteration. This was not a problem in the numpy implementation because I was manually calculating the gradients at each iteration, and overwriting the previous ones.

```python
n_epochs = 200
learning_rate = 0.01

for i in range(n_epochs):
    # 1. Forward pass
    y_pred = m * X_train_pt + b
    
    # 2. Calculate Loss
    loss = ((y_pred - Y_train_pt)**2).mean()

    # Clear gradients from previous step
    if m.grad is not None:
        m.grad.zero_()
    if b.grad is not None:
        b.grad.zero_()

    # 3. Backward Pass
    loss.backward()

    # 4. Update parameters
    # We wrap in torch.no_grad() because we don't want to track this update step in the gradient graph
    with torch.no_grad():
        m -= learning_rate * m.grad
        b -= learning_rate * b.grad

    if i % 10 == 0:
        print(f"Epoch {i}, Loss: {loss.item():.4f}")

print(f"\nFinal parameters: m={m.item():.2f}, b={b.item():.2f}")
```

```text
Epoch 0, Loss: 9.7988
Epoch 10, Loss: 9.6641
Epoch 20, Loss: 9.5789
Epoch 30, Loss: 9.5250
Epoch 40, Loss: 9.4908
Epoch 50, Loss: 9.4691
Epoch 60, Loss: 9.4554
Epoch 70, Loss: 9.4467
Epoch 80, Loss: 9.4412
Epoch 90, Loss: 9.4377
Epoch 100, Loss: 9.4355
Epoch 110, Loss: 9.4341
Epoch 120, Loss: 9.4332
Epoch 130, Loss: 9.4326
Epoch 140, Loss: 9.4322
Epoch 150, Loss: 9.4320
Epoch 160, Loss: 9.4318
Epoch 170, Loss: 9.4317
Epoch 180, Loss: 9.4317
Epoch 190, Loss: 9.4316

Final parameters: m=3.07, b=4.82
```

As expected, we achieve the same parameters as the analytical solution and as the numpy implementation! 

```python
plt.figure(figsize=(8, 6))
plt.scatter(X, Y)
plt.plot(X, m.item() * X + b.item(), color='red')
plt.title("Line of Best Fit")
plt.show()
```

<p align="center">
  <img src="/assets/images/linear_regression_backprop/output_cell31.png" width="500" />
</p>

You might be wondering *If the OLS matrix solution allows us to directly solve for the parameters, why bother with pytorch and backpropagation?* 

The honest answer is that for simple Linear Regression, you usually shouldn't. The direct solution is faster and exact. However, as models become deeper and non-linear, direct mathematical solutions simply don't exist. By trading the perfect mathematical solution for an iterative approximation, we gain the ability to train complex neural networks on massive, unstructured datasets.

I wanted to prove to you that with enough iterations, the *learning* approach converges to the same ground truth as the math.

The next time you see a headline about an AI model learning a new skill, just remember: at the bottom of it all, it's just trying to find the best $m$'s and $b$'s to minimize a loss function.
