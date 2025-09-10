# Deep Learning Advanced Concepts & Implementation Notes  

These notes summarize **key deep learning concepts** (from theoretical foundations to ANN & CNN practicals) with detailed definitions, mathematical formulas, and full pipelines.  

---

## 1. Chain Rule, Backpropagation & Gradient Descent  

### 🔹 Forward Propagation & Backpropagation  
- **Forward Propagation**:  
  Input data flows through layers of the network, applying weights, biases, and activation functions → final prediction.  

- **Backpropagation**:  
  The error (loss) is propagated backward from the output to earlier layers.  
  Gradients are computed using the **chain rule of derivatives**, updating weights in the direction that reduces loss.  

---

### 🔹 Chain Rule of Derivatives  
Backprop relies on chain rule because the loss depends on weights indirectly via multiple layers.  

**Formula for Weight Update:**  

\[
w_{new} = w_{old} - \eta \cdot \frac{\partial \text{Loss}}{\partial w_{old}}
\]  

Where:  
- \(w_{new}\): updated weight  
- \(w_{old}\): old weight  
- \(\eta\): learning rate  
- \(\frac{\partial \text{Loss}}{\partial w_{old}}\): gradient of loss wrt weight  

---

### 🔹 Gradient Descent on Loss Landscape  
- If gradient (slope) is **positive**, weight decreases.  
- If gradient (slope) is **negative**, weight increases.  
- Helps weights move “downhill” on the loss surface.  

**Learning Rate Importance:**  
- **Too high (large η)** → weights oscillate, fail to converge.  
- **Too low (small η)** → convergence is very slow.  

---

## 2. Vanishing Gradient Problem  

### 🔹 Explanation  
- Occurs in **deep networks** when gradients get very small as they are backpropagated.  
- Caused by multiplying many small derivatives (from sigmoid/tanh activations).  

**Example with Sigmoid:**  
- Derivative of sigmoid is in range \([0, 0.25]\).  
- In deep layers, multiplying repeatedly → gradient → 0.  
- Result: weights in earlier layers barely update → **network fails to learn**.  

---

## 3. Activation Functions  

Activation functions introduce **non-linearity** into neural networks.  

### 🔹 Sigmoid Function  
\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]  

- Range: (0, 1).  
- Smooth gradient.  
- **Problems**:  
  - Outputs not zero-centered.  
  - Causes **vanishing gradients**.  
  - Slow convergence.  

---

### 🔹 Tanh Function  
\[
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
\]  

- Range: (-1, 1).  
- Zero-centered → better than sigmoid.  
- Still suffers from vanishing gradients in deep nets.  

---

### 🔹 ReLU (Rectified Linear Unit)  
\[
f(x) = \max(0, x)
\]  

- For \(x < 0\), output = 0.  
- For \(x \geq 0\), output = x.  
- Solves vanishing gradient problem for positive side.  
- **Issue**: “Dead Neurons” when gradient = 0 for many inputs.  

---

### 🔹 Leaky ReLU  
\[
f(x) = \begin{cases}  
x & x > 0 \\  
0.01x & x \leq 0  
\end{cases}
\]  

- Allows a small slope for negative inputs.  
- Prevents dead neurons.  

---

### 🔹 Other Variants  
- **ELU**: Smooth curve for negatives, avoids sharp cutoff.  
- **PReLU**: Learnable negative slope.  
- **Swish**: Smooth, often better than ReLU in deep models.  

---

### 🔹 Guidelines for Usage  
- Hidden Layers → **ReLU** or its variants.  
- Output Layer → depends on task:  
  - **Binary Classification** → Sigmoid.  
  - **Multi-class Classification** → Softmax.  
  - **Regression** → Linear.  

---

## 4. Loss Functions  

Loss functions measure the difference between predicted and actual values.  

### 🔹 For Regression  
1. **Mean Squared Error (MSE):**  
\[
\text{MSE} = \frac{1}{n}\sum (y - \hat{y})^2
\]  

2. **Mean Absolute Error (MAE):**  
\[
\text{MAE} = \frac{1}{n}\sum |y - \hat{y}|
\]  

3. **Huber Loss:**  
- Combines MSE + MAE.  
- Robust to outliers.  

---

### 🔹 For Classification  
1. **Binary Cross Entropy (BCE):**  
\[
L = -\frac{1}{N}\sum \Big[ y \log(\hat{y}) + (1-y)\log(1-\hat{y}) \Big]
\]  

2. **Categorical Cross Entropy (CCE):**  
- For multi-class problems.  
- Uses **softmax outputs**.  

---

## 5. Optimizers  

Optimizers adjust weights based on gradients to minimize loss.  

### 🔹 Gradient Descent (GD)  
- Computes gradient using the **entire dataset**.  
- Accurate, but very slow and resource heavy.  

### 🔹 Stochastic Gradient Descent (SGD)  
- Updates after **each sample**.  
- Fast but noisy convergence.  

### 🔹 Mini-Batch SGD  
- Updates after a small batch of samples.  
- Balance between speed & stability.  

---

### 🔹 SGD with Momentum  
- Adds an exponentially weighted average of past gradients.  
- Smooths updates, accelerates convergence.  

---

### 🔹 Adaptive Methods  
1. **AdaGrad:**  
   - Individual learning rate per parameter.  
   - Learning rate decays over time.  

2. **RMSProp:**  
   - Maintains exponential moving average of squared gradients.  
   - Normalizes learning rate.  

3. **Adam (Most Popular):**  
   - Combines Momentum + RMSProp.  
   - Uses \(\beta_1, \beta_2\) for exponential averages.  
   - Best default optimizer in deep learning.  

---

## 6. Artificial Neural Networks (ANN): Practical Implementation  

### 🔹 Dataset Example: Customer Churn  

**Steps:**  
1. **Data Preprocessing**  
   - Separate features (X) and target (y).  
   - Handle categorical variables → one-hot encoding (`pandas.get_dummies`).  
   - Train-test split.  
   - Apply feature scaling (`StandardScaler`).  

2. **ANN Model Building**  
   - Use Keras `Sequential` model.  
   - Dense hidden layers with ReLU.  
   - Sigmoid activation in output (binary classification).  
   - Dropout layers to prevent overfitting.  

3. **Model Compilation & Training**  
   - Optimizer → Adam.  
   - Loss → Binary Crossentropy.  
   - Metric → Accuracy.  
   - Early stopping → stop when validation loss plateaus.  

4. **Evaluation**  
   - Confusion matrix.  
   - Accuracy score.  

---

### 🔹 White Box vs Black Box  
- **White box**: interpretable (e.g., Decision Trees).  
- **Black box**: hard to interpret (ANN, Random Forest, XGBoost).  

---

## 7. Convolutional Neural Networks (CNN)  

### 🔹 Motivation  
- Inspired by **human visual cortex**.  
- Excellent for **image recognition** tasks.  

---

### 🔹 Image Basics  
- **B/W images**: single channel (0–255).  
- **RGB images**: 3 channels (R, G, B).  

---

### 🔹 Convolution Operation  
- Apply filter (kernel) across image.  
- Detects features like edges, textures.  

**Output Size Formula:**  
\[
\text{Output Size} = \frac{(n + 2p - f)}{s} + 1
\]  

Where:  
- \(n\): input size  
- \(p\): padding  
- \(f\): filter size  
- \(s\): stride  

---

### 🔹 Padding  
- Adds border pixels to preserve spatial dimensions.  
- Example: “same” padding → input size = output size.  

### 🔹 Stride  
- Step size of filter movement.  

---

### 🔹 Activation after Convolution  
- Typically **ReLU** applied → introduces non-linearity.  

---

### 🔹 Pooling Layer  
- **Max Pooling (2×2):** selects maximum from each patch.  
- Reduces size, provides translation invariance.  
- Alternatives: average pooling, min pooling.  

---

### 🔹 Flattening  
- Converts pooled feature maps → 1D vector.  

---

### 🔹 Fully Connected Layer  
- Final dense layers perform classification.  

---

### 🔹 CIFAR-10 Example  
- Dataset: 60k images (32×32, 10 classes).  
- Architecture:  
  - Conv2D (32 filters) + ReLU  
  - Conv2D (64 filters) + ReLU  
  - MaxPooling2D  
  - Flatten  
  - Dense + Softmax output  

- Optimizer: Adam  
- Loss: Sparse Categorical Crossentropy  
- Achieved **~78% accuracy in 10 epochs**.  
- Use early stopping and transfer learning for better results.  

---

# 🔑 Summary  

- **Chain Rule**: foundation of backprop.  
- **Vanishing Gradients**: key issue with sigmoid/tanh in deep nets.  
- **ReLU & Variants**: preferred for hidden layers.  
- **Loss Functions**: different for regression vs classification.  
- **Optimizers**: Adam widely used due to stability & performance.  
- **ANN Pipeline**: preprocess → model → train → evaluate.  
- **CNN Pipeline**: convolution → activation → pooling → flatten → dense → output.  

---

