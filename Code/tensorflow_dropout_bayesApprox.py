import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import Bernoulli
import matplotlib.pyplot as plt


class VariationalDense:
    "Variational Dense Layer"
    
    def __init__(self, n_in, n_out, model_prob, lam):
        # Set the model probability and initialize the Bernoulli distribution
        self.model_prob = model_prob
        self.model_bern = Bernoulli(probs=model_prob, dtype=tf.float32)
        self.lam = lam

        # Initialize weights and bias
        self.model_M = tf.Variable(
            tf.random.truncated_normal([n_in, n_out], stddev=0.01))
        self.model_m = tf.Variable(tf.zeros([n_out]))

        # Sample from Bernoulli distribution for variational inference
        self.mask = self.model_bern.sample([n_in, n_out])  # Sample a mask
        self.model_W = tf.multiply(self.mask, self.model_M)  # Apply mask to weights

    def __call__(self, X, activation=tf.identity):
        output = activation(tf.matmul(X, self.model_W) + self.model_m)
        if self.model_M.shape[1] == 1:
            output = tf.squeeze(output)
        return output

    @property
    def regularization(self):
        return self.lam * (
            self.model_prob * tf.reduce_sum(tf.square(self.model_M) + tf.square(self.model_m))
        )

# Create sample data 
n_samples = 100
X = np.random.normal(size=(n_samples,1))
y = np.random.normal(5. * np.cos(5.*X) / (np.abs(X) + 1.), 0.1)
X_pred = np.atleast_2d(np.linspace(-3., 3., num=100)).T
X = np.hstack((X, X**2, X**3))
X_pred = np.hstack((X_pred, X_pred**2, X_pred**3))

# Create tensorflow model
# Convert numpy arrays to tf.Tensor
X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

# Define model parameters
n_feats = X.shape[1]
n_hidden = 100
model_prob = 0.9
model_lam = 1e-2

# Create the model (variational dense layer)
model_L_1 = VariationalDense(n_feats, n_hidden, model_prob, model_lam)
model_L_2 = VariationalDense(n_hidden, n_hidden, model_prob, model_lam)
model_L_3 = VariationalDense(n_hidden, 1, model_prob, model_lam)

# Forward pass
model_out_1 = model_L_1(X_tensor, tf.nn.relu)
model_out_2 = model_L_2(model_out_1, tf.nn.relu)
model_pred = model_L_3(model_out_2)

# Calculate the loss function
model_sse = tf.reduce_sum(tf.square(y_tensor - model_pred))
model_mse = model_sse / n_samples
model_loss = (
    # Negative log-likelihood.
    model_sse +
    # Regularization.
    model_L_1.regularization +
    model_L_2.regularization +
    model_L_3.regularization
) / n_samples

# Optimizer and training step
optimizer = tf.optimizers.Adam(learning_rate=1e-3)

# @tf.function
# def train_step():
#     with tf.GradientTape() as tape:
#         loss_value = model_loss
#         gradients = tape.gradient(loss_value, model_L_1.__dict__.values() + model_L_2.__dict__.values() + model_L_3.__dict__.values())
#         optimizer.apply_gradients(zip(gradients, model_L_1.__dict__.values() + model_L_2.__dict__.values() + model_L_3.__dict__.values()))
#     return loss_value

@tf.function
def train_step():
    with tf.GradientTape() as tape:
        loss_value = model_loss
    # Collecting all model variables manually
    variables = (
        list(model_L_1.__dict__.values()) +
        list(model_L_2.__dict__.values()) +
        list(model_L_3.__dict__.values())
    )
    gradients = tape.gradient(loss_value, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss_value

# Training loop
n_epochs = 10000
for epoch in range(n_epochs):
    loss_value = train_step()
    if epoch % 100 == 0:
        mse_value = model_mse.numpy()  # Convert the tensor to a numpy value for display
        print(f"Iteration {epoch}. Mean squared error: {mse_value:.4f}.")

# Sample from the posterior using the trained model
n_post = 1000
Y_post = np.zeros((n_post, X_pred.shape[0]))
for i in range(n_post):
    Y_post[i] = model_pred.numpy()


if True:
    plt.figure(figsize=(17,7))
    plt.plot(X[:,0], y, 'r.')
    plt.grid(True)
    plt.show()
