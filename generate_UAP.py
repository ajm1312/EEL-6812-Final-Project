import sys
import os
# sys.path.append('/content/trapdoor/trapdoor/')
current_dir = os.getcwd()
target_dir = os.path.join(current_dir, 'trapdoor', 'trapdoor')
trapdoor_path = os.path.abspath(target_dir)
if trapdoor_path not in sys.path:
    sys.path.append(trapdoor_path)


import numpy as np
import keras
import keras.backend as K
import tensorflow as tf
import matplotlib
import pickle
matplotlib.use('Agg') # Set backend for non-interactive plotting
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()

from trap_utils import load_dataset, preprocess, init_gpu

# ---- Environment setup ----
tf.compat.v1.set_random_seed(1234)
np.random.seed(1234)

target_path = os.path.join(current_dir, 'models', 'cifar_model.h5')
# MODEL_PATH = os.path.abspath(target_path)

# with open(MODEL_PATH, 'rb') as file:
#     protected_model_data = pickle.load(file)

# Extract the actual model file path
MODEL_FILE = os.path.abspath(target_path)
print(f'Loading honeypot model from: {MODEL_FILE}')

# FIRST initialize session
sess = init_gpu("0")
K.set_learning_phase(0)

# THEN load model into that session
model = keras.models.load_model(MODEL_FILE, compile=False)


train_X, train_Y, test_X, test_Y = load_dataset('cifar')  # or cifar
train_X = preprocess(train_X, method='raw')
test_X = preprocess(test_X, method='raw') # Added preprocessing for test_X

# Use a subset (UAPs do NOT need full dataset)
# train_X = train_X[:2000]

# Pre-softmax logits
if model.layers[-1].activation.__name__ == 'softmax':
    # No explicit pre-softmax tensor available
    logits = model.output
else:
    logits = model.layers[-1].input

input_tensor = model.input

# Gradient of logits w.r.t. input
grad_tensor = K.gradients(logits, input_tensor)[0]

############ Function Defs ############
# max_passes=40
def compute_universal_perturbation(
    X,
    grad_fn,
    pred_fn,
    xi=5.0,
    delta=0.2,
    max_passes=1):
    v = np.zeros_like(X[0])

    for epoch in range(max_passes):
        fooled = 0
        np.random.shuffle(X)
        for x in X:
            pred_clean = np.argmax(pred_fn([x[None]])[0])
            pred_adv = np.argmax(pred_fn([(x + v)[None]])[0])

            if pred_clean != pred_adv:
                fooled += 1
                continue

            dv = deepfool_step(x, v, grad_fn, pred_fn)
            v = project_l2(v + dv, xi)

        fooling_rate = fooled / float(len(X))
        print('Pass {} – Fooling rate: {:.3f}'.format(epoch, fooling_rate))

        if fooling_rate >= (1 - delta):
            break

    return v

def deepfool_step(x, v, grad_fn, pred_fn):
    x_adv = x + v
    logits = pred_fn([x_adv[None]])[0]
    label = np.argmax(logits)

    grads = grad_fn([x_adv[None]])[0][0]

    grad_norm = np.linalg.norm(grads) + 1e-8
    direction = grads / grad_norm

    step_size = 1e-3  # small, stable
    return step_size * direction

def project_l2(v, xi):
    norm = np.linalg.norm(v.ravel(), ord=2)
    if norm > xi:
        v = v * (xi / norm)
    return v

def show_original_vs_uap(x, v, i, title_left='Original', title_right='UAP Perturbed'):

    # Apply universal perturbation
    x_adv = x + v

    # Clip to valid range
    x_adv = np.clip(x_adv, 0.0, 1.0)

    plt.figure(figsize=(8, 4))

    # Original image
    plt.subplot(1, 2, 1)
    if x.shape[-1] == 1:  # MNIST
        plt.imshow(x[:, :, 0], cmap='gray')
    else:  # CIFAR
        plt.imshow(x)
    plt.title(title_left)
    plt.axis('off')

    # Perturbed image
    plt.subplot(1, 2, 2)
    if x_adv.shape[-1] == 1:
        plt.imshow(x_adv[:, :, 0], cmap='gray')
    else:  # CIFAR
        plt.imshow(x_adv)
    plt.title(title_right)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('original_vs_uap_{}.png'.format(i)) # Save the figure
    plt.close() # Close the figure to free up memory

def show_perturbation(v, i):
    v_vis = v - v.min()
    v_vis = v_vis / (v_vis.max() + 1e-8)

    plt.figure(figsize=(3, 3))
    if v.shape[-1] == 1:
        plt.imshow(v_vis[:, :, 0], cmap='gray')
    else:
        plt.imshow(v_vis)
    plt.title('Universal Perturbation_{} (scaled)'.format(i))
    plt.axis('off')
    plt.savefig('universal_perturbation_{}.png'.format(i)) # Save the figure
    plt.close() # Close the figure to free up memory

def generate_real_uaps(
    X,
    model,
    grad_fn,
    pred_fn,
    num_uaps=1,
    xi=5.0,
    delta=0.2):

    uaps = []

    for i in range(num_uaps):
        print('Generating real UAP {}/{}'.format(i + 1, num_uaps))

        np.random.shuffle(X)

        v = compute_universal_perturbation(
            X,
            grad_fn,
            pred_fn,
            xi=xi,
            delta=delta
        )

        uaps.append(v)

    return uaps

def normalize_uaps(uaps):
    normed = []
    for v in uaps:
        v_norm = np.linalg.norm(v.ravel(), ord=2)
        normed.append(v / (v_norm + 1e-8))
    return normed

def generate_synthetic_uaps(
    base_uaps,
    num_synthetic=50,
    xi=5.0,
    l2_threshold=None):
    synthetic_uaps = []

    # Stack normalized UAPs
    P = np.stack(base_uaps, axis=0)

    if l2_threshold is None:
        l2_threshold = np.mean([
            np.linalg.norm(v.ravel(), ord=2) for v in base_uaps
        ])

    for i in range(num_synthetic):
        v = np.zeros_like(base_uaps[0])

        while np.linalg.norm(v.ravel(), ord=2) < xi:
            # Random coefficients (positive orthant)
            z = np.random.uniform(0, 1, size=(len(P),))
            direction = np.tensordot(z, P, axes=(0, 0))
            v += 0.1 * direction  # small step

        # Accept only if norm is comparable
        if np.linalg.norm(v.ravel(), ord=2) >= l2_threshold:
            v = v * (xi / np.linalg.norm(v.ravel(), ord=2))
            synthetic_uaps.append(v)

    return synthetic_uaps


##############################################

# Keras functions (static graph)
grad_fn = K.function(
    inputs=[input_tensor],
    outputs=[grad_tensor]
)

pred_fn = K.function(
    inputs=[input_tensor],
    outputs=[logits]
)

# Step 1: Generate real UAPs
real_uaps = generate_real_uaps(
    train_X,
    model,
    grad_fn,
    pred_fn,
    num_uaps=1,
    xi=5.0,
    delta=0.2
)

# Step 2: Normalize
normed_uaps = normalize_uaps(real_uaps)

# Step 3: Generate synthetic UAPs
synthetic_uaps = generate_synthetic_uaps(
    normed_uaps,
    num_synthetic=50,
    xi=5.0
)

# Step 4: Combine
all_uaps = real_uaps + synthetic_uaps

# np.save('multi_universal_perturbations.npy', all_uaps)
all_uaps_loaded = np.load('multi_universal_perturbations.npy', allow_pickle=True)

# Randomly pick one UAP
# for i in range(5):
#   v_idx = np.random.randint(0, all_uaps_loaded.shape[0])
#   ua = all_uaps_loaded[i*10 - 1]
#   show_original_vs_uap(test_X[0], ua, i)
#   show_perturbation(ua, i)

# for i in range(4):
#   a = all_uaps_loaded[3*i] - all_uaps_loaded[3*i+1]
#   print(np.linalg.norm(a.ravel(), ord=2))

correct = 0
fooled = 0
for i, img in enumerate(test_X):
  v_idx = np.random.randint(0, all_uaps_loaded.shape[0])
  selected_uap = all_uaps_loaded[v_idx]
  img_perturbed = img + selected_uap # Apply UAP to current image

  # Clip to valid range (0-1) after perturbation
  img_perturbed = np.clip(img_perturbed, 0.0, 1.0)

  logits = pred_fn([img_perturbed[None]])[0]
  attk_pred_label = np.argmax(logits)
  cln_pred_label = np.argmax(pred_fn([img[None]])[0])
  true_label = np.argmax(test_Y[i]) # Correctly get true label for current image
  if i % 200 == 0:
    print('Clean Pred {}| Atk Pred {} | True val {}'.format(cln_pred_label, attk_pred_label, true_label))
  if cln_pred_label == true_label:
    correct += 1
    if attk_pred_label != true_label:
      fooled += 1
print('Test accuracy:')
print(correct/len(test_X))
print('Fooling rate:')
print(fooled/correct)