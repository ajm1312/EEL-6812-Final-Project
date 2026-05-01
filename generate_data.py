import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.abspath(os.path.join(current_dir, 'data'))
trapdoor_dir = os.path.join(current_dir, 'trapdoor', 'trapdoor')
attacks_folder = os.path.abspath(os.path.join(current_dir, 'attacks'))
os.makedirs(data_folder, exist_ok=True)
if trapdoor_dir not in sys.path:
    sys.path.append(trapdoor_dir)

from trap_utils import load_dataset, preprocess, init_gpu

uap_path = os.path.join(attacks_folder, 'multi_universal_perturbations.npy')
all_uaps_loaded = np.load(uap_path, allow_pickle=True)
# all_uaps_loaded = np.load('multi_universal_perturbations.npy', allow_pickle=True)

train_X, train_Y, test_X, test_Y = load_dataset('cifar')
train_X = preprocess(train_X, method='raw')
test_X = preprocess(test_X, method='raw')

def generate_mixed_dataset(X, Y, uaps, attack_prob=0.5):
    mixed_X = []
    attack_labels = []
    mixed_Y = []
    for i in range(len(X)):
        img = X[i]
        label = Y[i]

        if np.random.rand() < attack_prob:
            v_idx = np.random.randint(0, uaps.shape[0])
            selected_uap = uaps[v_idx]
            img_perturbed = img + selected_uap
            img_final = np.clip(img_perturbed, 0.0, 1.0)
            is_attacked = 1
        else:
            img_final = img
            is_attacked = 0

        mixed_X.append(img_final)
        attack_labels.append(is_attacked)
        mixed_Y.append(label)

    return np.array(mixed_X), np.array(attack_labels), np.array(mixed_Y)

mixed_train_X, train_attack_labels, mixed_train_Y = generate_mixed_dataset(train_X, train_Y, all_uaps_loaded, attack_prob=0.5)

np.save(os.path.join(data_folder, 'mixed_train_X.npy'), mixed_train_X)
np.save(os.path.join(data_folder, 'clean_train_X.npy'), train_X) 
np.save(os.path.join(data_folder, 'train_attack_labels.npy'), train_attack_labels)
np.save(os.path.join(data_folder, 'mixed_train_Y.npy'), mixed_train_Y) 

mixed_test_X, test_attack_labels, mixed_test_Y = generate_mixed_dataset(test_X, test_Y, all_uaps_loaded, attack_prob=0.5)

np.save(os.path.join(data_folder, 'mixed_test_X.npy'), mixed_test_X)
np.save(os.path.join(data_folder, 'clean_test_X.npy'), test_X)
np.save(os.path.join(data_folder, 'test_attack_labels.npy'), test_attack_labels)
np.save(os.path.join(data_folder, 'mixed_test_Y.npy'), mixed_test_Y)

print(f'Saved mixed_test_X.npy with shape: {mixed_test_X.shape}')