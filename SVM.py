import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import scipy.fftpack as fftpack
import joblib
import torch

def compute_dct_features(image_tensor, keep_coeffs=None):
    """
    Compute 2D DCT features from an image tensor of shape (B, C, H, W).
    If keep_coeffs is set, only the top-left keep_coeffs x keep_coeffs
    block of each channel is retained (these hold most of the energy).
    Returns a numpy array of shape (B, feature_dim).
    """

    batch_size, height, width, channels = image_tensor.shape

    features = []
    for batch in range(batch_size):
        if channels == 3:
            gray_diff = 0.299 * image_tensor[batch, :, :, 0] + 0.587 * image_tensor[batch, :, :, 1] + 0.114 * image_tensor[batch, :, :, 2]
        else:
            gray_diff = image_tensor[batch, :, :, 0]
        dct_result = fftpack.dct(
            fftpack.dct(gray_diff, axis=0, norm='ortho'),
            axis=1, norm='ortho'
        )
        log_abs_dct = np.log(np.abs(dct_result) + 1e-8)
        if keep_coeffs is not None:
            log_abs_dct = log_abs_dct[:keep_coeffs, :keep_coeffs]     
        features.append(log_abs_dct.flatten())

    return np.array(features)



def extract_features(prn_model, images, device, keep_coeffs=8, batch_size=64):
    """
    Full feature extraction: PRN -> residual (input - rectified) -> DCT.
    Returns a numpy matrix ready for the SVM.
    """
    all_features = []
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i : i + batch_size]
        rectified = prn_model.predict(batch_imgs)
        diff = batch_imgs - rectified
        batch_feats = compute_dct_features(diff, keep_coeffs=keep_coeffs)
        all_features.append(batch_feats)        

    return np.vstack(all_features)

class PerturbationDetectorSVM:
    """
    SVM detector that wraps around a PRN.
    Label 0 = clean, Label 1 = adversarial.
    """

    def __init__(self, prn_model, device, keep_coeffs=8,
                 kernel='rbf', C=1.0, gamma='scale'):
        self.prn_model = prn_model
        self.device = device
        self.keep_coeffs = keep_coeffs

        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel=kernel, C=C, gamma=gamma, probability=True))
        ])
        self.is_trained = False

    def extract(self, images):
        return extract_features(
            self.prn_model, images, self.device,
            keep_coeffs=self.keep_coeffs
        )

    def fit(self, clean_images, adversarial_images):
        clean_feats = self.extract(clean_images)
        adv_feats = self.extract(adversarial_images)

        X = np.vstack([clean_feats, adv_feats])
        y = np.concatenate([
            np.zeros(len(clean_feats)),
            np.ones(len(adv_feats))
        ])

        perm = np.random.permutation(len(y))
        X, y = X[perm], y[perm]

        self.pipeline.fit(X, y)
        self.is_trained = True

        train_acc = accuracy_score(y, self.pipeline.predict(X))
        print(f"Training accuracy: {train_acc:.4f}")
        return train_acc

    def predict(self, images):
        if not self.is_trained:
            raise RuntimeError("SVM must be trained before prediction")
        return self.pipeline.predict(self.extract(images))

    def predict_proba(self, images):
        if not self.is_trained:
            raise RuntimeError("SVM must be trained before prediction")
        return self.pipeline.predict_proba(self.extract(images))

    def evaluate(self, clean_images, adversarial_images):
        clean_preds = self.predict(clean_images)
        adv_preds = self.predict(adversarial_images)

        fpr = float(np.mean(clean_preds == 1))
        tpr = float(np.mean(adv_preds == 1))
        fnr = float(np.mean(adv_preds == 0))
        tnr = float(np.mean(clean_preds == 0))

        y_true = np.concatenate([
            np.zeros(len(clean_preds)),
            np.ones(len(adv_preds))
        ])
        y_pred = np.concatenate([clean_preds, adv_preds])

        print("Detector Evaluation")
        print("-" * 40)
        print(f"False Positive Rate (FPR): {fpr:.4f}")
        print(f"True Positive Rate  (TPR): {tpr:.4f}")
        print(f"False Negative Rate (FNR): {fnr:.4f}")
        print(f"True Negative Rate  (TNR): {tnr:.4f}")
        print("-" * 40)
        print(classification_report(
            y_true, y_pred, target_names=['clean', 'adversarial']
        ))

        return {
            'fpr': fpr, 'tpr': tpr,
            'fnr': fnr, 'tnr': tnr,
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }

    def save(self, path):
        joblib.dump(self.pipeline, path)

    def load(self, path):
        self.pipeline = joblib.load(path)
        self.is_trained = True


