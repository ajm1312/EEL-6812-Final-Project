# Neural Network Defense Architecture

This is a repository for the code for Group 3 in EEL-6812. 

## Quickstart
This code was developed and testing on Python 3.6.8. Conda is highly recommended for this code. The Conda documentation can be found [here](https://docs.conda.io/en/latest/). Once Conda is installed and running, a new environment can be made by running the following commands in your terminal:

```bash
conda create --name trapdoor python=3.6.8
conda activate trapdoor
```

## Repository Installation
### 1. Clone the main repository:
```bash
mkdir EEL-6812-Final-Project
git clone https://github.com/ajm1312/EEL-6812-Final-Project.git
cd EEL-6812-Final-Project
```

### 2. Install project dependencies:

``` bash
pip3 install -e .
```

### 3. Install the trapdoor repository
this repository relies on the [Honeypot repository](https://github.com/Shawn-Shan/trapdoor/tree/master). To clone this:
```bash
git clone https://github.com/Shawn-Shan/trapdoor/tree/master
```

## Data and Attack Generation
### 1. Attack and Data Generation

Before training the models, the Universal Adversarial Attacks and the CIFAR-10 mixed datasets must be generated.

**Generate the Universal Adversarial Attack**
```bash
python3 generate_UAP.py
```
This will create an `attacks/` folder containing `multi_universal_perturbations.npy` with the attacks.

**Generate the Data**

```bash
python3 generate_data.py
```

This will create a `data/` folder containing training and testing data samples for the model to train on.

## Training
The training file can be found in `train.ipynb.` 
1. Open `train.ipynb.` in [Visual Studio Code](https://code.visualstudio.com/).
2. Ensure the kernel (in the top right) is set to the trapdoor environment created earlier.
3. Click **"Run All"** at the top of the notebook.

Once the models have finished generating, they will be found in the `models/` folder.

## Testing
The testing file can be found in `test.ipynb.` 
1. Open `test.ipynb.` in [Visual Studio Code](https://code.visualstudio.com/).
2. Ensure the kernel (in the top right) is set to the trapdoor environment created earlier.
3. Click **"Run All"** at the top of the notebook.

Once the evaluation has finished, you can see the statistics printed at the bottom of the Jupityr Notebook cells.

# File Structure
```bash
.
├── attacks/
├── data/
├── demo/
├── models/
├── results/
├── trapdoor/
├── .gitignore
├── generate_data.py
├── generate_UAP.py
├── PRN_augmented.py
├── PRN.py
├── README.md
├── requirements.txt
├── SVM.py
├── test.ipynb
└── train.ipynb
```

