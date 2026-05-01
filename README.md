# Neural Network Defense Architecture

This is a repository for the code for Group 3 in EEL-6812. 

# Quickstart
This code was developed and testing on Python 3.6.8. Conda is highly recommended for this code. The Conda documentation can be found [here](https://docs.conda.io/en/latest/). Once Conda is installed and running, a new environment can be made by running the following commands in your terminal:

```bash
conda create --name trapdoor python=3.6.8
conda activate trapdoor
```

# Repository Installation
### 1. Clone the main repository:
```bash
mkdir EEL-6812-Final-Project
git clone https://github.com/ajm1312/EEL-6812-Final-Project.git
cd EEL-6812-Final-Project
```

### 2. Install project dependencies:

``` bash
pip3 install -r requirements.txt
```

### 3. Install the trapdoor repository
this repository relies on the [Honeypot repository](https://github.com/Shawn-Shan/trapdoor/tree/master). To clone this:
```bash
git clone https://github.com/Shawn-Shan/trapdoor.git
```

# Data and Attack Generation
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

# Training
The training file can be found in `train.ipynb.` 
1. Open `train.ipynb.` in [Visual Studio Code](https://code.visualstudio.com/).
2. Ensure the kernel (in the top right) is set to the trapdoor environment created earlier.
3. Click **"Run All"** at the top of the notebook.

Once the models have finished generating, they will be found in the `models/` folder.

# Testing
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

# Demo
The demo for this project was made using [Gradio](https://www.gradio.app/) and launched on a [Huggingface space](https://huggingface.co/).

## Online
The online version of this demo can be found [here](https://huggingface.co/spaces/ajm1312/NN_demo_docker). If the Huggingface space is sleeping, simply restart the environent and begin using the demo.

## Offline
The code for the demo can also be found in the `demo` folder. To launch the demo:

### 1. Installing Docker
The demo was built on a Docker environment. It is recommended to install [Docker Desktop](https://www.docker.com/).

### 2. Building the environment

```bash
cd demo
docker build -t uap-defense .
```
The Docker container will begin building and once the container is finished building, it should appear in the Docker Desktop app that it is running.

### 3. Launching the demo
To run the demo locally, the Docker container must be ran with:

```bash
docker run -it -p 7860:7860 uap-defense
```

Once the container is running, the demo can be access via http://localhost:7860/