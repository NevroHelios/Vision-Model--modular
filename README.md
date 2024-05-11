# Vision Model - Modular

Welcome to the Visioon Model repository! This repository contains modular code for training and testing computer vision models.

## Directory Structure

```
Visioon Model -modular/
│
├── data/
│   ├── train         # train data
│   ├── test          # test data
│
├── going_modular/
│   ├── __pycache__/
│   ├── data_setup.py        # Script for setting up the dataset
│   ├── engine.py            # Script containing training and testing engines
│   ├── get_data.py          # Script for downloading data
│   ├── model_builder.py     # Script for building the model architecture
│   ├── predict.py           # Script for making predictions
│   ├── train.py             # Script for training the model
│   └── utils.py             # Utility functions
│
├── models/                   # Directory for storing trained models
│
├── requirements.txt          # List of dependencies for the project
└── torchvision_classification_modular.ipynb  # Jupyter notebook for classification with torchvision

```

## Setting up the Environment

To set up the environment for running the code, follow these steps:

1. **Create and Activate Virtual Environment**:
   
   ```python -m venv venv```
   
   ```venv\Scripts\activate      # For Windows```

3. **Install Dependencies**: Install the required dependencies by running:
   
   ``` pip install -r requirements.txt ```

## Training the Model

To train the model, follow these steps:

1. **Data Setup**: Set up the dataset using the `data_setup.py` script or provide the necessary data in the appropriate format.

2. **Training Script**: Run the training script:
   
   ``` python going_modular/train.py --learning_rate 0.01 --batch_size 32 --num_epochs 5 ```

## Making Predictions

To make predictions using the trained model, you can use the `predict.py` script.


   ```  python going_modular/predict.py --img <file-name>.<type> ```
