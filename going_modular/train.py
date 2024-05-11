import torch
import torch.nn as nn
import os
from tqdm.auto import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import model_builder
from data_setup import create_dataloader
from engine import train
from utils import save_model
from get_data import download_data, image_path

# for command line usages
from argparse import ArgumentParser, Namespace
parser = ArgumentParser()
parser.add_argument("-lr", "--learning_rate", type=float, help="sets the learning rate for the following model", default=0.01)
parser.add_argument("-bs", "--batch_size", type=int, help="specifies the batch size", default=32)
parser.add_argument("-ne", "--num_epochs", type=int, help="sets the numbers of epochs the model will be trined", default=5)
args = parser.parse_args()

# Set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# sample data transform
transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()]
)

# batch size
batch_size = args.batch_size

# download the data
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
if not os.path.exists(data_dir):
    download_data(image_path)

# train and test dirs
train_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'pizza_steak_sushi', 'train')
test_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'pizza_steak_sushi', 'test')

train_dataloader, test_dataloader, class_names = create_dataloader(train_dir=train_dir,
                                                                   test_dir=test_dir,
                                                                   transform=transform,
                                                                   batch_size=batch_size)

# Set number of epochs
NUM_EPOCHS = args.num_epochs

# Recreate an instance of TinyVGG
model_0 = model_builder.TinyVGG(input_shape=3, # number of color channels (3 for RGB)
                  hidden_units=10,
                  output_shape=len(class_names)).to(device)

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=args.learning_rate)

# Start the timer
from timeit import default_timer as timer
start_time = timer()

# Train model_0
if __name__ == "__main__":
    model_0_results = train(model=model_0,
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            epochs=NUM_EPOCHS,
                            device=device)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

    # Save the model
    save_model(model=model_0,
            target_dir="models",
            model_name="model.pth")
