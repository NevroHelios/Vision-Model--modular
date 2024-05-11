
import torch
from torch import nn
from PIL import Image
from pathlib import Path
from torchvision import transforms
import model_builder

# get image name
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--img", type=str, help="specify thr image path")
args = parser.parse_args()
img_path = f"/content/{args.img}"

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = model_builder.TinyVGG(input_shape=3, hidden_units=10, output_shape=3).to(device)

model_path = "/content/models/model.pth"
model.load_state_dict(torch.load(model_path))


transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

img_pil = Image.open(img_path)

img = transform(img_pil)

model.eval()
with torch.inference_mode():
  y_pred = model(img.unsqueeze(0).to(device))
  y_label = y_pred.argmax(dim=1).cpu()

class_names = ['pizza', 'steak', 'sushi']
print(class_names[y_label])
