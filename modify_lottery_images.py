"""
This script modifies lottery images by applying a series of transformations
such as resizing, rotation, color jittering, perspective distortion, cropping,
and blurring. It uses PyTorch's torchvision library to perform these operations.
"""

from torchvision import transforms, datasets
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.RandomResizedCrop(800, scale=(0.8, 1.0)),
    transforms.GaussianBlur(2),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("images", transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

print(dataset.classes)