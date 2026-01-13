import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Subset

from PIL import Image

from ticket_classifier import SimpleCNN, load_model

train_ratio = 0.65
validation_ratio = 0.2
test_ratio = 0.15

#TRAIN IMAGES ONLY: defines the image transformations that will be applied to each image in the dataset
train_transform = transforms.Compose([
    transforms.Resize((112, 112)), #resizes each image to 224x224 pixels
    transforms.RandomHorizontalFlip(p=0.5), #randomly flips the image horizontally
    # transforms.RandomRotation(15), #randomly rotates the image by up to 15 degrees
    # transforms.ColorJitter(brightness=0.2, contrast=0.2), #randomly changes brightness and contrast
    # transforms.RandomResizedCrop(112, scale=(0.8, 1.0)), #randomly crops and resizes the image
    transforms.RandomPerspective(distortion_scale=0.1, p=0.3), #randomly applies perspective transformation
    transforms.ToTensor(), #converts the image to a PyTorch tensor
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406], #normalizes the image tensor with mean and std values
        std = [0.229, 0.224, 0.225]
    )
])

#TEST IMAGES ONLY: same as above but without the random augmentations
test_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])

# Create a single base dataset with a placeholder transform (e.g., test_transform as default)
full_dataset = ImageFolder(root="images", transform=test_transform)

# Determine sizes and split the dataset object directly using random_split
train_size = int(len(full_dataset) * train_ratio)
validation_size = int(len(full_dataset) * validation_ratio)
test_size = len(full_dataset) - train_size - validation_size

#Split the dataset into training, validation, and test subsets
generator = torch.Generator().manual_seed(42)  #for reproducibility
train_subset, validation_subset, test_subset = random_split(
    full_dataset, 
    [train_size, validation_size, test_size],
    generator=generator
)

#Here we create new Subset datasets that apply the correct transforms
train_dataset = Subset(
    ImageFolder(root="images", transform = train_transform),
    train_subset.indices
)
validation_dataset = Subset(
    ImageFolder(root="images", transform = test_transform),
    validation_subset.indices
)
test_dataset = Subset(
    ImageFolder(root="images", transform = test_transform),
    test_subset.indices
)

# Create the dataloaders for training and testing (they feed the data into the model in batches)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #checks to see if there is a GPU that is available for training, otherwise uses the CPU of the computer

def predict_image(model, image_path):
    """ Predict the class of a lottery ticket image and print all class confidences.
    Args:
        model (SimpleCNN): The trained CNN model.
        image_path (str): The path to the image file.
    Returns:
        str: The predicted class name.
    """
    img = Image.open(image_path).convert("RGB")
    img = test_transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1).squeeze(0)  # shape: [num_classes]

    class_names = train_dataset.dataset.classes
    pred_idx = probs.argmax().item()

    print(f"\nImage: {image_path}")
    print("Class confidences:")

    for i, class_name in enumerate(class_names):
        print(
            f"  {class_name:<15}: {probs[i].item() * 100:.2f}%"
        )

    print(f"\nPredicted Class: {class_names[pred_idx]}")

    return class_names[pred_idx]

def classify_all_images(model, image_root="images", test_logos=False):
    """
    Classify all images and print overall accuracy.
    Args:
        model (SimpleCNN): The trained CNN model.
        image_root (str): The root directory containing images.
    Returns:
        float: The overall accuracy percentage.
    """
    model.to(device)
    model.eval()

    dataset = ImageFolder(root=image_root, transform=test_transform)
    class_names = dataset.classes

    correct = 0
    total = 0

    with torch.no_grad():
        for idx, (img, label) in enumerate(dataset):
            img_path = dataset.samples[idx][0]

            # Skip images containing "logo" (case-insensitive)
            if "logo" in os.path.basename(img_path).lower() and not test_logos:
                continue

            img = img.unsqueeze(0).to(device)

            output = model(img)
            probs = torch.softmax(output, dim=1)
            pred = probs.argmax(dim=1).item()

            is_correct = pred == label
            correct += int(is_correct)
            total += 1

            print(
                f"Image: {img_path} | "
                f"True: {class_names[label]} | "
                f"Predicted: {class_names[pred]} | "
                f"Conf: {probs[0][pred].item():.2f} | "
                f"{'✓' if is_correct else '✗'}"
            )

    accuracy = 100 * correct / total
    print("\n-----------------------------------")
    print(f"Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print("-----------------------------------")

    return accuracy

def main():
    model = load_model("ticket_classifier_models/even_better_model.pth") #load a pre-trained model
    classify_all_images(model, test_logos=False) #classify all images in the dataset

    predict_image(model, "images/euromillions/img31.jpg") #path to a test image

    # model = load_model("ticket_classifier_models/37.50_model_weights.pth") #uncomment this line to load a pre-trained model instead of training a new one
    # accuracy = test_model(model) #tests the trained model

    # accuracy_avg = 0.0
    # num_tests = 10
    # for _ in range(num_tests):
    #     accuracy_avg += test_model(model)
    # accuracy_avg /= num_tests
    # print(f"Average Test Accuracy over {num_tests} runs: {accuracy_avg:.2f}%")

    # test_image_path = "images/megamillions/img10.jpg" #path to a test image
    # predict_image(model, test_image_path)

if __name__ == "__main__":
    main()