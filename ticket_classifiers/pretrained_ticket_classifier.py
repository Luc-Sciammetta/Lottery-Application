import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Subset
from torchvision import models

try: #so that the imports work whether this file is run directly or imported into api_main.py
    from ticket_classifiers.ticket_classifier import device, count_files_in_directory
except ModuleNotFoundError:
    from ticket_classifier import device, count_files_in_directory

train_ratio = 0.7
validation_ratio = 0.2
test_ratio = 0.1

include_logos = True 

train_transform = transforms.Compose([
    transforms.Resize((224, 224)), #resizes each image to 224x224 pixels
                                   #since this is what ResNet18 expects as input
    transforms.RandomHorizontalFlip(p=0.5), #randomly flips the image horizontally
    transforms.RandomRotation(5), 
    transforms.ColorJitter(brightness=0.1, contrast=0.1), #randomly changes brightness and contrast
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), #randomly crops and resizes the image
    transforms.RandomPerspective(distortion_scale=0.1, p=0.3), #randomly applies perspective transformation
    transforms.ToTensor(), #converts the image to a PyTorch tensor
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406], #normalizes the image tensor with mean and std values
        std = [0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])

#NOTE: this dataset code needs to be used here and cannot be imported since the imported code will use
#      images that are resized to 112x112 instead of 224x224 (it uses the ticket_classifier.py transforms)
full_dataset = ImageFolder(root="images", transform=test_transform)

if include_logos:
    non_logo_indicies = list(range(len(full_dataset))) #make a list of all the indicies in the dataset
else:
    #filter for non-logo images
    print("Filtering out logo images...")
    non_logo_indicies = [
        i for i, (path, _) in enumerate(full_dataset.samples)
        if "logo" not in os.path.basename(path).lower()
    ]

filtered_dataset = Subset(full_dataset, non_logo_indicies) #the filtered dataset 

train_size = int(len(filtered_dataset) * train_ratio)
validation_size = int(len(filtered_dataset) * validation_ratio)
test_size = len(filtered_dataset) - train_size - validation_size

generator = torch.Generator().manual_seed(42)
train_subset, validation_subset, test_subset = random_split(
    filtered_dataset, [train_size, validation_size, test_size], generator=generator
)

train_original_indices = [non_logo_indicies[i] for i in train_subset.indices]
validation_original_indices = [non_logo_indicies[i] for i in validation_subset.indices]
test_original_indices = [non_logo_indicies[i] for i in test_subset.indices]

train_dataset = Subset(ImageFolder(root="images", transform=train_transform), train_original_indices)
validation_dataset = Subset(ImageFolder(root="images", transform=test_transform), validation_original_indices)
test_dataset = Subset(ImageFolder(root="images", transform=test_transform), test_original_indices)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def load_model(filepath):
    """ Load the trained CNN model from a file.
    Args:
        filepath (str): The path to the file containing the model weights.
    Returns:
        SimpleCNN: The loaded CNN model.
    """
    model = create_model(num_classes=4)
    model.load_state_dict(torch.load(filepath))
    model.to(device)
    model.eval()
    return model

#NOTE: the above not is also the reason why this is copy pasted
def test_model(model):
    """ Test the CNN model on the test dataset.
    Args:
        model (SimpleCNN): The trained CNN model.
    Returns:
        float: The accuracy of the model on the test dataset.
    """
    model.eval() #sets the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad(): #tells PyTorch not to calculate gradients (saves memory and computations)
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device) #does things with putting the images and model on GPU/CPU

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * (correct / total)
    print(f"Test Accuracy: {accuracy:.2f}%")

    return accuracy

def predict_image(model, image_path, print_out=False):
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

    if print_out:
        print(f"\nImage: {image_path}")
        print("Class confidences:")

        for i, class_name in enumerate(class_names):
            print(
                f"  {class_name:<15}: {probs[i].item() * 100:.2f}%"
            )

        print(f"\nPredicted Class: {class_names[pred_idx]}")

    return class_names[pred_idx]

#NOTE: the above not is also the reason why this is copy pasted
#!--------------- REMOVE THIS LATER -----------------
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
#!--------------- END REMOVE THIS LATER -----------------

def create_model(num_classes = 4):
    """
    Creates a ResNet18 model that is pre-trained. Replaces the final layer to have only 4 outputs of the different
    lottery ticket games
    
    Args:
        num_classes (int): this should always be 4
    Returns:
        model: the model that it created
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) #loads a pretrained ResNet18 model

    for param in model.parameters():
        param.requires_grad = False #freezes the model's parameters so that they dont change during training

    model.fc = nn.Sequential(
        nn.Dropout(0.5), 
        nn.Linear(model.fc.in_features, num_classes) #replaces the final fully connected layer with a new one that has the correct number of output classes
    )

    return model

def train_model(epochs = 10, patience = 5, savepath="model_weights.pth"):
    """ train the CNN model on the dataset.
    Args:
        epochs (int): Number of training epochs.
        patience (int): Number of epochs to wait for improvement before stopping.
        savepath (str): Path to save the trained model weights.
    Returns:
        SimpleCNN: The trained CNN model.
    """
    model = create_model().to(device)

    #calculate class weights to handle class image data counts imbalance
    class_counts = torch.tensor([count_files_in_directory("images/powerball"), count_files_in_directory("images/euromillions"), count_files_in_directory("images/lottoamerica"), count_files_in_directory("images/megamillions")], dtype=torch.float)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(class_counts)
    loss_function = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.05)                                                                        

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=1e-4) 
                # the filter part tells the optimizer to only include the paramters that are not frozen in the model.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
                                                                                                    
    best_validation_loss = float('inf') #initialize the best validation loss to infinity
    num_epochs_no_improvement = 0 #counter for the number of epochs with no improvement

    num_epochs = epochs 
    for epoch in range(num_epochs):
        model.train() 

        running_loss = 0.0 #this is the loss over all batches for this epoch
        correct = 0 #number of correct predictions
        total = 0 #total number of predictions

        for images, labels in train_dataloader: #gives batches of 32 images and their corresponding labels
            images, labels = images.to(device), labels.to(device)  #does things with putting the images and model on GPU/CPU

            optimizer.zero_grad() #clears old gradients from the previous batch
            outputs = model(images) #feeds the batch images into the model to get the predictions

            loss = loss_function(outputs, labels) #calculates the loss between the predictions and the true labels

            loss.backward() #computes the gradients for the weights
            optimizer.step() #updates the weights based on the gradients

            running_loss += loss.item() #adds the loss for this batch to the running loss

            _, predicted = torch.max(outputs.data, 1) 
            total += labels.size(0) #updates the total number of predictions
            correct += (predicted == labels).sum().item() #updates the number of correct predictions

        accuracy = 100*(correct/total) #calculates the accuracy for this epoch
        print(f"Training Epoch {epoch+1}, Loss: {running_loss:.3f}, Accuracy: {accuracy:.2f}%")

        #validate the model on the validation dataset
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in validation_dataloader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = loss_function(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_accuracy = 100 * (correct / total)
        val_loss /= len(validation_dataloader)
        print(f"Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_accuracy:.2f}%")

        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']}")
        scheduler.step(val_loss) #adjusts the learning rate based on the validation loss (if the validation loss does not improve for a certain number of epochs, it reduces the learning rate)

        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            num_epochs_no_improvement = 0

            torch.save(model.state_dict(), f"ticket_classifier_models/validation_models/{savepath}") #saves the model with the best weights to a file (so that we keep the best one and we have it if we need to stop)
        else:
            num_epochs_no_improvement += 1
            if num_epochs_no_improvement >= patience:
                print("Early stopping due to no improvement in validation loss.")
                break

        print()

    return model


if __name__ == "__main__":
    savepath = "model_weights.pth"

    # model = load_model("ticket_classifier_models/pt_88.52_88.89_model_weights.pth")
    # classify_all_images(model, test_logos=False)

    print(device)

    for i in range(30):
        print(f" ----- Training Run {i+1} ----- ")
        savepath = "model_weights.pth"

        model = train_model(epochs = 100, savepath=savepath, patience=15) #trains the model

        model.load_state_dict(torch.load(f"ticket_classifier_models/validation_models/{savepath}")) 
        accuracy = test_model(model) #tests the trained model

        total_accuracy = classify_all_images(model)

        torch.save(model.state_dict(), f"ticket_classifier_models/pt_{total_accuracy:.2f}_{accuracy:.2f}_{savepath}") 

