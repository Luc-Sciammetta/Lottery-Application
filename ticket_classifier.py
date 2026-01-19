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

train_ratio = 0.65
validation_ratio = 0.2
test_ratio = 0.15

#TRAIN IMAGES ONLY: defines the image transformations that will be applied to each image in the dataset
train_transform = transforms.Compose([
    transforms.Resize((112, 112)), #resizes each image to 112x112 pixels
    transforms.RandomHorizontalFlip(p=0.5), #randomly flips the image horizontally
    transforms.RandomRotation(5), 
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

class SimpleCNN(nn.Module):
    """ A simple Convolutional Neural Network for image classification. 
    Args:
        num_classes (int): Number of output classes.

    Returns:
        torch.Tensor: The output class scores.
    """

    def __init__(self, num_classes = 4):
        """ Initialize the CNN model. 
        Args:
            num_classes (int): Number of output classes.
        """
        super().__init__()

        #convolutional layers that are used to extract 'features' from the images
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1) #3 input channels (RGB), 16 output channels, 3x3 kernel which a kernel is a filter that is slid over the image to produce the feature maps
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 256, 3, padding=1)


        self.pool = nn.MaxPool2d(2, 2) #pooling layer to reduce the spatial dimensions of the feature maps
                                       #also done to make the model focus on the most important features
        self.gap = nn.AdaptiveAvgPool2d((4, 4))  #global average pooling layer to reduce the spatial dimensions to 4x4
        self.dropout = nn.Dropout(p=0.4) #a dropout to prevent the model from overfitting
                                         #it works by randomly setting some of the neurons to zero during training

        self.fc1 = nn.Linear(256 * 4 * 4, 128) #the nn nodes that make the final classification decision (kinda like a traditional neural network)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """ Define the forward pass of the CNN.
        Args:
            x (torch.Tensor): The input image tensor.
        Returns:
            torch.Tensor: The output class scores.
        """
        x = self.pool(F.relu(self.conv1(x))) #apply conv1, then use the ReLU activation function, then apply pooling
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))

        x = self.gap(x) #apply global average pooling
        
        x = x.view(x.size(0), -1) #flatten the tensor into a 1D vector (was 64x4x4, now 1024x1)
       
        x = F.relu(self.fc1(x)) #go through the first nn layer
        x = self.dropout(x) #apply dropout
       
        x = self.fc2(x) #go through the second nn layer to get the final output (which is the class scores)
        return x

def train_model(epochs = 10, patience = 5, savepath="model_weights.pth"):
    """ train the CNN model on the dataset.
    Args:
        epochs (int): Number of training epochs.
        patience (int): Number of epochs to wait for improvement before stopping.
        savepath (str): Path to save the trained model weights.
    Returns:
        SimpleCNN: The trained CNN model.
    """
    model = SimpleCNN().to(device) #sends the model to the device (GPU or CPU)

    #calculate class weights to handle class image data counts imbalance
    class_counts = torch.tensor([count_files_in_directory("images/powerball"), count_files_in_directory("images/euromillions"), count_files_in_directory("images/lottoamerica"), count_files_in_directory("images/megamillions")], dtype=torch.float)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(class_counts)
    loss_function = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.05) #loss function for multi-class classification problems
                                                                                         #label_smoothing is used to prevent the model from becoming too confident in its predictions, which can help improve generalization

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=1e-4) #updates the model's weights and uses a learning rate (alpha) of 0.001
                                                                                    #weight_decay is used to prevent overfitting by penalizing model weights that are very large
    l1_lambda = 0.0 #regularization parameter for L1 regularization (WAS 1e-6)

    best_validation_loss = float('inf') #initialize the best validation loss to infinity
    num_epochs_no_improvement = 0 #counter for the number of epochs with no improvement

    num_epochs = epochs #number of times the model will go through the entire dataset
    for epoch in range(num_epochs):
        model.train() #sets the model to training mode

        running_loss = 0.0 #this is the loss over all batches for this epoch
        correct = 0 #number of correct predictions
        total = 0 #total number of predictions

        for images, labels in train_dataloader: #gives batches of 32 images and their corresponding labels
            images, labels = images.to(device), labels.to(device)  #does things with putting the images and model on GPU/CPU

            optimizer.zero_grad() #clears old gradients from the previous batch
            outputs = model(images) #feeds the batch images into the model to get the predictions

            loss = loss_function(outputs, labels) #calculates the loss between the predictions and the true labels

            l1_norm = model.fc1.weight.abs().sum() #calculates the L1 norm of the weights of the first fully connected layer

            loss = loss + l1_lambda * l1_norm #adds the L1 regularization term to the loss

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

        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            num_epochs_no_improvement = 0

            #! I have not included the loss because i dont want my computer to run out of storage holding many various model weights
            torch.save(model.state_dict(), f"ticket_classifier_models/validation_models/{savepath}") #saves the model with the best weights to a file (so that we keep the best one and we have it if we need to stop)
            # torch.save(model.state_dict(), f"ticket_classifier_models/validation_models/{val_loss:.2f}_{savepath}") #saves the model with the best weights to a file (so that we keep the best one and we have it if we need to stop)
        else:
            num_epochs_no_improvement += 1
            if num_epochs_no_improvement >= patience:
                print("Early stopping due to no improvement in validation loss.")
                break

        print()

    #& torch.save(model.state_dict(), f"ticket_classifier_models/{savepath}") #saves the model weights to a file

    return model

def load_model(filepath):
    """ Load the trained CNN model from a file.
    Args:
        filepath (str): The path to the file containing the model weights.
    Returns:
        SimpleCNN: The loaded CNN model.
    """
    model = SimpleCNN(num_classes=4)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model

def count_files_in_directory(directory):
    """ Count the number of files in a directory.
    Args:
        directory (str): The path to the directory.
    Returns:
        int: The number of files in the directory.
    """
    count = 0
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry) #get the full path of the entry (file)
        if os.path.isfile(full_path): #check if this entry is a file
            count += 1
    return count

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

def main():
    savepath = "model_weights.pth"

    # model = train_model(epochs = 50, savepath=savepath, patience=15) #trains the model

    # model.load_state_dict(torch.load(f"ticket_classifier_models/validation_models/{savepath}")) #loads the best validation model weights
        
    # accuracy = test_model(model) #tests the trained model
    # print(f"Model Test Accuracy: {accuracy:.2f}%")

    # torch.save(model.state_dict(), f"ticket_classifier_models/{accuracy:.2f}_{savepath}") #saves the model weights to a file

    for i in range(20):
        print(f" ----- Training Run {i+1} ----- ")
        savepath = "model_weights.pth"

        model = train_model(epochs = 55, savepath=savepath, patience=15) #trains the model

        model.load_state_dict(torch.load(f"ticket_classifier_models/validation_models/{savepath}")) #loads the best validation model weights
        
        accuracy = test_model(model) #tests the trained model
        print(f"Model Test Accuracy: {accuracy:.2f}%")

        torch.save(model.state_dict(), f"ticket_classifier_models/{accuracy:.2f}_{savepath}") #saves the model weights to a file

if __name__ == "__main__":
    main()