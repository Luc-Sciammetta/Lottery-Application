import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from PIL import Image

#defines the image transformations that will be applied to each image in the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)), #resizes each image to 224x224 pixels
    transforms.RandomHorizontalFlip(), #randomly flips the image horizontally
    transforms.RandomRotation(15), #randomly rotates the image by up to 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2), #randomly changes brightness and contrast
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), #randomly crops and resizes the image
    transforms.ToTensor() #converts the image to a PyTorch tensor
])

dataset = ImageFolder(root = "images", transform=transform) #makes the dataset
dataloader = DataLoader(dataset, batch_size=32, shuffle=True) #feeds the dataset into the model in batches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #checks to see if there is a GPU that is available for training, otherwise uses the CPU of the computer

print(dataset.classes) #prints the classes found in the dataset ['powerball', 'euromillions', 'megamillions', 'lottoamerica']

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
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1) #3 input channels (RGB), 16 output channels, 3x3 kernel which a kernel is a filter that is slid over the image to produce the feature maps
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        
        #pooling layer to reduce the spatial dimensions of the feature maps
        #also done to make the model focus on the most important features
        self.pool = nn.MaxPool2d(2, 2)

        #the nn nodes that make the final classification decision (kinda like a traditional neural network)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
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


        x = x.view(x.size(0), -1) #flatten the tensor into a 1D vector (was 64x28x28, now 50176x1)
        x = F.relu(self.fc1(x)) #go through the first nn layer
        x = self.fc2(x) #go through the second nn layer to get the final output (which is the class scores)
        return x

def train_model(epochs = 10, savepath="model_weights.pth"):
    """ train the CNN model on the dataset.
    Args:
        epochs (int): Number of training epochs.
        savepath (str): Path to save the trained model weights.
    Returns:
        SimpleCNN: The trained CNN model.
    """
    model = SimpleCNN().to(device) #sends the model to the device (GPU or CPU)
    loss_function = nn.CrossEntropyLoss() #loss function for multi-class classification problems
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) #updates the model's weights and uses a learning rate (alpha) of 0.001

    num_epochs = epochs #number of times the model will go through the entire dataset
    for epoch in range(num_epochs):
        running_loss = 0.0 #this is the loss over all batches for this epoch
        correct = 0 #number of correct predictions
        total = 0 #total number of predictions

        for images, labels in dataloader: #gives batches of 32 images and their corresponding labels
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
        print(f"Epoch {epoch+1}, Loss: {running_loss:.3f}, Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), savepath) #saves the model weights to a file

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

def predict_image(model, image_path):
    """ Predict the class of a lottery ticket image.
    Args:
        model (SimpleCNN): The trained CNN model.
        image_path (str): The path to the image file.
    Returns:
        str: The predicted class name.
    """
    img = Image.open(image_path).convert("RGB") #opens the image and converts it to RGB format
    img = transform(img).unsqueeze(0).to(device) #applies the transformations defined before

    model.eval() #sets the model to evaluation mode
    with torch.no_grad(): #tells PyTorch not to calculate gradients (saves memory and computations)
        output = model(img) #passes the image through the network
        predicted = output.argmax(dim = 1) #gets the index of the classification with the highest score
    
    print(f"Predicted Class: {dataset.classes[predicted.item()]}")
    return dataset.classes[predicted.item()] #returns the class name corresponding to the predicted index

def main():
    model = train_model(epochs = 45) #trains the model
    # model = load_model("model_weights.pth") #uncomment this line to load a pre-trained model instead of training a new one
    # test_image_path = "images/megamillions/img3.jpg" #path to a test image
    # predict_image(model, test_image_path)

main()