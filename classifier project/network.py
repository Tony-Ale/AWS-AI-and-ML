from torch import nn, optim
import torch.nn.functional as F
import torch
import sys

class Network(nn.Module):
    """
    This class creates a network based on user input
    """
    def __init__(self, in_features, hidden_units, output):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features, hidden_units[0]))
        fc_layers = zip(hidden_units[:-1], hidden_units[1:])
        self.layers.extend(nn.Linear(*units) for units in fc_layers)
        self.layers.append(nn.Linear(hidden_units[-1], output))
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        for layers in self.layers[:-1]:
            x = self.dropout(F.relu(layers(x)))
        x = F.log_softmax(self.layers[-1](x), dim = 1)
        
        return x

def train_network(model, classifier, classifier_name, trainloader, validloader, lr, epochs, device):
    """
    This function trains a network and prints out the losses and accuracy
    """
    if classifier_name:
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr)
    else:
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(), lr)

    criterion = nn.NLLLoss()
    
    
    model.to(device)

    # Training the model
    n_iter = 5
    batch = 0
    train_loss = 0
    
    print('\n***Setting up Model for Training***\n')
    for e in range(epochs):
        for images, labels in trainloader:
            batch += 1
            optimizer.zero_grad()

            images, labels = images.to(device), labels.to(device)

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss

            if batch % n_iter == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)

                        logits = model.forward(images)
                        prob = torch.exp(logits)
                        valid_loss += criterion(logits, labels)

                        value, prediction = prob.topk(1, dim =1)
                        equal = prediction == labels.view(*prediction.shape)
                        accuracy += torch.mean(equal.type(torch.FloatTensor))

                print("Epoch: {}/{}.. Training Loss: {:.3f}.. Validation loss: {:.3f}.. Accuracy: {:.3f}%"
                     .format(e+1, epochs, train_loss/n_iter, valid_loss/len(validloader), 100*accuracy/len(validloader)))
                model.train()
                train_loss = 0
    return model