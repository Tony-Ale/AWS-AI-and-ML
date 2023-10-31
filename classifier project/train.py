from torch import optim, nn
from io_functions import input_arg, checkgpu, has_dir
from network import Network, train_network
from model_functions import pretrained_model, save_model, check_modelname, freeMemory
from process_images import load_trainimages, load_testimages

# get user input
user_input = input_arg()
lr = user_input.learning_rate
epochs = user_input.epochs
n_classes = user_input.output # default is 102
mode = user_input.gpu
archi = user_input.arch
data_dir = user_input.data_dir
save_dir = user_input.save_dir
hidden_units = user_input.hidden_units

# Check if the input for model name is valid
check_modelname(archi)

# checks if gpu is available and allows user to use it 
device = checkgpu(mode)

# checks if data directory exists
has_dir(data_dir, save = False)

# checks if save directory exists
has_dir(save_dir)

# Directory to data
train_dir = data_dir + '/train'
test_dir = data_dir + '/test'
valid_dir = data_dir + '/valid'


# loading training data, pretrained models and initailizing model parameters
trainloader, train_datasets = load_trainimages(train_dir)
validloader = load_testimages(valid_dir, load_images = True)

# Getting pretrained model and other attributes
model, in_feature, classifier_name = pretrained_model(archi)

# Creating a custom classifier
classifier = Network(in_feature, hidden_units, n_classes)

# Training the network
model = train_network(model, classifier, classifier_name, trainloader, validloader, lr, epochs, device)

# save checkpoint            
save_model(save_dir, model, classifier_name, archi, train_datasets)

# free cuda memory
freeMemory(mode)

