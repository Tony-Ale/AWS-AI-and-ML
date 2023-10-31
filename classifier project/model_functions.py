import torch
import sys
from torchvision import models
from network import Network

def freeMemory(mode):
    # Free cuda memory
    if mode == 'cuda':
        torch.cuda.empty_cache()

def check_modelname(modelname):
    """ 
    This function ensures that model type inputed by user is one of type vgg, alexnet, densenet and resnet.
    The input "modelname" is a string 
    """
    for model_name in ['vgg', 'alexnet', 'densenet', 'resnet']:
        if model_name in modelname.lower() and model_name != modelname.lower():
            if hasattr(models, modelname.lower()):
                return
            
    print('\n***Input a valid model name of type vgg, alexnet, resnet, and densenet***\n')
    sys.exit()

def predict_image(model, input_img, topk, categories, device):
    """
    This function accepts a model, an image, a 'topk' parameter, a class mapping, and a computing mode (GPU or CPU) as inputs, 
    and it produces the model's top-k probabilities and associated class predictions.    
    """
    model.eval()
    print('\n***Making Inference***\n')
    with torch.no_grad():
        input_img = input_img.to(device)
        logp = model.forward(input_img.view(1, *input_img.shape))
        out_prob = torch.exp(logp)
        prob, class_index = out_prob.topk(topk, dim =1)
    # Invert class_to_idx dictionary
    idx_to_class = {i:c for c, i in model.class_to_idx.items()}
    prediction = [categories[idx_to_class[index]] for index in class_index[0].tolist()]
    prob = prob[0].tolist()
    return prob, prediction

def pretrained_model(model_name):
    """
    This function takes in the user input as model_name(str) and returns a loaded model, its classifier input features
    and a boolean which is used to know if the classifier name is "classifier" (True) or "fc" (False)
    """
    model_name = model_name.lower()
    model = getattr(models, model_name)(pretrained = True)
    for params in model.parameters():
        params.requires_grad = False
    
    if hasattr(model, 'classifier'):
        classifier = getattr(model, 'classifier')
        classifier_name = True
    elif hasattr(model, 'fc'):
        classifier = getattr(model, 'fc')
        classifier_name = False
   
    if isinstance(classifier, torch.nn.Sequential):
        for key, val in classifier.named_children():
            if hasattr(val, 'in_features'):
                in_features = getattr(val, 'in_features')
                break
    elif isinstance(classifier, torch.nn.Linear):
            in_features = getattr(classifier, 'in_features')
            
    return model, in_features, classifier_name

def save_model(save_dir, model, classifier_name, model_name, train_data):
    """
    This function saves the model as a pth file
    """
    if classifier_name:
        classifier = model.classifier
    else:
        classifier = model.fc
    # checks if pth extension is added
    if not save_dir[-4:] == '.pth':
        save_dir = save_dir + '.pth'
        
    checkpoint = {'classifier': classifier,
                  'pretrained_model': model_name.lower(),
                  'classifier_name': classifier_name,
                  'state_dict': model.state_dict(),
                  'class_to_idx': train_data.class_to_idx}
    
    torch.save(checkpoint, save_dir)
    print("\n***Model is Saved***\n")
def load_checkpoint(filepath, device):
    """
    Function loads a checkpoint into memory so that it can be used for inference
    """
    try:
        checkpoint = torch.load(filepath, map_location = device)
        model = getattr(models, checkpoint['pretrained_model'])(pretrained = True)
        for params in model.parameters():
            params.requires_grad = False
   
        if checkpoint['classifier_name']:
            model.classifier = checkpoint['classifier']
        else:
            model.fc = checkpoint['classifier']
        model = model.to(device)
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
    except Exception as error:
        print('\n***Error with checkpoint file:', type(error).__name__, '-', error, '***\n')
        sys.exit()
    else:
        print('\n***Model is loaded***\n')
        return model

