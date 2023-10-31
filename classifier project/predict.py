from process_images import load_testimages, get_categorymap
from model_functions import load_checkpoint, predict_image, freeMemory
from io_functions import predict_args, checkgpu, print_data, has_file

# Get user input data
user_input = predict_args()
checkpoint = user_input.checkpoint
image_path = user_input.image
categorymap_dir = user_input.category_names
topk = user_input.top_k
mode = user_input.gpu

# checks if gpu is available and allows user to use it 
device = checkgpu(user_input.gpu)

# checks if checkpoint file exists
if not checkpoint[-4:] == '.pth':
    checkpoint = checkpoint + '.pth'

has_file(checkpoint)

# checks if category map file exists
if not categorymap_dir[-5:] == '.json':
    categorymap_dir = categorymap_dir + '.json'

has_file(categorymap_dir)

# checks if image file exist
has_file(image_path)

# load checkpoint
model = load_checkpoint(checkpoint, device)

# load and process input to image to fit model input
input_img = load_testimages(image_path, load_images = False)

# Get category map as dictionary
category_map = get_categorymap(categorymap_dir)

# To get the topk probaility and class of input image
prob, prediction = predict_image(model, input_img, topk, category_map, device)

# Print out topk classes and probability
print_data(prob, prediction)

# free cuda memory
freeMemory(mode)


