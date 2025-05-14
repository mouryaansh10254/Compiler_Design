# To train and save the model
load_images from "dataset/"
define_model resnet18
train_model for 5 epochs save_model_to "models/model.pth"
