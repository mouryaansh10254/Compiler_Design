load_images from "dataset/"
define_model resnet18
load_model from "models/model.pth"
predict_images from "unclassified/"
organize_images into "organized_output/"