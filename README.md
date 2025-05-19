## Fish Species Classifier
- This Project is a deep learning image classifier that identifies aquarium fish species using transfer learning.
- The model was trained on a custom image dataset that was created through web scraping and photos taken at Petco, which where then cleaned and preprocessed.
- We tested multiple different convolutional neural netowrks (CNNs) and found that EfficientNet with its default parameters did the best
- This project was built to look at the entire machine learning pipeline, starting at image collection and preprocessing to model evaluation. 

## Features
- Transfer learning with MobileNet, ResNet, and EfficientNet architectures
- Custom dataset from Google/Bing images and real world images taken at Petco
- Image preprocessing including cropping, resizing, and padding using OpenCV
- Model evaluation
- Presentation slides are included

## Team Contribution
This was a two person project for CS-436 (Artificial Intelligence 1) completed by Joe Thompson and Will Thompson
## My Contributions ( Will Thompson ):
- Image collection, cleaning and preprocessing
- Transfer learning set up for MobileNet
- Final model evaluation
- Grid Search
- Designed and created final presentation slides

## Model Evaluation
- We tested MovileNetV2 and V3 as well as ResNet and EfficientNet architectures with custom top layers.
- Our best performance came from EfficientNet with base parameters, We attempted to use grid search (Dense units, dropout rate, learning rate), but these models tended to overfit. This was likely due to the dataset's limited size (283 images).
- The default configuration performed better and was selected for final use
## Final Model
- Test Accuracy: 83%
- Precision: 85%
- Recall: 82%
- F1: 82%
- See graphs like confusion matrix, ROC curve and Validation Curves in the presentation

## Sample Dataset
- A small sample dataset is included, it contains:
   - 5 training imgaes per class
   - 2 validation images per class
   - 2 test images per class
- The fill cleaned dataset of 283 total images is available upon request
## Dataset Info
- Images were manually collected and cleaned from online sources as well as taken from petco. All contest is used only for educational purposes
## Running the Project
1. Install dependencies:
   - pip install tensorflow keras opencv-python matplotlib numpy pandas scikit-learn
2. Open and run notebooks
   - Image_Processing_Fish_Recongnizer.ipynb, for scraping, cleaning, formatting images
   - CS436_FishRecognizer_JoeThompsonWillThompson.ipynb, for model training and evualtion
3. Load the best model
   - from tensorflow.keras.models import load_model
   - model = load_model("models/EfficientNetB0_fish_classifier.h5")

## Future Directions 
- Collect additional images to try and combat overfitting
- Streamline the process of adding images to the dataset
- Create a project that uses the model, like an app that can scan the fish 
