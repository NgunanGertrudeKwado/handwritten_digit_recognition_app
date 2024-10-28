Handwritten Digit Recognition
This project aims to recognize handwritten digits using Convolutional Neural Networks (CNN) trained on the MNIST dataset. The model is designed to classify images of handwritten digits from 0 to 9, achieving high accuracy.

Table of Contents
Data Collection
Data Preprocessing
Exploratory Data Analysis (EDA)
Model Building
Evaluation
Deployment
Data Source
Getting Started
License
Data Collection
The dataset used in this project is the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits. Each image is a grayscale 28x28 pixel representation of a digit.

Data Preprocessing
Before training the model, the data undergoes the following preprocessing steps:

Normalization: Pixel values are scaled to the range [0, 1] by dividing by 255.
Reshaping: The images are reshaped to include a channel dimension suitable for CNN input.
Exploratory Data Analysis (EDA)
Visual samples of the digits are displayed, and the distribution of the digits in the dataset is analyzed to ensure balanced representation.

Model Building
A Convolutional Neural Network (CNN) is constructed using the following layers:

Convolutional layers to extract features from the images.
Pooling layers to reduce dimensionality.
Fully connected layers for classification.
Evaluation
The model's performance is evaluated using:

Accuracy: The proportion of correctly classified images.
Confusion Matrix: A tool to visualize the performance of the model across all digit classes.
Deployment
The trained model can be deployed as a web application or mobile app, allowing users to input handwritten digits for recognition. The deployment process involves creating an interface that utilizes the model's prediction capabilities.

Data Source
The MNIST dataset can be accessed from the Yann LeCun's website.

## Getting Started

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/NgunanGertrudeKwado/handwritten_digit_recognition_app.git


2. Navigate to the project directory:

cd handwritten_digit_recognition_app


3. Install the required packages:
pip install -r requirements.txt


4. Run the training script:
python train.py


License
This project is licensed under the MIT License. See the LICENSE file for more details.

