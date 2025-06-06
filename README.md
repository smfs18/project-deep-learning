Image Classification with Transfer Learning using VGG16
This project demonstrates a powerful and common technique in computer vision: transfer learning. We use the VGG16 model, pre-trained on the extensive ImageNet dataset, and adapt it to a new, specific task: classifying images into one of six categories.

The training process is strategically divided into two key phases to maximize performance and training stability:

Feature Extraction: We start by freezing the convolutional base of the VGG16 model. This allows us to use its learned features without altering them. We then train only the new, custom classifier that we've added on top. This quickly adapts the model to our new dataset's classes.

Fine-Tuning: After the initial training, we unfreeze the entire model (or parts of it) and continue training with a very low learning rate. This step slightly adjusts the pre-trained weights to better fit the nuances of our specific dataset, further improving accuracy.

⚙️ Key Technologies
TensorFlow & Keras: For building and training the deep learning model.

VGG16: The pre-trained convolutional neural network (CNN) used as the base model.

ImageDataGenerator: For efficient data loading and real-time data augmentation.

🧠 Model Architecture
The final model is constructed by stacking new layers on top of the pre-trained VGG16 base.

Base Model (VGG16): We load the VGG16 model, discarding its original top classification layer (include_top=False) but retaining the weights learned from ImageNet.

Freezing Layers: Initially, all layers in the VGG16 base are frozen (base_model.trainable = False) so their weights are not updated during the first phase of training.

Custom Classifier: We add our own classification head:

A GlobalAveragePooling2D layer to reduce the spatial dimensions of the feature maps into a single vector per map, drastically reducing the number of parameters.

A final Dense layer with 6 units (one for each target class) and a softmax activation function to output the probability distribution across the classes.

📦 Data Preparation & Augmentation
To make the model more robust and prevent overfitting, we apply data augmentation to the training images on-the-fly using ImageDataGenerator. This creates modified versions of the images at each epoch. The applied transformations include:

Random rotations

Random zooming

Random horizontal and vertical shifts

Random horizontal flips

The validation data is not augmented; we only rescale it, just as we do with the training data. The data is loaded from structured directories where each sub-directory corresponds to a single class.

🚀 Training Process
Phase 1: Feature Extraction
The model is first compiled with the adam optimizer and CategoricalCrossentropy loss. We train it for 20 epochs. In this phase, only the weights of the GlobalAveragePooling2D and Dense layers are updated.

# Freeze the base model
base_model.trainable = False

# Compile the model
model.compile(
    optimizer='adam',
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.CategoricalAccuracy()]
)

# Train only the top layers
model.fit(train_it,
          validation_data=valid_it,
          epochs=20)

Phase 2: Fine-Tuning
Next, we unfreeze the base model to allow all weights to be trainable. The model is then re-compiled with a very low learning rate (0.0001) and the RMSprop optimizer. Using a low learning rate is crucial to prevent catastrophic forgetting—losing the valuable features learned from ImageNet.

The model is then trained for another 20 epochs, allowing the entire network to gently adapt to the new dataset.

# Unfreeze the base model to allow fine-tuning
base_model.trainable = True

# Re-compile with a very low learning rate
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate = 0.0001),
              loss = keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics =[keras.metrics.CategoricalAccuracy()])

# Continue training the entire model
model.fit(train_it,
          validation_data=valid_it,
          epochs=20)

📋 How to Use
Clone the Repository:

git clone <your-repository-url>
cd <repository-name>

Organize Your Data: Ensure your image dataset is structured in the following way:

dataset/
├── train/
│   ├── class_1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class_2/
│       ├── image3.jpg
│       └── image4.jpg
└── valid/
    ├── class_1/
    │   ├── image5.jpg
    │   └── image6.jpg
    └── class_2/
        ├── image7.jpg
        └── image8.jpg

Update File Paths: In the Python script, change the paths in the flow_from_directory calls to point to your train and valid directories.

train_it = datagen_train.flow_from_directory(
    'path/to/your/dataset/train',
    # ... other parameters
)

valid_it = datagen_valid.flow_from_directory(
    'path/to/your/dataset/valid',
    # ... other parameters
)

Run the script to start the training process.
