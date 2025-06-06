from tensorflow import keras

base_model = keras.applications.VGG16(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(224, 224, 3),
    include_top=False)
#base_model.summary()

base_model.trainable = False
inputs = keras.Input(shape=(224, 224, 3))

# base_model é, por exemplo, um modelo pré-treinado como MobileNet, ResNet, etc.
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)

# Camada final para 6 classes com ativação softmax
outputs = keras.layers.Dense(6, activation='softmax')(x)

# Define o modelo final
model = keras.Model(inputs, outputs)

#model.summary()
model.compile(
    optimizer='adam',
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.CategoricalAccuracy()]
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create a data generator
datagen_train = ImageDataGenerator(
    samplewise_center=True,  # set each sample mean to 0
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,
)  # we don't expect Bo to be upside-down so we will not flip vertically

# No need to augment validation data
datagen_valid = ImageDataGenerator(samplewise_center=True)

# load and iterate training dataset
train_it = datagen_train.flow_from_directory(
    '/home/silas/Documentos/project-deep-learning/fruits/train',
    target_size=(224,224),
    color_mode="rgb",
    class_mode="categorical",
)
# load and iterate validation dataset
valid_it = datagen_valid.flow_from_directory(
    '/home/silas/Documentos/project-deep-learning/fruits/valid',
    target_size=(224,224),
    color_mode="rgb",
    class_mode="categorical",
)
model.fit(train_it,
          validation_data=valid_it,
          steps_per_epoch=int(train_it.samples/train_it.batch_size),
          validation_steps=int(valid_it.samples/valid_it.batch_size),
          epochs=20)

# Unfreeze the base model
base_model.trainable = True

# Compile the model with a low learning rate
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate = 0.0001),
              loss = keras.losses.CategoricalCrossentropy(from_logits=False) , metrics =[keras.metrics.CategoricalAccuracy()])
model.fit(train_it,
          validation_data=valid_it,
          steps_per_epoch=int(train_it.samples/train_it.batch_size),
          validation_steps=int(valid_it.samples/valid_it.batch_size),
          epochs=20)
