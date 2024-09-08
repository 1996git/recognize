import io
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Dense, Flatten 
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam 
import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# グローバル変数の初期化
model = None

# 1. データの準備
def setup_data_generators(train_dir, test_dir):
    if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
        raise ValueError("Both training and test directories must be valid directories.")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'  # 'binary' for two classes
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'  # 'binary' for two classes
    )

    return train_generator, test_generator

# モデルの構築
def build_model(num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)  # sigmoid for binary classification

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',  # binary_crossentropy for two classes
                  metrics=['accuracy'])
    
    return model

# 画像の分類
def classify_image(img_path, model):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])  # Get the index of the highest probability
    result = f"Class index: {class_idx}, Probability: {predictions[0][class_idx]:.2f}"
   
    return result
    
# トレーニング履歴のプロット
def plot_training_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history['loss'], label='Training Loss')
    ax[0].plot(history.history['val_loss'], label='Validation Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].set_title('Loss Over Epochs')
    ax[1].plot(history.history['accuracy'], label='Training Accuracy')
    ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    ax[1].set_title('Accuracy Over Epochs')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# Streamlitアプリ
def main():
    global model
    st.title("Model Training and Classification")
    st.write("Upload directories and train the model, or classify an image.")

    # トレーニングとテストデータディレクトリのアップロード
    train_dir = st.text_input("Training Data Directory", "")
    test_dir = st.text_input("Test Data Directory", "")

    if st.button("Start Training"):
        if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
            st.error("Both training and test directories must be valid directories.")
        else:
            try:
              train_generator, test_generator = setup_data_generators(train_dir, test_dir)
              model = build_model(1)
              history = model.fit(
                train_generator,
                epochs=10,
                validation_data=test_generator
            )
              st.write("Training complete!")
              buf = plot_training_history(history)
              st.image(buf, use_column_width=True)
            except ValueError as e:
              st.error(str(e))

    # 画像分類機能
    uploaded_file = st.file_uploader("Upload Image for Classification", type=["jpg", "jpeg", "png"])
    if uploaded_file and model:
        img_path = uploaded_file.read()
        result = classify_image(img_path, model)
        st.write(result)

if __name__ == "__main__":
    main()