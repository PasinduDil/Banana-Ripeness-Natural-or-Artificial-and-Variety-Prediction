import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

input_dir = "data set/TEST /Anamalu_Artificial test"  
output_dir = "output"  
os.makedirs(output_dir, exist_ok=True)

# Set up ImageDataGenerator for Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,      # Rotate 
    width_shift_range=0.2,  # Shift horizontally 
    height_shift_range=0.2, # Shift vertically
    shear_range=0.2,        # Shearing transformation
    zoom_range=0.2,         # Zoom
    horizontal_flip=True,   # Flip horizontally
    fill_mode="nearest"     # Fill empty pixels with nearest
)


def augment_and_save_images(input_dir, output_dir, datagen, augmentations_per_image=20):

    image_filenames = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in image_filenames:
        img_path = os.path.join(input_dir, filename)
        img = tf.keras.preprocessing.image.load_img(img_path)  # Load image
        img_array = tf.keras.preprocessing.image.img_to_array(img)  # Convert to array
        img_array = img_array.reshape((1,) + img_array.shape)  # Add batch dimension

        # Generate augmented images
        counter = 0
        for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_dir,
                                  save_prefix=filename.split('.')[0], save_format='jpeg'):
            counter += 1
            if counter >= augmentations_per_image:  # Limit augmentations per image
                break

# Call the function
augment_and_save_images(input_dir, output_dir, datagen)

print(f"Augmented images saved to {output_dir}")