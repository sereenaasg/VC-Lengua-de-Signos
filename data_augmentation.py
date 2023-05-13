from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import os

# Create a new directory to save the augmented images
augmented_directory = 'augmented_data'
os.makedirs(augmented_directory, exist_ok=True)

# create image data augmentation generator
datagen = ImageDataGenerator(horizontal_flip=True, 
                             width_shift_range=0.1, 
                             height_shift_range=0.1, 
                             brightness_range=[0.2,1.0],
                             rotation_range=0.5)
# prepare iterator
it = datagen.flow_from_directory('mans', batch_size=1)

# Generate samples and save the augmented images
for i in range(len(it.filenames)*3):
    # Generate batch of images
    batch = it.next()

    # Convert to unsigned integers for viewing
    image = batch[0][0].astype('uint8')

    # Get the original image's subdirectory and filename
    if batch[1][0][0] == 1: original_subdirectory = 'negatiu'
    elif batch[1][0][1] == 1: original_subdirectory = 'positiu'
    elif batch[1][0][2] == 1: original_subdirectory = 'tijeras'

    # Create the corresponding subdirectory in the augmented directory
    augmented_subdirectory = os.path.join(augmented_directory, original_subdirectory)
    os.makedirs(augmented_subdirectory, exist_ok=True)

    # Save the augmented image
    plt.imsave(augmented_subdirectory + "\\IMG_%04d.png" % i, image)