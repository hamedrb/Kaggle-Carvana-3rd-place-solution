import threading
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from model import get_dilated_unet

WIDTH = 1024
HEIGHT = 1024
BATCH_SIZE = 2

INPUT_DATA_PATH = '/home/analysisstation3/projects/CNNForCarSegmentation/Input/kaggle_Carvana_Image_Masking_Challenge/all'
OUTPUT_DATA_PATH = '/home/analysisstation3/projects/CNNForCarSegmentation/output'

class ThreadSafeIterator:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    """
    A piece of code is thread-safe if it functions correctly during 
    simultaneous execution by multiple threads.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    
    """

    def g(*args, **kwargs):
        return ThreadSafeIterator(f(*args, **kwargs))

    return g


@threadsafe_generator
def train_generator(df):
    """
    A generator of image batches for trainning
    
    Input: 
        df: a list of image IDs, 
    
    Return:
        x_batch: a batch of trainning images
        y_batch: the corresponding masks
    """
    while True: # since the generator is not reseted after each epoch, so we 
        #need this "while" to reset the "for-loop" at the beginning of each 
        #epoch. Otherwise, at the end of the first epoch, the for-loop 
        #reaches to end and couldn't begin again in the beginning of the 2nd
        #epoch. 
        shuffle_indices = np.arange(len(df))
        shuffle_indices = np.random.permutation(shuffle_indices)
        
        for start in range(0, len(df), BATCH_SIZE):
            x_batch = []
            y_batch = []
            
            end = min(start + BATCH_SIZE, len(df))
            ids_train_batch = df.iloc[shuffle_indices[start:end]]
            
            for _id in ids_train_batch.values:
                img = cv2.imread(INPUT_DATA_PATH+'/train_hq/{}.jpg'.format(_id))
                img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                
                mask = cv2.imread(INPUT_DATA_PATH+'/train_masks/{}_mask.png'.format(_id), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                mask = np.expand_dims(mask, axis=-1)
                assert mask.ndim == 3
                
                # === You can add data augmentations here. === #
                if np.random.random() < 0.5:
                    img, mask = img[:, ::-1, :], mask[..., ::-1, :]  # random horizontal flip
                
                x_batch.append(img)
                y_batch.append(mask)
            
            x_batch = np.array(x_batch, np.float32) / 255.
            y_batch = np.array(y_batch, np.float32) / 255.
            
            yield x_batch, y_batch # each time a generator reaches to "yield",
                                   #it yields the values and paused. Next time
                                   #that the generator is called, it starts 
                                   #exactly from the point that it was paused i.e.
                                   #the next line after the "yield".


@threadsafe_generator
def valid_generator(df):
    while True:
        for start in range(0, len(df), BATCH_SIZE):
            x_batch = []
            y_batch = []

            end = min(start + BATCH_SIZE, len(df))
            ids_train_batch = df.iloc[start:end]

            for _id in ids_train_batch.values:
                img = cv2.imread(INPUT_DATA_PATH+'/train_hq/{}.jpg'.format(_id))
                img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

                mask = cv2.imread(INPUT_DATA_PATH+'/train_masks/{}_mask.png'.format(_id),
                                  cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                mask = np.expand_dims(mask, axis=-1)
                assert mask.ndim == 3
                
                x_batch.append(img)
                y_batch.append(mask)

            x_batch = np.array(x_batch, np.float32) / 255.
            y_batch = np.array(y_batch, np.float32) / 255.

            yield x_batch, y_batch


if __name__ == '__main__':

    df_train = pd.read_csv(INPUT_DATA_PATH+'/train_masks.csv')
    ids_train = df_train['img'].map(lambda s: s.split('.')[0])

    ids_train, ids_valid = train_test_split(ids_train, test_size=0.1)

    model = get_dilated_unet(
        input_shape=(1024, 1024, 3),
        mode='cascade',
        filters=32,
        n_class=1
    )

    callbacks = [EarlyStopping(monitor='val_dice_coef',
                               patience=10,
                               verbose=1,
                               min_delta=1e-4,
                               mode='max'),
                 ReduceLROnPlateau(monitor='val_dice_coef',
                                   factor=0.2,
                                   patience=5,
                                   verbose=1,
                                   epsilon=1e-4,
                                   mode='max'),
                 ModelCheckpoint(monitor='val_dice_coef',
                                 filepath=OUTPUT_DATA_PATH+'model_weights.hdf5',
                                 save_best_only=True,
                                 mode='max')]

    model.fit_generator(generator=train_generator(ids_train),
                        steps_per_epoch=np.ceil(float(len(ids_train)) / float(BATCH_SIZE)),
                        epochs=100,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=valid_generator(ids_valid),
                        validation_steps=np.ceil(float(len(ids_valid)) / float(BATCH_SIZE)))
