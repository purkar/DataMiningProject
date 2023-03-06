import glob
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import vgg16,VGG19,DenseNet201,VGG16,ResNet50,InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, MaxPool2D
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Flatten, Dropout, BatchNormalization,AveragePooling2D
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os
import shutil
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics import accuracy_score,classification_report, roc_curve, confusion_matrix
import tensorflow as tf
from sklearn.metrics import roc_curve, auc

def seperate_into_directories(basepath):
    if os.path.exists(basepath +'VIRAL') is False:
        os.mkdir(basepath +'VIRAL')
    else:
        return
    
        
    pneunomnia_images = glob.glob(basepath +'PNEUMONIA/*')
    for img in pneunomnia_images:
        category=img.split('_')[2]
        if(category=='virus'):
            shutil.copy(img,basepath +'VIRAL')
            os.remove(img)
     
    os.rename(basepath +'PNEUMONIA', basepath +'BACTERIA')
    
    
def read_data(base_path):
    #Variables Declaration
    image_size=224
    labels=[]
    
    #Normal Images
    normal_images = glob.glob(base_path + 'NORMAL/*')
    normal_images_array = [img_to_array(load_img(img, target_size=(image_size,image_size))) for img in normal_images]
    
    normal_images_labels=['normal']* len(normal_images_array)
    labels=labels + normal_images_labels
    
    
    # Baterial Pneumonia Images
    pneumonia_images = glob.glob(base_path + 'BACTERIA/*')
    pneumonia_images_array = [img_to_array(load_img(img, target_size=(image_size,image_size))) for img in   pneumonia_images]
    
    
    pneumonia_labels=['bacteria']* len(pneumonia_images_array)
    labels=labels + pneumonia_labels
    
    #Viral Pneumonia Images
    viral_images = glob.glob(base_path + 'VIRAL/*')
    viral_images_array = [img_to_array(load_img(img, target_size=(image_size,image_size))) for img in viral_images]
    
    
    viral_labels=['viral']* len(viral_images_array)
    labels=labels + viral_labels
    
    #Label Encoder
    le = LabelEncoder()
    le.fit(labels)
    #Train_Label
    labels_enc = le.transform(labels)
    
   
    #Train_data
    images=normal_images_array + pneumonia_images_array + viral_images_array
    
  
    
    
   
    return labels_enc,images
    
  
def ImageAugmentation(train_base_path,val_base_path,test_base_path):
    image_size = 224
    batch_size = 32
    train_datagen = ImageDataGenerator(zoom_range=0.1,     
                                    horizontal_flip = True,
                                    fill_mode = 'constant',
                                    validation_split=0.1,
                                    preprocessing_function = preprocess_input)
    
    val_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
    
    train_gen = train_datagen.flow_from_directory(train_base_path, 
                                              target_size=(image_size, image_size), 
                                              batch_size=batch_size)
    val_gen = val_datagen.flow_from_directory(val_base_path, 
                                              target_size=(image_size, image_size), 
                                              batch_size=batch_size) 
                                              

    test_gen = test_datagen.flow_from_directory(test_base_path, 
                                              target_size=(image_size, image_size), 
                                              batch_size=batch_size )
    
    return train_gen,val_gen,test_gen
                                               

    
def pretrained_mode_without_ImageAugmentation(train_data_scaled,val_data_scaled,train_labels_enc,val_labels_enc):
    INIT_LR = 1e-3
    EPOCHS = 5
    BS = 8
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                     classes = np.unique(train_labels_enc),
                                                     y=train_labels_enc)
    class_weights=dict(zip(np.unique(train_labels_enc), class_weights))
    #baseModel = VGG16(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
    baseModel = VGG19(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
    #baseModel = DenseNet201(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
    #baseModel = ResNet50(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
    #baseModel = InceptionV3(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
      
      
    
    # construct the head of the model that will be placed on top of the
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dense(64, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(3, activation="softmax")(headModel)
    
    model = Model(inputs=baseModel.input, outputs=headModel)
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=8,verbose=1),ModelCheckpoint(filepath='best_model_w_imgA.h5', monitor='val_loss', save_best_only=True)]
    
    for layer in baseModel.layers:
        layer.trainable = False
        
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
# train the head of the network
    H = model.fit(train_data_scaled, train_labels_enc, validation_data=(val_data_scaled,val_labels_enc),batch_size = BS, epochs=EPOCHS ,callbacks=callbacks,verbose=1,class_weight = class_weights)
    
    baseModel.trainable = True
    opt = Adam(lr=INIT_LR/10, decay=INIT_LR / EPOCHS)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

    H = model.fit(train_data_scaled, train_labels_enc, validation_data=(val_data_scaled,val_labels_enc),batch_size = BS, epochs=EPOCHS ,callbacks=callbacks,verbose=1,class_weight = class_weights)
   
    print("history",H)
    return H,model

def InceptionNet(train_data_scaled,val_data_scaled,train_labels_enc,val_labels_enc):
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                     classes = np.unique(train_labels_enc),
                                                     y=train_labels_enc)
    class_weights=dict(zip(np.unique(train_labels_enc), class_weights))
    INIT_LR = 1e-3
    BS=8
    baseModel = InceptionV3(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
    model = Sequential([
        baseModel,
        GlobalAveragePooling2D(),
        Dense(512, activation="relu"),
        
       
        Dense(128, activation="relu"),

        
        Dense(64,activation="relu"),
        
        Dropout(0.3),
        Dense(3,activation="sigmoid")
    ])
    
   
    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam',metrics=["accuracy"])
# train the head of the network
    H = model.fit(train_data_scaled, train_labels_enc, validation_data=(val_data_scaled,val_labels_enc),batch_size = BS, epochs=5,verbose=1,class_weight = class_weights)
    print("history",H)
    return H,model
    

def basic_CNN(train_data_scaled,val_data_scaled,train_labels_enc,val_labels_enc):
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                     classes = np.unique(train_labels_enc),
                                                     y=train_labels_enc)
    class_weights=dict(zip(np.unique(train_labels_enc), class_weights))
    INIT_LR = 1e-3
    BS=50
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(224, 224, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(3, activation='sigmoid'))
   
    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam',metrics=["accuracy"])
# train the head of the network
    H = model.fit(train_data_scaled, train_labels_enc, validation_data=(val_data_scaled,val_labels_enc),batch_size = BS, epochs=1,verbose=1,class_weight = class_weights)
    print("history",H)
    return H,model
    
# def pretrained_Feature_Extractor(train_data_scaled,val_data_scaled,train_labels_enc,val_labels_enc):
#     print("Transfer Learning Feature")
#     input_shape = (150, 150, 3)
#     batch_size = 50
#     epochs=20
    
#     vgg = vgg16.VGG16(include_top=False, weights='imagenet', 
#                                      input_shape=input_shape)
#     output = vgg.layers[-1].output
#     output = keras.layers.Flatten()(output)
#     vgg_model = Model(vgg.input, output)
#     output = vgg.layers[-1].output
    
#     vgg_model.trainable = False
#     for layer in vgg_model.layers:
#         layer.trainable = False
      
#     print("beforemodel prediction")    
#     train_data_features=vgg_model.predict(train_data_scaled,verbose=1)
#     val_data_features=vgg_model.predict(val_data_scaled,verbose=1)
    
#     print('Train Bottleneck Features:', train_data_features.shape, 
#       '\tValidation Bottleneck Features:', val_data_features.shape)
    
#     input_shape = vgg_model.output_shape[1]

#     model = Sequential()
#     model.add(InputLayer(input_shape=(input_shape,)))
#     model.add(Dense(512, activation='relu', input_dim=input_shape))
#     model.add(Dropout(0.3))
#     model.add(Dense(512, activation='relu'))
#     model.add(Dropout(0.3))
#     model.add(Dense(1, activation='sigmoid'))
    
#     print("Before compile")
#     model.compile(loss='binary_crossentropy',
#                   optimizer=optimizers.RMSprop(lr=1e-4),
#                   metrics=['accuracy'])
    
#     print("After compile")
#     allRuns = model.fit(x=train_data_features, y=train_labels_enc,
#                     validation_data=(val_data_features, val_labels_enc),
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1)
    
#     return allRuns
    
def PlotGraph(allRuns):
    print("Model fit")
    acc = allRuns.history['accuracy']
    val_acc = allRuns.history['val_accuracy']
    
    loss = allRuns.history['loss']
    val_loss = allRuns.history['val_loss']
    
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([min(plt.ylim()),3])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
    
def pretrained_Feature_Extractor_withImgAugmentation(train_data_scaled,val_data_scaled,train_labels_enc,val_labels_enc,train_data,val_data):

    train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
                                   horizontal_flip=True, fill_mode='nearest')

    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow(train_data, train_labels_enc, batch_size=30)
    val_generator = val_datagen.flow(val_data, val_labels_enc, batch_size=20)
    
    input_shape = (220, 220, 3)
    batch_size = 50
    epochs=20
    
    vgg = vgg16.VGG16(include_top=False, weights='imagenet', 
                                     input_shape=input_shape)
    output = vgg.layers[-1].output
    output = keras.layers.Flatten()(output)
    vgg_model = Model(vgg.input, output)
    output = vgg.layers[-1].output
    
    vgg_model.trainable = True
    set_trainable = False
    for layer in vgg_model.layers:
        if layer.name in ['block5_conv1', 'block4_conv1']:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
            
    print("beforemodel prediction")    
    train_data_features=vgg_model.predict(train_data_scaled,verbose=1)
    val_data_features=vgg_model.predict(val_data_scaled,verbose=1)
    
    print('Train Bottleneck Features:', train_data_features.shape, 
      '\tValidation Bottleneck Features:', val_data_features.shape)
    
    input_shape = vgg_model.output_shape[1]
    
    model = Sequential()
    model.add(vgg_model)
    model.add(Dense(512, activation='relu', input_dim=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=2e-5),
                  metrics=['accuracy'])
                  
    history = model.fit_generator(train_generator, steps_per_epoch=10, epochs=20,
                                  validation_data=val_generator, validation_steps=50, 
                                  verbose=1)     
    
    return history
    


def DENSENET121(train_gen,val_gen):
    
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                     classes = np.unique(train_labels_enc),
                                                     y=train_labels_enc)
    class_weights=dict(zip(np.unique(train_labels_enc), class_weights)),
    base_learning_rate = 0.00001
    print(class_weights)
    base_model = DenseNet121(weights='imagenet', include_top=False)

    x = base_model.output
    
    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)
    
    #Dropout Layer
    x = Dropout(0.2)(x) # Regularize with dropout
    
    # and a logistic layer
    predictions = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    

    #Compile model
    base_model.trainable = True
    model.compile(optimizer=Adam(lr= base_learning_rate), loss= "mse", metrics = ['accuracy'])
   
    final_ckpt_filename= "chest_xray/best_model.hdf5"
    checkpoint = ModelCheckpoint(filepath=final_ckpt_filename, save_best_only=True, save_weights_only=True, verbose = 1)

    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, mode='min', verbose = 1)
    
    fine_tune_epochs = 5

    # Fitting the model
    history_unfreeze = model.fit(train_gen,
                             validation_data=val_gen,
                             epochs= fine_tune_epochs,
                             callbacks=[checkpoint,early_stop, lr_reduce],
                             class_weight = class_weights
                        )
    
    return history_unfreeze,model

if __name__=="__main__":
    
    # train_nosal_path='chest_xray/train/NORMAL/*'
    # train_pneumonia_path ='chest_xray/train/PNEUMONIA/*'
    # val_normal_path='chest_xray/val/NORMAL/*'
    # val_pneumonia_path ='chest_xray/val/PNEUMONIA/*'
    train_base_path='chest_xray/train/'
    val_base_path='chest_xray/val/'
    test_base_path='chest_xray/test/'
    
    seperate_into_directories(train_base_path)
    seperate_into_directories(val_base_path)
    seperate_into_directories(test_base_path)
    
    
    
    train_gen,val_gen,test_gen= ImageAugmentation(train_base_path,val_base_path,test_base_path)
    # #Train Data
    train_labels_enc,train_data=read_data(train_base_path)
    
    # #Validation Data
    val_labels_enc,val_data=read_data(val_base_path)
    
    #Test Data
    test_labels_enc,test_data=read_data(test_base_path)
    
    
    print(len(train_labels_enc))
    print(len(train_data))
    print(len(val_labels_enc))
    print(len(val_data))
    
    train_data=np.array(train_data)
    train_data_scaled = train_data.astype('float32')
    val_data=np.array(val_data)
    val_data_scaled  = val_data.astype('float32')
    test_data_scaled = np.array(test_data)
    val_data_scaled  = val_data.astype('float32')
    train_data_scaled /= 255
    val_data_scaled /= 255
    test_data_scaled /=255
    
       
    # #PreTrained Feature Extractor
    # # allRuns=pretrained_Feature_Extractor(train_data_scaled,val_data_scaled,train_labels_enc,val_labels_enc)
    
    # #PreTrainedFeatureExtractor with Image Augmentation
    # #allRuns=pretrained_Feature_Extractor_withImgAugmentation(train_data_scaled,val_data_scaled,train_labels_enc,val_labels_enc,train_data,val_data)
    
    # #DenseNET
    #allRuns,model=DENSENET121(train_gen,val_gen)
    
    
   
    
    # acc = accuracy_score(test_labels_enc, np.round(preds))*100
    # print("Test data accuracy : "+str(acc))
    # print("Classification report")
    # print(classification_report(test_labels_enc,np.round(preds)))
        
    allRuns,model= pretrained_mode_without_ImageAugmentation(train_data_scaled,val_data_scaled,train_labels_enc,val_labels_enc)
    predIdxs = model.predict(test_data_scaled, batch_size=8)
    predIdxs = np.argmax(predIdxs, axis=1)
    print(predIdxs)
    print(classification_report(test_labels_enc, predIdxs,digits=4))
    
    cm = confusion_matrix(test_labels_enc,predIdxs,labels=[0,1,2])
    print('Confusion matrix : \n',cm)
    
    PlotGraph(allRuns)
    
    
                                                                                                        
    