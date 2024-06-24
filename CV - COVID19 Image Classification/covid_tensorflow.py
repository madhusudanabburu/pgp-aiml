from typing import Dict, List
import kfp
from kfp import compiler
from kfp import dsl
import kfp.components as comp
from typing import NamedTuple
import json
from kfp.dsl import Input, InputPath, Output, OutputPath, Dataset, Model, component
from kfp import kubernetes

@dsl.component(packages_to_install=['pandas', 'numpy', 'minio'])
def load_data(trainImages_url:str, trainLabel_url:str, trainImages:Output[Dataset], trainLabels:Output[Dataset]):
    # Loading data into pandas dataframe
    
    import pandas as pd
    import numpy as np
    import os
    
    cwd = os.getcwd()
    
    print('Current working directory' + cwd)
    # Open the file as readonly
    trainImages_v = np.load(trainImages_url)
    trainLabels_v = pd.read_csv(trainLabel_url)

    print('After pd.read_csv')

    with open(trainImages.path, "wb") as f:
        np.save(file=f, arr=trainImages_v)
    
    with open(trainLabels.path, "w") as f:
        trainLabels_v.to_csv(f, index=False)


"""
## Data Preprocessing
"""
@dsl.component(packages_to_install=['pandas', 'numpy', 'opencv-python-headless', 'matplotlib', 'scikit-learn'])
def pre_process_data(trainImages: Dataset, trainLabels: Dataset, train_out: Output[Dataset], y_out: Output[Dataset]):
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import cv2
    from sklearn.preprocessing import LabelBinarizer
    
    #trainImages_l = np.load(trainImages)
    
    with open(trainImages.path, 'rb') as f:
        trainImages_l = np.load(f)
    
    with open(trainLabels.path) as f:
        trainLabels_l = pd.read_csv(f)
        
    new_train = []
    sets = []; getEx = True
    for i in trainImages_l:
        blurr = cv2.GaussianBlur(i,(5,5),0)
        hsv = cv2.cvtColor(blurr,cv2.COLOR_BGR2HSV) #Using BGR TO HSV conversion. reason is mentioned above
        #HSV Bou daries for the Green color (GREEN PARAMETERS)
        lower = (25,40,50)
        upper = (75,255,255)
        mask = cv2.inRange(hsv,lower,upper) # create a mask 
        boolean = mask>0
        new = np.zeros_like(i,np.uint8)
        new[boolean] = i[boolean]
        new_train.append(blurr)
        if getEx:
            plt.subplot(2,3,1);plt.imshow(i) # ORIGINAL
            plt.subplot(2,3,2);plt.imshow(blurr) # BLURRED
            plt.subplot(2,3,3);plt.imshow(hsv) # HSV CONVERTED
            plt.subplot(2,3,6);plt.imshow(new) # NEW PROCESSED IMAGE
            plt.show()
            getEx = False
    new_train = np.asarray(new_train)
    print("# CLEANED IMAGES")
    for i in range(8):
        plt.subplot(2,4,i+1)
        plt.imshow(new_train[i])
        
    # Normalization of the image data
    new_train = new_train / 255
    
    enc = LabelBinarizer()
    y = enc.fit_transform(trainLabels_l)
        
    with open(train_out.path, "wb") as f:
        np.save(file=f, arr=new_train)
        
    with open(y_out.path, "wb") as f:
        np.save(file=f, arr=y)

"""
## Train Model
"""
@dsl.component(packages_to_install=['pandas', 'numpy', 'opencv-python-headless', 'scikit-learn', 'seaborn', 'matplotlib', 'joblib', 'tensorflow==2.11.0', 'tensorflow-io==0.29.0'])
def train(new_train_in: Dataset, y_in: Dataset, cnn_model: Output[Model], X_test_out: Output[Dataset], y_test_out: Output[Dataset]):  

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import joblib
    import tensorflow as tf
    from tensorflow import keras
    
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical # convert to one-hot-encoding
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalMaxPooling2D
    from tensorflow.keras.optimizers import RMSprop
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.optimizers import Adam
    
    with open(new_train_in.path, 'rb') as f:
        new_train = np.load(f)
        
    with open(y_in.path, 'rb') as f:
        y = np.load(f)
    
    X_train, X_test, y_train, y_test = train_test_split(new_train,y , test_size=0.1, random_state=7,stratify=y)
    
    #Check the class proportion as we have used stratify to have an equal distribution of the dataset
    pd.DataFrame(y_train.argmax(axis=1)).value_counts()/pd.DataFrame(y_train.argmax(axis=1)).value_counts().sum()
    pd.DataFrame(y_test.argmax(axis=1)).value_counts()/pd.DataFrame(y_test.argmax(axis=1)).value_counts().sum()
    
    #Build the CNN
    # Set the CNN model 
    batch_size=5


    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', batch_input_shape = (batch_size,128, 128, 3)))


    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))


    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same', activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', activation ='relu'))
    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.4))

    model.add(GlobalMaxPooling2D())
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation = "softmax"))
    model.summary()
    
    #Defining the optimizer and loss function 
    optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, weight_decay=0.0)
    model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])
    
    # fitting the model with epochs = 5 
    history = model.fit(X_train, y_train, epochs = 5, validation_split=0.2, batch_size = batch_size)
    
    # Evaluate the model.
    score = model.evaluate(X_test, y_test, verbose=0, batch_size = 5)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    # save
    joblib.dump(model, cnn_model.path, compress=1)
    print(f"Compressed CNN Model: {np.round(os.path.getsize(cnn_model.path) / 1024 / 1024, 2) } MB")
        
    with open(X_test_out.path, "wb") as f:
        np.save(file=f, arr=X_test)
        
    with open(y_test_out.path, "wb") as f:
        np.save(file=f, arr=y_test)
        
        
@dsl.component(packages_to_install=['pandas', 'numpy', 'opencv-python-headless', 'scikit-learn', 'seaborn', 'matplotlib', 'joblib', 'tensorflow==2.11.0', 'tensorflow-io==0.29.0'])
def predict(X_test_in: Dataset, y_test_in: Dataset, cnn_model_in: Input[Model]): 
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import joblib
    import tensorflow as tf
    from tensorflow import keras
    
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical # convert to one-hot-encoding
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalMaxPooling2D
    from tensorflow.keras.optimizers import RMSprop
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.optimizers import Adam
    
    with open(X_test_in.path, 'rb') as f:
        X_test = np.load(f)
        
    with open(y_test_in.path, 'rb') as f:
        y_test = np.load(f)
        
    #Load the saved model
    model = joblib.load(cnn_model_in.path)
    
    plt.figure(figsize=(2,2))
    plt.imshow(X_test[3],cmap="gray")
    plt.show()
    print('Predicted Label', np.argmax(model.predict(X_test[3].reshape(1,128,128,3))))
    print('True Label', np.argmax(y_test[3]))

    plt.figure(figsize=(2,2))
    plt.imshow(X_test[2],cmap="gray")
    plt.show()
    print('Predicted Label', np.argmax(model.predict(X_test[2].reshape(1,128,128,3))))
    print('True Label', np.argmax(y_test[2]))

    plt.figure(figsize=(2,2))
    plt.imshow(X_test[16],cmap="gray")
    plt.show()
    print('Predicted Label', np.argmax(model.predict(X_test[16].reshape(1,128,128,3))))
    print('True Label', np.argmax(y_test[16]))

    plt.figure(figsize=(2,2))
    plt.imshow(X_test[20],cmap="gray")
    plt.show()
    print('Predicted Label', np.argmax(model.predict(X_test[20].reshape(1,128,128,3))))
    print('True Label', np.argmax(y_test[20]))

    plt.figure(figsize=(2,2))
    plt.imshow(X_test[25],cmap="gray")
    plt.show()
    print('Predicted Label', np.argmax(model.predict(X_test[25].reshape(1,128,128,3))))
    print('True Label', np.argmax(y_test[25]))

        
"""
## Train Model with Learning Rate Reduction
"""
@dsl.component(packages_to_install=['pandas', 'numpy', 'opencv-python-headless', 'scikit-learn', 'seaborn', 'matplotlib', 'joblib', 'tensorflow==2.11.0', 'tensorflow-io==0.29.0'])
def train_lr_reduction(new_train_in: Dataset, y_in: Dataset, cnn_lr_reduction_model: Output[Model], X_test_out: Output[Dataset], y_test_out: Output[Dataset]):  

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import joblib
    import tensorflow as tf
    from tensorflow import keras
    
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical # convert to one-hot-encoding
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalMaxPooling2D
    from tensorflow.keras.optimizers import RMSprop
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.optimizers import Adam
    from keras.callbacks import ReduceLROnPlateau
    
    with open(new_train_in.path, 'rb') as f:
        new_train = np.load(f)
        
    with open(y_in.path, 'rb') as f:
        y = np.load(f)

    
    X_train_1, X_test, y_train_1, y_test = train_test_split(new_train, y , test_size=0.1, random_state=7,stratify=y)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train_1, y_train_1, test_size=0.2, random_state=7,stratify=y_train_1)
    
    #Check the class proportion as we have used stratify to have an equal distribution of the dataset
    pd.DataFrame(y_train.argmax(axis=1)).value_counts()/pd.DataFrame(y_train.argmax(axis=1)).value_counts().sum()
    pd.DataFrame(y_test.argmax(axis=1)).value_counts()/pd.DataFrame(y_test.argmax(axis=1)).value_counts().sum()
    
    #Build the CNN
    # Set the CNN model 
    batch_size=5

    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', batch_input_shape = (batch_size,128, 128, 3)))

    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same', activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', activation ='relu'))
    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.4))

    model.add(GlobalMaxPooling2D())
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation = "softmax"))
    model.summary()

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
    epochs = 5
    batch_size = 5
    
    #Defining the optimizer and loss function 
    optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, weight_decay=0.0)
    model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])    

    # fitting the model with epochs = 5 
    history = model.fit(X_train, y_train, epochs = 5, validation_data = (X_val,y_val), batch_size = batch_size, verbose = 2, callbacks=[learning_rate_reduction])

    # save
    joblib.dump(model, cnn_lr_reduction_model.path, compress=1)
    print(f"Compressed CNN Model: {np.round(os.path.getsize(cnn_lr_reduction_model.path) / 1024 / 1024, 2) } MB")   
        
    with open(X_test_out.path, "wb") as f:
        np.save(file=f, arr=X_test)
        
    with open(y_test_out.path, "wb") as f:
        np.save(file=f, arr=y_test)


        
"""
## Train Model with Learning Rate Reduction
"""
@dsl.component(packages_to_install=['pandas', 'numpy', 'opencv-python-headless', 'scikit-learn', 'seaborn', 'matplotlib', 'joblib', 'tensorflow==2.11.0', 'tensorflow-io==0.29.0'])
def compare_cnn_models(cnn_model: Input[Model], cnn_lr_reduction_model: Input[Model], X_test_in: Input[Dataset], y_test_in: Input[Dataset]): 
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import joblib
    import tensorflow as tf
    from tensorflow import keras
    
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    
    with open(X_test_in.path, 'rb') as f:
        X_test = np.load(f)
        
    with open(y_test_in.path, 'rb') as f:
        y_test = np.load(f)
    
    # Evaluate the first CNN model
    
    cnn_model1 = joblib.load(cnn_model.path)
    
    score1 = cnn_model1.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score1[0])
    print('Test accuracy:', score1[1])
    
    # Predict the values from the validation dataset
    Y_pred_cnn_model1 = cnn_model1.predict(X_test)
    # Convert predictions classes to one hot vectors 
    result_cnn_model1 = np.argmax(Y_pred_cnn_model1, axis=1)
    # Convert validation observations to one hot vectors
    Y_true_cnn_model1 = np.argmax(y_test, axis=1)

    conf_mat_cnn_model1 = confusion_matrix(Y_true_cnn_model1, result_cnn_model1)
        
    #accuracy_score(X_test, Y_pred_cnn_model1)
    
    #cnn_model1.attrs['accuracy'] = score_cnn_model_1[1]
    #cnn_model1.attrs['confusion_matrix'] = conf_mat_cnn_model1
    
    print('Prediction using the first model')
    
    plt.figure(figsize=(2,2))
    plt.imshow(X_test[3],cmap="gray")
    plt.show()
    print('Predicted Label', np.argmax(cnn_model1.predict(X_test[3].reshape(1,128,128,3))))
    print('True Label', np.argmax(y_test[3]))

    plt.figure(figsize=(2,2))
    plt.imshow(X_test[2],cmap="gray")
    plt.show()
    print('Predicted Label', np.argmax(cnn_model1.predict(X_test[2].reshape(1,128,128,3))))
    print('True Label', np.argmax(y_test[2]))

    plt.figure(figsize=(2,2))
    plt.imshow(X_test[16],cmap="gray")
    plt.show()
    print('Predicted Label', np.argmax(cnn_model1.predict(X_test[16].reshape(1,128,128,3))))
    print('True Label', np.argmax(y_test[16]))

    plt.figure(figsize=(2,2))
    plt.imshow(X_test[20],cmap="gray")
    plt.show()
    print('Predicted Label', np.argmax(cnn_model1.predict(X_test[20].reshape(1,128,128,3))))
    print('True Label', np.argmax(y_test[20]))

    plt.figure(figsize=(2,2))
    plt.imshow(X_test[25],cmap="gray")
    plt.show()
    print('Predicted Label', np.argmax(cnn_model1.predict(X_test[25].reshape(1,128,128,3))))
    print('True Label', np.argmax(y_test[25]))
    
    # Evaluate the second CNN model
    
    cnn_model2 = joblib.load(cnn_lr_reduction_model.path)
    
    # Predict the values from the validation dataset
    Y_pred_cnn_model2 = cnn_model2.predict(X_test)
    # Convert predictions classes to one hot vectors 
    result_cnn_model2 = np.argmax(Y_pred_cnn_model2, axis=1)
    # Convert validation observations to one hot vectors
    Y_true_cnn_model2 = np.argmax(y_test, axis=1)

    conf_mat_cnn_model2 = confusion_matrix(Y_true_cnn_model2, result_cnn_model2)
    
    #score_cnn_model_2 = cnn_model2.evaluate(X_test, y_test, verbose=0, batch_size = 5)
    #print('Test loss:', score_cnn_model_2[0])
    #print('Test accuracy:', score_cnn_model_2[1])
    
    #accuracy_score(X_test, Y_pred_cnn_model2)
    
    #cnn_model2.attrs['accuracy'] = score_cnn_model_2[1]
    #cnn_model2.attrs['confusion_matrix'] = conf_mat_cnn_model2
    
    print('Loss and Accuracy score of Models')
    
    score_cnn_model_1 = cnn_model1.evaluate(X_test, y_test, verbose=0, batch_size = 5)
    print('Default Model Test loss:', score_cnn_model_1[0])
    print('Default Model Test accuracy:', score_cnn_model_1[1])
    
    score_cnn_model_2 = cnn_model2.evaluate(X_test, y_test, verbose=0, batch_size = 5)
    print('Tuned Model with LearningRate Test loss:', score_cnn_model_2[0])
    print('Tuned Model with LearningRate Test accuracy:', score_cnn_model_2[1])
    
    print('Prediction using the second model - tuned with Learning Rate Reduction')
    
    plt.figure(figsize=(2,2))
    plt.imshow(X_test[3],cmap="gray")
    plt.show()
    print('Predicted Label', np.argmax(cnn_model2.predict(X_test[3].reshape(1,128,128,3))))
    print('True Label', np.argmax(y_test[3]))

    plt.figure(figsize=(2,2))
    plt.imshow(X_test[2],cmap="gray")
    plt.show()
    print('Predicted Label', np.argmax(cnn_model2.predict(X_test[2].reshape(1,128,128,3))))
    print('True Label', np.argmax(y_test[2]))

    plt.figure(figsize=(2,2))
    plt.imshow(X_test[16],cmap="gray")
    plt.show()
    print('Predicted Label', np.argmax(cnn_model2.predict(X_test[16].reshape(1,128,128,3))))
    print('True Label', np.argmax(y_test[16]))

    plt.figure(figsize=(2,2))
    plt.imshow(X_test[20],cmap="gray")
    plt.show()
    print('Predicted Label', np.argmax(cnn_model2.predict(X_test[20].reshape(1,128,128,3))))
    print('True Label', np.argmax(y_test[20]))

    plt.figure(figsize=(2,2))
    plt.imshow(X_test[25],cmap="gray")
    plt.show()
    print('Predicted Label', np.argmax(cnn_model2.predict(X_test[25].reshape(1,128,128,3))))
    print('True Label', np.argmax(y_test[25]))
    
    
"""
## pipeline function
"""
@dsl.pipeline(pipeline_root='', name='covid_pipeline', display_name='Covid XRay Image Recognition Pipeline')
def covid_pipeline():
        
    trainImages_url = "/files/trainimage.npy"
    trainLabel_url = "/files/trainLabels.csv"
    volume_name = 'covidmgmt-workspace'
    load_task = load_data(trainImages_url=trainImages_url, trainLabel_url=trainLabel_url)
    load_task.set_caching_options(False)
    pre_process_task = pre_process_data(trainImages=load_task.outputs['trainImages'], trainLabels=load_task.outputs['trainLabels'])
    pre_process_task.set_caching_options(False)
    train_task = train(new_train_in=pre_process_task.outputs['train_out'], y_in=pre_process_task.outputs['y_out'])
    train_task.set_caching_options(False)
    #predict_task = predict(X_test_in=train_task.outputs['X_test_out'], y_test_in=train_task.outputs['y_test_out'], cnn_model_in=train_task.outputs['cnn_model'])
    train_lr_task = train_lr_reduction(new_train_in=pre_process_task.outputs['train_out'], y_in=pre_process_task.outputs['y_out'])
    train_lr_task.set_caching_options(False)
    compare_learning_task = compare_cnn_models(cnn_model=train_task.outputs['cnn_model'], cnn_lr_reduction_model=train_lr_task.outputs['cnn_lr_reduction_model'], X_test_in=train_task.outputs['X_test_out'],y_test_in=train_task.outputs['y_test_out'])
    compare_learning_task.set_caching_options(False)
    
    kubernetes.mount_pvc(
        load_task,
        pvc_name='covid-workspace',
        mount_path='/files',
    )
        
    kubernetes.mount_pvc(
        pre_process_task,
        pvc_name='covid-workspace',
        mount_path='/files',
    )
    
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(covid_pipeline, __file__ + '.yaml')
