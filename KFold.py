# Data preprocessing
from sklearn.preprocessing import LabelBinarizer
# for binary encoding
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import math
import time

# GPU
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# import tensorflow as tf
# sess=tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))


#training
from Models import * 
from send_email import send_email

######### Metrics #######
# set the matplotlib backend so figures can be saved in the background
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd


from sklearn.model_selection import KFold, StratifiedKFold
from keras import backend as K
import gc

import os
import pandas as pd
from sklearn.utils import shuffle

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-n_c", "--num_classes", required=True,
                help="Num of classes")
ap.add_argument("-m_n", "--model_name", required=True,
                help="Model Name")
ap.add_argument("-t", "--traindir", required=True)
ap.add_argument("-o", "--logfile", required=True)
# do not add / at the end of the passed traindir argument

ap.add_argument("-opt", "--optimizer", required=False,
                help="Optimizer")
ap.add_argument("-d_n", "--dense_neurons", required=True,
                help="Num of dense layer neurons")
ap.add_argument("-lr", "--learning_rate", required=True,
                help="Num of dense layer neurons")
ap.add_argument("-epochs", "--epochs", required=True,
                help="Num of epochs")
ap.add_argument("-batch", "--batch", required=True,
                help="batch size")

args = vars(ap.parse_args())


print (os.getcwd())
train_set_dir = args["traindir"]

# get all subfolders within the train_set_dir
subfolders = [f.path for f in os.scandir(train_set_dir) if f.is_dir()]

print(subfolders)

# get list of files in each subfolder
file_list = []
for subfolder in subfolders:
    file_list.append(os.listdir(subfolder))

# print number of files in each subfolder
for i in range(len(subfolders)):
    print("{} has {} files".format(subfolders[i], len(file_list[i])))

#get list of labels from subfolder
labels = []
for subfolder in subfolders:
    labels.append(os.path.basename(subfolder))
print('labels: ', labels)

df = pd.DataFrame(columns=['images','labels'])

#add path of all images to df['images'] and labels to df['labels']
# df['images']=[]
# df['labels']=[]
img_lst = []
label_lst = []

for i in range(len(subfolders)):
    for j in range(len(file_list[i])):
        print(subfolders[i])
        print(i)
        img_lst.append(subfolders[i]+'/'+file_list[i][j])
        label_lst.append(labels[i])

df['images']=img_lst
df['labels']=label_lst

print("image paths:\n", df['images'])
print("labels:\n", df['labels'])


#df['images']=[ac_dir+str(x) for x in images_train_ac]+[ad_dir+str(x) for x in images_train_ad]+[ah_dir+str(x) for x in images_train_ah]+[camel_dir+str(x) for x in images_train_camel]+[cat_dir+str(x) for x in images_train_cat]+[dog_dir+str(x) for x in images_train_dog]+[donkey_dir+str(x) for x in images_train_donkey]+[es_dir+str(x) for x in images_train_es]+[fox_dir+str(x) for x in images_train_fox]+[ghost_dir+str(x) for x in images_train_ghost]+[goat_dir+str(x) for x in images_train_goat]+[oa_dir+str(x) for x in images_train_oa]+[rat_dir+str(x) for x in images_train_rat]+[sheep_dir+str(x) for x in images_train_sheep]+[ss_dir+str(x) for x in images_train_ss]
#df['labels']=[str('AC') for x in images_train_ac]+[str('AD') for x in images_train_ad]+[str('AH') for x in images_train_ah]+[str('Camel') for x in images_train_camel]+[str('Cat') for x in images_train_cat]+[str('Dog') for x in images_train_dog]+[str('Donkey') for x in images_train_donkey]+[str('ES') for x in images_train_es]+[str('Fox') for x in images_train_fox]+[str('Ghost') for x in images_train_ghost]+[str('Goat') for x in images_train_goat]+[str('OA') for x in images_train_oa]+[str('Rat') for x in images_train_rat]+[str('Sheep') for x in images_train_sheep]+[str('SS') for x in images_train_ss]
df = shuffle(df)
df.reset_index(drop=True, inplace=True) 
print(df.columns)
df.to_csv('train.csv')

# ------------------------------------------------------------- #
# ------------------ Variables Definition --------------------- #
# ------------------------------------------------------------- #

# initialize the data and labels
print("[INFO] loading images...")
#data = []
#labels = []

num_classes=int(args["num_classes"])
if (num_classes==2):
    activation="sigmoid"
    class_mode='binary'
else:
    activation="softmax"
    class_mode='categorical'
learning_rate=float(args["learning_rate"])
optimizer = 'SGD'
num_epochs=int(args["epochs"])
batch_size = int(args["batch"])
dense_neurons = int(args["dense_neurons"])
model_name = args["model_name"]

train_data = pd.read_csv('train.csv')
Y = train_data[['labels']]

kf = KFold(n_splits = 10)   
fold_num=1

# ------------------------------------------------------------- #
# -------------------- DATA PREPARATION ----------------------- #
# ------------------------------------------------------------- #

def Average(lst): 
    return sum(lst) / len(lst)

# total number of images in folder
n = len(df['images'])

#n=len(images_train_ac)+len(images_train_ad)+len(images_train_ah)+len(images_train_camel)+len(images_train_cat)+len(images_train_dog)+len(images_train_donkey)+len(images_train_es)+len(images_train_fox)+len(images_train_ghost)+len(images_train_goat)+len(images_train_oa)+len(images_train_rat)+len(images_train_sheep)+len(images_train_ss)
print('n = ', n)
# ac_fscores = []
# ad_fscores = []
# ah_fscores = []
# camel_fscores = []
# cat_fscores = []
# dog_fscores = []
# donkey_fscores = []
# es_fscores = []
# fox_fscores = []
# ghost_fscores = []
# goat_fscores = []
# oa_fscores = []
# rat_fscores = []
# sheep_fscores = []
# ss_fscores = []
# accuracies = []

file = open("./" + args["logfile"], "w") 
file.write("#################################\n")
file.write(model_name)
file.write(" K-fold trial results\n")
file.write("#################################\n\n")

file.write("Number of classes: ")
file.write(str(num_classes))
file.write("\n")
file.write("Learning Rate: ")
file.write(str(learning_rate))
file.write("\n")
file.write("Optimizer: ")
file.write(optimizer)
file.write("\n")
file.write("Epochs: ")
file.write(str(num_epochs))
file.write("\n")
file.write("Batch Size: ")
file.write(str(batch_size))
file.write("\n")
file.write("Dense Neurons: ")
file.write(str(dense_neurons))
file.write("\n\n")

for train_index, val_index in kf.split(np.zeros(n),Y):
	
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Fold #%d" % fold_num)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
	
    # ------------------------------------------------------------- #
    # -------------------- MODEL DEFINITION ----------------------- #
    # ------------------------------------------------------------- #

    if (model_name=="MobileNetV2"): 
        # learning_rate=0.01, num_classes=2, dense_neurons=512,activation = 'softmax', optimizer='SGD'
        model = create_MobileNetV2(num_classes=num_classes,learn_rate=learning_rate,activation = activation, optimizer=optimizer, dense_neurons=dense_neurons)
        image_size = 224

    elif (model_name=="InceptionV3"): 
        # learning_rate=0.01,num_classes=2, dense_neurons=512,activation = 'softmax', optimizer='SGD'
        model = create_InceptionV3(num_classes=num_classes,learn_rate=learning_rate,activation = activation, optimizer=optimizer, dense_neurons=dense_neurons)
        image_size = 299

    elif (model_name=="ResNet18"): 
        # learning_rate=0.01,num_classes=2, dense_neurons=512,activation = 'softmax', optimizer='SGD'
        model = create_ResNet18(num_classes=num_classes,learn_rate=learning_rate,activation = activation, optimizer=optimizer, dense_neurons=dense_neurons)
        image_size = 224

    elif (model_name=="Xception"): 
        # learning_rate=0.01,num_classes=2, dense_neurons=1024,activation = 'softmax', optimizer='SGD'
        model = create_Xception(num_classes=num_classes,learn_rate=learning_rate,activation = activation, optimizer=optimizer, dense_neurons=dense_neurons)
        image_size = 299

    elif (model_name=="DenseNet121"): 
        # learning_rate=0.01,num_classes=2, dense_neurons=512,activation = 'softmax', optimizer='SGD'
        model = create_densenet121(num_classes=num_classes,learn_rate=learning_rate,activation = activation, optimizer=optimizer, dense_neurons=dense_neurons)
        image_size = 224

    elif (model_name=="EfficientNetB1"): 
        # learning_rate=0.01,num_classes=2, dense_neurons=512,activation = 'softmax', optimizer='SGD'
        model = create_EfficientNetB1(num_classes=num_classes,learn_rate=learning_rate,activation = activation, optimizer=optimizer, dense_neurons=dense_neurons)
        image_size = 224

    else: 
        print("Wrong Model Name!")

#     print('np.zeros(n),Y',np.zeros(n))
#     print('Y',Y)
#     print()
  
    training_data = train_data.iloc[train_index]
    validation_data = train_data.iloc[val_index]
    print('train_index',train_index) 
    print('val_index',val_index)

#     print('len(training_data)',len(training_data))
#     print('(training_data)',(training_data))
#     print('len(validation_data)',len(validation_data))
#     print('(validation_data)',(validation_data))




    # load training data
    idg = ImageDataGenerator(rescale=1./255)
    train_data_generator = idg.flow_from_dataframe(training_data,directory=None,
                            x_col = "images", y_col = "labels",
                            target_size=(image_size, image_size),
                            batch_size=batch_size,class_mode=class_mode,shuffle=True)
    valid_data_generator = idg.flow_from_dataframe(validation_data,directory=None,
			    x_col = "images", y_col = "labels",
                            target_size=(image_size, image_size),
                            batch_size=batch_size,class_mode=class_mode,shuffle=False)

#     for i in range(10):
#         entry = next(train_data_generator)
#         image = entry[0][0]
#         label = entry[1]
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         cv2.imwrite('./kfold_investigation/image_{}.png'.format(i), image)
#         print(np.shape(image))
#         print(np.shape(label))
#         print('Label: ', label)

    # FIT THE MODEL
    history = model.fit_generator(
        train_data_generator,
        epochs=num_epochs,
        workers=16,
        use_multiprocessing=True,
 	max_queue_size=40)

    # EVALUATE THE MODEL
    predictions = model.predict_generator(
        valid_data_generator,
        workers=16,
        use_multiprocessing=True,
 	max_queue_size=60)
    lb = LabelBinarizer()
    
    # print('y_true before ravel and fit', valid_data_generator.classes)
    # print('y_true before ravel and fit len --> ', len(valid_data_generator.classes))
    # y_true = lb.fit_transform(valid_data_generator.classes)
    # y_true = y_true.ravel()
    y_true = valid_data_generator.classes

    # print('predictions', predictions)
    # print('predictions len --> ', len(predictions))
	
    if num_classes == 2:
        target_names = labels
        y_pred = np.array([int(np.round(p)) for p in predictions])
    else:
        target_names = labels
        y_pred=np.argmax(predictions,axis=1)

    print('y_true', y_true)
    print('y_pred', y_pred)
    print('y_true len --> ', len(y_true))
    print('y_pred len --> ', len(y_pred))
    report = classification_report(y_true, y_pred, target_names=target_names)
    print('Classification Report')
    print(report)
    dict_report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    

    file.write("*~~~* Fold #%d results *~~~*\n" % fold_num)
    file.write(report)
    file.write("\n\n")


    tf.keras.backend.clear_session()
    fold_num = fold_num + 1
    del model
    K.clear_session()
    gc.collect()

file.close()

# ------------------------------------------------------------- #
# --------------- TIME STATS AND EMAIL NOTIF ------------------ #
# ------------------------------------------------------------- #
def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

my_ip = str(get_ip())
user = os.getlogin()

current_time = time.ctime()
# send email notification
body = "\nMessage from: "+user+"@"+my_ip+"\nModel: " + model_name + " - K-Fold complete @ " + current_time
to = ['b00068908@aus.edu']
send_email(body, to)

print("K-Fold complete!")


	
