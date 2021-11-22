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

ac_dir = train_set_dir + 'AC/'
ad_dir = train_set_dir + 'AD/'
ah_dir = train_set_dir + 'AH/'
camel_dir = train_set_dir + 'Camel/'
cat_dir = train_set_dir + 'Cat/'
dog_dir = train_set_dir + 'Dog/'
donkey_dir = train_set_dir + 'Donkey/'
es_dir = train_set_dir + 'ES/'
fox_dir = train_set_dir + 'Fox/'
ghost_dir = train_set_dir + 'Ghost/'
goat_dir = train_set_dir + 'Goat/'
oa_dir = train_set_dir + 'OA/'
rat_dir = train_set_dir + 'Rat/'
sheep_dir = train_set_dir + 'Sheep/'
ss_dir = train_set_dir + 'SS/'

files_in_train_ac = sorted(os.listdir(ac_dir))
files_in_train_ad = sorted(os.listdir(ad_dir))
files_in_train_ah = sorted(os.listdir(ah_dir))
files_in_train_camel = sorted(os.listdir(camel_dir))
files_in_train_cat = sorted(os.listdir(cat_dir))
files_in_train_dog = sorted(os.listdir(dog_dir))
files_in_train_donkey = sorted(os.listdir(donkey_dir))
files_in_train_es = sorted(os.listdir(es_dir))
files_in_train_fox = sorted(os.listdir(fox_dir))
files_in_train_ghost = sorted(os.listdir(ghost_dir))
files_in_train_goat = sorted(os.listdir(goat_dir))
files_in_train_oa = sorted(os.listdir(oa_dir))
files_in_train_rat = sorted(os.listdir(rat_dir))
files_in_train_sheep = sorted(os.listdir(sheep_dir))
files_in_train_ss = sorted(os.listdir(ss_dir))

images_train_ac=[i for i in files_in_train_ac]
images_train_ad=[i for i in files_in_train_ad]
images_train_ah=[i for i in files_in_train_ah]
images_train_camel=[i for i in files_in_train_camel]
images_train_cat=[i for i in files_in_train_cat]
images_train_dog=[i for i in files_in_train_dog]
images_train_donkey=[i for i in files_in_train_donkey]
images_train_es=[i for i in files_in_train_es]
images_train_fox=[i for i in files_in_train_fox]
images_train_ghost=[i for i in files_in_train_ghost]
images_train_goat=[i for i in files_in_train_goat]
images_train_oa=[i for i in files_in_train_oa]
images_train_rat=[i for i in files_in_train_rat]
images_train_sheep=[i for i in files_in_train_sheep]
images_train_ss=[i for i in files_in_train_ss]

print(len(images_train_ac))
print(len(images_train_ad))
print(len(images_train_ah))
print(len(images_train_camel))
print(len(images_train_cat))
print(len(images_train_dog))
print(len(images_train_donkey))
print(len(images_train_es))
print(len(images_train_fox))
print(len(images_train_ghost))
print(len(images_train_goat))
print(len(images_train_oa))
print(len(images_train_rat))
print(len(images_train_sheep))
print(len(images_train_ss))

df = pd.DataFrame(columns=['images','labels'])

df['images']=[ac_dir+str(x) for x in images_train_ac]+[ad_dir+str(x) for x in images_train_ad]+[ah_dir+str(x) for x in images_train_ah]+[camel_dir+str(x) for x in images_train_camel]+[cat_dir+str(x) for x in images_train_cat]+[dog_dir+str(x) for x in images_train_dog]+[donkey_dir+str(x) for x in images_train_donkey]+[es_dir+str(x) for x in images_train_es]+[fox_dir+str(x) for x in images_train_fox]+[ghost_dir+str(x) for x in images_train_ghost]+[goat_dir+str(x) for x in images_train_goat]+[oa_dir+str(x) for x in images_train_oa]+[rat_dir+str(x) for x in images_train_rat]+[sheep_dir+str(x) for x in images_train_sheep]+[ss_dir+str(x) for x in images_train_ss]
df['labels']=[str('AC') for x in images_train_ac]+[str('AD') for x in images_train_ad]+[str('AH') for x in images_train_ah]+[str('Camel') for x in images_train_camel]+[str('Cat') for x in images_train_cat]+[str('Dog') for x in images_train_dog]+[str('Donkey') for x in images_train_donkey]+[str('ES') for x in images_train_es]+[str('Fox') for x in images_train_fox]+[str('Ghost') for x in images_train_ghost]+[str('Goat') for x in images_train_goat]+[str('OA') for x in images_train_oa]+[str('Rat') for x in images_train_rat]+[str('Sheep') for x in images_train_sheep]+[str('SS') for x in images_train_ss]
df = shuffle(df)
df.reset_index(drop=True, inplace=True) 
print(df.columns)
df.to_csv('train.csv')

# ------------------------------------------------------------- #
# ------------------ Variables Definition --------------------- #
# ------------------------------------------------------------- #

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

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

n=len(images_train_ac)+len(images_train_ad)+len(images_train_ah)+len(images_train_camel)+len(images_train_cat)+len(images_train_dog)+len(images_train_donkey)+len(images_train_es)+len(images_train_fox)+len(images_train_ghost)+len(images_train_goat)+len(images_train_oa)+len(images_train_rat)+len(images_train_sheep)+len(images_train_ss)
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
        target_names = ['Animal', 'Ghost']
        y_pred = np.array([int(np.round(p)) for p in predictions])
    else:
        target_names = ['AC', 'AD', 'AH', 'Camel', 'Cat', 'Dog', 'Donkey', 'ES', 'Fox', 'Ghost', 'Goat', 'OA', 'Rat', 'Sheep', 'SS']
        y_pred=np.argmax(predictions,axis=1)

    print('y_true', y_true)
    print('y_pred', y_pred)
    print('y_true len --> ', len(y_true))
    print('y_pred len --> ', len(y_pred))
    report = classification_report(y_true, y_pred, target_names=target_names)
    print('Classification Report')
    print(report)
    dict_report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    # ac_fscores.append(dict_report['AC']['f1-score'])
    # ad_fscores.append(dict_report['AD']['f1-score'])
    # ah_fscores.append(dict_report['AH']['f1-score'])
    # camel_fscores.append(dict_report['Camel']['f1-score'])
    # cat_fscores.append(dict_report['Cat']['f1-score'])
    # dog_fscores.append(dict_report['Dog']['f1-score'])
    # donkey_fscores.append(dict_report['Donkey']['f1-score'])
    # es_fscores.append(dict_report['ES']['f1-score'])
    # fox_fscores.append(dict_report['Fox']['f1-score'])
    # goat_fscores.append(dict_report['Goat']['f1-score'])
    # oa_fscores.append(dict_report['OA']['f1-score'])
    # rat_fscores.append(dict_report['Rat']['f1-score'])
    # sheep_fscores.append(dict_report['Sheep']['f1-score'])
    # ss_fscores.append(dict_report['SS']['f1-score'])
    # accuracies.append(dict_report['accuracy'])

    file.write("*~~~* Fold #%d results *~~~*\n" % fold_num)
    file.write(report)
    file.write("\n\n")

#     results = model.evaluate(valid_data_generator)
#     results = dict(zip(model.metrics_names,results))

#     file.write("*~~~* Fold #%d results *~~~*\n" % fold_num)
#     file.write("Test Accuracy: ")
#     file.write(str(results['accuracy']))
#     file.write("\n")
#     file.write("Test Loss: ")
#     file.write(str(results['loss']))
#     file.write("\n\n")

    tf.keras.backend.clear_session()
    fold_num = fold_num + 1
    del model
    K.clear_session()
    gc.collect()

# avg_ac_fscore = round(Average(ac_fscores),2)
# avg_ad_fscore = round(Average(ad_fscores),2)
# avg_ah_fscore = round(Average(ah_fscores),2)
# avg_camel_fscore = round(Average(camel_fscores),2)
# avg_cat_fscore = round(Average(cat_fscores),2)
# avg_dog_fscore = round(Average(dog_fscores),2)
# avg_donkey_fscore = round(Average(donkey_fscores),2)
# avg_es_fscore = round(Average(es_fscores),2)
# avg_fox_fscore = round(Average(fox_fscores),2)
# avg_goat_fscore = round(Average(goat_fscores),2)
# avg_oa_fscore = round(Average(oa_fscores),2)
# avg_rat_fscore = round(Average(rat_fscores),2)
# avg_sheep_fscore = round(Average(sheep_fscores),2)
# avg_ss_fscore = round(Average(ss_fscores),2)

# file.write('Average AC F1-Score: ')
# file.write(str(avg_ac_fscore))
# file.write("\n")
# file.write('Average AD F1-Score: ')
# file.write(str(avg_ad_fscore))
# file.write("\n")
# file.write('Average AH F1-Score: ')
# file.write(str(avg_ah_fscore))
# file.write("\n")
# file.write('Average Camel F1-Score: ')
# file.write(str(avg_camel_fscore))
# file.write("\n")
# file.write('Average Cat F1-Score: ')
# file.write(str(avg_cat_fscore))
# file.write("\n")
# file.write('Average Dog F1-Score: ')
# file.write(str(avg_dog_fscore))
# file.write("\n")
# file.write('Average Donkey F1-Score: ')
# file.write(str(avg_donkey_fscore))
# file.write("\n")
# file.write('Average ES F1-Score: ')
# file.write(str(avg_es_fscore))
# file.write("\n")
# file.write('Average Fox F1-Score: ')
# file.write(str(avg_fox_fscore))
# file.write("\n")
# file.write('Average Goat F1-Score: ')
# file.write(str(avg_goat_fscore))
# file.write("\n")
# file.write('Average OA F1-Score: ')
# file.write(str(avg_oa_fscore))
# file.write("\n")
# file.write('Average Rat F1-Score: ')
# file.write(str(avg_rat_fscore))
# file.write("\n")
# file.write('Average Sheep F1-Score: ')
# file.write(str(avg_sheep_fscore))
# file.write("\n")
# file.write('Average SS F1-Score: ')
# file.write(str(avg_ss_fscore))
# file.write("\n")

# avg_accuracy = round(Average(accuracies),2)

# file.write('Average Accuracy: ')
# file.write(str(avg_accuracy))
# file.write("\n")
file.close()

# ------------------------------------------------------------- #
# --------------- TIME STATS AND EMAIL NOTIF ------------------ #
# ------------------------------------------------------------- #

current_time = time.ctime()
# send email notification
body = "Model: " + model_name + " - K-Fold complete @ " + current_time
to = ['b00067871@aus.edu','b00068908@aus.edu', 'g00071496@aus.edu', 'g00068368@aus.edu']
send_email(body, to)

print("K-Fold complete!")


	
