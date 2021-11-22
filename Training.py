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
# from sklearn.utils import class_weight

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

# calculate time
def time_elapsed(start, end):
  millis = end - start
  hoursElapsed = (millis/(1000*60*60))%24
  hoursElapsed = int(hoursElapsed)
  minutesElapsed = (millis/(1000*60))%60
  minutesElapsed = int(minutesElapsed)
  secondsElapsed = (millis/1000)%60
  return f"{hoursElapsed} hours, {minutesElapsed} minutes, and {secondsElapsed} seconds"

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-o", "--output", required=True,
#                 help="path to output log file")
# ap.add_argument("-w", "--weights", required=True,
#                 help="path to output trained model weights")
# ap.add_argument("-p", "--plot", required=True,
#                 help="path to output accuracy/loss plot")

ap.add_argument("-o", "--outputdir", required=True,
                help="path to output directory")
ap.add_argument("-t", "--traindir", required=True)
ap.add_argument("-v", "--validationdir", required=True)
ap.add_argument("-x", "--testdir", required=True)


# ap.add_argument("-r", "--ROC", required=True,
#                 help="path to output ROC plot")
# ap.add_argument("-z", "--ROCz", required=True,
#                 help="path to output zoomed ROC plot")
# ap.add_argument("-conf", "--CONF", required=True,
#                 help="path to output  Confusion Matrix")
# ap.add_argument("-pr", "--PR", required=True,
#                 help="path to output  PR plot")


ap.add_argument("-n_c", "--num_classes", required=True,
                help="Num of classes")
ap.add_argument("-opt", "--optimizer", required=False,
                help="Optimizer")
# ap.add_argument("-act", "--activation_function", required=False,
#                 help="Activation function")
ap.add_argument("-d_n", "--dense_neurons", required=False,
                help="Num of dense layer neurons")
ap.add_argument("-lr", "--learning_rate", required=False,
                help="Num of dense layer neurons")
ap.add_argument("-epochs", "--epochs", required=False,
                help="Num of epochs")
ap.add_argument("-batch", "--batch", required=False,
                help="batch size")
ap.add_argument("-m_n", "--model_name", required=True,
                help="Model Name")



args = vars(ap.parse_args())

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


if args["learning_rate"] is not None:
    learning_rate=float(args["learning_rate"])
else: 
    learning_rate = 0.01


if args["optimizer"] is not None:
    optimizer=args["optimizer"]
else: 
    optimizer = 'SGD'


if args["epochs"] is not None:
    num_epochs=int(args["epochs"])
else: 
    num_epochs = 100


if args["batch"] is not None:
    training_batch_size = validaton_batch_size = testing_batch_size =int(args["batch"])
else: 
    # batch size
    training_batch_size =validaton_batch_size = testing_batch_size =  16

if args["dense_neurons"] is not None:
    dense_neurons = int(args["dense_neurons"])
else:
    dense_neurons = 512

# ------------------------------------------------------------- #
# -------------------- MODEL DEFINITION ----------------------- #
# ------------------------------------------------------------- #

if (args["model_name"]=="MobileNetV2"): 
    # learning_rate=0.01, num_classes=2, dense_neurons=512,activation = 'softmax', optimizer='SGD'
    model = create_MobileNetV2(num_classes=num_classes,learn_rate=learning_rate,activation = activation, optimizer=optimizer, dense_neurons=dense_neurons)
    image_size = 224

elif (args["model_name"]=="InceptionV3"): 
    # learning_rate=0.01,num_classes=2, dense_neurons=512,activation = 'softmax', optimizer='SGD'
    model = create_InceptionV3(num_classes=num_classes,learn_rate=learning_rate,activation = activation, optimizer=optimizer, dense_neurons=dense_neurons)
    image_size = 299

elif (args["model_name"]=="ResNet18"): 
    # learning_rate=0.01,num_classes=2, dense_neurons=512,activation = 'softmax', optimizer='SGD'
    model = create_ResNet18(num_classes=num_classes,learn_rate=learning_rate,activation = activation, optimizer=optimizer, dense_neurons=dense_neurons)
    image_size = 224

elif (args["model_name"]=="Xception"): 
    # learning_rate=0.01,num_classes=2, dense_neurons=1024,activation = 'softmax', optimizer='SGD'
    model = create_Xception(num_classes=num_classes,learn_rate=learning_rate,activation = activation, optimizer=optimizer, dense_neurons=dense_neurons)
    image_size = 299

elif (args["model_name"]=="DenseNet121"): 
    # learning_rate=0.01,num_classes=2, dense_neurons=512,activation = 'softmax', optimizer='SGD'
    model = create_densenet121(num_classes=num_classes,learn_rate=learning_rate,activation = activation, optimizer=optimizer, dense_neurons=dense_neurons)
    image_size = 224

elif (args["model_name"]=="EfficientNetB1"): 
    # learning_rate=0.01,num_classes=2, dense_neurons=512,activation = 'softmax', optimizer='SGD'
    model = create_EfficientNetB1(num_classes=num_classes,learn_rate=learning_rate,activation = activation, optimizer=optimizer, dense_neurons=dense_neurons)
    image_size = 224

else: 
    print("Wrong Model Name!")


# ------------------------------------------------------------- #
# -------------------- DATA PREPARATION ----------------------- #
# ------------------------------------------------------------- #




# load training data
train_set_dir = args["traindir"]
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_set_dir,
        target_size=(image_size, image_size),
        batch_size=training_batch_size,
        class_mode=class_mode)
# print("[INFO] Training set successfully loaded %d images!" % len(train_generator))

# # class_weights to account for imbalanced data distribution
# class_weights = {0:52, 1:67, 2:13, 3:101, 4:47, 5:22, 6:3, 7:54, 8:1, 9:1, 10:35, 11:73, 12:13, 13:31}
# print('Class Weights --> ', class_weights)

# load validation data
validation_set_dir = args["validationdir"]
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
        validation_set_dir,
        target_size=(image_size, image_size),
        batch_size=validaton_batch_size,
        shuffle=False,
        class_mode=class_mode)
# print("[INFO] Validation set successfully loaded %d images!" % len(validation_generator))

# load testing data
test_set_dir = args["testdir"]
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_set_dir,
        target_size=(image_size, image_size),
        batch_size=testing_batch_size,
        shuffle=False,
        class_mode=class_mode)
# print("[INFO] Testing set successfully loaded %d images!" % len(test_generator))


# num of samples
import os
num_train_samples = sum(len(files) for _, _, files in os.walk(train_set_dir))
print(num_train_samples)

num_val_samples = sum(len(files) for _, _, files in os.walk(validation_set_dir))
print(num_val_samples)

num_test_samples = sum(len(files) for _, _, files in os.walk(test_set_dir))
print(num_test_samples)


# step size
training_steps_per_epoch = round((num_train_samples)/training_batch_size,0)
print("Training steps per epoch = ", training_steps_per_epoch)
validation_steps_per_epoch = round((num_val_samples)/validaton_batch_size,0)
print("Validation steps per epoch = ", validation_steps_per_epoch)
testing_steps = round((num_test_samples)/testing_batch_size,0)
print("Testing steps = ", testing_steps)




# train the neural network
print("[INFO] training network...")

startmillis = int(round(time.time() * 1000))  # get start time

print(train_generator)
H = model.fit_generator(
        train_generator,
        steps_per_epoch=training_steps_per_epoch,
        epochs=num_epochs,
        workers=16,
        use_multiprocessing=True,
 	      max_queue_size=60,
        validation_data=validation_generator,
        validation_steps=validation_steps_per_epoch)

endmillis = int(round(time.time() * 1000))  # get end time

# save the network
model.save(str(args["outputdir"])+"/"+str(args["model_name"])+"_model.h5")
print("Model weights have been saved!")

# ------------------------------------------------------------- #
# ------- EVALUATION AND LOSS/ACC/CONFUSION PLOTS ------------- #
# ------------------------------------------------------------- #

# plot the training loss and accuracy (if --plot output file specified in arguments)
N = np.arange(0, num_epochs)
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
# plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch")
print("Plot of training/validation loss and accuracy saved!")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(str(args["outputdir"])+"/"+str(args["model_name"])+"_trainingplot.png")

# evaluate the network

print("[INFO] evaluating network...")

predictions = model.predict_generator(test_generator, steps=testing_steps)

lb = LabelBinarizer()
y_true = lb.fit_transform(test_generator.classes)

if num_classes == 2:
    target_names = ['animal', 'ghost']
    y_pred = np.array([int(np.round(p)) for p in predictions])
else:
    target_names = test_generator.class_indices
    y_pred=predictions

# to get rid of ValueError: Found input variables with inconsistent numbers of samples: [6379, 6376] or ([13291, 13288]) for batch size 8
if training_batch_size == 8:
    test_generator.classes.resize(13288)
    print('Shape before is ', y_true.shape)
    y_true.resize(13288, 15)
    print('Shape after is ', y_true.shape)
    # for i in range(0,3):
        # test_generator.classes.pop()

predicted_class_indices=np.argmax(predictions,axis=1)
print('y_true --> ', test_generator.classes)
print('y_pred --> ', predicted_class_indices)
matrix = confusion_matrix(test_generator.classes, predicted_class_indices)
report = classification_report(test_generator.classes, predicted_class_indices, target_names=target_names)

# print evaluation results and confusion matrix onto terminal and into a txt file
print('Confusion Matrix')
print(matrix)
print('Classification Report')
print(report)
file = open(str(args["outputdir"])+"/"+str(args["model_name"])+"_report.txt", "w")      # open (or create) .txt file for writing
file.write('Classification Report\n')
file.write(report)                    # output classification report onto the text file
file.write('\n')                      # write new line to file
file.write('Confusion Matrix\n')
file.write(str(matrix))                   # output confusion matrix onto the text file


# ------------------------------------------------------------- #
# ----- Saving History and test results to regenrate plots -----#
# ------------------------------------------------------------- #
# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(H.history) 
hist_csv_file = str(args["outputdir"])+"/"+str(args["model_name"])+'_lr'+str(learning_rate)+'_epochs'+str(num_epochs)+'_batchSize'+str(training_batch_size)+'_history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

predictions_df = pd.DataFrame(y_pred) 
predictions_file =  str(args["outputdir"])+"/"+str(args["model_name"])+'_lr'+str(learning_rate)+'_epochs'+str(num_epochs)+'_batchSize'+str(training_batch_size)+'_predictions.csv'
with open(predictions_file, mode='w') as f:
    predictions_df.to_csv(f)


y_true_df = pd.DataFrame(y_true) 
y_true_file = str(args["outputdir"])+"/"+str(args["model_name"])+'_lr'+str(learning_rate)+'_epochs'+str(num_epochs)+'_batchSize'+str(training_batch_size)+'_y_true.csv'
with open(y_true_file, mode='w') as f:
    y_true_df.to_csv(f)


# ------------------------------------------------------------- #
# ----------------------- ROC CURVE --------------------------- #
# ------------------------------------------------------------- #
print("[INFO] Plotting ROC curve...")

# Plot linewidth.
lw = 2

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

if (num_classes==2):
    plot_range=1
else: 
    plot_range=num_classes

for i in range(plot_range):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), predictions.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(plot_range)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(plot_range):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= num_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
plt.figure(dpi=300)
plt.plot(fpr["micro"], tpr["micro"],
        label='micro-average ROC curve (area = {0:0.2f})'
            ''.format(roc_auc["micro"]),
        color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
        label='macro-average ROC curve (area = {0:0.2f})'
            ''.format(roc_auc["macro"]),
        color='navy', linestyle=':', linewidth=4)

colors = cycle(['dimgray', 'rosybrown', 'lightcoral','brown','peru','darkorange','olivedrab','limegreen','springgreen','turquoise','cyan','deepskyblue','dodgerblue','darkviolet']) 

target_names_roc = list(target_names.keys())

if plot_range==1:
    plt.plot(fpr[0], tpr[0], color=next(colors), lw=lw,
                label='ROC curve of animal class (area = {0:0.2f})'
                ''.format(roc_auc[0]))
else:
    for i, color in zip(range(plot_range), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of {0} class (area = {1:0.2f})'
                ''.format(target_names_roc[i], roc_auc[i]))

#     for i, color in zip(range(plot_range), colors):
#         if plot_range==1:
#             plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#                     label='ROC curve of animal class (area = {1:0.2f})'.format(roc_auc[i]))
#         else:
#             plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#                     label='ROC curve of {0} class (area = {1:0.2f})'.format(target_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curves')
plt.legend(loc="best",fontsize="xx-small")
plt.savefig(str(args["outputdir"])+"/"+str(args["model_name"])+"_roc.png")

# plot zoomed ROC

# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot(fpr["micro"], tpr["micro"],
        label='micro-average ROC curve (area = {0:0.2f})'
            ''.format(roc_auc["micro"]),
        color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
        label='macro-average ROC curve (area = {0:0.2f})'
            ''.format(roc_auc["macro"]),
        color='navy', linestyle=':', linewidth=4)

colors = cycle(['dimgray', 'rosybrown', 'lightcoral','brown','peru','darkorange','olivedrab','limegreen','springgreen','turquoise','cyan','deepskyblue','dodgerblue','darkviolet'])

if plot_range==1:
    plt.plot(fpr[0], tpr[0], color=next(colors), lw=lw,
                label='Area = {0:0.2f})'
                ''.format(roc_auc[0]))
else:
    for i, color in zip(range(plot_range), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of {0} class (area = {1:0.2f})'
                ''.format(target_names_roc[i], roc_auc[i]))

#     for i, color in zip(range(num_classes), colors):
#         plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#                 label='ROC curve of {0} class (area = {1:0.2f})'
#                 ''.format(target_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for InceptionV3_34k')
plt.legend(loc="best",fontsize="xx-small")
plt.savefig(str(args["outputdir"])+"/"+str(args["model_name"])+"_rocz.png")


plt.figure(3)
# precision recall curve
precision = dict()
recall = dict()
pr_auc = dict()

for i in range(plot_range):
    precision[i], recall[i], _ = precision_recall_curve(y_true[:, i],
                                                        predictions[:, i])
    pr_auc[i] = auc(recall[i], precision[i])

colors = cycle(['dimgray', 'rosybrown', 'lightcoral','brown','peru','darkorange','olivedrab','limegreen','springgreen','turquoise','cyan','deepskyblue','dodgerblue','darkviolet'])

if plot_range==1:
    plt.plot(recall[0], precision[0], color=next(colors), lw=lw,
         label='Area = {0:0.2f})'
         ''.format( pr_auc[0]))
else:
    for i, color in zip(range(plot_range), colors):
        plt.plot(recall[i], precision[i], color=color, lw=lw,
         label='PR curve of class {0} (area = {1:0.2f})'
         ''.format(target_names_roc[i], pr_auc[i]))


# plt.plot([1, 0], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])    
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best",fontsize="xx-small")
plt.title("precision vs. recall curve")
plt.savefig(str(args["outputdir"])+"/"+str(args["model_name"])+"_prcurve.png")


# Plot non-normalized confusion matrix
plt.figure(5)

df_cm = pd.DataFrame(matrix,  index = [i for i in target_names],
                columns = [i for i in target_names])
plt.figure(figsize=(10,7))
sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

plt.savefig(str(args["outputdir"])+"/"+str(args["model_name"])+"_matrix.png")

# ------------------------------------------------------------- #
# --------------- Plot Model ------------------ #
# ------------------------------------------------------------- #

from keras.utils import plot_model
plot_model(model, to_file=str(args["outputdir"])+"/"+str(args["model_name"])+"_model.png", show_shapes=True)
    
# ------------------------------------------------------------- #
# --------------- TIME STATS AND EMAIL NOTIF ------------------ #
# ------------------------------------------------------------- #

current_time = time.ctime()
file.write("Python code of training completed execution on %s " % current_time)
file.close()
# send email notification
body = "Model: " + args["model_name"] + " - Training complete @ " + current_time
to = ['b00067871@aus.edu','b00068908@aus.edu', 'g00071496@aus.edu', 'g00068368@aus.edu']
send_email(body, to)
