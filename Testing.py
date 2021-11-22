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
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# import tensorflow as tf
# sess=tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

#training
from Models import * 
from send_email import send_email
from keras.models import load_model

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

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--outputdir", required=True)
ap.add_argument("-x", "--testdir", required=True)
ap.add_argument("-w", "--weights", required=True)
ap.add_argument("-n", "--num_classes", required=True)
ap.add_argument("-m", "--model_name", required=True)
ap.add_argument("-b", "--batch_size", required=True)
args = vars(ap.parse_args())

# ------------------------------------------------------------- #
# ------------------ Variables Definition --------------------- #
# ------------------------------------------------------------- #

num_classes=int(args["num_classes"])

if (num_classes==2):
    class_mode='binary'
else:
    class_mode='categorical'

testing_batch_size = int(args["batch_size"])

# ------------------------------------------------------------- #
# -------------------- MODEL IMAGE SIZE ----------------------- #
# ------------------------------------------------------------- #

if (args["model_name"]=="MobileNetV2"): 
    image_size = 224
elif (args["model_name"]=="InceptionV3"): 
    image_size = 299
elif (args["model_name"]=="ResNet18"): 
    image_size = 224
elif (args["model_name"]=="Xception"): 
    image_size = 299
elif (args["model_name"]=="DenseNet121"): 
    image_size = 224
elif (args["model_name"]=="EfficientNetB1"): 
    image_size = 224
else: 
    print("Wrong Model Name!")
    
# ------------------------------------------------------------- #
# -------------------- DATA PREPARATION ----------------------- #
# ------------------------------------------------------------- #

print("[INFO] Loading test images...")

print(class_mode)

# load testing data
test_set_dir = args["testdir"]
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_set_dir,
        target_size=(image_size, image_size),
        batch_size=testing_batch_size,
        shuffle=False,
        class_mode=class_mode)

# num of samples
import os
num_test_samples = sum(len(files) for _, _, files in os.walk(test_set_dir))
print(num_test_samples)

# step size
testing_steps = math.ceil((num_test_samples)/testing_batch_size)
print("Testing steps = ", testing_steps)

# ------------------------------------------------------------- #
# ------------------------- EVALUATION ------------------------ #
# ------------------------------------------------------------- #

print("[INFO] Evaluating model...")
model = load_model(args["weights"])

# from keras.utils import plot_model
# plot_model(model, to_file=str(args["outputdir"])+"/"+str(args["model_name"])+"_model.png", show_shapes=True)

predictions = model.predict_generator(test_generator, steps=testing_steps)

lb = LabelBinarizer()
y_true = lb.fit_transform(test_generator.classes)

if num_classes == 2:
    target_names = ['animal', 'ghost']
    y_pred = np.array([int(np.round(p)) for p in predictions])
else:
    target_names = test_generator.class_indices
    y_pred=predictions

# to get rid of ValueError: Found input variables with inconsistent numbers of samples: [6379, 6376] for batch size 8
if testing_batch_size == 8:
    test_generator.classes.resize(6376)
    # print('Shape before is ', y_true.shape)
    y_true.resize(6376, 14)
    # print('Shape after is ', y_true.shape)
    # for i in range(0,3):
        # test_generator.classes.pop()

print('y_pred before argmax --> ', predictions)
print('y_pred shape before argmax --> ', predictions.shape)
print('y_pred[0] shape before argmax --> ', predictions[0].shape)
predicted_class_indices=np.argmax(predictions,axis=1)
print('y_true --> ', test_generator.classes)
print('y_pred --> ', predicted_class_indices)
print('y_true shape --> ', test_generator.classes.shape)
print('y_pred shape --> ', predicted_class_indices.shape)
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
file.write("Python code of testing completed execution on %s " % current_time)
file.close()
# send email notification
body = "Model: " + args["model_name"] + " - Testing complete @ " + current_time
to = ['b00067871@aus.edu','b00068908@aus.edu', 'g00071496@aus.edu', 'g00068368@aus.edu']
send_email(body, to)
