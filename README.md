# KerasCNN_Automated

This repository contains generalized code for running common CNN models using keras and outputting a range of result metrics.


## Information

The models included in this repository are as follows:

• InceptionV3\
• MobileNetV2\
• ResNet18\
• EfficientNetB1\
• DenseNet121\
• Xception

To train using a predefined split of training-validation-testing, your directory structure should be as follows:

```bash
├── training
│   ├── class1
│   ├── class2
│   ├── ...
│   └── classn
├── validation
│   ├── class1
│   ├── class2
│   ├── ...
│   └── classn
├── testing
│   ├── class1
│   ├── class2
│   ├── ...
│   └── classn
```

To train using a 10-fold approach the directory structure should simply be as follows:
```bash
├── full_dataset
│   ├── class1
│   ├── class2
│   ├── ...
│   └── classn
```

## Usage

The flags that should be set are below:

```bash
-t Path to training data
-v Path to validation data
-x Path to testing data
-m_n Name of model to use
-n_c Number of classes
-batch Batch size
-lr Learning Rate
-epochs Number of Epochs
-d_n Number of dense neurons in penultimate layer
-o Path of directory to output to
```

Sample launch commands can be found in the **multi_commands_list.txt** file

## Citation
```

@Article{computers11010013,
AUTHOR = {Zualkernan, Imran and Dhou, Salam and Judas, Jacky and Sajun, Ali Reza and Gomez, Brylle Ryan and Hussain, Lana Alhaj},
TITLE = {An IoT System Using Deep Learning to Classify Camera Trap Images on the Edge},
JOURNAL = {Computers},
VOLUME = {11},
YEAR = {2022},
NUMBER = {1},
ARTICLE-NUMBER = {13},
URL = {https://www.mdpi.com/2073-431X/11/1/13},
ISSN = {2073-431X},
DOI = {10.3390/computers11010013}
}
```
