# AgeClassification-ForAwesomeness
Spring 2018 - Repository for Team SIM
This project uses faces to determine if the person in the photo is between 0-17 years old, 18 to 60 years old, or over 60 years old. 
Noted below are the requisites to run it, how to run the training, and how to run the testing/predicting of the model. 

## Requisites

|Name         | version        |
|-------------|----------------|
|Tensorflow   |1.5             |
|Keras        |2.1.5           |
|h5py         |2.7.1           |
|Pillow       |5.1.0           |
|Python       |3.6             |

## Installing requisites

```
conda install -c conda-forge tensorflow
```

```
conda install pillow
```

```
conda install keras
```

## Running the code (Accuracy of predictions)

```
set "KERAS_BACKEND=tensorflow"
python AISIMTest.py
```

## Training the model
```
set "KERAS_BACKEND=tensorflow"
python AISIMTrain.py
```




### Notes 
AISIMTest.py is used to test/predict using the model. The AISIMTrain.py builds the model using the images folder. 
There are keras UserWarnings but the code will run and there are futureWarning conversion on float vs np.float.
The code will still run. 

### Team Members
Lauro Perez Jr.
Moyosore Akinrinmade Afolasade
Justin De Luna
Jocelyn Rocha
Shawn Roberts