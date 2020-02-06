---
title:    
subtitle: Machine Learning Engineer Nanodegree Capstone Project  
date:     Aug 15, 2019
author: Sam Hiatt
bibliography: bibliography.bib
autoSectionLabels: true
numbersections: true
urlcolor: cyan
#secPrefixTemplate: $$p$$&nbsp;$$i$$
listingTitle: Algorithm
lstPrefixTemplate: $$listingTitle$$&nbsp;$$i$$
csl: ieee.csl
#abstract: |
#    There's not much of an abstract here.
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{H}
---


# Project Overview  


## Backround  

Many social animals communicate using vocalizations that can be used to identify their species. The ability to automatically classify audio recordings of animal vocalizations opens up countless opportunities for sound-aware computer applications and could help accelerate studies of these animals. For example, a classifier trained to recognize the call of a specific species of bird could be used to trigger a camera recording, or automatically tag a live audio stream containing avian calls with the species of the bird that made it, producing a time-series record of the presence of this species. 

[Xeno-Canto.org](https://www.xeno-canto.org/)(@xenocanto) is an online community and crowd-sourced Creative Commons database containing audio recordings of avian vocalizations from around the world, indexed by species. It presents a good opportunity for experimentation with machine learning for classification of audio signals. The [Xeno-Canto Avian Vocalizations CA/NV, USA](https://www.kaggle.com/samhiatt/xenocanto-avian-vocalizations-canv-usa)[@xc_ca_nv] dataset was procured for the purpose of jumpstarting exploration into this space. It contains a small subset of the available data, including 30 varying-length audio samples for each of 91 different bird species common in California and Nevada, USA.

Spectrograms (also called sonograms) map audio signals into 2-dimensional frequency-time space, and have long been used for studying animal vocalizations. In the book [Bird Song Research: The Past 100 Years](https://courses.washington.edu/ccab/Baker%20-%20100%20yrs%20of%20birdsong%20research%20-%20BB%202001.pdf)(@baker2001bird) Myron Baker describes how a device called the Sona-Graph™, developed by Kay Electric in 1948, began to be used by ornithologists in the early 1950s and accelerated avian bioacoustical research. The project [DeepSqueak](https://github.com/DrCoffey/DeepSqueak)(@deepsqueak) at the University of Washington in Seattle applies deep learning for classifying audio recordings using spectrograms of ultrasonic vocalizations of mice. Their publication in Nature, [DeepSqueak: a deep learning-based system for detection and analysis of ultrasonic vocalizations](https://www.nature.com/articles/s41386-018-0303-6)(@coffey2019deepsqueak), shows how their classifier was used to study correlations between types of vocalizations and specific behaviors.

[Mel-frequency Cepstral Coefficients (MFCCs)](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) also map audio signals into 2-dimensional space and are commonly used in voice recognition tasks. Inspired by the DeepSqueak's use of spectrograms as inputs to convolutional neural networks, this project uses spectrograms and MFCCs to train a classifier for avian vocalizations.

## Problem Statement

This project defines and trains a digital audio classifier to predict bird species when given an audio sample of avian vocalizations. 

By taking a neural-network approach to  After being trained on a preprocessed dataset of labeled samples, the classifier will be able to read an mp3 file and return a single label representing the common English name of the most prevalent predicted species in the recording.

This effort focuses on just a small subset of the available xeno-canto.org data. While many additional samples are available and could be used to improve the accuracy for any particular species of interest, or more species could be added, such refinement is left for future work.

## Metrics

Performance is evaluated during model selection and training by calculating the [accuracy score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) on a 3-fold cross-validation data split of the training data. Originally a 5-fold cross-validation was planned, but after some experimentation it was determined that using 3 folds was sufficient as results were stable between splits.

Evaluation of accuracy gives equal weight to each class, considering them all equally important to identify, and is simply defined as the portion of samples correctly classified. So, for example, a model that correctly predicts the label (one out of 91 species) half of the time, would have a score of `0.50`. 

Final model performance is evaluated by first training the best model chosen during the model selection phase on the entire training dataset (without cross-validation splits), and final test accuracy is evaluated by predicting labels on the designated test dataset and comparing it to their true values.

# Analysis

## Data Exploration

![Number of Audio Samples per Species](avian-vocalizations-report_files/avian-vocalizations-report_2_1.png){#fig:samples_per_species}

@Fig:samples_per_species shows that the dataset has a balanced distribution in terms of the number of samples per species, with 30 samples for each of 91 species.

The fact that the number of samples per class is balanced is an important consideration as each species should be represented by recordings with a variety of different environmental conditions. If, say, a single sample was chopped up and used as multiple examples for training the model would likely end up overfitting to environmental factors specific to that recording, for example by picking up on the sound of a waterfall in the background instead of listening to the birds. Having a balanced number of samples per class should help regularize environmental factors like these. 

![Total Duration of Audio per Species](avian-vocalizations-report_files/avian-vocalizations-report_4_0.png){#fig:seconds_per_species}

Looking at how the data is distributed in terms of the total duration of audio samples per class shows that each species is not equally represented in the dataset. @Fig:seconds_per_species shows that sample lengths range from 185 seconds to 1877 seconds of audio, with a mean of 80 seconds. This imbalance is due to the process used to compile the list of audio samples for each species. In particular, the 30 _shortest_ samples for each species recorded in California and Nevada were downloaded from xeno-canto.org, with the intention of reducing the load on the servers. This results in a dataset containing shorter samples for species that are more commonly recorded.  This will be an important consideration when evaluating model performance. 

## Exploratory Visualization


![XC119222.mp3: Abert's Towhee (Melozone aberti), contributed by: Ryan P. O'Donnell](avian-vocalizations-report_files/avian-vocalizations-report_7_6.png)



![XC79575.mp3: American Grey Flycatcher (Empidonax wrightii), contributed by: Ryan P. O'Donnell](avian-vocalizations-report_files/avian-vocalizations-report_7_7.png)



![XC79577.mp3: Ash-throated Flycatcher (Myiarchus cinerascens cinerascens), contributed by: Ryan P. O'Donnell](avian-vocalizations-report_files/avian-vocalizations-report_7_8.png)


After visualizing the spectrograms it is clear that their values have an exponential distribution. Scaling them with `np.log()` allows us to visualize the textures of the vocalizations. 

The MFCC values are mostly normally distributed, but it appears the bottom row has values that fall well below all the rest.  

The mean pixel and standard deviation of the spectrogram, log(melspectrogram), and mfcc arays are calculed here. 



    Total number of melspectrogram pixels: 412313856, mean: 0.29080, std. dev: 15.13515
    log(melspectrogram_agg) mean: -7.34926, std. dev: 3.56474


`   Total number of MFCC pixels: 64424040, mean: -19.00330, std. dev: 86.45496`

## Algorithms and Techniques

The log scaled spectrograms produce visualizations with distinctive shapes and textures. The inherent interdependence of pixels that are near each other in the spectrogram makes it an appropriate task for a convolutional neural network as this essentially turns this problem into a classic image classification problem. A similar model to that which was used in the [dog species classifier](https://www.kaggle.com/samhiatt/udacity-dog-project-model-selection-and-tuning) project. This model seems to perform well when classifying images of dogs and using it should test the hypothesis that a CNN will perform better than the benchmark Naive Bayes model, and similarly, it will be trained using gradient descent. 

The MFCC features contain information about the vocal characteristics of the frame. Adding them to the feature space can perhaps help improve predictions. Since they are correlated in time with the spectrogram, a convenience technique is applied to concatenate the two input arrays to produce a single 2-dimensional array for input to the CNN. 

Data augmentation will be employed by using a data generator that crops samples from equal-length windows of input data with a random offset. 

In order to evaluate the performance of models during experimentation, experimental models are trained and evaluated on a 3-fold stratified and shuffled split to help evaluate stability and prevent model over-fitting. Final performance is evaluated by re-training the model on the entire training dataset and then evaluating against the test dataset. 




## Benchmark

A purely random predictor would be correct 1.1% of the time (1/91 classes). A [Gaussian Naive Bayes classifier](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) applied to the spectrogram pixels should perform better than random guessing and is used as a benchmark predictor. It is expected that this predictor will become sensitive to certain frequency bands that are common in a particular species' vocalizations and that this will give it some predictive power. The naive assumption of feature independence is expected to limit this model's performance, but it should still provide a good baseline. 


# Methodology

## Data Preprocessing

The Kaggle kernel [Avian Vocalizations: Data Preprocessing](https://www.kaggle.com/samhiatt/avian-vocalizations-data-preprocessing) documents the data preprocessing methodology used to decode audio input files, generate spectral features, calculate statistics, and then scale and normalize data. The same steps are taken for decoding and visualizing the samples in the Exploratory Visualization section above. 

Mp3s are first decoded and Spectrograms and MFCCs are computed using librosa. The resulting arrays are stored as memory-mapped data files and saved in the Kaggle dataset [Avian Vocalizations: Spectrograms and MFCCs](https://www.kaggle.com/samhiatt/avian-vocalizations-spectrograms-and-mfccs) and used as input for further processing steps. 

Log scaling the spectrograms accounts for their exponential distribution, and normalization zero-centers the data, making it ready for input into a neural network. The dataset [Avian Vocalizations: Melspectrograms Log Norm](https://www.kaggle.com/samhiatt/avian-vocalizations-melspectrograms-log-norm) contains the scaled and normalized spectrograms preprocessed by calculating the `log`, subtracting the mean log scaled pixel (`mean(log(melspectrograms)) = -7.34926`, and dividing by the standard deviation (`std(log(melspectrogram)) = 3.56474`) This preprocessing is documented in the kernel [Fork of Avian Vocalizations: Data Transformation](https://www.kaggle.com/samhiatt/fork-of-avian-vocalizations-data-transformation?scriptVersionId=18761461) Use of this preprocessed data is optional as data scaling and normalization can alternatively be done in the data generation step.

A data generator modeled after [Shervine Amidi's example](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly) is used to read the mem-mapped spectrograms and MFCCs and produce optionally shuffled batches of equal-length normalized samples with one-hot encoded labels. A seed is used for shuffling to allow reproducibility. One-hot encoding the labels enables categorical classification as it removes the ordinality of the encoded labels. Normalization can be done in this step if the data on disk is not already normalized. This is the case with the MFCC data due to storage space errors encountered during Kaggle kernel execution. It's fast enough to do during data generation, so for convenience MFCC normalization is done in this phase. 

The data generator is also responsible for combining the spectrogram and MFCC inputs into a single 2-dimensional array by either concatenating the MFCCs to the top of the spectrograms, or by overwriting the lower frequency bands of the spectrograms with the MFCC data. Both of these approaches for combining the arrays were evaluated for performance.

In order to select a random window of a specified length from the input sample, the data generator randomly selects (again using a seed, for reproducibility) a window offset for each sample. If the input file is shorter than the crop window then the output array is padded with the dataset mean pixel value, or 0 in the case of a normalized dataset. This choice for padding the samples has implications that are discussed in the results section. 

The dataset was first partitioned with [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) reserving 1/3 of the dataset for testing, and again supplying a seed value for reproducibility. This output of this split is saved in the dataset [Avian Vocalizations: Partitioned Data](https://www.kaggle.com/samhiatt/avian-vocalizations-partitioned-data) and used for training / testing in subsequent steps. 

Let's load the partitioned data take a look at some outputs from the generator.



![png](avian-vocalizations-report_files/avian-vocalizations-report_18_0.png)



![png](avian-vocalizations-report_files/avian-vocalizations-report_18_1.png)



![png](avian-vocalizations-report_files/avian-vocalizations-report_18_2.png)


We see that the generator is producing equal-length clips of scaled, zero-centered 2-dimensional data. Notice how the first sample has a recording that is shorter than the clip window length and so it has been padded with zeros. 

Let's take a look to see how data augmentation is functioning. Does it produce different clips from the same sample? Let's call the generator again and compare it to the clips above.



![png](avian-vocalizations-report_files/avian-vocalizations-report_20_0.png)



![png](avian-vocalizations-report_files/avian-vocalizations-report_20_1.png)



![png](avian-vocalizations-report_files/avian-vocalizations-report_20_2.png)


Notice how the samples are shifted along the time axis. So each time the generator is called a new clip is created. This technique should help the model better generalize to new data by making it sensitive to the patterns at whatever time step they occur in the sample. 

## Implementation

The model is implemented using a similar architecture as used in the [dog species classifier](https://www.kaggle.com/samhiatt/udacity-dog-project-model-selection-and-tuning) project. This model contains three stacks of 2-d Convolution and MaxPooling layers followed by a Dropout layer with a rate of `0.2`. Convolutional layers use a ReLU activation function. The Dropout layer masks 20% of the input neurons in each layer and effectively causes the model to develop redundant neural pathways which will help the model generalize better to unseen data. The output of these stacks is fed into a global average pooling layer, followed by a fully-connected layer with a softmax activation function. The position of the maximum value of this output corresponds to the predicted label.



    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 126, 126, 64)      640       
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 42, 42, 64)        0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 42, 42, 64)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 40, 40, 64)        36928     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 13, 13, 64)        0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 13, 13, 64)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 11, 11, 64)        36928     
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 3, 3, 64)          0         
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 3, 3, 64)          0         
    _________________________________________________________________
    global_average_pooling2d_1 ( (None, 64)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 91)                5915      
    =================================================================
    Total params: 80,411
    Trainable params: 80,411
    Non-trainable params: 0
    _________________________________________________________________


A [Stratified Shuffle Split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html) is created from the training data, and then each training split is used to train the network using the data generator implemented above. The [keras Sequence.fit_generator](https://keras.io/models/sequential/#fit_generator) method is used to train and evaluate the model after each training epoch using instances of the generator implemented above. Evaluation is done on a single batch containing all of the validation samples. The validation data generator does not shuffle the data, however it does still augment the data by producing different cropped windows for each epoch. 

Let's try out the pipeline and train the model with cross-validation for just 3 epochs of 3 different splits.


    
    Epoch 00001: val_loss improved from inf to 4.50232, saving model to weights.best.cnn.split00.hdf5
    Split 1: min loss: 4.50232, accuracy at min loss: 0.02198
    Cross Validation Accuracy: mean(val_acc[argmin(val_loss)]): 0.0220


The model is able to retrieve batches from the training generator and evaluate accuracy against the validation data. The same structure above is used to train and evaluate different versions of the model in separate Kaggle kernels. The results of these experiments are reported in the next section.


## Refinement

Several different model configurations were trained and evaluated. It was observed in general that increasing the number of neurons per layer improved accuracy, as was expected.

Frame filtering was implemented following the methodology presented in Edoardo Ferrante's notebook [Extract features with Librosa, predict with NB](https://www.kaggle.com/fleanend/extract-features-with-librosa-predict-with-nb), however after some initial experimentation it didn't seem to improve results. It was apparent that it resulted in many more short samples requiring padding and also removed information related to tempo, distorting many of the distinguishing characteristics of the vocalizations in the spectrograms. So this approach was abandoned.

Models with different kernel and max pooling sizes, including 1-row tall convolutional kernels and MaxPooling layers were tried. The hypothesis was that the pitch of each vocalization is important and that convolution applied only to the time dimension might preserve these frequencies. This approach was tried and evaluated in the notebook [Version 16: CNN Classifier](https://www.kaggle.com/samhiatt/avian-vocalizations-cnn-classifier/output?scriptVersionId=18872310). This model includes 64 filters for each convolutional layer and uses 1x4 convolutional kernels and max pooling sizes of 1x4, 1x3, and 1x2, respectively for each layer. It is evaluated on 3 splits for 100 epochs and achieves a score of: `0.0762`. Not much better than the benchmark. 

In [Version 18](https://www.kaggle.com/samhiatt/avian-vocalizations-cnn-classifier/output?scriptVersionId=18878731) a similar model was evaluated, except with 3x3 convolutional kernels and 2x2 max pooling. It was trained on 3 splits to 100 epochs each and achieved a score of: `0.1238`. 

The model in [Version 17](https://www.kaggle.com/samhiatt/avian-vocalizations-cnn-classifier/output?scriptVersionId=18872556) similarly has 60 filters per layer and uses 3x3 convolutions, but 3x3 max pooling. It achieves a score of: `0.18680`.

All of the versions above use a generator that concatenates the MFCCs to the top of the spectrograms. Another approach was evaluated using an alternative method to combine the input arrays, simply overwriting the lowest 20 frequencies of the spectrograms with the MFCCs. The hypothesis was that the lower frequencies are unimportant for avian vocalization identification, and the results seemed to support this, as the model in [Version 20](https://www.kaggle.com/samhiatt/avian-vocalizations-cnn-classifier?scriptVersionId=18897485) has an identical structure to that in version 17, except that it uses this alternative method of combining inputs. It achieved the top score of: `0.1927`. This model architecture and data generation method was chosen for final evaluation.

Shown below are the learning curves from the output of this training session. The minimum loss is achieved after about 80-100 epochs, and this point is indicated in the plots with a red vertical line. 



# Results

## Model Evaluation and Validation

The model from [CNN Classifier: Version 20](https://www.kaggle.com/samhiatt/avian-vocalizations-cnn-classifier?scriptVersionId=18897485) achieved the best score during cross-validation, and it is trained in the kernel [CNN Classifier - Train & Test](https://www.kaggle.com/samhiatt/avian-vocalizations-cnn-classifier-train-test?scriptVersionId=18943170) on the entire training dataset (without cross-validation). The resulting weights are saved in the dataset [CNN Classifier weights](https://www.kaggle.com/samhiatt/avian-vocalizations-cnn-classifier-weights). Let's load them and test the model.


    Test accuracy score: 0.25385


The final test accuracy actually exceeds the accuracy evaluated during training. This is somewhat surprising; however, considering that the model was trained on the entire training dataset, as opposed to only 1/3 of the training data the cross-validation models saw. Having more training examples is apparently improving the model's predictive power.

To evaluate the sensitivity of the model, let's do some more rounds of testing. Successive batches of test data will be cropped with different windows, so let's see how this stability.


    Epoch 1 test accuracy score: 0.23736
    Epoch 2 test accuracy score: 0.23297
    Epoch 3 test accuracy score: 0.23516
    Mean test score: 0.23516, standard deviation: 0.00179


The model appears to be stable, consistently scoring around `0.24`. 

Let's download a new mp3 and try it out. Let's try a sample of an [Elegant Tern](https://www.xeno-canto.org/449570), contributed by [Richard E. Webster](https://www.xeno-canto.org/contributor/KZYUWIRZVH) which the model has not seen before.




    The vocalization is predicted to be from a Phainopepla



![png](avian-vocalizations-report_files/avian-vocalizations-report_32_1.png)


The Elegant Tern had the highest accuracy in testing so it should get this one right. Unfortunately, although the model did make a prediction, it incorrectly predicted the sample to be a Phainopepla. There's still room for improvement.

## Justification

Using the data generator defined above, the benchmark model is trained and tested in the cells below. 


```python
training_generator = AudioFeatureGenerator(X_train, y_train, batch_size=len(X_train), 
                                           shuffle=True, seed=37, n_frames=128, 
                                           n_classes=n_classes)
scores=[]
nb = GaussianNB()
Xs, ys = training_generator[0] #  batch_size=len(X_test), so just the first batch
Xs = Xs.reshape(Xs.shape[0],Xs.shape[1]*Xs.shape[2])
ys = np.argmax(ys,axis=1)
nb.partial_fit(Xs, ys, classes=range(n_classes))
predictions = nb.predict(Xs) 
training_accuracy = accuracy_score(ys, predictions)
print("Training accuracy of benchmark model: %.5f"%training_accuracy)
```

    Training accuracy of benchmark model: 0.18022



```python
test_generator = AudioFeatureGenerator(X_test, y_test, batch_size=len(X_test),
                                       seed=37, n_frames=128, n_classes=n_classes)
Xs, ys = test_generator[0] # batch_size=len(X_test), so just the first batch
Xs = Xs.reshape(Xs.shape[0],Xs.shape[1]*Xs.shape[2])
ys = np.argmax(ys,axis=1)
predictions = nb.predict(Xs) 
test_accuracy = accuracy_score(ys, predictions)
print("Test accuracy of benchmark model: %.5f"%test_accuracy)
```

    Test accuracy of benchmark model: 0.05275


While the benchmark model achieves a training score of `0.18022`, when evaluated aginst the test dataset its accuracy only reaches `0.05275`. The test accuracy of the CNN-based model was `0.23663`, outperforming the benchmark model by a factor of `4.5 X`. 

Let's see a breakdown of how the predictor fares for each species by plotting a confusion matrix. 


```python
# Draw a confusion matrix
conf_matrix = confusion_matrix(y_true, y_predicted, labels=range(n_classes))
plt.figure(figsize=(20,20))
plt.imshow(conf_matrix)
plt.xticks(range(n_classes), label_encoder.classes_, rotation='vertical')
plt.xlabel("true label")
plt.yticks(range(n_classes), label_encoder.classes_)
plt.xlabel("predicted label")
plt.colorbar(shrink=.25);
```


![png](avian-vocalizations-report_files/avian-vocalizations-report_38_0.png)


Visualizing the confusion matrix shows that the accurate predictions along the diagonal are starting to line up. 

Recall that an artifact of the data collection process used to create the original dataset was that species with the most samples available actually end up having shorter samples in the dataset. It is possible that the classifier is somehow picking up on this clue. Knowing that padded clips likely came from one of these classes with shorter samples is a big clue to the classifier, one it won't have when being tested in the wild. This would be a form of data leakage. 

To see if there is a correlation between the total duration of audio per class and class accuracy, we can take a look at a scatter plot. 



![png](avian-vocalizations-report_files/avian-vocalizations-report_40_0.png)



![png](avian-vocalizations-report_files/avian-vocalizations-report_40_1.png)



![png](avian-vocalizations-report_files/avian-vocalizations-report_40_2.png)


It is evident that there is a negative correlation between the total duration of audio samples and the species class accuracy. The species with shorter samples in the collection end up with greater test accuracy. The `Elegant Tern` species appears to be an outlier. It's scoring 100%. It is also a species that is under-represented in terms of total duration of audio. This is further evidence of data leakage that should be fixed.


# Conclusion


## Reflection

A convolutional neural network was trained to predict bird species heard in input audio using audio samples collected from xeno-canto.org. The samples were transformed into spectrograms and MFCCs and fed into a data generator that creates batches of equal length samples clipped from random windows of input data. Several classifiers and network configurations were evaluated using 3-fold cross-validation. The best performing classifier, as measured by calculating overall prediction accuracy, was selected and trained on the entire training dataset. Final classifier performance was evaluated against the test dataset. 

The naive assumption of feature independence inherent in the benchmark Naive Bayes classifier prevents it from learning the distinguishing patterns present in the spectrograms. The translational invariance of the CNN allows it to learn from these patterns even when they appear in different regions of the input data. This is in line with expected results, and in the end a CNN-based classifier was found to achieve a roughly 3X increase in accuracy over the benchmark Naive Bayes model. 

Initial results are encouraging, but they also uncovered some issues with the dataset collection process and the methodology used by the data generator to standardize input sample lengths. 


## Improvement

Several improvements could be made to increase the accuracy of this classifier. The model architecture could be refined, experimenting with different convolution kernel and pooling sizes or by increasing the network depth and increasing memory requirements. Rigorous hyperparameter tuning could further improve accuracy. 

Transfer learning could be employed. An initial attempt was made in the kernel [Avian Vocalizations: Transfer Learning](https://www.kaggle.com/samhiatt/avian-vocalizations-transfer-learning?scriptVersionId=18845751), but found little success, likely because the spectrograms don't resemble any of the classes in the pre-trained network.

However, It is anticipated that the best gains would result from addressing the data leakage issue presented by zero-padding missing values in short samples, identified in the reflections above. Modifying the generator to simply loop short clips to fill the window would be a simple technique that could address this. Additionally, balancing the training dataset during the data collection process by setting an appropriate lower limit for each audio clip duration would circumvent the need for padding.  

Additionally, improving the data generator methods with appropriate frame filtering to ensure that each cropped sample contains an identifiable vocalization would likely improve predictor performance. Any frame filtering should ensure that any rhythmic features of vocalizations are not distorted, perhaps by setting a buffer around any retained frame. Exploration into more robust audio filtering and window selection methods would likely be rewarded with additional accuracy gains.

The project DeepSqueak addresses this by using a recurrent convolutional neural network (FasterR-CNN) with object region proposals that identify the locations of the vocalizations in the spectrograms. The xeno-canto.org training data does not have object regions labeled and so it can not be readily used in the same way. However, an effort could be made to create such a training dataset with expert input helping to define the regions containing vocalizations. 

More experimentation could be done with the MFCCs. They could be used independently as inputs and analyzed for their predictive power, and they could be combined with the spectrograms in a lower layer of the network. The decision to stack them onto the spectrograms was made out of convenience as it was compatible with the model from the dog breed classifier project.


# References

