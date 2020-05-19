# A Comparison of Classification models for Traffic Signs
A Comparison of Classification models for Traffic Signs
Anchal Sojatiya
Abstract—Increase in the density of traffic across the globe has necessitated the need for automated cars and this has become a sought field of research. These are self-driving cars that require a combination of a number of sensors to identify its surroundings and take appropriate action. They are required to identify and choose suitable navigation paths, being mindful of the surroundings and adhere to the traffic signals. For passengers to travel with high confidence in self-driving cars, it is required to have a high accuracy of judgment of the traffic signs and take decisions accordingly. There are a number of classes of traffic signs such as school area, children passing, turn right or left, road narrowing, speed limit, one way, traffic signal, etc. The project focuses on the classification of these traffic signs using algorithm such as CNN, SVM and Random Forest and compare their accuracies to select the best performing algorithm. Advantage of CNN over other algorithms for image classification is discussed, so as to improve the classification accuracy in order to reduce the chances of accidents in self-driving cars.

——————————      ——————————

## INTRODUCTION 
THE German Traffic Sign Benchmark is a multi-class, single-image classification challenge held at the Inter-national Joint Conference on Neural Networks (IJCNN) 2011 and is available on Kaggle: https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign. The dataset contains more than 50000 images of various traffic signals fur-ther divided into 43 different classes. The dataset has been divided into train and test datasets. The aim was to explore and process the dataset to build a Convolu-tion Neural Network (CNN) and classify the images into appropriate categories using the train dataset and check the accuracy of model on test dataset and to com-pare accuracy of CNN to other Machine Learning algo-rithms such as Support Vector Machine (SVM) and Random Forest (RF) to understand the advantages of CNN for image classification over other algorithms. Python libraries like Pandas, NumPy were used to pro-cess the images, Keras, Scikit-learn to fit the models and Matplotlib, Seaborn was used for visualizations.
## EXPLORATORY DATA ANALYSIS
 Data was already split in train and test sets with 39209 and 12630 images in train and test sets respec-tively across 43 different classes. Figure 1 shows distri-bution of these 43 classes across train dataset. For visu-alization purpose, images were scaled to 1x1. Figure 2 shows a 3-D scatterplot of images, with different colors representation different classes. As there are 43 classes it is difficult to understand the plot, however, when ro-tated and observed clusters of different classes can be seen clearly in this plot.
## DATA PROCESSING
Cv2 library in python converts the images to NumPy arrays. Converting images makes it easier to integrate it with libraries such as Scikit-learn and Keras that use NumPy. Images in dataset were of different size, to make integration of algorithm easier, all images were scaled to 32 x 32. Output of Cv2 was in form of a 4-D array, with first dimension representing number of rec-ords, second and third representing height and width of image and fourth representing RGB. In order to fit SVM and RF, the 4-D NumPy array was resized to 2-D and normalized, whereas 4-D data was normalized to fit CNN.
## METHODS

### Support Vector Machine (SVM)
SVMs are essentially binary classifies but can be adopt-ed to fit multiclass problems as well. nuSVC function from scikit-learn library was used to fit the model. Ad-vantage of nuSVC over SVC is that it allows to select a parameter to control the number of support vectors (nu) and uses one-vs-rest (ovr) decision function shape. RBF kernel with gamma=0.00001 and nu=0.05 was used to fit SVM. It approximately requires 32 mins 32 seconds to train this model. Accuracy of model on test set was calculated to be 81.35%. Though it changes with use of different parameters, computational complexity of training SVM is generalized by O (n2 p + n3), where ‘n’ is the number of training samples and ‘p’ is number of features. With 39209 training samples, and 32 x 32 RGB image which equals 3072 features, use of SVM may re-sult in a higher computational cost which is clearly re-flected by the time used to train SVM
Classification report for SVM classifier NuSVC(cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma=1e-05, kernel='rbf',max_iter=-1, nu=0.05, probability=False, random_state=121,shrinking=True,tol=0.001, ver-bose=False):
     class precision recall  f1-score  support

           0       0.83      0.33      0.48        60
           1       0.71      0.88      0.79       720
           2       0.78      0.89      0.83       750
           3       0.64      0.72      0.68       450
           4       0.77      0.77      0.77       660
           5       0.70      0.82      0.75       630
           6       0.77      0.50      0.60       150
           7       0.85      0.74      0.79       450
           8       0.84      0.83      0.84       450
           9       0.92      0.84      0.88       480
          10       0.91      0.94      0.92       660
          11       0.82      0.90      0.86       420
          12       0.93      0.92      0.93       690
          13       0.97      0.98      0.97       720
          14       0.95      0.86      0.90       270
          15       0.88      0.82      0.85       210
          16       0.85      0.95      0.89       150
          17       0.91      0.76      0.83       360
          18       0.83      0.57      0.68       390
          19       0.38      0.50      0.43        60
          20       0.43      0.49      0.46        90
          21       0.61      0.54      0.58        90
          22       0.92      0.92      0.92       120
          23       0.38      0.40      0.39       150
          24       0.58      0.54      0.56        90
          25       0.88      0.78      0.83       480
          26       0.75      0.83      0.79       180
          27       0.49      0.50      0.50        60
          28       0.85      0.57      0.68       150
          29       0.76      0.93      0.84        90
          30       0.62      0.38      0.47       150
          31       0.76      0.83      0.79       270
          32       0.73      0.68      0.71        60
          33       0.90      0.98      0.94       210
          34       0.97      0.97      0.97       120
          35       0.93      0.84      0.88       390
          36       0.99      0.82      0.89       120
          37       0.98      0.68      0.80        60
          38       0.91      0.93      0.92       690
          39       0.98      0.67      0.79        90
          40       0.93      0.48      0.63        90
          41       0.38      0.72      0.50        60
          42       0.68      0.89      0.77        90

    accuracy                           0.81     12630
   macro avg       0.78      0.74      0.75     12630
weighted avg       0.82      0.81      0.81     12630

###  Random Forest
RF is an ensemble method used for both classification and regression problems which uses ‘n’ number of de-cision trees created using bagging technique to sample trees. Each of this tree predicts a class, and the class with most votes is selected as final prediction. Compu-tational complexity of RF is given by O (n2 p ntrees), where ‘n’ is number of training samples, ‘p’ is number of features and ‘ntrees‘ is number of trees. As the number of trees increases, the computational complexity of RF goes on increasing.
RF model was fit with number of trees - 50, 100, 200, 300, 500, where entropy calculates the information gain for each split. Figure 3 shows accuracy of model with time required to train each model. It can be seen that although the time required to train model significantly increases after 300 trees, the accuracy remains almost the same without any significant change. 
Model with 300 trees and accuracy of 77.78% was se-lected, the time required to train this RF model was found to be 13 mins and 13 seconds.
Classification report for RF classifier Random-ForestClassifier(bootstrap=True, class_weight=None, criterion='entropy', max_depth=None, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2,min_weight_fraction_leaf=0.0, n_estimators=300,n_jobs=None,oob_score=False,random_state=121, verbose=0, warm_start=False):
     class  precision  recall  f1-score support

           0       1.00      0.08      0.15        60
           1       0.67      0.84      0.75       720
           2       0.55      0.70      0.61       750
           3       0.60      0.65      0.62       450
           4       0.68      0.77      0.72       660
           5       0.56      0.59      0.57       630
           6       0.68      0.54      0.60       150
           7       0.69      0.74      0.71       450
           8       0.64      0.45      0.52       450
           9       0.95      0.85      0.90       480
          10       0.90      0.96      0.93       660
          11       0.85      0.95      0.89       420
          12       0.97      0.92      0.94       690
          13       0.97      0.99      0.98       720
          14       0.99      1.00      0.99       270
          15       1.00      0.82      0.90       210
          16       0.99      0.95      0.97       150
          17       1.00      0.83      0.91       360
          18       0.70      0.61      0.65       390
          19       0.93      0.42      0.57        60
          20       0.38      0.53      0.45        90
          21       1.00      0.47      0.64        90
          22       0.98      0.77      0.86       120
          23       0.70      0.54      0.61       150
          24       1.00      0.42      0.59        90
          25       0.75      0.94      0.84       480
          26       0.66      0.72      0.69       180
          27       1.00      0.07      0.12        60
          28       0.75      0.71      0.73       150
          29       0.81      0.48      0.60        90
          30       0.52      0.36      0.43       150
          31       0.66      0.74      0.70       270
          32       0.58      0.23      0.33        60
          33       0.95      0.87      0.91       210
          34       0.99      0.96      0.97       120
          35       0.94      0.89      0.91       390
          36       1.00      0.73      0.85       120
          37       1.00      0.87      0.93        60
          38       0.86      0.98      0.92       690
          39       0.98      0.67      0.79        90
          40       0.97      0.93      0.95        90
          41       0.41      0.55      0.47        60
          42       0.71      0.28      0.40        90

    accuracy                           0.78     12630
   macro avg       0.81      0.68      0.71     12630
weighted avg       0.79      0.78      0.77     12630

### Convolutional Neural Network (CNN)
The idea behind CNN algorithm came from visual per-ception of humans in recognizing things. The main ad-vantage of using CNN for image classification is its high accuracy, and it automatically detects important features in data. It reduces the number of parameters in model without losing information and makes feature detection more robust. Python library keras was used for CNN. Training data was split into train and valida-tion sets, with 90% of data in train set and 10% in vali-dation set, and response variable was converted into a categorical variable. Function to build CNN was de-fined with 3 convolution layers, epoch was set to 10, categorical crossentropy loss was used with adadelta optimizer. Adadelta is a more robust extension of Adag-rad that adapts learning rates based on a moving win-dow of gradient updates, instead of accumulating all past gradients. 
Figure 4 shows training and validation error after each epoch. Model can be seen converging at the end of 10 epochs.  Figure 5 shows train, validation loss of CNN. By the end of 10 epochs it can be seen that loss function is tending towards zero. Computational complexity of CNN considerably reduces as the number of features are reduced by algorithm. The most memory is used by fully connected layer. Model was fitted and time re-quired to fit CNN was 10 mins 10 sec with an accuracy of 96.73%

Classification report for CNN classifier <keras.engine.sequential.Sequential object at 0x0000018998CCBA48>:
    class precision  recall  f1-score support

           0       0.98      0.93      0.96        60
           1       0.94      0.99      0.96       720
           2       0.97      0.98      0.98       750
           3       0.99      0.97      0.98       450
           4       0.99      0.98      0.99       660
           5       0.97      0.95      0.96       630
           6       1.00      0.89      0.94       150
           7       0.99      0.97      0.98       450
           8       0.97      0.96      0.97       450
           9       0.98      0.99      0.99       480
          10       1.00      1.00      1.00       660
          11       0.96      0.94      0.95       420
          12       0.98      0.95      0.96       690
          13       0.99      1.00      0.99       720
          14       0.99      1.00      1.00       270
          15       0.94      1.00      0.97       210
          16       0.99      0.99      0.99       150
          17       1.00      0.96      0.98       360
          18       0.99      0.86      0.92       390
          19       1.00      0.95      0.97        60
          20       0.68      0.99      0.81        90
          21       0.91      0.74      0.82        90
          22       0.98      0.99      0.99       120
          23       0.97      1.00      0.98       150
          24       0.78      0.98      0.87        90
          25       0.97      0.96      0.97       480
          26       0.97      0.95      0.96       180
          27       0.56      0.50      0.53        60
          28       0.95      0.98      0.97       150
          29       0.90      0.99      0.94        90
          30       0.85      0.81      0.83       150
          31       0.99      0.98      0.99       270
          32       0.67      1.00      0.81        60
          33       0.99      1.00      0.99       210
          34       0.98      1.00      0.99       120
          35       1.00      1.00      1.00       390
          36       0.99      0.97      0.98       120
          37       0.95      1.00      0.98        60
          38       0.98      1.00      0.99       690
          39       0.99      0.96      0.97        90
          40       0.94      0.98      0.96        90
          41       0.98      0.88      0.93        60
          42       0.99      0.99      0.99        90

    accuracy                           0.97     12630
   macro avg       0.94      0.95      0.95     12630
weighted avg       0.97      0.97      0.97     12630

## RESULTS
Table 1 shows comparison of SVM, RF and CNN in terms of their performance and test accuracy, with CNN model resulting to highest test accuracy and least com-putational cost reflected by time required to train the model. Although time complexity also depends on the memory of system being used, but still CNN comes out to be a clear winner.


## Model	Accura-cy	Time
1	Support Vector Machine	81.3539%	32.32 mins
2	Random Forest	77.7830%	13.13 mins
3	Convolutional Neural Net-work	96.7379%	10.10 mins
Table 1

## DISCUSSIONS
Figure 6 shows a detailed comparison of SVM, RF and CNN with their recall, precision, F-1 and accuracy scores. Recall indicates how many actual positives were detected by the model; it is a ratio of correctly predicted positives to all observations in original class. Precision is ratio of correctly predicted positive to all predicted positives, whereas F-1 score is weighted average of pre-cision and recall. Though the precision score is higher than accuracy in RF model, recall score is low when compared to its accuracy indicating that predicted posi-tive when compared to observation in original class was only about 68.28%, that is the number of true posi-tives predicted by model were low, thus affecting the F-1 score of model as well. Similarly, for SVM, recall, preci-sion and F-1 score were considerably less than accura-cy. Whereas in CNN along with accuracy; recall, preci-sion and F-1 scores are also high, thus supporting our claim of CNN being better performer than SVM and RF

## CONCLUSIONS
For automated cars accuracy of detecting traffic signs accurately is very essential as minute inaccuracy might lead to a fatal accident. Not only CNN performs better in terms of accuracy but also time computational com-plexity of it is less than that of SVM and RF, thus taking least time to train the model. Also, CNN outperforms in terms of recall, precision and F-1 scores, indicating, CNN can be a good choice to classify traffic signs for automated cars.























