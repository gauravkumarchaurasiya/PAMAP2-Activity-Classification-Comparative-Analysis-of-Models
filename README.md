# PAMAP2-Activity-Classification-Comparative-Analysis-of-Models
Activity Classification on PAMAP2 dataset





	
Title of Project :             PAMAP2 Activity Classification: Comparative Analysis of Models
Student Name :             GAURAV KUMAR CHAURASIYA
Enrollment Number :    00919011921
Signature:
Email ID :                         Gaurav.919011921@ipu.ac.in
Contact Number :           9873390197
Google Drive Link :           https://colab.research.google.com/drive/1C-kFBWT83YzutdsHAEsbhtesa-v2k85g?usp=sharing
Google Website Link:
 
PAMAP2 Activity Classification: 
Comparative Analysis of Models
Abstract:
This study presents a comparative analysis of various models for activity classification on the PAMAP2 dataset. The PAMAP2 dataset is a widely used dataset for human activity recognition, containing sensor data from wearable devices collected during different activities. The objective of this analysis is to evaluate the performance of different classification models in accurately predicting activities based on sensor data.
In this research, we employ a range of classification models, including decision tree classifier, random forests, support vector machines (SVM), k-nearest neighbors (KNN), logistic regression, naive Bayes, and  gradient boosting.Each model is trained and evaluated on the PAMAP2 dataset using appropriate evaluation metrics such as accuracy, precision, recall, and F1 score.
The comparative analysis aims to identify the most suitable model for activity classification on the PAMAP2 dataset. The evaluation results provide insights into the strengths and weaknesses of each model and their performance in differentiating between activities. Additionally, the analysis explores the impact of feature selection, pre-processing techniques, and parameter tuning on the performance of the models.
The findings of this study can aid researchers and practitioners in choosing an appropriate model for activity classification tasks using the PAMAP2 dataset. The comparative analysis sheds light on the effectiveness of different models and provides valuable guidance for selecting the most accurate and efficient approach for activity recognition in various applications, such as healthcare monitoring, sports performance analysis, and human-computer interaction.

 
Keywords:
PAMAP2 dataset, activity classification, comparative analysis, classification models, evaluation metrics


Introduction
The PAMAP2 (Physical Activity Monitoring and Assessment System) dataset has emerged as a valuable resource for researchers and practitioners in the field of human activity recognition. With the proliferation of wearable sensor technology, the PAMAP2 dataset provides a comprehensive collection of multimodal sensor data. This enables the analysis and classification of various physical activities.

In this report, I present an in-depth investigation into activity classification on the PAMAP2 dataset. This study developed an accurate and robust model capable of identifying and categorizing different activities based on sensor readings. Accurately classifying activities has numerous practical applications, such as fitness tracking, health monitoring, and personalized coaching.

The PAMAP2 dataset consists of data recorded from 9 inertial sensors worn by participants during various physical activities. These activities include walking, running, cycling, ascending and descending stairs, rope jumping, lying down, and vacuuming, among others. Additionally, the dataset provides contextual information, such as heart rate, temperature, and participant ID, which can enhance classification.

To tackle the task of classification, we used machine learning techniques, feature engineering, and ML models. We preprocessed the raw sensor data, handled missing values, extracted relevant features, and constructed a comprehensive feature set to capture the intrinsic characteristics of each activity. Subsequently, we trained and fine-tuned several classification models, evaluating their performance using various metrics and cross-validation strategies.

Accurate activity classification enhances our understanding of human behavior. This enables personalized interventions and tailored recommendations in areas such as healthcare, sports performance, and rehabilitation. By analyzing the patterns and dynamics of different activities, we can gain insights into the physiological and biomechanical aspects associated with each task. This will facilitate the development of targeted interventions and improve well-being.

In this report, I provide a detailed description of our experimental setup, including the preprocessing steps, feature extraction techniques, and the selection of classification algorithms. I present an analysis of different models' performance. Furthermore, we highlight the challenges encountered during the classification process and suggest potential avenues for future research.

Overall, this report aims to contribute to the growing body of knowledge in the field of activity classification using the PAMAP2 dataset. By examining the effectiveness of various machine learning approaches, we seek to advance the development of accurate and practical activity recognition systems that can be applied in real-world settings, ultimately promoting healthier lifestyles and improved quality of life.

Dataset
The PAMAP2 (Physical Activity Monitoring and Assessment System) dataset was developed by a team of researchers at the University of Twente in the Netherlands.
The dataset used is PAMAP2 which is an Activity Monitoring dataset that covers 18 different physical activities which are taken by 9 different subjects (8 men and 1 woman) taken using 3 inertial measurement units and a heart rate monitor.
Inertial measurement units contain an accelerometer, gyroscope, and magnetometer. The accelerometer measures acceleration, while the gyroscope measures angular velocity. Each of these measurements is represented in a three-axis coordinate system.



 
- Sensor position:
1 IMU over the wrist on the dominant arm
1 IMU on the chest
1 IMU on the dominant side's ankle
The data files contain 54 columns: each line consists of a timestamp, an activity label, and 52 attributes of raw sensory data(from sensors devices). 
Data used in this notebook can be found and downloaded from:
https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring
PREPROCESSING
It can be seen that various data is missing and as the readme file comments on, there were some wireless disconnections in data collection therefore the missing data has to be accounted for and made up in a way that our data analysis will not be impacted.
The raw data from the PAMAP 2 dataset, because it is realistic, has imperfections. There are 3 main issues.
•	UNREQUIRED ATTRIBUTES OR ROWS
o	Dropping  “activityID == 0”,  since this is transient period where the subject was not doing any particular activity
o	removal of orientation columns as they are not needed for activity classification
•	MISSING HEART RATES CELLS due to equipment malfunction
o	We are going to focus on heart rate as it is our most precise meter of check for tracking subjects during activities 
o	Heart rate box plot (has outliers)
	 
o	The bar chart shows that Rope Jumping and Running are the most cumbersome activities out of all the activities.
	 
o	Apply mean of different activity to their respective columns 
•	HAVING NULLS VALUES
o	The data set has less missing value than the recorded data. So removing the rows having null values and checking its effect on data. (no big change).
o	Hence dropping rows with null values
o	After this data does contains any missing values.

FEATURES MODELING
Selecting dependent attributes (Target)
Classification is a supervised learning which have input data and target.
As activity recognition we use activityId as Target or dependent attributes which depends on other independent attributes (sensory data).
SPLITING DATA INTO TRAIN AND TEST
As the data uses classification for activity recognition. Model should be train on data and test on other data from training data for avoiding memorization. 
We divided data into 75:25 (75% data for training and 25% for test).
FEATURES SCALING
Standardization can become skewed or biased if the input variable contains outlier values. To overcome this, the median and interquartile range can be used when standardizing numerical input variables.
 The dataset attributes have different units and contains outliers. The formula of RobustScaler is
 		(Xi-Xmedian) / Xiqr, 
so it is not affected by outliers. Since it uses the interquartile range, it absorbs the effects of outliers while scaling.
Hence Standarizing data using robustScaler.
MODELLING 
Applying different multiclass Classifer on training data such as (DecisionTreeClassifier,  GaussianNB, KNeighborsClassifier,LogisticRegression, GradientBoostingClassifier,  AdaBoostClassifier )
And Calculating Accuracy, Precision, Recall and F1-score
This help to identify the model accuracy of models.	
RESULT and CONCLUSION
In our study, we explored the effectiveness of various classification models for activity recognition on the PAMAP2 dataset. We trained and evaluated several models using different machine learning algorithms and techniques. Here are the results obtained from our experimentation:
Classifier: <class 'sklearn.tree._classes.DecisionTreeClassifier'>
Accuracy: 0.9993
Precision: 0.9993
Recall: 0.9993
F1-Score: 0.9993
--------------------------------------------
Classifier: <class 'sklearn.naive_bayes.GaussianNB'>
Accuracy: 0.9277
Precision: 0.9326
Recall: 0.9277
F1-Score: 0.9286
--------------------------------------------
Classifier: <class 'sklearn.neighbors._classification.KNeighborsClassifier'>
Accuracy: 0.9886
Precision: 0.9887
Recall: 0.9886
F1-Score: 0.9886
Classifier: <class 'sklearn.linear_model._logistic.LogisticRegression'>
Accuracy: 0.9304
Precision: 0.9306
Recall: 0.9304
F1-Score: 0.9305
--------------------------------------------
Classifier: <class 'sklearn.svm._classes.SVC'>
Accuracy: 0.9842
Precision: 0.9842
Recall: 0.9842
F1-Score: 0.9842
Overall, our experiments demonstrated that machine learning  models, Decision Tree Classifier, SVM and KNN in activity classification on the PAMAP2 dataset. The DecsionTreeClassifier and SVC model achieved an accuracy of above 98%.
FUTURE WORK
Although this study provides valuable insights into activity classification on the PAMAP2 dataset, there are several avenues for future research that can further enhance the accuracy and applicability of activity recognition systems. Some potential areas for future work include:
1.	Enhanced Feature Engineering: Exploring more advanced feature engineering techniques to capture the subtle nuances of different activities. This could involve extracting higher-level features or using advanced signal processing algorithms to extract more informative features from the sensor data.
2.	Deep Learning Approaches: Investigating the application of deep learning models, such as convolutional neural networks (CNNs) or recurrent neural networks (RNNs), for activity classification on the PAMAP2 dataset. Deep learning models have shown promising results in various domains and could potentially improve the classification accuracy in this context.
In conclusion, this report presents a comparative analysis of different models for activity classification on the PAMAP2 dataset. The study highlights the performance of various classification algorithms and provides insights into their strengths and weaknesses. Through experimentation, we have identified the most accurate models and evaluated their effectiveness in differentiating between activities.

Overall, this report serves as a comprehensive analysis of activity classification on the PAMAP2 dataset, offering valuable insights and recommendations for future research. It is hoped that the findings presented here will contribute to the development of innovative solutions in areas such as healthcare monitoring, sports performance analysis, and human-computer interaction, ultimately improving the quality of life for individuals through accurate activity recognition.
References:
•	IMU Sensor https://builtin.com/internet-things/inertial-measurement-unit
•	Handling Missing Values in Machine Learning: https://towardsdatascience.com/7-ways-to-handle-missing-values-in-machine-learning-1a6326adf79e
•	Data Preprocessing in Machine Learning: https://www.javatpoint.com/data-preprocessing-machine-learning
•	RobustScaler in Scikit-learn: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
•	Supervised Learning in Scikit-learn: https://scikit-learn.org/stable/supervised_learning.html
•	OpenAI ChatGPT: https://openai.com/blog/chatgpt


