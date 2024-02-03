# Wine-Quality-Prediction-by-Machine-Learning
<br>
Analyzed the 1599 samples of wine dataset for the quality  of wine. I've used the Supervised machine learning model Random Forest Classifier model to check the wine quality.
<br>
<br>
We are using a wine dataset that contains chemical parameters value & labeled values. Here the label means quality of the wine. We’ll train our ML model with the labeled dataset. This type of Machine learning is known as Supervised learning. Unlabeled data set fall under unsupervised learning.
<br>
<br>
We’ll check the relationship between the chemical parameters & wine quality in data analysis. We also use plots & graphs to get a better idea about the data set through visualization.
<be>
<br>
**Work Flow**
<img width="960" alt="work flow" src="https://github.com/Akshay3190/Wine-Quality-Prediction-by-Machine-Learning/blob/main/Workflow_1706942192275.png">
<br>
<br>
# Libraries we used for the Processing the data-
<br>
<br>
1-Numpy
<br>
2-Pandas
<br>
3-Matplotlib
<br>
4-Seaborn
<br>
5-Scikit learn
<br>
<br>
# Relation between Volatile acidity & Wine quality-
<br>
Volatile acidity is inversely proportional to the wine quality. High volatile acidity results in  low wine quality  & vice versa.
<br>
<br>
# Relation between Citric acid & Wine quality-
<br>
Citric acid is directly proportional to the wine quality. High citric acid contents result in high wine quality & vice versa.
<br>
<br>
There are 2 types of correlation. Positive & Negative.
<br>
<br>
**A positive correlation** is a relationship between two variables that move in tandem—that is, in the same direction. A positive correlation exists when one variable decreases as the other variable decreases, or one variable increases while the other increases.
<br>
<br>
**Negative correlation** is a relationship between two variables in which one variable increases as the other decreases, and vice versa.
<br>
<br>
# Parameters of Heatmap-
<br>
<br>
**Cbar**- colour bar.
<br>
<br>
**Fmt**- format. The fmt argument determines the format of the annotations.
<br>
<br>
**Annotations** are an important way to add additional information to a heatmap, such as the exact values of each data point. Using the annot and fmt arguments in the heatmap () function, we can add annotations to a heatmap. 
<br>
<br>
**Correlation** function finds the correlation values in entire columns.
<br>
<br>
**Annot**-Annot values are nothing but the chemical parameters mentioned in the dataset according to the columns.
<br>
<br>
**annot_kws**= is used to change the text properties, typically the font size. But you also could change the font itself, or the alignment if you'd used multiline text. You can even rotate the text by using the function ‘rotation’.
<br>
<br>
**Cmap**= You can customize the colors in your heatmap with the cmap parameter of the heatmap() function in seaborn. It is also possible to set maximum and minimum values for the colour bar on a seaborn heatmap by giving values to vmax and vmin parameters in the function.
<br>
<br>
The colour bar in the heatmap represents the relation between the parameters. Dark the colour represents the parameters that are directly proportionate & light colour represents the parameters that are inversely proportionate.
<br>
<br>
While dropping a column in data preprocessing you need to mention the axis=1 & while dropping a row you need to mention the axis=o.
<br>
<br>
# Label binarization-
<br>
The Label Binarizer works by converting each label into a binary vector representation. Let's say we have a set of labels: [“red”, “blue”, “green”]. After applying the Label Binarizer, each label is converted into a binary vector where the length is equal to the total number of unique labels.
<br>
Here we’ve binarize the value into 2 categories. Like good & bad according to the values mentioned in the quality column. If the value is above 7 then wine quality  is good & if the value is below 7 then wine quality is bad. So instead of having 6 different values as labeled values of wine quality, we will have 2 values i.e. good & bad. The same we have mentioned in terms of numeric values means 0 will be bad & 1 will be good. To replace these values we need to use Python function “lambda”.
<br>
<br>
**test_size**=0.2_ means 20% of data will be test data.
<br>
<br>
**random_state**= controls randomness of the sample. The model will always produce the same results if it has a definite value of random state and has been given the same hyperparameters and training data. If you don't specify the random_state in the code, then every time you run(execute) your code a new random value is generated and the train and test datasets would have different values each time. However, if a fixed value is assigned like random_state = 0 or 1 or 42 or any other integer then no matter how many times you execute your code the result would be the same i.e., the same values in train and test datasets.
<br>
<br>
**Random Forest Classifier Model-**
<br>
It is an ensemble model, means it uses more than 2-3 models in combination for prediction.
<br>
Ensemble learning is a machine learning technique that enhances accuracy and resilience in forecasting by merging predictions from multiple models. It aims to mitigate errors or biases that may exist in individual models by leveraging the collective intelligence of the ensemble.
<br>
<br>
<img width="960" alt="Random Forest Classifier Model" src="https://github.com/Akshay3190/Wine-Quality-Prediction-by-Machine-Learning/blob/main/Random%20Forest%20Classifier%20model.png">
<br>
<br>
A single decision tree is like checking wine quality  according  to the content of citric acid. If it is high then it will come to one of the branch of tree & if it is low then it will go to another branch. Like good & bad quality branches. Random forest is nothing but the ensemble model of  a combination of more than 2,3 decision trees to get a more accurate prediction. It will check the majority value or mean value of output given by decision trees to  predict the result. 
<br>
<br>
More decision trees then more accurate will be the prediction.
<br>
<br>
**Fit function**= Fit the values in the model-Random Forest Classifier .
<br>
<br>
After checking the accuracy of both the train & test we need to check the implementation of the model with new data.
<br>
<br>
End of the project. Thank You !!!
<br>
<br>
<a href = "https://github.com/Akshay3190">My Project Portfolio </a>

