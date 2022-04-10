# Problem Statement:
Predicting news category based on headline and the body.

# Dataset:
The dataset for the project was obtained from Kaggle.

[https://www.kaggle.com/datasets/rmisra/news-category-dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)

This dataset contains around 200k news headlines from the year 2012 to 2018 obtained from HuffPost. There are 41 news categories in the datset.
For this project I am going to take a subset of this data with 6 news categories: 'BUSINESS', 'DIVORCE','COMEDY', 'CRIME','SPORTS', 'FOOD & DRINK'.

# Key Metric(KPI) 
The metric used for assesing the model performance is going to be accuracy.

# Importing Libraries:
![](/images/importing.png)

# Loading the dataset:
![](/images/loading.png)

# Exploratory Data Analysis
![](/images/EDA.png)
# Distribution of categories
![](/images/Dist.png)
The plot shows that majority of the news articles belong to the category Politics followed by Wellness and Entertainment. The dataset is imbalanced.

# Data cleaning and preprocessing

![](/images/cleaning1.png)
### Using regular expressions and beautifulsoup to remove characters, digits and urls.
![](/images/cleaning2.png)
### Using lemmatization and tokenization 
![](/images/cleaning3.png)

# Encoding Labels
![](/images/encoding_labels.png)

# Splitting data into train and test sets
![](/images/data_split.png)

# Vectorizing text data using tfidf
![](/images/vectorizing.png)

# Using Word2Vec Embedding
![](/images/gensim_preprocess.png)
![](/images/w2v1.png)
![](/images/w2v2.png)
![](/images/w2v3.png)

# Modeling

## Multinomial Naive Bayes
It is a probabilistic classifier that is based on Bayes theorem.
𝑃(𝑐| 𝑥) = 𝑃(𝑥 | 𝑐) 𝑃(𝑐)/𝑃(𝑥)
Multinomial naive bayes classifier is suitable for multiclass classification tasks and it works well with high dimensional data.
Using Randomized Search to finding best parameter

![](/images/mnb.png)

### Model Performance
![](/images/mnb_acc.png)

### Confusion Matrix
![](/images/mnb_cm.png)

## Random Forest Classifier using Word2Vec Embeddings
![](/images/rf.png)

### Performing Hyperparameter tuning.
![](/images/rf_tuned.png)
 
 ### Making Predictions
 ![](/images/rf_preds.png)

### Confusion Matrix
![](/images/rf_cm.png)

##  A simple neural network model
![](/images/nn.png)

### Model performance
![](/images/nn_preds.png)

### Confusion Matrix
![](/images/nn_cm.png)

## Bidirectional LSTM
![](/images/lstm1.png)
![](/images/lstm2.png)

### Model Performance
![](/images/lstm_preds.png)

### Confusion Matrix
![](/images/lstm_cm.png)
