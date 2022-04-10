# Problem Statement:
Predicting news category based on headline and the body.

# Dataset:
The dataset for the project was obtained from Kaggle.

[https://www.kaggle.com/datasets/rmisra/news-category-dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)

This dataset contains around 200k news headlines from the year 2012 to 2018 obtained from HuffPost. There are 41 news categories in the datset.
For this project I am going to take a subset of this data with 6 news categories: 'BUSINESS', 'DIVORCE','COMEDY', 'CRIME','SPORTS', 'FOOD & DRINK'.

# Key Metric(KPI) 
### The metric used for assesing the model performance is going to be accuracy.

# Importing Libraries:
![](/images/importing.png)

# Loading the dataset:
![](/images/loading.png)

# Exploratory Data Analysis
![](/images/EDA.png)
# Distribution of categories
![](/images/Dist.png)

### The plot shows that majority of the news articles belong to the category Politics followed by Wellness and Entertainment. The dataset is imbalanced.

# Data cleaning and preprocessing

![](/images/cleaning1.png)
### Using regular expressions and BeautifulSoup to remove characters, digits and urls.
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
### Word2Vec is used for word embeddings that take the semantic meaning of the words into consideration unlike other vectorizers like tfidf and bag of words. It provides a dense vector representation.There are two methods for learning word represiontations: 
### Continuous bag of words: It takes the surrounding words into consideration for predicting the middle word.
### Skipgram method: It takes a context words and predicts the surrounding words.
### I am going to use the skipgram method for this task.
![](/images/gensim_preprocess.png)
![](/images/w2v1.png)
![](/images/w2v2.png)
![](/images/w2v3.png)

# Modeling

## Multinomial Naive Bayes
### It is a probabilistic classifier that is based on Bayes theorem.
### 𝑃(𝑐| 𝑥) = 𝑃(𝑥 | 𝑐) 𝑃(𝑐)/𝑃(𝑥)
### Multinomial naive bayes classifier is suitable for multiclass classification tasks and it works well with high dimensional data.
### Using Randomized Search to finding best parameter

![](/images/mnb.png)

### Model Performance
![](/images/mnb_acc.png)

### Confusion Matrix
![](/images/mnb_cm.png)

## Random Forest Classifier using Word2Vec Embeddings
## Random Forest Model is an ensemble learning model that is constructed using multiple decision trees. It works for multiclass classification. But it doesn't work well with sparse feature vector representions like tfidf. Because of that I am using Word2Vec Embedding.
![](/images/rf.png)

### Performing Hyperparameter tuning.
![](/images/rf_tuned.png)
 
 ### Making Predictions
 ![](/images/rf_preds.png)

### Confusion Matrix
![](/images/rf_cm.png)

##  A simple neural network model
### Since this is a multiclass classification task, A simple neural network model with a softmax classifier will work well for this task.
### A softmax classifier is basically a generalization of logistic regression for multiclass classification. It provides probability labels for each class.
![](/images/nn.png)

### Model performance
![](/images/nn_preds.png)

### Confusion Matrix
![](/images/nn_cm.png)

## Bidirectional LSTM

### Bidirectional LSTM is a popular neural network model used for NLP tasks. It uses sequence information from both directions simultaneously.
![](/images/lstm1.png)
![](/images/lstm2.png)

### Model Performance
![](/images/lstm_preds.png)

### Confusion Matrix
![](/images/lstm_cm.png)

# Conclusion: 
### It would be better to use a simple naive bayes model for this task as it requires low computational power and performs equally well as other models. 
