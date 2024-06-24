
"""
## Background and Context:

Twitter possesses 330 million monthly active users, which allows businesses to reach a broad population and connect with customers without intermediaries. On the other hand, there’s so much information that it’s difficult for brands to quickly detect negative social mentions that could harm their business.

That's why sentiment analysis/classification, which involves monitoring emotions in conversations on social media platforms, has become a key strategy in social media marketing.


Listening to how customers feel about the product/service on Twitter allows companies to understand their audience, keep on top of what’s being said about their brand and their competitors, and discover new trends in the industry.
"""

"""
## Data Description:

A sentiment analysis job about the problems of each major U.S. airline. Twitter data was scraped from February of 2015 and contributors were asked to first classify positive, negative, and neutral tweets, followed by categorizing negative reasons (such as "late flight" or "rude service").

 

## Dataset:

The dataset has the following columns:

* tweet_id                                                           
* airline_sentiment                                               
* airline_sentiment_confidence                               
* negativereason                                                   
* negativereason_confidence                                    
* airline                                                                    
* airline_sentiment_gold                                              
* name     
* negativereason_gold
* retweet_count
* text
* tweet_coord
* tweet_created
* tweet_location 
* user_timezone
"""

"""
## Supervised models 
### Some popular techniques used for encoding text:
* Bag of Words
* TF-IDF (Term Frequency - Inverse Document Frequency)

## Unsupervised models 
### Some popular techniques used for unsupervised Sentiment Analysis:
* TextBlob
* VADER Sentiment
"""

"""
## Importing the Libraries
"""

from typing import Dict, List
import kfp
from kfp import compiler
from kfp import dsl
import kfp.components as comp
from typing import NamedTuple
import json
from kfp.dsl import Input, InputPath, Output, OutputPath, Dataset, Model, component
from kfp import kubernetes

"""   
## plot the distribution of the class label
"""
def bar_plot(data, feature):
    plot = sns.countplot(x =feature, data = data)
    total = len(data)
    for p in plot.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_y() + p.get_height()
        plot.annotate(percentage, (x, y),ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points")
    plt.show()

    
@dsl.component(packages_to_install=['pandas', 'numpy'])
def read_csv(csv_url:str, data:Output[Dataset]):
    # Loading data into pandas dataframe
    
    import pandas as pd
    import os
    
    cwd = os.getcwd()
    
    print('Current working directory' + cwd)
    
    df = pd.read_csv(csv_url)
    print('After pd.read_csv')
    with open(data.path, "w") as f:
        df.to_csv(f, index=False)

"""
## Exploratory Data Analysis (EDA) and Data Cleanup
"""
@dsl.component(packages_to_install=['pandas', 'numpy', 'nltk', 'wordcloud', 'fsspec'])
def exploratory_data_analysis(dataset: Input[Dataset], dataset_out: Output[Dataset]) -> Dict:
    
    import pandas as pd
    import matplotlib.pyplot as plt

    import nltk                                             # Import Natural Language Tool-Kit.
    
    nltk.download('wordnet')
    nltk.download('stopwords')
    
    from nltk.corpus import stopwords
    
    """   
    ## Calculate the missing zeros
    """
    def missing_zero_values_table(df):
            zero_val = (df == 0.00).astype(int).sum(axis=0)                                            # Nu of zero in each column
            mis_val = df.isnull().sum()                                                                #Missing value in each column
            mis_val_percent = 100 * df.isnull().sum() / len(df)                                        #Missing value percent accross column
            mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)                         #Concatenation of above aoutput
            mz_table = mz_table.rename(
            columns = {0 : 'Zero Values', 1 : 'Missing Values', 2 : '% of Total Values'})               #Renaming of each coumn
            mz_table['Total Zero Missing Values'] = mz_table['Zero Values'] + mz_table['Missing Values']  #column having total of zero value and missing values
            mz_table['% Total Zero Missing Values'] = 100 * mz_table['Total Zero Missing Values'] / len(df) # Column having percentage of totalof zero and missing value
            mz_table['Data Type'] = df.dtypes
            mz_table = mz_table[
                mz_table.iloc[:,1] != 0].sort_values(                                   #Selecting and sorting those column which have at not a zero value in % of Total Values column
            '% of Total Values', ascending=False).round(1)
            print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      
                "There are " + str(mz_table.shape[0]) +
                  " columns that have missing values.")
            return mz_table

    
    with open(dataset.path) as f:
        data = pd.read_csv(f)
        
    #data = pd.read_csv(data_csv)
    
    data.shape

    data.info()

    data.isnull().sum(axis=0)          # Check for NULL values.
    
    print('exploratory_data_analysis::After checking null values')

    missing_zero_values_table(data)
    
    print('exploratory_data_analysis::After checking missing zero values')

    """
    ## Drop the columns that have high percentage of missing values
    """

    # Drop the column which have got too many missing values or does not relevent information
    data.drop(['negativereason_gold', 'tweet_id','airline_sentiment_gold','tweet_coord'],axis=1,inplace=True)
    data.shape

    """
    ## Convert the date field into multiple columns for better analysis
    """

    year=[]
    month=[]
    date=[]
    hour=[]
    for x in data['tweet_created']:
        year.append(int(x.split("-")[0]))                                          #extraction of year from date column and appending into list
        month.append(int(x.split("-")[1]))                                         #extraction of month from date column and appending into list
        date.append(int(x.split("-")[2].split(" ")[0]))                            #extraction of date of the month from date column and appending into list
        hour.append(int(x.split("-")[2].split(" ")[1].split(":")[0]))              #extraction of hour of that day from date column and appending into list

    data['year']=year
    data['month']=month
    data['dates']=date
    data['hour']=hour
    data.head()
    
    print('exploratory_data_analysis::After converting date field into multiple columns')

    """
    ## Cleanup the rows that are having missing values 
    """

    data["negativereason_confidence"] = data["negativereason_confidence"].transform(
        lambda x: x.fillna(x.median())
    )
    data["negativereason"].fillna("Can't Tell", inplace=True)
    data["tweet_location"].fillna("USA", inplace=True)
    data = data[data["user_timezone"].notna()]


    data["airline_sentiment"] = data["airline_sentiment"].replace("neutral", 0)
    data["airline_sentiment"] = data["airline_sentiment"].replace("positive", 1)
    data["airline_sentiment"] = data["airline_sentiment"].replace("negative", 2)
    
    print('exploratory_data_analysis::After cleaning up the rows having missing values')


    """
    ## Word Cloud for Negative Reviews
    """

    from wordcloud import WordCloud,STOPWORDS

    #creating word cloud for negative reviews
    negative_reviews=data[data['airline_sentiment']==2]
    negative_words = ' '.join(negative_reviews['text'])
    negative_cleaned_word = " ".join([word for word in negative_words.split()])

    negative_wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='black',
                          width=3000,
                          height=2500
                         ).generate(negative_cleaned_word)

    plt.figure(1,figsize=(12, 12))
    plt.imshow(negative_wordcloud)
    plt.axis('off')
    plt.show()
    
    print('exploratory_data_analysis::After plotting the number of negative reviews')

    """
    ## Word Cloud for Positive Reviews
    """

    #creating word cloud for positive reviews
    positive_reviews=data[data['airline_sentiment']==1]
    positive_words = ' '.join(positive_reviews['text'])
    positive_cleaned_word = " ".join([word for word in positive_words.split()])

    positive_wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='black',
                          width=3000,
                          height=2500
                         ).generate(positive_cleaned_word)

    plt.figure(1,figsize=(12, 12))
    plt.imshow(positive_wordcloud)
    plt.axis('off')
    plt.show()
    
    print('exploratory_data_analysis::After plotting the number of positive reviews')

    """
    ## Word Cloud for Neutral Reviews
    """

    #creating word cloud for neutral reviews
    neutral_reviews=data[data['airline_sentiment']==0]
    neutral_words = ' '.join(neutral_reviews['text'])
    neutral_cleaned_word = " ".join([word for word in neutral_words.split()])

    neutral_wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='black',
                          width=3000,
                          height=2500
                         ).generate(neutral_cleaned_word)

    plt.figure(1,figsize=(12, 12))
    plt.imshow(neutral_wordcloud)
    plt.axis('off')
    plt.show()
    
    print('exploratory_data_analysis::After plotting the number of neutral reviews')

    data['airline_sentiment'].unique()         #check the labels

    data['airline_sentiment'].value_counts()    
    
    print('exploratory_data_analysis::After checking unique values of airline_sentiment')

    metrics = {
      'metrics': [{
          'name': 'Positive Wordcloud',
          'value':  positive_wordcloud.to_array().size,
        }, {
          'name': 'Negative Wordcloud',
          'value':  negative_wordcloud.to_array().size,
        }, {
          'name': 'Neutral Wordcloud',
          'value':  neutral_wordcloud.to_array().size,
        }]}
    
    print('exploratory_data_analysis::After setting metrics field')
        
    #with open(dataset_out.path, 'w') as f:
    #    f.write(data)
        
    with open(dataset_out.path, "w") as f:
        data.to_csv(f, index=False)

    return metrics

"""
## Data Preprocessing
"""
@dsl.component(packages_to_install=['pandas', 'numpy', 'nltk', 'wordcloud', 'contractions', 'beautifulsoup4'])
def pre_process_data(dataset: Dataset, dataset_out: Output[Dataset]):
    
    import pandas as pd
    import contractions                                     # Import contractions library.
    import nltk                                             # Import Natural Language Tool-Kit.
    import re, string, unicodedata                          # Import Regex, string and unicodedata.
    
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('punkt')

    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer                 # Stemmer
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem.wordnet import WordNetLemmatizer         # Import Lemmatizer.
    from bs4 import BeautifulSoup                           # Import BeautifulSoup.

    with open(dataset.path) as f:
        data = pd.read_csv(f)
    
    """
    * Remove html tags
    * Replace contractions in string. (e.g. replace I'm --> I am)
    * Remove numbers
    * Remove non-ascii
    * Tokenization
    * Remove Stopwords
    * Lemmatized data
    * We have used the NLTK library to tokenize words, remove stopwords and lemmatize the remaining words
    """

    #remove the html tags
    def strip_html(text):
        soup = BeautifulSoup(text, "html.parser")                    
        return soup.get_text()

    #expand the contractions
    def replace_contractions(text):
        """Replace contractions in string of text"""
        return contractions.fix(text)

    #remove the numericals present in the text
    def remove_numbers(text):
      text = re.sub(r'\d+', '', text)
      return text

    def clean_text(text):
        text = strip_html(text)
        text = replace_contractions(text)
        text = remove_numbers(text)
        return text

    data['text'] = data['text'].apply(lambda x: clean_text(x))
    data.head()

    """
    ### Tokenization
    """

    data['text'] = data.apply(lambda row: nltk.word_tokenize(row['text']), axis=1) # Tokenization of data
    data.head()
    
    print('pre_process_data::After Tokenization')

    """
    ## Lemmatization
    """
    print(stopwords)
    stopwords = stopwords.words('english')
    stopwords = list(set(stopwords)) 
    lemmatizer = WordNetLemmatizer()

    #remove the non-ASCII characters
    def remove_non_ascii(words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words

    # convert all characters to lowercase
    def to_lowercase(words):
        """Convert all characters to lowercase from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words

    # Remove the punctuations
    def remove_punctuation(words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words

    # Remove the stop words
    def remove_stopwords(words):
        """Remove stop words from list of tokenized words"""
        new_words = []
        for word in words:
            if word not in stopwords:
                new_words.append(word)
        return new_words

    # lemmatize the words
    def lemmatize_list(words):
        new_words = []
        for word in words:
          new_words.append(lemmatizer.lemmatize(word, pos='v'))
        return new_words

    def normalize(words):
        words = remove_non_ascii(words)
        words = to_lowercase(words)
        words = remove_punctuation(words)
        words = remove_stopwords(words)
        words = lemmatize_list(words)
        return ' '.join(words)

    #data['review'] = data['review'].astype(str)
    data['text'] = data.apply(lambda row: normalize(row['text']), axis=1)
    
    print('pre_process_data::After Lemmatization')

    data.head()
    
    #with open(dataset_out.path, 'w') as f:
    #    f.write(data)

    with open(dataset_out.path, "w") as f:
        data.to_csv(f, index=False)


"""
### Bag of words (CountVectorizer)
"""
@dsl.component(packages_to_install=['pandas', 'numpy', 'nltk', 'wordcloud', 'scikit-learn', 'seaborn', 'matplotlib', 'joblib'])
def bag_of_words(dataset: Dataset, clf_bow_model: Output[Model], ytest_bow: Output[Dataset], count_vectorizer_pred: Output[Dataset]):

    import pandas as pd
    import numpy as np
    import joblib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Vectorization (Convert text data to numbers).
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from wordcloud import WordCloud
    from sklearn.ensemble import RandomForestClassifier       # Import Random forest Classifier
    from sklearn.metrics import classification_report         # Import Classification report
    from sklearn.model_selection import cross_val_score  
    from sklearn.metrics import accuracy_score
    
    with open(dataset.path) as f:
        data = pd.read_csv(f)
    
    global plt
    

    Count_vec = CountVectorizer(max_features=500)                # Keep only 500 features as number of features will increase the processing time.
    data_features = Count_vec.fit_transform(data['text'])
    
    print('bag_of_words::After CountVectorizer.fit_transform') 
          
    data_features = data_features.toarray()    

    data_features.shape

    X = data_features

    y = data.airline_sentiment

    # Split data into training and testing set.


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    
    print('bag_of_words::After train_test_split')

    # Finding optimal number of base learners using k-fold CV ->
    base_ln = np.arange(100,400,100).tolist()
    base_ln

    # K-Fold Cross - validation .
    cv_scores = []
    for b in base_ln:
        clf = RandomForestClassifier(n_estimators = b)
        scores = cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'accuracy')
        cv_scores.append(scores.mean())
        
    print('bag_of_words::After K-Fold cross validation')

    # plotting the error as k increases
    error = [1 - x for x in cv_scores]                                 #error corresponds to each nu of estimator
    optimal_learners = base_ln[error.index(min(error))]                #Selection of optimal nu of n_estimator corresponds to minimum error.
    plt.plot(base_ln, error)                                           #Plot between each nu of estimator and misclassification error
    xy = (optimal_learners, min(error))
    plt.annotate('(%s, %s)' % xy, xy = xy, textcoords='data')
    plt.xlabel("Number of base learners")
    plt.ylabel("Misclassification Error")
    plt.show()
    
    print('bag_of_words::After Plotting the error as K increases')

    # Training the best model and calculating accuracy on test data .
    clf = RandomForestClassifier(n_estimators = optimal_learners)
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    count_vectorizer_predicted = clf.predict(X_test)
    print(classification_report(y_test , count_vectorizer_predicted , target_names = ['neutral', 'positive', 'negative']))
    print("Accuracy of the model is : ",accuracy_score(y_test, count_vectorizer_predicted))
    
    print('bag_of_words::After training the RandomForestClassifier model')

    # Print and plot Confusion matirx to get an idea of how the distribution of the prediction is, among all the classes.


    conf_mat = confusion_matrix(y_test, count_vectorizer_predicted)

    print(conf_mat)
    
    print('bag_of_words::After printing the confusion matrix')

    df_cm = pd.DataFrame(conf_mat, index = [i for i in ['neutral', 'positive', 'negative']],
                      columns = [i for i in ['neutral', 'positive', 'negative']])
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True, fmt='g')

    all_features = Count_vec.get_feature_names()              #Instantiate the feature from the vectorizer
    top_features=''                                            # Addition of top 40 feature into top_feature after training the model
    feat=clf.feature_importances_
    features=np.argsort(feat)[::-1]
    for i in features[0:40]:
        top_features+=all_features[i]
        top_features+=','

    print(top_features)  
    
    print('bag_of_words::After Identifying the top features')

    print(" ") 
    print(" ")     

    wordcloud = WordCloud(background_color="white",colormap='viridis',width=2000, 
                              height=1000).generate(top_features)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.figure(1, figsize=(14, 11), frameon='equal')
    plt.title('Top 40 features WordCloud', fontsize=20)
    plt.axis("off")
    plt.show()
    
    print('bag_of_words::After Plotting the top features')
    
    #clf.save(clf_bow_model.path)
    
    # save
    joblib.dump(clf, clf_bow_model.path, compress=1)
    print(f"Compressed Random Forest: {np.round(os.path.getsize(clf_bow_model.path) / 1024 / 1024, 2) } MB")
    
    clf_bow_model.metadata['accuracy'] = accuracy_score(y_test, count_vectorizer_predicted)
    
    #with open(ytest_bow.path, 'w') as f:
    #    f.write(str(y_test))
        
    with open(ytest_bow.path, "w") as f:
        y_test.to_csv(f, index=False)
    
    with open(count_vectorizer_pred.path, 'w') as f:
        pd.DataFrame(count_vectorizer_predicted).to_csv(f, index=False)

"""
## TF-IDF - Term Frequency - Inverse Document Frequency
"""
@dsl.component(packages_to_install=['pandas', 'numpy', 'nltk', 'wordcloud', 'scikit-learn', 'seaborn', 'matplotlib', 'joblib'])
def tf_idf(dataset: Dataset, clf_tf_model: Output[Model], ytest_tf: Output[Dataset], tf_idf_pred: Output[Dataset]):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import joblib
    
    from wordcloud import WordCloud
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from wordcloud import WordCloud
    from sklearn.ensemble import RandomForestClassifier       # Import Random forest Classifier
    from sklearn.metrics import classification_report         # Import Classification report
    from sklearn.model_selection import cross_val_score  
    from sklearn.metrics import accuracy_score
    
    # Using TfidfVectorizer to convert text data to numbers.
    from sklearn.feature_extraction.text import TfidfVectorizer          #For TF-IDF
    
    global plt
    
    with open(dataset.path) as f:
        data = pd.read_csv(f)
        
    tfidf_vect = TfidfVectorizer(max_features=500)
    data_features = tfidf_vect.fit_transform(data['text'])
    
    print('tf_idf::After TfidfVectorizer.fit_transform')

    data_features = data_features.toarray()

    data_features.shape     #feature shape

    X = data_features

    y = data.airline_sentiment

    # Split data into training and testing set.

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    
    print('tf_idf::After test_train_split')

    # Finding optimal number of base learners using k-fold CV ->
    base_ln = np.arange(100,400,100).tolist()
    base_ln

    # K-Fold Cross - validation .
    cv_scores = []
    for b in base_ln:
        clf = RandomForestClassifier(n_estimators = b)
        scores = cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'accuracy')
        cv_scores.append(scores.mean())
        
    print('tf_idf::After K-Fold cross validation')

    # plotting the error as k increases
    error = [1 - x for x in cv_scores]                                 #error corresponds to each nu of estimator
    optimal_learners = base_ln[error.index(min(error))]                #Selection of optimal nu of n_estimator corresponds to minimum error.
    plt.plot(base_ln, error)                                           #Plot between each nu of estimator and misclassification error
    xy = (optimal_learners, min(error))
    plt.annotate('(%s, %s)' % xy, xy = xy, textcoords='data')
    plt.xlabel("Number of base learners")
    plt.ylabel("Misclassification Error")
    plt.show()
    
    print('tf_idf::After plotting the error as K increases')

    # Training the best model and calculating accuracy on test data .
    clf = RandomForestClassifier(n_estimators = optimal_learners)
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    tf_idf_predicted = clf.predict(X_test)
    print(classification_report(y_test , tf_idf_predicted , target_names = ['neutral', 'positive', 'negative']))
    print("Accuracy of the model is : ",accuracy_score(y_test, tf_idf_predicted))
    
    print('tf_idf::After training the RandomForestClassifier model')

    # Print and plot Confusion matirx to get an idea of how the distribution of the prediction is, among all the classes

    conf_mat = confusion_matrix(y_test, tf_idf_predicted)

    print(conf_mat)
    
    print('tf_idf::After printing the confusion matrix')

    df_cm = pd.DataFrame(conf_mat, index = [i for i in ['neutral', 'positive', 'negative']],
                      columns = [i for i in ['neutral', 'positive', 'negative']])
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True, fmt='g')

    all_features = tfidf_vect.get_feature_names()              #Instantiate the feature from the vectorizer
    top_features=''                                            # Addition of top 40 feature into top_feature after training the model
    feat=clf.feature_importances_
    features=np.argsort(feat)[::-1]
    for i in features[0:40]:
        top_features+=all_features[i]
        top_features+=', '

    print(top_features)  
    
    print('tf_idf::After Identifying the top features')

    print(" ") 
    print(" ") 

    wordcloud = WordCloud(background_color="white",colormap='viridis',width=2000, 
                              height=1000).generate(top_features)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.figure(1, figsize=(14, 11), frameon='equal')
    plt.title('Top 40 features WordCloud', fontsize=20)
    plt.axis("off")
    plt.show()
    
    print('tf_idf::After plotting the top features')
    
    #clf.save(clf_tf_model.path)
    
    # save
    joblib.dump(clf, clf_tf_model.path, compress=1)
    print(f"Compressed Random Forest: {np.round(os.path.getsize(clf_tf_model.path) / 1024 / 1024, 2) } MB")
    
    clf_tf_model.metadata['accuracy'] = accuracy_score(y_test, tf_idf_predicted)
    
    #with open(ytest_tf.path, 'w') as f:
    #    f.write(str(y_test))
        
    with open(ytest_tf.path, "w") as f:
        y_test.to_csv(f, index=False)
    
    with open(tf_idf_pred.path, 'w') as f:
        pd.DataFrame(tf_idf_predicted).to_csv(f, index=False)
    
"""
## Compare Supervised Learning Methods - CountVectorizer (Bag of Words), TF-IDF methods
"""
@dsl.component(packages_to_install=['pandas', 'numpy', 'nltk', 'wordcloud', 'scikit-learn', 'seaborn', 'matplotlib'])
def compare_supervised_learning(y_test_input: Input[Dataset], count_vectorizer_predicted_input: Input[Dataset], tf_idf_predicted_input: Input[Dataset], bow_score: OutputPath(str), tf_score: OutputPath(str), chosen_model: OutputPath(str)):

    """
    ## Compare Supervised Learning Models
    """
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from wordcloud import WordCloud
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from wordcloud import WordCloud
    from sklearn.ensemble import RandomForestClassifier       # Import Random forest Classifier
    from sklearn.metrics import classification_report         # Import Classification report
    from sklearn.model_selection import cross_val_score  
    from sklearn.metrics import accuracy_score
    
    # Using TfidfVectorizer to convert text data to numbers.
    from sklearn.feature_extraction.text import TfidfVectorizer          #For TF-IDF
    
    global plt
    
    #with open(dataset.path) as f:
    #    data = f.read()

    with open(y_test_input.path) as f:
        y_test = pd.read_csv(f)
        
    with open(count_vectorizer_predicted_input.path) as f:
        count_vectorizer_predicted = pd.read_csv(f)
        
    with open(tf_idf_predicted_input.path) as f:
        tf_idf_predicted = pd.read_csv(f)
    
    #convert the test samples into a dataframe where the columns are
    #the y_test(ground truth labels),tf-idf model predicted labels(tf_idf_predicted),Count Vectorizer model predicted labels(count_vectorizer_predicted)
    df = pd.DataFrame()
    df['y_test'] = y_test
    df['count_vectorizer_predicted'] = count_vectorizer_predicted
    df['tf_idf_predicted'] = tf_idf_predicted
    print(df.head())
    
    print('compare_supervised_learning::After converting the test samples into DataFrame')

    #create bar plot to compare the accuaracies of Count Vectorizer and TF-IDF
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_axes([0,0,1,1])
    subjects = ['Count_Vectorizer', 'TF-IDF']
    
    print('compare_supervised_learning::After creating the bar plot to compare accuracies of CountVectorizer and TF-IDF')

    # calculation accuracies of Count Vectorizer and TF-IDF using accuracy_score metrics
    scores = [accuracy_score(y_test,count_vectorizer_predicted),accuracy_score(y_test,tf_idf_predicted)]
    
    bow_score_val = accuracy_score(y_test,count_vectorizer_predicted)
    tf_score_val = accuracy_score(y_test,tf_idf_predicted)
    
    bow_score_str = str(bow_score_val)
    tf_score_str = str(tf_score_val)
    
    with open(bow_score, 'w') as f:
        f.write('Accuracy score of Count Vectorizer model - ' + bow_score_str)
        
    with open(tf_score, 'w') as f:
        f.write('Accuracy score of TF-IDF model - ' + tf_score_str)
      
    if accuracy_score(y_test, count_vectorizer_predicted) > accuracy_score(y_test, tf_idf_predicted):
        chosen_model_val = 'count_vectorizer_predicted'
        print('Count Vectorizer model has higher Accuracy Score')
    else:
        chosen_model_val = 'tf_idf_predicted'
        print('TF-IDF model has higher Accuracy Score')
  
    with open(chosen_model, 'w') as f:
        f.write('Chosen Model is - ' + chosen_model_val)
    
    ax.bar(subjects,scores)
    ax.set_ylabel('scores',fontsize= 12)    # y axis label
    ax.set_xlabel('models',fontsize= 12)    # x axis label
    ax.set_title('Accuracies of Supervised Learning Methods')  # tittle
    for i, v in enumerate(scores):
        ax.text( i ,v+0.01, '{:.2f}%'.format(100*v), color='black', fontweight='bold')     
        plt.savefig('Accuracies_of_Supervised_learning_models.png',dpi=100, format='png', bbox_inches='tight')
    plt.show()
    
    print('compare_supervised_learning::After plotting the accuracies of supervised learning methods')

"""
## pipeline function
"""
@dsl.pipeline(pipeline_root='', name='twitter_supervised_pipeline', display_name='Twitter Pipeline')
def twitter_supervised_pipeline():
        
    csv_url = "/files/Tweets.csv"
    volume_name = 'twittermgmt-workspace'
    csv_task = read_csv(csv_url=csv_url)
    eda_task = exploratory_data_analysis(dataset=csv_task.outputs['data'])    
    pre_process_task = pre_process_data(dataset=eda_task.outputs['dataset_out'])
    bag_of_words_task = bag_of_words(dataset=pre_process_task.outputs['dataset_out'])
    #dataset: Dataset, clf_bow_model: Output[Model], ytest_bow: Output[Dataset], count_vectorizer_predicted: Output[Dataset]
    tf_idf_task = tf_idf(dataset=pre_process_task.outputs['dataset_out'])
    #dataset: Dataset, clf_tf_model: Output[Model], ytest_tf: Output[Dataset], tf_idf_predicted: Output[Dataset]
    compare_learning_task = compare_supervised_learning(y_test_input=bag_of_words_task.outputs['ytest_bow'], count_vectorizer_predicted_input=bag_of_words_task.outputs['count_vectorizer_pred'], tf_idf_predicted_input=tf_idf_task.outputs['tf_idf_pred'])
    #chosen_model = supervised_learning(dataset=pre_process_task.outputs['dataset_out'])
    #print('chosen_model is - ' + chosen_model) 
    
    kubernetes.mount_pvc(
        csv_task,
        pvc_name='twittermgmt-workspace',
        mount_path='/files',
    )
    
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(twitter_supervised_pipeline, __file__ + '.yaml')
