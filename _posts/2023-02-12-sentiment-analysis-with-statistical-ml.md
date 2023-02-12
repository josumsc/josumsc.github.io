---
layout: post
title: "Sentiment Analysis with Statistical Methods"
date: 2023-02-12 00:00:00-0100
description: "Using statistical methods to perform sentiment analysis on a dataset of reviews might be simpler than you think."
tags: 
  - sentiment analysis
  - python
giscus_comments: true
---

{% include figure.html path="/assets/img/sentiment-statistical-01.jpg" class="img-fluid rounded z-depth-1" %}

> [This repository](https://github.com/josumsc/classic-ml-sentiment-analysis/blob/master/src/IMDB_Sentiment_Analysis.ipynb) shows a quick implementation of the concepts in this article.

Given all the fuss about chatbots, large language models (LLMs) and Generative Models, I was one of those excited ML practitioners that decided to give NLP a deeper look than before and see if I could apply all those promising technologies to my career and create something useful.

So I enrolled on the [Deep Learning Specialization by DeepLearning.ai](https://www.deeplearning.ai/courses/natural-language-processing-specialization/) expecting to see how to speak with a machine to make it do my job for me. But, to my surprise, the course didn't just jump straight ahead to those fancy technologies, but instead, it started with the basics: sentiment analysis and statistical learning.

It was a revelation to see these concepts again after my Master's degree. I had forgotten how simple and powerful they are when dealing with day-to-day tasks and to establish baselines that are sometimes really difficult to beat. So I decided to write this article to share my experience and hopefully help someone else to get started with NLP.

## Sentiment Analysis

Sentiment analysis is the task of classifying a text into a positive or negative sentiment. It is a very common task in NLP and it is a good starting point to learn about the field. In teams dealing with Customer Service, for example, sentiment analysis is used to classify customer reviews and complaints into positive or negative. This way, the team can focus on the negative reviews and improve the customer experience. In politics or brand management departments, it can be used to leverage the public opinion about a topic using social media posts.

Sentiment analysis is one of the most basic tasks in NLP, and for that reason it's currently widespread in industry as well as in academia. For these reasons it is a good benchmarking tool to compare different models and techniques. The current state-of-the-art is focused on using LLMs, big models trained using a lot of data and compute power. These models can often be downloaded using public hubs as [HuggingFace](https://huggingface.co/) and require little to none finetuning to be used. However, when the task at hand is too specific that we cannot find an appropriate pre-trained model or the task is too small to justify the use of a big model, we can use statistical methods to perform sentiment analysis.

In this article, we will use the [IMDB dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) from Kaggle, which contains 50,000 movie reviews from IMDB, labeled as positive or negative.

## Statistical Methods

Among the different statistical methods that can be used to perform sentiment analysis, we will focus on the following:

- **Logistic Regression**, a linear model that uses the sigmoid function to fit the parameters to a binary classification. This binary feature can be surpassed using [One-vs-All or similar techniques](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/) in order to fit it to K classes.
- **Naive Bayes**, a probabilistic model that uses Bayes' theorem to calculate the probability of a class given a set of features. It is a very simple model that can be used to perform sentiment analysis in a very fast way.

These methods are very simple and easy to implement, and they can be used as a baseline to compare with more complex models. In addition, they have been proven to be very effective in sentiment analysis, as they were the state-of-the-art alongside SVMs in the early 2000s.

## Data Preparation

In every NLP pipeline there are different steps that need to be performed before feeding the data to the model. This is necessary as our models don't understand the strings that compose our data, but rather numbers. So we need to transform our data into a format that our models can understand. Also, we can use this step to clean our data and remove unnecessary information, such as blank spaces, punctuation or stopwords (as prepositions and very common words).

Apart from the tokenization (convert text to numbers) and the cleansing of noisy characters, we may also stem our words. Stemming is the process of reducing a word to its root form. For example, the words "running", "runs", "ran" and "run" would be reduced to "run". This is useful as it reduces the number of features that our model needs to learn, and thus it also helps to reduce the noise in our data.

{% include figure.html path="https://devopedia.org/images/article/218/8583.1569386710.png" class="img-fluid rounded z-depth-1" %}

In a nutshell, our pipeline will look like this:

1. Lowering the text, so uppercase and lowercase letters are considered the same and we don't have to deal with different representations of the same word.
2. Tokenization, using blank spaces as separators we separate our strings into tokens that we may look up in a learnt mapping of `{string: integer}` often called *vocabulary*.
3. Removing punctuation and stopwords, to reduce the noise.
4. Stemming the words to their root form thanks to the [Porter Stemmer](https://tartarus.org/martin/PorterStemmer/), to reduce the number of features.

```python
def preprocess_text(
    text: str,
    stopwords: list,
) -> str:
    # Lowercase text
    preprocessed_text = text.lower()

    # Tokenize text
    preprocessed_text = word_tokenize(preprocessed_text)

    # Remove punctuation
    preprocessed_text = [w for w in preprocessed_text if w not in string.punctuation]

    # Remove stopwords
    preprocessed_text = [w for w in preprocessed_text if w not in stopwords]

    # Stem text
    stemmer = PorterStemmer()
    preprocessed_text = [stemmer.stem(w) for w in preprocessed_text]

    return ' '.join(preprocessed_text)
```

---

## Modelling

### Logistic Regression

The first step in implementing this method is to create a frequency table of the appearance of each token in the different classes. In this case, we will have two records for each token: one for the positive class and one for the negative class. This table will be used to calculate the probability of a class given a token, that we can later on aggregate to the probability of a class given a set of tokens.

```python
def get_word_dict(result, X, y):
    for label, sentence in zip(y, X):
        for word in word_tokenize(sentence):
            # define the key, which is the word and label tuple
            pair = (word, label)
            
            # if the key exists in the dictionary, increment the count
            if pair in result:
                result[pair] += 1

            # else, if the key is new, add it to the dictionary and set the count to 1
            else:
                result[pair] = 1

    return result
```

After creating the frequency table, we can then transform our sentences into a matrix of $N*K$ dimensions, where $N$ is the number of sentences and $K$ is the number of classes in our problem, which will represent the sum of appearances of each token in each class.

```python
def extract_features(text, freqs, preprocess=False):
    if preprocess:
      word_l = preprocess_text(text, stopwords)
    else:
      word_l = word_tokenize(text)

    x = np.zeros((1, 3))
    #bias term is set to 1
    x[0, 0] = 1 
    
    # loop through each token
    for word in word_l:
        if (word, 1.0) in freqs.keys():
            x[0, 1] += freqs[(word, 1.0)]
        if (word, 0.0) in freqs.keys():
            x[0, 2] += freqs[(word, 0.0)]
    return x


X_train = np.zeros((len(preprocessed_train), 3))
for i, row in enumerate(preprocessed_train.values):
    X_train[i, :]= extract_features(row, freqs)

X_test = np.zeros((len(preprocessed_test), 3))
for i, row in enumerate(preprocessed_test.values):
    X_test[i, :]= extract_features(row, freqs)
```

Now we just have to fit our `LogisticRegression` model, which will learn a bias parameter (depending on the proportion of each class in the train corpus) and the weights of each feature (the sum of appearances of each token in each class). For this model we will use the `sklearn` implementation:

```python
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
```

### Naive Bayes

As a second traditional method, we select [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier). This method is based on the conditional probability of an example to belong to a certain class given the probabilities of each of the tokens, which will be summarized in what's called *loglikelihoods*. To calculate this probability, we will be using the same frequency table that we calculated for the Logistic Regression:

$$ P(W_{pos}) = \frac{freq_{pos} + 1}{N_{pos} + V} $$
$$ P(W_{neg}) = \frac{freq_{neg} + 1}{N_{neg} + V} $$

$$\text{loglikelihood} = \log \left(\frac{P(W_{pos})}{P(W_{neg})} \right)$$

Besides, a bias term is fitted which represents the overall likelihood of an example of a certain class to be draft from the corpus. This bias term is called *logprior*.

$$\text{logprior} = \log (D_{pos}) - \log (D_{neg})$$

Finally, after the calculation of these both terms we just need to add them to form the final probability of an example to belong to a particular class, and then perform a *softmax* operation to retrieve the index with the higher probability. As in this case the problem is binary, we can summarize the equation as:

$$ p = logprior + \sum_i^N (loglikelihood_i)$$

Although there are several implementations of Naive Bayes in Python, given the relative simpleness of this case we can create our own functions in order to get a better grasp of the inner workings of the algorithm.

```python
def train_naive_bayes(freqs, train_x, train_y):
    loglikelihood = {}
    logprior = 0

    vocab = {w[0] for w in freqs.keys()}
    V = len(vocab)

    # calculate N_pos, N_neg, V_pos, V_neg
    N_pos = N_neg = 0
    for pair in freqs.keys():
        # if the label is positive (greater than zero)
        if pair[1] > 0:
            N_pos += freqs[pair]
        # else, the label is negative
        else:
            N_neg += freqs[pair]
    
    D = len(train_x)
    D_pos = train_y[train_y == 1].shape[0]
    D_neg = train_y[train_y == 0].shape[0]

    # Calculate logprior
    logprior = np.log(D_pos) - np.log(D_neg)
    
    for word in vocab:
        # get the positive and negative frequency of the word
        freq_pos = freqs[word, 1.0] if (word, 1.0) in freqs.keys() else 0
        freq_neg = freqs[word, 0.0] if (word, 0.0) in freqs.keys() else 0
        # calculate the probability that each word is positive, and negative
        p_w_pos = (freq_pos + 1)/(N_pos + V)
        p_w_neg = (freq_neg + 1)/(N_neg + V)
        # calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_pos / p_w_neg)

    return logprior, loglikelihood


logprior, loglikelihood = train_naive_bayes(freqs, preprocessed_train, y_train)
```

Once we get our parameters `logprior` and `loglikelihood`, that represent the bias and the weights of this model, we can use them to predict the class of a new example by aggregating the scores of their tokens:

```python
def naive_bayes_predict(text, logprior, loglikelihood):
    word_l = word_tokenize(text)

    # start by adding our logprior
    p = 0
    p += logprior

    # words that are not in vocabulary simply get ignored
    for word in word_l:
        if word in loglikelihood:
            p += loglikelihood[word]
    return p


def test_naive_bayes(test_x, test_y, logprior, loglikelihood, naive_bayes_predict=naive_bayes_predict):
    y_hats = []
    for text in test_x:
        if naive_bayes_predict(text, logprior, loglikelihood) > 0:
            y_hat_i = 1
        else:
            y_hat_i = 0
        y_hats.append(y_hat_i)

    error = np.mean(np.abs(test_y - y_hats))
    accuracy = 1 - error
    return y_hats, accuracy
```

As we can see, this approach is very similar than the one taken in the Logistic Regression, but it is often more efficient in terms of computational cost and in model performance. In fact, in the case of the IMDb dataset, the Naive Bayes model achieves a better accuracy than even a more complex model based on RNNs ([reference](https://github.com/josumsc/classic-ml-sentiment-analysis/blob/master/src/IMDB_Sentiment_Analysis.ipynb)).

## Conclusion

As we just saw, the implementation of a simple model based on the frequency of the tokens in the corpus can be easily explained, implemented and trained. Furthermore, their efficiency allow us to fit these models to a large vocabulary of custom tokens to adapt them to our specific needs without the need of a large amount of data.

We should not be narrow-minded and think that due to the emergence of far more powerful algorithms these models are not useful anymore, as they both serve as a good starting point to understand our datasets and as a baseline to compare the performance of more complex models.
