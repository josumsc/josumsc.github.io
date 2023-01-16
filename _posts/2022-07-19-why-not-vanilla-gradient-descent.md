---
layout: post
title: "Beyond Gradient Descent Optimizer"
date: 2022-07-19 00:00:00-0100
description: What are those fancy optimizers as ADAM or RMSprop and why should I use them instead of the old trusty SGD?
tags: 
  - deep learning 
  - optimization
giscus_comments: true
---

{% include figure.html path="/assets/img/gradient-descent-01.png" class="img-fluid rounded z-depth-1" %}

In almost every learning resource, books, references, or guides about Machine Learning is based on the optimization of a cost function by a learning algorithm.

On almost every occasion, with the particular exception of the normal equation on linear regression problems, whenever we talk about optimization we are refering to the Gradient Descent algorithm. This algorithm is based on the idea of calculating the partial derivative of the different weights of the model with respect to the cost function in a point, which indicates the adjustment necessary to be performed in that weight to minimize the cost function. As the partial derivative is subject to the point in which the cost function is located, we multiply the value of the cost function by a learning rate to guide how much we will move the value of our weights.

{% include figure.html path="/assets/img/gradient-descent-02.png" class="img-fluid rounded z-depth-1" %}

Thus this learning rate parameter, often called $\alpha$ or simply `lr`, has a huge impact on the results of our model. If we decide to make our learning rate too low, we may need more steps for our model to converge, and if we decide to make it too big, our model may not converge at all.

{% include figure.html path="/assets/img/gradient-descent-03.png" class="img-fluid rounded z-depth-1" %}

In simple models, we can allow a low value for this function since the training times are low, but in complicated models such as deep learning where the values to update are of order hundreds or thousands, we need alternatives:

## Stochastic Gradient Descent

The very same word *stochastic* refers to the randomness of this method, which is based on selecting randomly K examples from our training set and computing the cost with them instead of with the total set, which allow us to perform updates in the weights as soon as possible and accelerate the training process. The problem of this method is its randomness, because if we had taken the pure gradient descent approach we would have taken into account all the inputs of data, but instead we are only considering a small subset of the data and adjusting the weights according to their loss, which could lead to different results and eventually lead to a non-optimal solution.

## Mini-batch Gradient Descent

The solution to the randomness problem on stochastic gradient descent is based on dividing our training set into K sets and adjusting the weights after each set, so that we are sure that our model will have to look at all the examples of the training set on each iteration. Each step of the full training set is called an epoch and this is a standard method in the automatic learning models.

{% include figure.html path="/assets/img/gradient-descent-04.png" class="img-fluid rounded z-depth-1" %}

## Momentum Gradient Descent

Nevertheless, despite realizing faster adjustments thanks to the batch division of the dataset, we sometimes see that our cost function is not getting closer to the minimum, because the different sets of data may still give us results contrary to the desired result.

In order to reduce the impact of this problem, the idea behind the momentum algorithm is to give a weight to the previous partial derivatives before performing our adjustment, which allows us to accelerate and smooth the adjustments to be made by our algorithm. The idea to follow is the same as in the smoothing of time series, reducing the impact of individual observations to get a tendency and directionality inside the function.

{% include figure.html path="/assets/img/gradient-descent-05.png" class="img-fluid rounded z-depth-1" %}

## RMSProp

On the next iteration, the *Root Mean Squared Propagation* method appears. It corrects the negative effects of the accumulation of the values passed through the derivatives, by converting them into a moving average adjusted. Furthermore, it allow us to select a different learning rate for each weight, which is a particularly good approach for deep learning models, where we can have thousands or millions of values to optimize.

{% include figure.html path="/assets/img/gradient-descent-06.png" class="img-fluid rounded z-depth-1" %}

## ADAM

Finally, now that we now the basics of the momentum gradient descent and the RMSProp, we can combine the strenghts of both methods to get a more efficient optimizer that reduce the negative counterparts of selecting each of those. With this idea we reached the ADAM algorithm, which uses the quotient of the adjustments made by the two previous methods to get a more adjusted value, adding an epsilon parameter that allows to modify the weight of the momentum trend. In practice, this $/epsilon$ parameter is not modified and is left as it is on the default implementations used in libraries as `Keras` or `PyTorch`.

{% include figure.html path="/assets/img/gradient-descent-07.png" class="img-fluid rounded z-depth-1" %}

## Final remarks

Now that we have seen the different alternatives to vanilla gradient descent, it is easy to understand why we often find ourselves with one of the optimizers seen in this text more and more often. If even though our guide was not exhaustive, and there are still other optimizers to consider (such as Adagrad or Adadelta), in the practice each time more practitioners of automatic learning are enthused by ADAM, RMSProp or SGD, as those 3 are the most often seen in web competitions hosted in sites like [Kaggle](www.kaggle.com).

If even after selecting an optimizer as ADAM or RMSProp you still find yourself with a sluggish model, you can consider the application of other optimization techniques such as Learning Rate Decay, which would allow us to reduce the learning rate according to the progress of the model in order to make large adjustments at the beginning and precise and small adjustments at the end:

{% include figure.html path="/assets/img/gradient-descent-08.png" class="img-fluid rounded z-depth-1" %}

The combination of this technique with the use of a more advanced optimizer like ADAM allows our models and networks to evolve faster towards the desired minimum of our cost function.

I hope the explanation was clear and that your next models will use one of the optimizers explained here. Keep reading, practicing, and experimenting, and come back to share with me your advances!
