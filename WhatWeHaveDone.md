# directory
## img
Stored images of deaths of some counties on several days.

## models
Stored RNN models trained with death data directly.

## processed_data
### DTW
Stored the output of `DTW.py`. Could be loaded as a dataframe.

### kmeans
Stored the output of `kmeans.py`. Could be loaded as a dataframe.

### Stationary
Stored the output of `StationarityTest.py`. Could be loaded as a dataframe.

# scripts
## DTW
Calculate the distance of time series between every two counties' death, using DTW algorithm.

Do clustering according to DTW distance. And add a label to each county according to the cluster they belongs to.

Create an instance of `DTW`, and call `compare_similarity()` method to calculate the results of DTW. Then call `output()` method to store the result as a pickle file.
```
DTWClass = DTW()
DTWClass.compare_similarity()
DTWClass.output()
```

## kmeans
Using kmeans algorithm to classify counties into 2 clusters. Add a label to each county according to the cluster they belongs to.

Create an instance of `KMeansClassifier` with the number of clusters you want. And call `train()` method to acquire the result.
```
clf = KMeansClassifier(2)
clf.train()
```

## RNN
Basic RNN or LSTM networks. The input is each county's death data.

Recursively predict the death of each county in the future.

Due to the lack of data, we can't achieve a great RNN model. Both the loss and the accuracy is bad at this time.

Create an instance of `RNNClassifier`. And call `train()` method to train the model. Then call `test()` method to predict.
```
clf = RNNClassifier()
clf.train()
clf.test()
```

Future work could be combining RNN with other models. RNN could act as the output part of our final model.

## StationarityTest
Check the stationarity of the time series of each county's death data with ADF test.

Add a label which indicates if the time series is stationary.

Calculate second order difference if the time series is not stationary.

# THOUGHTS
## Method 1
Using exponential function to fit the death data of each county.

Works well！

## Method 2
Using SEIRS model. We focus on finding the best parameters for the model. This could be achieved by searching resources online and experimenting on death data.

Works pretty well! Our first submission.

## Method 3
Using RNN based models. Since the death data of each county is less than 100, it's difficult to train a good model.

Works bad! Could be explored deeply in May!

## Method 4
Using regression models. This is a traditional way to fit time series.

At the beginning, we need to check the stationarity of our time series.

Then we need to modify our data to make them stationary.

Finally, we fit our data with regression models.

Should work well! Let's see! Hopefully will be our second submission!
