# directory
## img
Stored images of deaths of some counties on several days.

## models
Stored RNN models trained with death data directly.

## processed_data
### DTW
Stored the output of DTW.py. Could be loaded as a dataframe.

### kmeans
Using kmeans algorithm to classify counties into 2 clusters. Add a label to each county according to the cluster they belongs to.

### Stationary
Check the stationarity of the time series of each county's death data with ADF test.

Add a label which indicates if the time series is stationary.

Calculate second order difference if the time series is not stationary.

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



