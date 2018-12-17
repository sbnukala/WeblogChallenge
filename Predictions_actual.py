
#So for this part, I'd initially get inspired from my earlier solution of using Spark datframes and then use Pandas with scikit-learn
#for the predictions
# The analysis is going to get more interesting now. So let's outline the way we'd want to attack any ML problem here:
# 1) Load the data, get the relevant entries from each Web log. For this I used the standard format here: https://www.w3.org/TR/WD-logfile
# and mapped it to the given raw data (~450 MB)
# 2) For each problem at hand
#    a) Get the relevant features needed along with the labels (supervised)
#    b) Normalize or Standardize
#    c) Split the feature+label dataset into train and test sets
#    d) Fit a classifier or regression model (depending on the problem)
#    e) Predict on the test dataset and get the classification metrics. 
#    f) Serialize if needed for future predictions
# We can pretty much use scikit-learn's pipeline for most of the steps above and for that I am more comfortable using Pandas as 
# data input before venturing into scikit-learn. 

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import TimestampType
from pyspark.sql.functions import window, unix_timestamp, col, avg
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.window import Window
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score, median_absolute_error
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVR

#Create a spark session
spark = SparkSession.builder \
    .master("local[4]") \
    .appName("IP Address Sessionize") \
    .config("spark.executor.memory", "2gb") \
    .getOrCreate()
sc = spark.sparkContext

#Now create a resilient distributed dataset from the dataset provided.
rdd = sc.textFile('data/2015_07_22_mktplace_shop_web_log_sample.log')
rdd = rdd.map(lambda line: line.split(" "))
#Get the relevant features needed
df = rdd.map(lambda line: Row(time_stamp=line[0], ip_address=line[2].split(':')[0],url=line[12])).toDF()

df = df.withColumn('timestamp_new_format', df['time_stamp'].cast(TimestampType()))
df = df.drop('time_stamp')
df = df.withColumnRenamed("timestamp_new_format", "time_stamp")
df_original = df
#Irrespective of the IP, let's get a df that has the hits per minute in a given time window of 1 minute
window_df  = df.select(window("time_stamp", "60 seconds").alias('time_window'),"time_stamp","ip_address").groupBy('time_window').count().withColumnRenamed('count', 'hit_per_min')

#Now, lets have another df, where we also see how the hit rates are looking like for a group of both ip_address and time_window. i.e. in a given time window of 60 seconds, how many times has this particular IP been hit?

ip_window_df = df.select(window("time_stamp", "60 seconds").alias('time_window'),'time_stamp',"ip_address").groupBy('time_window','ip_address').count().withColumnRenamed('count', 'ip_address_hit_per_min')

#With this https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b as inspiration, 
# to simplify things, now we know we have a dataframe with just a 60 second time window as one column and it's hit count as another column. I've basically taken the general statistics that one can do with numerical data
# like this (also inspired from pandas df.describe() method equivalent in spark!) and transformed into one "feature" dataframe like this: 
#PART 1: 

standard_deviation_df = ip_window_df.groupBy("time_window").agg({'ip_address_hit_per_min':'stddev'}).alias("ip_address_hit_standard_deviation")
standard_deviation_df = standard_deviation_df.withColumnRenamed("stddev(ip_address_hit_per_min)", "ip_address_hit_standard_deviation")
mean_df = ip_window_df.groupBy("time_window").agg({'ip_address_hit_per_min': 'mean'}).alias("ip_address_hit_mean")
mean_df = mean_df.withColumnRenamed("avg(ip_address_hit_per_min)", "ip_address_hit_mean")
max_df = ip_window_df.groupBy("time_window").agg({'ip_address_hit_per_min': 'max'}).alias("ip_address_hit_max_value")
max_df = max_df.withColumnRenamed("max(ip_address_hit_per_min)", "ip_address_hit_max_value")
#we already have our window_df computed above that acts as our "count" dataframe of all the hits for each time window.

#Lets combine everything into one feature dataframe
predict_hit_features_df = standard_deviation_df.join(mean_df,["time_window"])
predict_hit_features_df = predict_hit_features_df.join(max_df,["time_window"])
predict_hit_features_df = predict_hit_features_df.join(window_df,["time_window"])
predict_hit_features_df = predict_hit_features_df.orderBy('time_window', ascending=True)
predict_hit_features_df = predict_hit_features_df.withColumn("tagId", monotonically_increasing_id())

print("After cleaning and manipulations the DF looks like this for Part 1:")
predict_hit_features_df.show(20, False)

#Get the load in the next minute using the window function
window_load = Window.orderBy('tagId').rowsBetween(1,1) 
avg_hit = avg(predict_hit_features_df['hit_per_min']).over(window_load)
df_with_next_min_load = predict_hit_features_df.select(predict_hit_features_df['ip_address_hit_standard_deviation'],predict_hit_features_df['ip_address_hit_mean'],predict_hit_features_df['ip_address_hit_max_value'],predict_hit_features_df['hit_per_min'],avg_hit.alias("load_next_min"))

print("Further wrangling the data to make sure we have labels(load) and numerical features, DF looks like:")
df_with_next_min_load.show(20, False)

#Let's move to Pandas now
df_with_next_min_load_pandas = df_with_next_min_load.toPandas()

df = df_with_next_min_load_pandas
df = df.dropna()
features= [c for c in df.columns.values if c  not in ['load_next_min']]
target = 'load_next_min'

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.20, random_state=42)
   
param_grid = {
    'clf__n_estimators': [10, 20, 50, 100, 200],
    'clf__bootstrap': [True, False],
    #'clf__max_depth': ["auto", "sqrt", "log2"],
}

rf_regression_pipe = Pipeline([
        ('scale', StandardScaler()),
        ('clf', RandomForestRegressor(random_state=42))])
print("Training a Random Forest Regressor with Gridsearch..")
gs_rf = GridSearchCV(estimator=rf_regression_pipe, param_grid=param_grid,cv = 3, n_jobs = -1, verbose = 2 )
gs_rf.fit(X_train, y_train)
y_pred = gs_rf.predict(X_test)
print(f"Best obtained parameters for Part 1 were:{gs_rf.best_params_} with score {gs_rf.best_score_}")
print(f"Only regression metric taken was the variance score: {explained_variance_score(y_test, y_pred, multioutput='raw_values')}")


#Part 2: 

session_df = ip_window_df
session_df = session_df.withColumn("session_id", monotonically_increasing_id())
df_original = df_original.select(window("time_stamp", "60 seconds").alias('time_window'),"time_stamp","ip_address","url")
session_df = df_original.join(session_df,['time_window','ip_address'])

first_visit_time_stamp = session_df.groupBy("session_id").agg({'time_stamp':'min'}).alias('first_visit_time_stamp')
session_df = first_visit_time_stamp.join(session_df,['session_id'])
session_df = session_df.withColumnRenamed("min(time_stamp)", "first_visit_time_stamp")

#Get the timestamp differences between the current time stamp and the first time a site was visited by this IP
timeDiff = (unix_timestamp(session_df.time_stamp)-unix_timestamp(session_df.first_visit_time_stamp))
session_df = session_df.withColumn("time_difference", timeDiff)
max_time_difference = session_df.groupBy("session_id").agg({'time_difference': 'max'})
session_df = session_df.join(max_time_difference,['session_id'])
session_df = session_df.withColumnRenamed("max(time_difference)", "session_length")

session_length_df = session_df.select('ip_address', 'session_length')

print("Alright, so for part 2, we have this dataframe that has the session length and IP address")
session_length_df.show(20, False)

#Let's move to Pandas now
session_length_df_pandas = session_length_df.toPandas()

df_session = session_length_df_pandas
df_session = df_session.dropna()
#NOTE: Keeping things simple here, otherwise we can extract a lot more info from IP addresses like lat and longitude as well..

X_train, X_test, y_train, y_test = train_test_split(df_session['ip_address'], df_session['session_length'], test_size=0.20, random_state=42)
ip_encoder = LabelBinarizer()
X_train_encoded = ip_encoder.fit_transform(X_train)
X_test_encoded = ip_encoder.transform(X_test)

print("Training part 2 here using an SVR regression algorithm..")
svr_estimator = SVR()
svr_estimator.fit(X_train_encoded, y_train)

y_pred = svr_estimator.predict(X_test_encoded)
print(f"Part 2: Only regression metric taken was the variance score: {explained_variance_score(y_test, y_pred, multioutput='raw_values')}")



#Part 3:

unique_url_hit_count = session_df.groupBy("ip_address","URL").count().distinct()
unique_url_hit_count = unique_url_hit_count.withColumnRenamed("URL\ncount", "unique_url_hit_count")

unique_url_hit_count_pandas = unique_url_hit_count.toPandas()
X_train, X_test, y_train, y_test = train_test_split(unique_url_hit_count_pandas['ip_address'], unique_url_hit_count_pandas['count'], test_size=0.20, random_state=42)
ip_encoder = LabelBinarizer()
X_train_encoded = ip_encoder.fit_transform(X_train)
X_test_encoded = ip_encoder.transform(X_test)
print("Training part 3 here using an SVR regression algorithm..")
svr_estimator = SVR()
svr_estimator.fit(X_train_encoded, y_train)
y_pred = svr_estimator.predict(X_test)
print(f"Part 3: Only regression metric taken was the variance score: {explained_variance_score(y_test, y_pred, multioutput='raw_values')}")
