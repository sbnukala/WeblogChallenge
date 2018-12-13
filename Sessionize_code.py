from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import TimestampType
from pyspark.sql.functions import window, unix_timestamp, col
from pyspark.sql.functions import monotonically_increasing_id

#Create a spark session
spark = SparkSession.builder     .master("local[4]")     .appName("IP Address Sessionize")     .config("spark.executor.memory", "2gb")     .getOrCreate()
sc = spark.sparkContext

#Now create a resilient distributed dataset from the dataset provided
rdd = sc.textFile('data/2015_07_22_mktplace_shop_web_log_sample.log')
#Convert each line into a list of entities so that we can retrieve the ones we want later
rdd = rdd.map(lambda line: line.split(" "))

#Observation 1: RDD was taking forever, especially rdd.collect() as I believe it is an action function and local system doesn't have enough processing power/memory
# After some research figured dataframes are much faster! The Row method can convert tuples to rows for easier access. After inspecting the data, the 3 fields needed were url, ip_address and time_stamp for further analysis
df = rdd.map(lambda line: Row(time_stamp=line[0], ip_address=line[2].split(':')[0],url=line[12])).toDF()

#Note that we have the dataframes columns as strings, we need to conver the time_stamp one to TimestampType
df = df.withColumn('timestamp_new_format', df['time_stamp'].cast(TimestampType()))
df = df.drop('time_stamp')
df = df.withColumnRenamed("timestamp_new_format", "time_stamp")

#Use the window function in spark.sql.function to make the current timestamp fall into a 15-min window bucket

#So now we have the data like this: We have a time_stamp when the URL was hit by a user (identified by IP address) also we have a bucket in which this time_stamp falls in a given 15 minute window.
#Now our task is to get the count of the number of times a particular IP has visited any site in a given time window. For e.g. 1.1.1.1 might have visited ww.abcd.com at t1 and www.dfgh.com at t2. Both t1 and t2 are within the 15 minute range
# => The count of 1.1.1.1 is 2 for that window.

session_df  = df.select(window("time_stamp", "15 minutes").alias('15_min_window'),"time_stamp","ip_address").groupBy('15_min_window','ip_address').count()


#To uniquely identify an index, I researched here: https://stackoverflow.com/questions/43406887/spark-dataframe-how-to-add-a-index-column-aka-distributed-data-index
#Solving part 1..
session_df = session_df.withColumn("session_id", monotonically_increasing_id())
df_original = df.select(window("time_stamp", "15 minutes").alias('15_min_window'),"time_stamp","ip_address","url")
session_df = df_original.join(session_df,['15_min_window','ip_address'])

filename = "analysis_part1_sessionize.csv"
part1_df = session_df
print(f"Printing Part 1: Sessionizing IP addresses")
part1_df.show(20, False)
#df.toPandas().to_csv(filename, header=True)

#Solving part 2..
first_visit_time_stamp = session_df.groupBy("session_id").agg({'time_stamp':'min'}).alias('first_visit_time_stamp')
session_df = first_visit_time_stamp.join(session_df,['session_id'])
session_df = session_df.withColumnRenamed("min(time_stamp)", "first_visit_time_stamp")

#Get the timestamp differences between the current time stamp and the first time a site was visited by this IP
timeDiff = (unix_timestamp(session_df.time_stamp)-unix_timestamp(session_df.first_visit_time_stamp))
session_df = session_df.withColumn("time_difference", timeDiff)
max_time_difference = session_df.groupBy("session_id").agg({'time_difference': 'max'})
session_df = session_df.join(max_time_difference,['session_id'])
session_df = session_df.withColumnRenamed("max(time_difference)", "session_duration")

part2_df = session_df
print(f"Printing PART 2: Session duration of each IP address/User")
part2_df.show(20, False)
#session_df.select(col("session_id"),col("ip_address"),col("max(time_difference)")).show(20)

unique_url_hit_count = session_df.groupBy("session_id","URL").count().distinct()
unique_url_hit_count = unique_url_hit_count.withColumnRenamed("URL\ncount", "unique_url_hit_count")
part3_df = unique_url_hit_count
print("Printing PART 3: How many URLs did each user/IP hit?")
part3_df.show(20, False)

#Part 4
users_with_most_session_duration = session_df.select("ip_address","session_id","session_duration").sort(col("session_duration").desc()).distinct()
print("printing PART 4: Users with the most session duration (browsing time)")
users_with_most_session_duration.show(20, False)

#Observations: 1) Spark involved a lot more learning curve than Pandas, I come from a Pandas background. The whole map reduce notion of thinking was new
# 2) RDD's were significantly slower to process and through some research found that dataframes were better
# 3) Spark uses the lazy computing approach and I could notice that. When I define the manipulations, the notebook cell would get processed fast but when I applied functions like
# collect() or show(), it would start computing
# 4) I am aware of a cache() method that I would like to use potentially at some point
# 5) I couldn't find a direct way of storing the results as a csv (per part), the only way I found was through converting the spark dataframe to a panda dataframe and doing it. I didn't want to use Pandas.
