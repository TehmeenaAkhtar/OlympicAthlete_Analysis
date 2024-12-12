import pyspark
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import avg, stddev, min, max
from pyspark.sql.functions import col
from pyspark.sql import functions as F
import pandas as pd
import seaborn as sns

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Olympic Athlete ML Project") \
    .config("spark.some.config.option", "config-value") \
    .getOrCreate()

print("This is the analysis of OLYMPIC ATHLETE DATASET!")

# Load CSV files into DataFrames
athletes_df = spark.read.csv('athletes.csv', header=True, inferSchema=True)
coaches_df = spark.read.csv('coaches.csv', header=True, inferSchema=True)
medals_df = spark.read.csv('medals.csv', header=True, inferSchema=True)
teams_df=spark.read.csv('teams.csv', header=True, inferSchema=True)
basketball3_df=spark.read.csv('3x3 Basketball.csv', header=True, inferSchema=True)
archery_df=spark.read.csv('Archery.csv', header=True, inferSchema=True)
football_df=spark.read.csv('Football.csv', header=True, inferSchema=True)
technicals_df=spark.read.csv('technical_officials.csv', header=True, inferSchema=True)
schedulesPrem_df=spark.read.csv('schedules_preliminary.csv', header=True, inferSchema=True)
medallists_df=spark.read.csv('medallists.csv', header=True, inferSchema=True)

# Display sample data
athletes_df.show(5)
coaches_df.show(5)
medals_df.show(5)

# Drop rows with missing values
athletes_df = athletes_df.dropna()
coaches_df = coaches_df.dropna()
medals_df = medals_df.dropna()

from pyspark.sql.functions import col


# 1 -->  Gold Medal
# 2 -->  Silver Medal
# 3 -->  Bronze Medal
medals_df = medals_df.withColumn("total_medals", col("medal_code"))

medals_df.show(5)

# Preview data
medals_df.select("name", "total_medals").show(5)

# Create a StringIndexer object
indexer = StringIndexer(inputCol="gender", outputCol="gender_index")

# Fit the indexer and transform the DataFrame
athletes_df = indexer.fit(athletes_df).transform(athletes_df)

# Show the DataFrame with the new numeric column
athletes_df.show()

# Gender analysis (how many males and females)
print("\nGender Distribution:")
athletes_df.groupBy("gender").count().show()

# Country analysis
print("\nNumber of Unique Countries Represented:")
unique_count = athletes_df.select("country_long").distinct().count()
print(f" {unique_count}")

# Top countries with most athletes
print("\nTop 10 Countries with Most Athletes:")
athletes_df.groupBy("country_long") \
    .count() \
    .orderBy("count", ascending=False) \
    .show(10)


# Custom statistics for 'height'
stats = athletes_df.agg(
    max("height").alias("Maximum Height")
)
stats.show()

# Custom statistics for 'weight'
stats = athletes_df.agg(
    max("weight").alias("Maximum Weight")
)
stats.show()

# Count unique countries in 'country' column
unique_values = medals_df.select("country").distinct().collect()
print("Unique countries:")
for row in unique_values:
    print(row["country"])


# Count of Medals by Type
medal_counts = medals_df.groupBy("medal_type").count()
medal_counts.show()

# Grouping by medal_type 
medal_counts_pandas = medal_counts.toPandas()
sns.barplot(
    x='medal_type', 
    y='count', 
    data=medal_counts_pandas, 
    palette='viridis',
    hue='medal_type'
)
plt.title('Count of Medals by Type')
plt.xlabel('Medal Type')
plt.ylabel('Count')

# Visualization: Top 10 Countries with Most Medals
top_countries = medals_df.groupBy("country").count()
top_countries_pandas = top_countries.toPandas()
top_countries_pandas = top_countries_pandas.sort_values('count', ascending=False).head(10)
sns.barplot(
    x='count', 
    y='country', 
    data=top_countries_pandas, 
    palette='viridis',
    hue='count', 
    legend=False 
)
plt.title('Top 10 Countries with Most Medals')
plt.xlabel('Number of Medals')
plt.ylabel('Country')
plt.show()

# Total number of teams
print(f"Total teams: {teams_df.count()}")

# Count teams by gender
teams_df.groupBy("team_gender").count().show()

# Total athletes and coaches
teams_df.agg({"num_athletes": "sum", "num_coaches": "sum"}).show()

 

   # Game: 3x3 Basketball
# View schema to verify data structure
basketball3_df.printSchema()
# Count participants by country
print("Total Participants by Country (3x3 Basketball Game)")
participants_by_country = basketball3_df.groupBy("participant_country").count()
participants_by_country.show()
# Aggregate wins, losses, and ties by country
country_performance = basketball3_df.groupBy("participant_country_code", "result_WLT") \
    .count() \
    .orderBy("participant_country_code", "result_WLT")
country_performance.show()


  # Game: Archery
#Count participants by Country
print("Total Participants by Country (Archery)")           
participants_by_country = archery_df.groupBy("participant_country").count()
participants_by_country.show()

# Game: Football
# Count participants by Country   
print("Total Participants by Country (Football)")           
participants_by_country = football_df.groupBy("participant_country").count()
participants_by_country.show()


# Group by the 'function' column and count the occurrences
function_count = technicals_df.groupBy('function').count()
function_count.show()

# Plot the count by function
function_count_pandas = function_count.toPandas()
function_count_pandas.plot(kind='bar', x='function', y='count', color='lightcoral', legend=False)
# plt.show()

# Count the number of technical officials by organization
organization_count = technicals_df.groupBy('organisation').count()
organization_count.show()

# Count the number of events by sport
sport_count = schedulesPrem_df.groupBy('sport').count()
sport_count.show()

# Calculate the duration of each event in hours
schedulesPrem_df = schedulesPrem_df.withColumn(
    'event_duration', 
    (F.unix_timestamp('date_end_utc') - F.unix_timestamp('date_start_utc')) / 3600
)
pandas_df = schedulesPrem_df.toPandas()
plt.figure(figsize=(10, 6))
plt.hist(pandas_df['event_duration'], bins=30, color='lightcoral', edgecolor='black')
plt.xlabel('Event Duration (hours)')
plt.ylabel('Frequency')
plt.title('Distribution of Event Duration')
# plt.show()

# Count the number of medals by country
country_medals = medallists_df.groupBy('country').count()
top_15_countries = country_medals.orderBy('count', ascending=False).limit(15)
top_15_countries.show()
top_15_countries_pandas = top_15_countries.toPandas()
top_15_countries_pandas.plot(kind='bar', x='country', y='count', color='lightcoral', legend=False)
plt.xlabel('Country')
plt.ylabel('Medals Count')
plt.title('Top 15 Countries by Medals Count')
plt.xticks(rotation=90)  
plt.show()

# Count the number of medals by discipline
discipline_medals = medallists_df.groupBy('discipline').count()
discipline_medals.show()
discipline_medals_pandas = discipline_medals.toPandas()
discipline_medals_pandas.plot(kind='bar', x='discipline', y='count', color='lightblue', legend=False)
plt.xlabel('Discipline')
plt.ylabel('Medals Count')
plt.title('Medal Count by Discipline')
plt.xticks(rotation=90)  
plt.show()

# Count the number of medals by gender
gender_medals = medallists_df.groupBy('gender').count()
gender_medals.show()
gender_medals_pandas = gender_medals.toPandas()
gender_medals_pandas.plot(kind='bar', x='gender', y='count', color='lightcoral', legend=False)
plt.xlabel('Gender')
plt.ylabel('Medals Count')
plt.title('Medal Count by Gender')
plt.xticks(rotation=0)  
plt.show()
































