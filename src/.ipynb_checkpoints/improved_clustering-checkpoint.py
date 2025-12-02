#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import functions
import matplotlib.pyplot as plt
import pandas as pd
import os

# start spark session
spark = SparkSession.builder.appName("ImprovedClustering").getOrCreate()


# In[2]:

# load sample data
df_raw = spark.read.csv("../data/raw/US_Accidents_March23.csv", header=True, inferSchema=True)
print(f"Loaded {df_raw.count()} rows")
print(f"Columns: {len(df_raw.columns)}")


# In[3]:


# extract temporal features from Start_Time
df_temporal = df_raw.withColumn("hour", hour(col("Start_Time")))
df_temporal = df_temporal.withColumn("day_of_week", dayofweek(col("Start_Time")))

# add rush hour flag (7-9am and 4-7pm are rush hours)
df_temporal = df_temporal.withColumn(
    "rush_hour",
    when((col("hour") >= 7) & (col("hour") <= 9), 1)
    .when((col("hour") >= 16) & (col("hour") <= 19), 1)
    .otherwise(0)
)

# add weekend flag (day 1=Sunday, 7=Saturday)
df_temporal = df_temporal.withColumn(
    "is_weekend",
    when((col("day_of_week") == 1) | (col("day_of_week") == 7), 1)
    .otherwise(0)
)

print("Added temporal features: hour, day_of_week, rush_hour, is_weekend")
df_temporal.select("Start_Time", "hour", "day_of_week", "rush_hour", "is_weekend").show(5)


# In[4]:


# create weather condition flags
from pyspark.sql.functions import lower

# get weather condition column and convert to lowercase for matching
df_weather = df_temporal.withColumn("weather_lower", lower(col("Weather_Condition")))

# create binary flags for different weather conditions
df_weather = df_weather.withColumn(
    "is_clear",
    when(col("weather_lower").contains("clear") | col("weather_lower").contains("fair"), 1).otherwise(0)
)

df_weather = df_weather.withColumn(
    "is_rainy",
    when(col("weather_lower").contains("rain") | col("weather_lower").contains("drizzle") | col("weather_lower").contains("shower"), 1).otherwise(0)
)

df_weather = df_weather.withColumn(
    "is_foggy",
    when(col("weather_lower").contains("fog") | col("weather_lower").contains("mist") | col("weather_lower").contains("haze"), 1).otherwise(0)
)

df_weather = df_weather.withColumn(
    "is_snowy",
    when(col("weather_lower").contains("snow") | col("weather_lower").contains("ice") | col("weather_lower").contains("sleet"), 1).otherwise(0)
)

# drop temporary column
df_weather = df_weather.drop("weather_lower")

print("Added weather flags: is_clear, is_rainy, is_foggy, is_snowy")
df_weather.select("Weather_Condition", "is_clear", "is_rainy", "is_foggy", "is_snowy").show(5)


# In[5]:


# create road type flags from street names
df_road = df_weather.withColumn("street_lower", lower(col("Street")))

# highway flag (interstate or highway)
df_road = df_road.withColumn(
    "is_highway",
    when(col("street_lower").contains("i-") | 
         col("street_lower").contains("interstate") | 
         col("street_lower").contains("hwy") | 
         col("street_lower").contains("highway"), 1).otherwise(0)
)

# local street flag
df_road = df_road.withColumn(
    "is_local",
    when(col("street_lower").contains(" st") | 
         col("street_lower").contains(" ave") | 
         col("street_lower").contains(" rd") | 
         col("street_lower").contains(" ln") | 
         col("street_lower").contains(" dr"), 1).otherwise(0)
)

# drop temporary column
df_road = df_road.drop("street_lower")

print("Added road type flags: is_highway, is_local")
df_road.select("Street", "is_highway", "is_local").show(5)


# In[6]:


# add spatial features
from pyspark.ml.feature import StringIndexer

# encode state as numeric feature
state_indexer = StringIndexer(inputCol="State", outputCol="state_encoded", handleInvalid="keep")
state_model = state_indexer.fit(df_road)
df_spatial = state_model.transform(df_road)

# identify high accident states (top 5 by count in sample)
# CA, TX, FL, NY, PA are typically highest
df_spatial = df_spatial.withColumn(
    "high_accident_state",
    when(col("State").isin("CA", "TX", "FL", "NY", "PA"), 1).otherwise(0)
)

print("Added spatial features: state_encoded, high_accident_state")
df_spatial.select("State", "state_encoded", "high_accident_state").show(5)
df_spatial.coalesce(1).write.option("header", True).csv("../data/processed/engineeredParams")


# In[7]:


# top 10 features from siddhants analysis
top_features = [
    "Temperature(F)",
    "Wind_Chill(F)", 
    "Humidity(%)",
    "Pressure(in)",
    "Traffic_Signal",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Traffic_Calming",
    "Precipitation(in)",
    "Stop"
]

# temporal features
temporal_features = ["rush_hour", "is_weekend"]

# weather features
weather_features = ["is_clear", "is_rainy", "is_foggy", "is_snowy"]

# road type features
road_features = ["is_highway", "is_local"]

# spatial features
spatial_features = ["state_encoded", "high_accident_state"]

# combine all features
all_features = top_features + temporal_features + weather_features + road_features + spatial_features

print(f"Using {len(all_features)} features:")
print(f"  Top 10 importance: {len(top_features)}")
print(f"  Temporal: {len(temporal_features)}")
print(f"  Weather: {len(weather_features)}")
print(f"  Road type: {len(road_features)}")
print(f"  Spatial: {len(spatial_features)}")
print(f"  Total: {len(all_features)}")

# select features and severity
df_selected = df_spatial.select(all_features + ["Severity"])


# In[8]:


# assemble features into vector column
assembler = VectorAssembler(
    inputCols=all_features,
    outputCol="features",
    handleInvalid="skip"
)

df_vector = assembler.transform(df_selected)
print(f"Vectorized features, rows after removing nulls: {df_vector.count()}")


# In[9]:


# test different k values
k_values = [3, 4, 5, 6]
results = []

print("Testing different k values...")
for k in k_values:
    kmeans = KMeans().setK(k).setSeed(38)
    model = kmeans.fit(df_vector)
    
    predictions = model.transform(df_vector)
    
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    wcss = model.summary.trainingCost
    
    results.append({
        'k': k,
        'silhouette': silhouette,
        'wcss': wcss
    })
    
    print(f"k={k}: silhouette={silhouette:.4f}, wcss={wcss:.2f}")

# find best k based on silhouette score
best_k = max(results, key=lambda x: x['silhouette'])['k']
print(f"\nBest k: {best_k}")


# In[10]:


# train with best k
print(f"Training final model with k={best_k}")
kmeans_final = KMeans().setK(best_k).setSeed(38)
model_final = kmeans_final.fit(df_vector)

predictions_final = model_final.transform(df_vector)
evaluator_final = ClusteringEvaluator()
final_silhouette = evaluator_final.evaluate(predictions_final)
final_wcss = model_final.summary.trainingCost

print(f"Final silhouette score: {final_silhouette:.4f}")
print(f"Final WCSS: {final_wcss:.2f}")


# In[15]:


# analyze what each cluster represents
print("\nCluster Analysis:")
print("=" * 60)

for cluster_id in range(best_k):
    cluster_data = predictions_final.filter(col("prediction") == cluster_id)
    count = cluster_data.count()
    
    
    print(f"\nCluster {cluster_id} ({count} accidents):")
    
    # get average values for each feature
    for feature in top_features[:5]:  # show top 5 features
        col_expr = functions.col(feature)
        if dict(cluster_data.dtypes)[feature] == 'boolean':
            col_expr = col_expr.cast("int")  # True:1, False:0
        
        avg_val = cluster_data.select(functions.avg(col_expr)).collect()[0][0]
        if avg_val is not None:
            print(f"  {feature}: {avg_val:.2f}")
    
    # show rush hour percentage
    rush_pct = cluster_data.filter(col("rush_hour") == 1).count() / count * 100
    print(f"  Rush hour: {rush_pct:.1f}%")
    
    # show severity distribution
    severity_dist = cluster_data.groupBy("Severity").count().orderBy("Severity").collect()
    severity_str = ", ".join([f"S{row.Severity}:{row['count']}" for row in severity_dist])
    print(f"  Severity: {severity_str}")


# In[16]:


# convert to pandas for visualization
pandas_df = predictions_final.select(
    "Temperature(F)", 
    "Humidity(%)", 
    "prediction"
).toPandas()

# scatter plot
plt.figure(figsize=(10, 6))
for cluster_id in range(best_k):
    cluster_points = pandas_df[pandas_df["prediction"] == cluster_id]
    plt.scatter(
        cluster_points["Temperature(F)"], 
        cluster_points["Humidity(%)"],
        label=f"Cluster {cluster_id}",
        alpha=0.6
    )

plt.xlabel("Temperature(F)")
plt.ylabel("Humidity(%)")
plt.title(f"K-Means Clustering (k={best_k})")
plt.legend()
plt.grid(True, alpha=0.3)
if not os.path.exists("K-Means Clustering (k={best_k}).png"):
    plt.savefig("K-Means Clustering (k={best_k}).png")
plt.close()


# In[17]:


# another visualization with different features
pandas_df2 = predictions_final.select(
    "Visibility(mi)", 
    "Wind_Speed(mph)", 
    "prediction"
).toPandas()

plt.figure(figsize=(10, 6))
for cluster_id in range(best_k):
    cluster_points = pandas_df2[pandas_df2["prediction"] == cluster_id]
    plt.scatter(
        cluster_points["Visibility(mi)"], 
        cluster_points["Wind_Speed(mph)"],
        label=f"Cluster {cluster_id}",
        alpha=0.6
    )

plt.xlabel("Visibility(mi)")
plt.ylabel("Wind_Speed(mph)")
plt.title(f"K-Means Clustering (k={best_k})")
plt.legend()
plt.grid(True, alpha=0.3)
if not os.path.exists("K-Means Clustering (k={best_k}).png"):
    plt.savefig("K-Means Clustering (k={best_k}).png")
plt.close()


# In[18]:


# plot elbow curve
k_vals = [r['k'] for r in results]
wcss_vals = [r['wcss'] for r in results]
silhouette_vals = [r['silhouette'] for r in results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(k_vals, wcss_vals, marker='o', linewidth=2)
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("WCSS")
ax1.set_title("Elbow Method")
ax1.grid(True, alpha=0.3)

ax2.plot(k_vals, silhouette_vals, marker='o', linewidth=2, color='orange')
ax2.set_xlabel("Number of Clusters (k)")
ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Score by k")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
if not os.path.exists("elbow curve.png"):
    plt.savefig("elbow curve.png")
plt.close()


# In[19]:


# compare to dominics original results
print("\n" + "=" * 60)
print("COMPARISON TO BASELINE")
print("=" * 60)

print("\nDominic's Original Clustering (k=4, all 7 features):")
print("  Silhouette Score: 0.4713")
print("  WCSS: 186248.60")
print("  Features: Severity, Temperature, Humidity, Pressure, Visibility, Wind_Speed, Precipitation")

print(f"\nImproved Clustering (k={best_k}, top {len(all_features)} features):")
print(f"  Silhouette Score: {final_silhouette:.4f}")
print(f"  WCSS: {final_wcss:.2f}")
print(f"  Features: {', '.join(all_features[:3])}... ({len(all_features)} total)")

improvement = ((final_silhouette - 0.4713) / 0.4713) * 100
print(f"\nSilhouette improvement: {improvement:+.1f}%")

if final_silhouette > 0.4713:
    print("Result: Improved clustering quality")
elif final_silhouette > 0.45:
    print("Result: Similar clustering quality")
else:
    print("Result: Lower clustering quality (may need more features)")


# In[20]:


# show which features we used
print("\n" + "=" * 60)
print("FEATURES USED")
print("=" * 60)

feature_importance = {
    "Temperature(F)": 0.8496,
    "Wind_Chill(F)": 0.5705,
    "Humidity(%)": 0.5078,
    "Pressure(in)": 0.3684,
    "Traffic_Signal": 0.2706,
    "Visibility(mi)": 0.2491,
    "Wind_Speed(mph)": 0.2225,
    "Traffic_Calming": 0.2145,
    "Precipitation(in)": 0.1942,
    "Stop": 0.1809
}

print("\nTop 10 features (from Siddhant's analysis):")
for i, (feat, imp) in enumerate(feature_importance.items(), 1):
    print(f"{i:2d}. {feat:25s} {imp:.4f}")

print("\nTemporal features (2):")
print("  - rush_hour")
print("  - is_weekend")

print("\nWeather condition flags (4):")
print("  - is_clear")
print("  - is_rainy")
print("  - is_foggy")
print("  - is_snowy")

print("\nRoad type flags (2):")
print("  - is_highway")
print("  - is_local")

print("\nSpatial features (2):")
print("  - state_encoded")
print("  - high_accident_state")

print(f"\nTotal features: {len(all_features)}")


# In[21]:


spark.stop()
print("Spark session stopped")


# In[ ]:





# In[ ]:




