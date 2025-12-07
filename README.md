\# DS/CMPSC 410 \-- U.S. Car Accidents Clustering Project

This repo has our big data analysis of the \*\*U.S. Accidents dataset\*\* using \*\*PySpark\*\* on the \*\*ROAR cluster\*\*. We're doing clustering to group similar accidents together based on weather, road conditions, and other features.

\#\# What We're Doing

We wanted to see if we could improve on the baseline clustering approach by using more features and better methods. The main goal is to group accidents into meaningful clusters that could help understand different types of accident scenarios.

\*\*Main approach:\*\*  
\- Use K-Means clustering with PySpark  
\- Test different numbers of clusters (k=3 to k=6)  
\- Compare baseline (7 features, k=4) vs improved (20 features, k=3)  
\- Run on full dataset (\~7.7M rows) using ROAR

\#\# Dataset

\*\*U.S. Accidents (Kaggle)\*\*    
https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

\- \~7.7 million accident records  
\- 46 features including weather, location, time, road conditions  
\- We cleaned it down to \~5.2M usable rows

\#\# Repo Structure

\`\`\`  
Ds410-Project/  
│  
├── data/  
│   ├── processed/     \# sample\_df.csv (20k rows for local testing)  
│   └── raw/           \# full dataset (not in git, too big)  
│  
├── notebooks/  
│   ├── feature\_importance\_analysis.ipynb  \# finding best features  
│   ├── graphingEngineeredFeatures.py      \# visualization code  
│   └── \*.png                               \# all our charts  
│  
├── src/  
│   ├── gatherData.py                    \# downloads data, creates sample  
│   ├── kNearestNeightbor.ipynb          \# baseline clustering (Dominic's original)  
│   ├── improved\_clustering.py           \# MAIN FILE \- production run on full data  
│   ├── improved\_clustering.ipynb        \# testing improved approach locally  
│   ├── improved\_clustering\_sklearn.ipynb \# sklearn version for comparison  
│   └── train\_model.py                   \# quick ML test  
│  
└── README.md  
\`\`\`

\#\# How to Run

\#\#\# On ROAR:

\`\`\`bash  
git clone https://github.com/Domistreng/Ds410-Project  
cd Ds410-Project  
module load anaconda3  
module load spark

\# for notebooks:  
jupyter lab

\# for production run:  
spark-submit src/improved\_clustering.py  
\`\`\`

\#\#\# Locally (with sample data):

Just run the notebooks in \`src/\` \- they use the 20k sample from \`data/processed/sample\_df.csv\`

\#\# What We Did

\#\#\# 1\. Feature Analysis (\`notebooks/feature\_importance\_analysis.ipynb\`)  
\- Used Random Forest to find which features matter most for severity  
\- Found top 20 features: Temperature, Wind\_Chill, Humidity, Pressure, Visibility, etc.  
\- Created a bunch of charts showing correlations and distributions

\#\#\# 2\. Baseline Clustering (\`src/kNearestNeightbor.ipynb\`)  
\- Dominic's original approach  
\- K-Means with k=4 clusters  
\- Used 7 basic weather features  
\- \*\*Silhouette Score: 0.4713\*\* (this is what we're trying to beat)

\#\#\# 3\. Improved Clustering (\`src/improved\_clustering.py\`)  
\- Our improved approach  
\- K-Means with k=3 clusters  
\- Used top 20 features (including engineered features like rush\_hour, is\_highway, etc.)  
\- Tested on sample first, then ran on full 5.2M rows on ROAR  
\- \*\*Silhouette Score: 0.5082\*\* (+7.8% improvement\!)

\#\# Results

\#\#\# Clustering Performance:

| Approach | k | Features | Silhouette Score | Dataset Size |  
|----------|---|----------|------------------|--------------|  
| Baseline | 4 | 7 | 0.4713 | 20k sample |  
| Improved | 3 | 20 | 0.5082 | 5.2M full |

\#\#\# Final Clusters (from full dataset run):  
\- \*\*Cluster 0:\*\* 1,997,183 accidents (38%)  
\- \*\*Cluster 1:\*\* 1,877,621 accidents (36%)  
\- \*\*Cluster 2:\*\* 1,372,007 accidents (26%)

The clusters are pretty balanced which is good. Each one represents different combinations of weather/road/time conditions.

\#\#\# What the improvement means:  
\- Higher silhouette score \= better defined clusters  
\- Accidents in each cluster are more similar to each other  
\- Could be useful for predicting accident types or allocating emergency resources

\#\# Files Explained

\*\*Main analysis files:\*\*  
\- \`src/improved\_clustering.py\` \- the big production run, this is our main result  
\- \`notebooks/feature\_importance\_analysis.ipynb\` \- figuring out which features to use  
\- \`src/kNearestNeightbor.ipynb\` \- baseline to compare against

\*\*Testing files:\*\*  
\- \`src/improved\_clustering.ipynb\` \- testing locally before ROAR run  
\- \`src/improved\_clustering\_sklearn.ipynb\` \- sklearn version (didn't work as well)

\*\*Output:\*\*  
\- \`src/spark-46704906.out\` \- full output from the production run  
\- All the PNGs in \`notebooks/\` \- visualizations

\#\# Limitations

\- Dataset is biased toward certain states (California, Texas, etc.)  
\- Missing weather data for some accidents  
\- Silhouette score improved but still not amazing (0.5 is decent but not great)  
\- Don't really know what the clusters "mean" in practical terms yet  
\- Would need more analysis to make this actually useful

\#\# Requirements

\- Python 3.x  
\- PySpark  
\- pandas, numpy  
\- matplotlib, seaborn  
\- sklearn (for feature analysis)  
\- Jupyter

On ROAR just do:  
\`\`\`bash  
module load anaconda3  
module load spark  
\`\`\`

\#\# Team

\- Osama Al-khanjry  
\- Omer Kandemir  
\- Dominic Sitto  
\- Faisal El-Mutawalli  
\- Siddhant Baghat

