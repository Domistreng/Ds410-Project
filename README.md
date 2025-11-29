# DS/CMPSC 410 -- U.S. Car Accidents Big Data Project

This repository contains our big data analysis of the **U.S. Accidents
(Kaggle)** dataset using **PySpark** and the **ROAR computing cluster**.
The goal of the project is to understand when and where car accidents
are most likely to occur in the United States and how conditions like
weather, visibility, and road features affect crash frequency and
severity.

# 1. Project Overview

We study accident patterns across: 
- **Time** (hour of day, weekday,
month)
- **Location** (cities, states, regions)
- **Environment**(temperature, visibility, weather)
- **Road conditions**(surface/lighting)

The dataset contains **millions of accident reports**, so Spark + ROAR
are used for efficient processing.

# 2. Repository Structure

    Ds410-Project/
    │
    ├── data/          # Dataset (downloaded manually from Kaggle)
    ├── notebooks/     # Jupyter notebooks (cleaning, EDA, temporal, weather, viz)
    ├── src/           # PySpark scripts (loading, cleaning, aggregation)
    └── README.md      # Main documentation (this file)

# 3. Dataset

**U.S. Accidents (Kaggle)**
https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents


# 4. Requirements

**Python Dependencies** - Python 3.x - PySpark - Pandas - Matplotlib -
Seaborn - Jupyter Notebook / Lab

**ROAR Cluster Modules**

    module load anaconda3
    module load spark

# 5. How to Run the Project

## Running on ROAR

    git clone https://github.com/Domistreng/Ds410-Project
    cd Ds410-Project
    module load anaconda3
    module load spark
    jupyter lab

# 6. Code Components Explained

## notebooks/

-   Cleaning
-   EDA
-   Temporal patterns
-   Weather analysis
-   Visualizations

## src/

-   load_data.py → Spark loading
-   cleaning.py → preprocessing
-   analysis.py → aggregations
-   utils.py → helper functions

# 7. Results Summary

Key findings: 
- Peak accident hours: late afternoon & early evening 
- Mid-week has the most accidents 
- Large cities have the highest volume
- Weather + low visibility increase severity

# 8. Limitations

-   Uneven dataset coverage
-   Missing weather values
-   Local execution may be slow
-   City/state differences may reflect population


# 9. Team

-   Osama Al-khanjry
-   Omer Kandemir
-   Dominic Sitto
-   Faisal El-Mutawalli
-   Siddhant Baghat
