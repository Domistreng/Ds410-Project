# **Project title**

**Course:** DS/CMPSC 410  
**Semester:** Fall 2025  
**Team Members: Osama Al-khanjry, Omer Kandemir, Dominic Sitto, Faisal El-Mutawalli, Yehe qui, Siddhant Baghat**  
**Team Name:** 

---

## **1\. Introduction and Motivation**

Road accidents are a major public safety concern in the United States, with millions of incidents reported annually. While numerous studies investigate accident causes, most rely on limited regional datasets. The availability of large-scale accident data provides an opportunity to study temporal, spatial, and environmental factors influencing accident risk at national scale.

**Key Question:**

* When and where are drivers most at risk of accidents in the U.S.?

**Why Big Data?**  
 Conventional methods struggle with the volume and complexity of accident records (millions of rows). Big data approaches with PySpark and Roar computing allow scalable analysis and meaningful discovery of trends across large, heterogeneous datasets.

---

## **2\. Project Objectives**

Collect and preprocess the U.S. Car Accidents dataset (Kaggle).

Build PySpark workflows for cleaning, transformation, and feature extraction.

Conduct large-scale temporal and spatial analysis of accidents:

* By day of week

* By time of day

* By city/region

* By weather and road conditions  
  Visualize accident patterns (heatmaps, bar charts, trend lines).

---

## **3\. Data Description**

* **Data Source:** U.S. Accidents Dataset from Kaggle (Sobhan Moosavi).

* **Data Size:** \~7.5 million accident records; several GB of CSV files.

* **Characteristics:** Structured tabular data with temporal (timestamps), spatial (latitude/longitude, city, state), and environmental (weather, severity, road conditions) attributes.

* **Challenges:** Missing data in weather/road conditions, skewed distribution (more data from populated areas), and potential high dimensionality in categorical attributes.

---

## **4\. Technical Approach**

* **Tools & Frameworks:** PySpark (RDDs, DataFrames, MLlib, GraphX if relevant), possibly integration with external libraries (e.g., Pandas, Matplotlib, scikit-learn for evaluation).

* **Cluster Usage Plan:**

  1. Number of nodes (up to 4 CPU nodes).

  2. Expected job types (batch processing, iterative ML training, streaming if applicable).

  3. Storage requirements and file formats (e.g., Parquet, CSV, JSON).

* **Pipeline Overview:**

  1. Data ingestion and preprocessing.

  2. Exploratory analysis (PySpark SQL, summary statistics).

  3. Model building or large-scale computation.

  4. Visualization and interpretation of results.

---

## 

## **5\. Expected Outcomes**

* Clean, documented PySpark scripts for large-scale accident analysis.

* Scalable visualizations of accident risks by time, location, and conditions.

* Summary dashboard highlighting safest/dangerous driving conditions.

* Final presentation and report.

**Success Metrics:**

* Accurate trend identification across large datasets.

* Runtime improvements when scaling across cluster nodes.

* Clear, interpretable visualizations.

---

## **6\. Work Plan & Timeline**

Break down project tasks into weeks

---

## **7\. Division of Labor**

Assign roles among team members, e.g.:

* Data acquisition & preprocessing lead.

* PySpark pipeline & ML lead.

* Visualization & results lead.

* Documentation & presentation lead.

---

## **8\. Potential Challenges & Mitigation**

Identify possible risks (data too large, PySpark API learning curve, cluster resource limits) and how the team will address them (data sampling, fallback tools, scheduling batch jobs).

---

## **9\. References**

List any datasets, papers, or tutorials you are drawing inspiration from.

---

