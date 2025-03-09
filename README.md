# 🏡 California Housing Clustering  

A data-driven approach to segmenting the California housing market using K-Means clustering. This project preprocesses housing data, handles missing values, scales numerical features, and encodes categorical variables to improve clustering accuracy.  

## 📌 Table of Contents  
- [Overview](#overview)  
- [Features](#features)  
- [Dataset](#dataset)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Results](#results)  
- [Technologies Used](#technologies-used)  
- [Contributions](#contributions)  

---

## 📖 Overview  
This project applies K-Means clustering to analyze housing market segmentation in California. A custom implementation of K-Means using the Minkowski distance metric optimizes clustering accuracy, reducing SSE from **175,000 to 75,000** by tuning k-values between **2 and 10**.  

---

## ✨ Features  
✅ **Data Preprocessing**: Handles missing values, scales numerical features, and encodes categorical data.  
✅ **Custom K-Means Implementation**: Built from scratch using Minkowski distance metrics.  
✅ **Cluster Optimization**: Tuned k-values (2 to 10) to improve segmentation accuracy.  
✅ **Visualization**: Seaborn & Matplotlib used for cluster insights and data distribution.  

---

## 📊 Dataset  
The dataset used in this project comes from the [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html).  

- **Features**: Median income, house age, population, latitude, longitude, etc.  
- **Target Variable**: Median house value.  

---

## ⚙ Installation  
To run this project, install the necessary dependencies:  

```bash
git clone https://github.com/yourusername/california-housing-clustering.git
cd california-housing-clustering
pip install -r requirements.txt
