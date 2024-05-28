### Log Clustering/Classification Analisys

<p align="center"><img src=" https://img.shields.io/badge/give-you_like-blue" alt="shields"></p>

# Overview
This project is designed to perform clustering and classification on log data using machine learning algorithms. It consists of two main parts:

Log Clustering: Clustering log data using the K-Means algorithm.
Log Classification: Classifying the clustered log data into categories such as "ERROR", "INFO", "DEBUG", and "WARNING".
Log Analisys: Analys anomaly logs

# Project Structure
```commandline
.
├── algo
│   ├── AEL
│   │   └── clustering.py
│   ├── AGGLOMERATIVE
│   │   ├── classification.py
│   │   └── clustering.py
│   ├── BRAIN
│   │   └── clustering.py
│   ├── COSINE_KMEANS
│   │   ├── classification.py
│   │   └── clustering.py
│   ├── DBSCAN
│   │   ├── classification.py
│   │   └── clustering.py
│   ├── KMEANS
│   │   ├── classification.py
│   │   └── clustering.py
│   └── anomaly
│       ├── anomaly_IsolationForest.py
│       ├── anomaly_One_Class_SVM.py
│       └── anomaly_Local_Outlier_Factor.py
├── main.py
├── requirements.txt
└── README.md

```

# Dependencies
Before running the project, make sure you have the necessary dependencies installed. You can install them using the following command:
```
pip install -r requirements.txt
```

**Files and Their Purpose**
- algo/AEL/clustering.py:

**Contains clustering algorithms related to the AEL method.**
- algo/AGGLOMERATIVE/classification.py:

**Contains classification algorithms related to the AGGLOMERATIVE method.**
- algo/AGGLOMERATIVE/clustering.py:

**Contains clustering algorithms related to the AGGLOMERATIVE method.**
- algo/BRAIN/clustering.py:

**Contains clustering algorithms related to the BRAIN method.**
- algo/COSINE_KMEANS/classification.py:

**Contains classification algorithms related to the COSINE_KMEANS method.**
- algo/COSINE_KMEANS/clustering.py:

**Contains clustering algorithms related to the COSINE_KMEANS method.**
- algo/DBSCAN/classification.py:

**Contains classification algorithms related to the DBSCAN method.**
- algo/DBSCAN/clustering.py:

**Contains clustering algorithms related to the DBSCAN method.**
- algo/KMEANS/classification.py:

**Contains classification algorithms related to the KMEANS method.**
- algo/KMEANS/clustering.py:

**Contains clustering algorithms related to the KMEANS method.**
- algo/anomaly/anomaly_IsolationForest.py:

**Contains anomaly detection algorithms using the Isolation Forest method.**
- algo/anomaly/anomaly_One_Class_SVM.py:

**Contains anomaly detection algorithms using the One-Class SVM method.**
- algo/anomaly/anomaly_Local_Outlier_Factor.py:

**Contains anomaly detection algorithms using the Local Outlier Factor method.**
- main.py:



Main script to demonstrate the usage of the clustering and classification classes.

# Running the Project
To run the complete example, execute the main.py script:
```commandline
python main.py
```

## Conclusion
This project provides a structured approach to clustering and classifying log data using multiple machine learning algorithms. The use of classes and object-oriented programming enhances the readability and reusability of the code. The provided

<p align="center"><img src="https://img.shields.io/github/watchers/Craxti/log-parser" alt="shields"></p>

## License

[MIT](https://choosealicense.com/licenses/mit/)


<p align="center"><img src="https://img.shields.io/github/repo-size/Craxti/log-parser" alt="shields"></p>

<p align="center"><img src="https://img.shields.io/github/languages/top/Craxti/log-parser" alt="shields"></p>

<p align="center"><img src="https://img.shields.io/github/commit-activity/m/Craxti/log-parser" alt="shields"></p>