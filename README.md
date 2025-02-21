
# Transaction Monitoring Software

## Overview
Transaction Monitoring Software is designed to detect and prevent fraudulent financial transactions using machine learning techniques. It provides real-time anomaly detection for insurance fraud and money laundering cases, ensuring financial security and compliance.

## Features
- **Fraud Detection:** Identifies suspicious transactions using Spark ML.
- **Anomaly Detection:** Flags unusual activity based on transaction patterns.
- **Scalability:** Built on Apache Spark for handling large datasets.
- **Real-time Processing:** Supports streaming data for continuous monitoring.
- **User-Friendly Interface:** Provides clear visual insights and reports.
- **Data Storage:** Integrates with HDFS, Hive, and Spark SQL for efficient data management.

## Tech Stack
- **Programming Language:** Python
- **Big Data Technologies:** Apache Spark, Spark ML, HDFS, Hive
- **Database:** Cassandra, HBase
- **Containerization:** Docker
- **Orchestration:** Kubernetes (Optional)
- **Cloud Services:** AWS (Optional)

## Installation & Setup
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- Apache Spark
- HDFS & Hive (if using big data storage)
- Docker (if using containerization)
- AWS CLI (if deploying to the cloud)

### Steps to Install
```sh
# Clone the repository
git clone https://github.com/saisrinivas6888/Transaction-Monitoring-Software.git
cd Transaction-Monitoring-Software

# Install dependencies
pip install -r requirements.txt

# Set up the environment variables (if required)
export SPARK_HOME=/path/to/spark
export HADOOP_HOME=/path/to/hadoop

# Run the application
python main.py 
