# HS Student GPA Regression with PySpark

This project applies machine learning techniques using Apache Spark MLlib to predict high school students' academic performance. Specifically, we build a regression model to predict GPA (Grade Point Average) based on a variety of demographic, academic, and behavioral features from the High School Student Performance dataset.

## Author

- Emmanuel De Vera
- Student ID: 300602434 Deveremma
- Course: AIML427 (Fourth Year)


## Required files 

* Python program: `student.py` 
* Project dataset: `student.csv` 
* Hadoop setup: `SetupSparkClasspath.sh` 


All three are obtainable via the submission system and are to be downloaded.

In case size limits are exceeded, the original dataset can be obtained through: <br>
https://www.kaggle.com/datasets/neuralsorcerer/student-performance/data

`validation.csv` is sufficient for training and testing due to its 168MB size


## Environment Setup

As per the brief, `student.py`  is required to run on the University Hadoop cluster. The following steps set up the environment: <br>

The downloaded required files are to be saved in a directory within the university servers. Below is a command to transfer from a local device to university servers.
```bash
scp student.py SetupSparkClasspath.sh student.csv <USERNAME>@barretts.ecs.vuw.ac.nz:/path/to/desired/directory
```

Remotely access the university servers, replacing `USERNAME` with an approved ECS username (or access the university servers physically) 
```bash
ssh <USERNAME>@barretts.ecs.vuw.ac.nz
```

Enter a university cluster node, replacing 8 with any integer from 1-9 depending on preference/queue-times
```bash
ssh co246a-8
```

Navigate to the directory containing the downloaded `SetupSparkClasspath.sh` and run `source` to update the environment variables
```bash
cd <download_directory>
source SetupSparkClasspath.sh
```

Apache Spark needs Java as it is primarily written in Scala which runs on the Java Virtual Machine 
```bash
need java8
```

Verify environment variables are successfully updated
```bash
echo $HADOOP_VERSION
# Should output: 3.3.6
```

Assuming the current directory contains `student.csv`, upload it to hdfs under input, replacing `USERNAME` accordingly. Verify the upload was successful.
```bash
hdfs dfs -mkdir -p /user/<USERNAME>/input
hdfs dfs -put student.csv input
hdfs dfs -ls input
```



## Program Usages

1. Explicitly run on the cluster node using Hadoop's resource manager YARN, specifying the expanded dataset filename, seed, and mode with appropriate `USERNAME`
```bash
spark-submit --master yarn --deploy-mode cluster student.py --dataset hdfs:///user/<USERNAME>/input/student.csv --seed 7 --mode 2
```

| **Argument** | **Failure Behavior**                                | **Mode Details**                     |
|--------------|------------------------------------------------------|--------------------------------------|
| `--dataset` can take only string value of accurate filepath  | Datatype failure will result in program termination  | 1 = No Scaling / No PCA              |
| `--seed` can take only integer values (0 - MAX_INTEGER)             | Datapath failure will result in program termination  | 2 = Scaling / No PCA                 |
| `--mode` can take only integer values in [1, 2, 3]             | Seed unspecified or beyond domain will default to 7                 | 3 = Scaling / PCA                    |
|              | Mode unspecified or beyond domain will default to 2                 |                                      |
|              |                                                      |                                      |
|              |                                                      |                                      |


2. Explicitly run on the cluster node using Hadoop's resource manager YARN, specifying shorthand filepath (`USERNAME` not required) with default seed and mode. 

```bash
spark-submit --master yarn --deploy-mode cluster student.py --dataset input/student.csv
```
3. Shorter command to run locally with specified shorthand filepath, seed and mode

```bash
spark-submit student.py --dataset input/student.csv --seed 42 --mode 1
```

4. Shortest command to run locally with default filepath `input/student.csv`, seed and mode

```bash
spark-submit student.py
```
