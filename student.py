# AIML427 Big Data Assignment 3
# Deveremma 300602434

from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, sum as sparksum
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler,
    OneHotEncoderModel,
    StringIndexerModel,
    MinMaxScaler, PCA
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pyspark.ml.regression import (
    LinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GeneralizedLinearRegression,
    GBTRegressor
)
import subprocess
import argparse
import time
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='input/student.csv')
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--mode', type=int, default=2)
args = parser.parse_args()

if args.mode not in [1, 2, 3]: args.mode = 2
if not (0 <= args.seed <= 2**32 - 1): args.seed = 7


# Initialize spark session, logging preference, and program seed
spark = SparkSession.builder.appName("GPARegression").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")  # or "WARN"
seed = args.seed


# ===============================
# Load Dataset and Sample to 25MB
# ===============================


# Path to CSV in HDFS (change if local or different user)
data_path = args.dataset
df = spark.read.csv(data_path, header=True, inferSchema=True)

target_mb = 40
# Get size of the file in MB using Hadoop fs
try:
    hdfs_size_output = subprocess.check_output(["hdfs", "dfs", "-du", "-s", data_path]).decode("utf-8")
    size_mb = int(hdfs_size_output.strip().split()[0]) / (1024 * 1024)
except:
    # fallback size
    size_mb = 160

sample_fraction = min(target_mb / size_mb, 1.0)
df = df.sample(withReplacement=False, fraction=sample_fraction, seed=seed)
print("\n" + "=" * 60+ "\n" + f"Dataset sampled to ~{target_mb:.1f}MB from ~{size_mb:.1f}MB using fraction={sample_fraction:.3f}" + "\n" + "=" * 60 + "\n")

#df_local = df.dropna().toPandas()
#df_local.to_csv("student.csv", index=False)

null_exprs = [sparksum(col(c).isNull().cast("int")).alias(c) for c in df.columns]
null_counts_row = df.select(null_exprs).collect()[0]
null_counts = null_counts_row.asDict()

# Display data schema statistics
print(f"Total rows: {df.count()}")
summary = spark.createDataFrame([
    Row(Column=c, DataType=dt, NullCount=null_counts.get(c, 0))
    for c, dt in df.dtypes
])
summary.show(n=summary.count(), truncate=False)


# ================================
# Categorical & numerical features
# ================================


# Feature Engineering: Hypothesis 1: STEM students have better performance Hypothesis 2: All rounders outperform
df = df.withColumn("Stem_Ratio", ((col("TestScore_Math") + col("TestScore_Science")) / 2) / col("TestScore_Reading"))
df = df.withColumn("TestScore_AVG", ((col("TestScore_Math") + col("TestScore_Science") + col("TestScore_Reading")) / 3))

# One-Hot Encode Categorical Features as per recommended pre-processing
categorical_cols = ["Gender", "Race", "ParentalEducation", "SchoolType", "Locale"]

# Scale numeric double variables. Scaling method to be determined by feature distribution.
numerical_cols = ["Age", "Grade", "SES_Quartile", "TestScore_Math", "TestScore_Reading", "TestScore_Science", "AttendanceRate", "StudyHours", "FreeTime", "GoOut", "Stem_Ratio", "TestScore_AVG"]

# Acceptable binary integer variables.
binary_cols = ["InternetAccess", "Extracurricular", "PartTimeJob", "ParentSupport", "Romantic"]

# Split values according to Pareto Principle
train_df, test_df = df.randomSplit([0.8, 0.2], seed=seed)
label_col = "GPA"

# Plot histograms of numerical cols
os.makedirs("plots", exist_ok=True)
numerical_sample = train_df.select(numerical_cols + binary_cols + ["GPA"]).dropna().toPandas()

for col in numerical_sample.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(numerical_sample[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"plots/{col}_distribution.png")
    # Close the figure to avoid memory buildup
    plt.close()


# =============================
# Feature transformation stages
# =============================


# StringIndexers for categoricals
indexers = [
    StringIndexer(inputCol=col, outputCol=f"{col}_Index", handleInvalid='keep')
    for col in categorical_cols
]
# OneHotEncoder for indexed categoricals
encoder = OneHotEncoder(
    inputCols=[f"{col}_Index" for col in categorical_cols],
    outputCols=[f"{col}_Vec" for col in categorical_cols]
)
# Assemble feature vector
assembler = VectorAssembler(
    inputCols=[f"{col}_Vec" for col in categorical_cols] + binary_cols + numerical_cols,
    outputCol="features_vec"
)
# Scale features
scaler = StandardScaler(inputCol="features_vec", outputCol="scaled_features")


# =============================
# Determine the optimal PCA K
# =============================


# Partially Preprocess Training Data
partial_pipeline = Pipeline(stages=indexers + [encoder, assembler, scaler])
partial_model = partial_pipeline.fit(train_df)
transformed_df = partial_model.transform(train_df)

print(f"Total features/predictors in the fully preprocessed data: {len(transformed_df.first())}" + "\n" + "=" * 60)

# PCA to reduce dimensionality (adjust k as desired)
pca = PCA(k=len(transformed_df.first()), inputCol="scaled_features", outputCol="pca_features")
pca_model = pca.fit(transformed_df)

# Get cumulative explained variance
explained_variance = pca_model.explainedVariance.toArray()
cumulative_variance = explained_variance.cumsum()

target_variance = 0.95

# Choose smallest k that explains at least target variance
optimal_k = next(i + 1 for i, v in enumerate(cumulative_variance) if v >= target_variance)
print(f"Optimal number of PCA components to retain {target_variance*100}% variance: {optimal_k}\n")

pca = PCA(k=optimal_k, inputCol="scaled_features", outputCol="pca_features")

# Scree Plot EV visualized
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.axhline(0.95, color='red', linestyle='--', label="95% Threshold")
plt.title("PCA Explained Variance (Scree Plot)")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plots/pca_scree_plot.png")
plt.close()


# ===============================
# User specified Mode of Operation
# ===============================


mode = args.mode
# mode 1 = No Scaling / No PCA, mode 2 = Scaling / No PCA, mode 3 = Scaling / PCA
out = "features_vec" if mode == 1 else "scaled_features" if mode == 2 else "pca_features"
op = "non" if mode == 1 else "scaled" if mode == 2 else "pca"
if mode == 1:
    pipe = [encoder, assembler]
elif mode == 2:
    pipe = [encoder, assembler, scaler]
else:
    pipe = [encoder, assembler, scaler, pca]


# =============================
# Classifiers and Evaluators
# =============================


regressors = {
    "LinearRegression": LinearRegression(
        featuresCol=out,
        labelCol=label_col,
        regParam=0.1,            # Ridge regularization (L2)
        elasticNetParam=0.0,      # ElasticNet mix (0 = Ridge, 1 = Lasso)
        maxIter=50
    ),
    "DecisionTree": DecisionTreeRegressor(
        featuresCol=out,
        labelCol=label_col,
        maxDepth=10,              # More depth for complexity
        minInstancesPerNode=5
    ),
    "RandomForest": RandomForestRegressor(
        featuresCol=out,
        labelCol=label_col,
        numTrees=20,              # More trees for stability
        maxDepth=5,
        subsamplingRate=0.8
    ),
    "Gradient-Boosted": GBTRegressor(
        featuresCol=out,
        labelCol=label_col,
        maxIter=50,              # More boosting rounds
        maxDepth=5,
        stepSize=0.1              # Conservative learning rate
    ),
    "GeneralizedLR": GeneralizedLinearRegression(
        featuresCol=out,
        labelCol=label_col,
        family="gaussian",
        link="identity",
        regParam=0.1,
        maxIter=100
    )
}
# Evaluation metrics
evaluator_rmse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2")
evaluator_mae = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="mae")


# =============================
# Train and Evaluation Loop
# =============================

#'''
results = []
for name, regressor in regressors.items():

    # Variable pipeline dependent on specified program mode
    pipeline = Pipeline(stages=indexers + pipe + [regressor])
    feature_cols = next(stage for stage in pipe if isinstance(stage, VectorAssembler)).getInputCols()

    start_time = time.time()
    model = pipeline.fit(train_df)
    print(f"Training: {name}\n")

    predictions_train = model.transform(train_df)
    predictions_test = model.transform(test_df)
    # Elapsed time to train and predict
    end_time = time.time()

    # Evaluate metrics on train data using trained model
    rmse_train = round(evaluator_rmse.evaluate(predictions_train), 4)
    mae_train = round(evaluator_mae.evaluate(predictions_train), 4)
    r2_train = round(evaluator_r2.evaluate(predictions_train), 4)

    # Evaluate metrics on test data using trained model
    rmse_test = round(evaluator_rmse.evaluate(predictions_test), 4)
    mae_test = round(evaluator_mae.evaluate(predictions_test), 4)
    r2_test = round(evaluator_r2.evaluate(predictions_test), 4)
    duration = round(end_time - start_time, 4)

    results.append(Row(
        BaseModel=name,
        RMSE_train=rmse_train,
        RMSE_test=rmse_test,
        MAE_train=mae_train,
        MAE_test=mae_test,
        R2_train=r2_train,
        R2_test=r2_test,
        TimeSec=duration))

# Visually present metrics with a dataframe
metrics_df = spark.createDataFrame(results)
metrics_df.orderBy("RMSE_test").show(truncate=False)
#'''

# =============================
# Cross Validated Fine Tuning
# =============================


print("=" * 60 + "\n" + "Fine-Tuning Linear Regression Using 5-Fold Cross-Validation" + "\n" + "=" * 60)

# Redefine a fresh model
lr = LinearRegression(
    featuresCol=out,
    labelCol=label_col
)
# Build new pipeline
lr_pipeline = Pipeline(stages=indexers + pipe + [lr])

# Define a parameter grid to search
paramGrid = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.0, 0.01, 0.1])\
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
    .addGrid(lr.maxIter, [50, 100])\
    .build()

# CrossValidator with 5 folds
crossval = CrossValidator(
    estimator=lr_pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator_rmse,
    numFolds=5,
    parallelism=2,
    seed=seed
)

# Train with cross-validation
cv_model = crossval.fit(train_df)
predictions_train = cv_model.bestModel.transform(train_df)
predictions_test = cv_model.bestModel.transform(test_df)

# Evaluate metrics on train data using final trained model
rmse_train = round(evaluator_rmse.evaluate(predictions_train), 4)
mae_train = round(evaluator_mae.evaluate(predictions_train), 4)
r2_train = round(evaluator_r2.evaluate(predictions_train), 4)

# Evaluate metrics on test data using final trained model
rmse_test = round(evaluator_rmse.evaluate(predictions_test), 4)
mae_test = round(evaluator_mae.evaluate(predictions_test), 4)
r2_test = round(evaluator_r2.evaluate(predictions_test), 4)

# Show best hyperparameters and metrics
best_lr_model = cv_model.bestModel.stages[-1]
print("\nBest Hyperparameters for Linear Regression:")
print(f"  regParam:         {best_lr_model._java_obj.getRegParam()}")
print(f"  elasticNetParam:  {best_lr_model._java_obj.getElasticNetParam()}")
print(f"  maxIter:          {best_lr_model._java_obj.getMaxIter()}")

print("\nCross-Validated Linear Regression Results:")
print(f"  RMSE (Test): {rmse_test}")
print(f"  MAE (Test) : {mae_test}")
print(f"  R2 (Test)  : {r2_test}")


# =============================
# Plotting Performance Results
# =============================


# Convert test predictions to Pandas for plotting
test_pred_df = predictions_test.select("prediction", label_col).dropna().toPandas()

# Actual vs Predicted
#KDE density-based shading
plt.figure(figsize=(6, 4))
sns.kdeplot(
    data=test_pred_df,
    x="prediction",
    y=label_col,
    fill=True,
    cmap="Blues",
    thresh=0.05,   # Minimum level of density to show
    levels=100     # More levels = smoother shading
)
plt.plot(
    # Ideal line (y = x)
    [test_pred_df[label_col].min(), test_pred_df[label_col].max()],
    [test_pred_df[label_col].min(), test_pred_df[label_col].max()],
    'r--', label='Ideal'
)
plt.title("Actual vs Predicted GPA")
plt.xlabel("Predicted GPA")
plt.ylabel("Actual GPA")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/actual_predicted.png")
plt.close()

# Histogram of Residuals
test_pred_df["residual"] = test_pred_df[label_col] - test_pred_df["prediction"]
plt.figure(figsize=(6, 4))
sns.histplot(test_pred_df["residual"], bins=30, kde=True)
plt.title("Histogram of Residuals")
plt.xlabel("Residual (Error)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/residual_distribution.png")
plt.close()


# =============================
# Extracting Top LR Coefficients
# =============================


print("\n" + "=" * 60 + "\n" + "Top 10 Features from Final LR Model (Coefficient Magnitude)" + "\n" + "=" * 60 + "\n")

encoder_model = None
indexer_models = {}

# Get OneHotEncoderModel and StringIndexerModel
for stage in cv_model.bestModel.stages:
    if isinstance(stage, OneHotEncoderModel):
        encoder_model = stage
    elif isinstance(stage, StringIndexerModel):
        indexer_models[stage.getOutputCol()] = stage  # Map by output (e.g., Gender_Index)

if encoder_model is None:
    raise Exception("OneHotEncoderModel not found in pipeline.")

# Expand One-hot cols
expanded_features = []
for input_col, size in zip(encoder_model.getInputCols(), encoder_model.categorySizes):

    indexer_model = indexer_models.get(input_col)
    if indexer_model is None:
        raise Exception(f"No StringIndexerModel found for {input_col}")

    # Specify each unique value
    labels = indexer_model.labels
    base_name = input_col.replace("_Index", "")
    expanded_features.extend([f"{base_name}_{label}" for label in labels])

# Add binary and numerical (same assembler order)
expanded_features += binary_cols + numerical_cols

# Get coefficients of the final linear regression model
coefficients = cv_model.bestModel.stages[-1].coefficients.toArray()

# Label Coeffs and sort by highest absolute magnitude first
coef_with_names = list(zip(expanded_features, coefficients))
top10 = sorted(coef_with_names, key=lambda x: abs(x[1]), reverse=True)[:10]
for name, coef in top10: print(f"  {name:25s}: {coef:+.4f}")

# Sort and extract top 10 coefficients
top10_df = pd.DataFrame(top10, columns=["Feature", "Coefficient"])

# Plot model coefficients
plt.figure(figsize=(8, 5))
sns.barplot(x="Coefficient", y="Feature", data=top10_df, palette="coolwarm", hue="Feature", dodge=False, legend=False)
plt.title("Top 10 Most Influential Features")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"plots/{op}_coefficients.png")
plt.close()


print()
spark.stop()

