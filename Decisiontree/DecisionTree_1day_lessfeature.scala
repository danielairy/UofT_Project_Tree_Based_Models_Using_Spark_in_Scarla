// Databricks notebook source
//This code builds a basic decision tree model with raw features provided by the dataset
//This is used to compare if adding seasonality features and moving averages would affect the performances of the model


import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
//read csv file from databrick database
val df1=spark.table("aapl_csv")

// COMMAND ----------

df1.show(10)

// COMMAND ----------

//add more features: difference between high and low, open and close, and log of trading volume
val df2 = df1.withColumn("high_low",df1("High") - df1("Low")).withColumn("open_close",df1("Open")-df1("Close")).withColumn("log_vol",log(df1("Volume")))
//add unique ID column for the ease of ordering later
val df3 = df2.withColumn("UniqueID", monotonicallyIncreasingId)

// COMMAND ----------

df3.show(10)

// COMMAND ----------

//include seasonality
//split Date into year, month, day, since spark ml only takes numeric value
//also get day of the week
import org.apache.spark.sql.types._
import org.apache.spark.sql._
val df4 = df3.withColumn("year", year(col("Date"))) .withColumn("month", month(col("Date"))) .withColumn("day", dayofmonth(col("Date")))
df4.show(10)

// COMMAND ----------

//add new feature: close price change between current day and previous day, which is the return of 1 day
//creating new column that is the target varibale 'label' for 1-day time horizon model
//define a window
val particionwindow = Window.partitionBy($"year").orderBy("UniqueID")
//define a transformation that needs to be applied within a window
val lag1 =lag("Close",1,0).over(particionwindow)
//create a new column with the close price of the day after
val lag2 = lag("Close",-1,0).over(particionwindow)
val df5 = df4.select($"*", lag1 as "prev_close", lag2 as "label" ).orderBy("UniqueID")
val df6 = df5.withColumn("return1",df5("Close")-df5("prev_close")).orderBy("UniqueID").drop("prev_close")
df6.show(10)

// COMMAND ----------

//create new features: moving average and cumulative return of different time intervals (10,20,50,100,200)

val wSpec1 = Window.partitionBy("year").orderBy("UniqueID").rowsBetween(-2, 2)
// Calculate the 5-day moving average,5-day cumulative return
val df7 = df6.withColumn("movingAvg5",avg(df6("Close")).over(wSpec1)).orderBy("UniqueID")

val wSpec2 = Window.partitionBy("year").orderBy("UniqueID").rowsBetween(-9, 10)
// Calculate the 20-day moving average
val df8 = df7.withColumn("movingAvg20",avg(df6("Close")).over(wSpec2)).orderBy("UniqueID")


val wSpec3 = Window.partitionBy("year").orderBy("UniqueID").rowsBetween(-19, 20)
// Calculate the 50-day moving average
val df9 = df8.withColumn("movingAvg50",avg(df6("Close")).over(wSpec3)).orderBy("UniqueID")


val wSpec4 = Window.partitionBy("year").orderBy("UniqueID").rowsBetween(-49, 50)
// Calculate the 100-day moving average
val df10 = df9.withColumn("movingAvg100",avg(df6("Close")).over(wSpec4)).orderBy("UniqueID")


val wSpec5 = Window.partitionBy("year").orderBy("UniqueID").rowsBetween(-99, 100)
// Calculate the 200-day moving average
val df11 = df10.withColumn("movingAvg200",avg(df6("Close")).over(wSpec5)).orderBy("UniqueID")

val wSpec6 = Window.partitionBy("year").orderBy("UniqueID").rowsBetween(-4, 5)
// Calculate the 10-day moving average,10-day cumulative return
val df12 = df11.withColumn("movingAvg10",avg(df6("Close")).over(wSpec6)).orderBy("UniqueID")



// COMMAND ----------

df12.printSchema()

// COMMAND ----------

//drop rows with any NA values
val df_final = df12.na.drop

// COMMAND ----------

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.feature.VectorAssembler

// COMMAND ----------

//get all feature names
val featurename = df_final.columns

// COMMAND ----------

//define the feature columns to put in the feature vector
//set the input and output column names
//"movingAvg5","movingAvg10","movingAvg20","movingAvg100","movingAvg200","dayofwk"
val assembler = new VectorAssembler()
  .setInputCols(Array("Open", "High", "Low", "Adj Close","Close", "Volume"))
  .setOutputCol("features")
//return a dataframe with all of the feature columns in  a vector column
// the transform method produced a new column: features
val data_ts1 = assembler.transform(df_final)
val data_ts = data_ts1.select("features","label")


// COMMAND ----------

data_ts.show(10)

// COMMAND ----------

//split data into train dataset 80%, 20% test dataset
val df_tv = data_ts.where($"UniqueID".between(0,7675)).orderBy("Date")
val splits = df_tv.randomSplit(Array(0.75,0.25), seed = 12345L) 
val df_train = splits(0)
val df_val = splits(1)
val df_test = data_ts.where($"UniqueID">=7676).orderBy("Date")
df_test.show()

// COMMAND ----------

println(df_test.count())
println(df_val.count())
println(df_train.count())

// COMMAND ----------

// initiate a decision tree and specifiy target variable and feature vector
// decision tree currently only support tree-depth <= 30
val dt = new DecisionTreeRegressor()
  .setLabelCol("label")
  .setFeaturesCol("features")
  .setMaxDepth(30)
  .setMinInstancesPerNode(5)
  .setMinInfoGain(0.05)

val dtmodel = dt.fit(df_train)

// COMMAND ----------

//define a function that can compute sMAPE value
def SMAPE(a:Array[Double],f:Array[Double]):Double = {
  import scala.math.abs
  var index = 0;
  var length = a.size;
  var total = 0.0;
  for (index <- 0 until length){
    if (a(index) != 0){
//      println(f(index));
//      println(a(index));
      var diff = Math.abs(f(index) - a(index));
      var sum1 = Math.abs(f(index)) + Math.abs(a(index));
      var temp = 2 * diff / sum1;
//      println(temp);
      total = total + temp;
//      println(total);
    }
  }
  val result = (100/length.toDouble) * total;
  return result
  
}

// COMMAND ----------

//make predictions on validation set
val predictions = dtmodel.transform(df_val).select("features","label","prediction")
// Select (prediction, true label) and compute rmse in validation set
val evaluator = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Root Mean Squared Error (RMSE) on val data = " + rmse)

// COMMAND ----------

//convert prediction and real value to array of double
val forec = predictions.select("prediction").collect().map(_(0).toString.toDouble)
val actual = predictions.select("label").collect().map(_(0).toString.toDouble)
// compute sMAPE value on validation set
val smape_val = SMAPE(forec,actual)
println("smape on val data = " + smape_val)

// COMMAND ----------

// Select (prediction, true label) and compute rmse in testing set
val predictions = dtmodel.transform(df_test).select("features","label","prediction")
val evaluator = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)

// COMMAND ----------

//convert prediction and real value to array of double
val forec2 = predictions.select("prediction").collect().map(_(0).toString.toDouble)
val actual2 = predictions.select("label").collect().map(_(0).toString.toDouble)
// compute sMAPE value on testing set
val smape_test = SMAPE(forec,actual)
println("smape on test data = " + smape_test)
