// Databricks notebook source
//define a function that can compute sMAPE value
def SMAPE(a:Array[Double],f:Array[Double]):Double = {
  import scala.math.abs
  var index = 0;
  var length = a.size;
  var total = 0.0;
  for (index <- 0 until length){
    if (a(index)!=0) {
    //    println(f(index));
//    println(a(index));
      var diff = Math.abs(f(index) - a(index));
      var sum1 = Math.abs(f(index)) + Math.abs(a(index));
      var temp = 2 * diff / sum1;
  //    println(temp);
      total = total + temp;
  //    println(total);
    }
  }
  val result = (100/length.toDouble) * total;
  return result
  
}

// COMMAND ----------

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.ml.feature.{Imputer,IndexToString, StringIndexer, VectorIndexer, VectorAssembler}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.functions._


val wSpec1 = Window.partitionBy("year").orderBy("id").rowsBetween(-2, 2)
// Calculate the 5-day moving average,5-day cumulative return
val wSpec2 = Window.partitionBy("year").orderBy("id").rowsBetween(-9, 10)
val wSpec3 = Window.partitionBy("year").orderBy("id").rowsBetween(-19, 20)
val wSpec4 = Window.partitionBy("year").orderBy("id").rowsBetween(-49, 50)
val wSpec5 = Window.partitionBy("year").orderBy("id").rowsBetween(-99, 100)
val wSpec6 = Window.partitionBy("year").orderBy("id").rowsBetween(-4, 5)

val InputDF1 = spark.sqlContext.sql("select Date,Open,High,Low,Close,Volume from AAPL_CSV where Close is not NULL")
          .withColumn("id",monotonically_increasing_id)
          .withColumn("Day",dayofmonth($"Date"))
          .withColumn("Month",month($"Date"))
          .withColumn("Year",year($"Date"))
          .withColumn("Day_of_Week",dayofweek($"Date"))
          .withColumn("Quarter",quarter($"Date"))
          .toDF
//create new features: moving average and cumulative return of different time intervals (10,20,50,100,200)
val InputDF = InputDF1.withColumn("movingAvg5",avg(InputDF1("Close")).over(wSpec1)).orderBy("id")
// Calculate the 20-day moving average
          .withColumn("movingAvg20",avg(InputDF1("Close")).over(wSpec2)).orderBy("id")
// Calculate the 50-day moving average
          .withColumn("movingAvg50",avg(InputDF1("Close")).over(wSpec3)).orderBy("id")
// Calculate the 100-day moving average
          .withColumn("movingAvg100",avg(InputDF1("Close")).over(wSpec4)).orderBy("id")
// Calculate the 200-day moving average
          .withColumn("movingAvg200",avg(InputDF1("Close")).over(wSpec5)).orderBy("id")
// Calculate the 10-day moving average
          .withColumn("movingAvg10",avg(InputDF1("Close")).over(wSpec6)).orderBy("id")
          .toDF
// split input into training & Validation set and Test set. The data is split on "2011-05-13".
val DF1 = InputDF.filter(InputDF("date").lt(lit("2011-05-13")))
val TestDF = InputDF.filter(InputDF("date").gt(lit("2011-05-13")))
// is model performance poor after 2016?
//val Test2016DF = InputDF.filter(InputDF("date").gt(lit("2011-05-13")))


//Create DF with one day lead close for training & validation and test
val particionwindow1 = Window.orderBy($"id")
val tomorrowclose =lead("Close",1,0).over(particionwindow1)
val DayDF= DF1.select($"*", tomorrowclose as "Label")
val DayDFTest= TestDF.select($"*", tomorrowclose as "Label") 

//Create DF with one week lead close for training & validation and test
val particionwindow2 = Window.orderBy($"id")
val nextweekclose =lead("Close",5,0).over(particionwindow2)
val WeekDF= DF1.select($"*", nextweekclose as "Label")
val WeekDFTest = TestDF.select($"*", nextweekclose as "Label")

//Create DF with two week lead close for training & validation and test
val particionwindow3 = Window.orderBy($"id")
val nexttwoweekclose =lead("Close",10,0).over(particionwindow3)
val TwoWeekDF= DF1.select($"*", nexttwoweekclose as "Label")
val TwoWeekDFTest = TestDF.select($"*", nexttwoweekclose as "Label")
//Create DF with one month lead close
val particionwindow4 = Window.orderBy($"id")
val nextmonthclose =lead("Close",20,0).over(particionwindow4)
val MonthDF= DF1.select($"*", nextmonthclose as "Label")
val MonthDFTest= TestDF.select($"*", nextmonthclose as "Label")

//Create DF with four month lead close for training & validation and test
val particionwindow5 = Window.orderBy($"id")
val nextfourmonthclose =lead("Close",80,0).over(particionwindow5)
val FourMonthDF= DF1.select($"*", nextfourmonthclose as "Label")
val FourMonthDFTest = TestDF.select($"*", nextfourmonthclose as "Label")

// Filling Missing (FM) Values with Imputer1 
val Imputer1 = new Imputer()
  .setInputCols(Array("Open","High","Close","Low","Volume","Label"))
  .setOutputCols(Array("Open_FM","High_FM","Close_FM","Low_FM","Volume_FM","Label_FM"))

  
// assemble other columns to features
val assembler = new VectorAssembler()
.setInputCols(Array("Open_FM","High_FM","Close_FM","Low_FM","Volume_FM","Year","Month","Day","Day_of_Week","Quarter","movingAvg5","movingAvg10","movingAvg20","movingAvg50","movingAvg100","movingAvg200"))
//  .setInputCols(Array("Open_FM","High_FM","Close_FM","Low_FM","Volume_FM"))
  .setOutputCol("features")

// Index features
val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
  
// Split the data into training and validation sets (25% held out for validation)
// Change MonthDF to the time horizon of interst
val Array(trainingData, valData) = FourMonthDF.randomSplit(Array(0.75, 0.25),seed =12345)

// Train a RandomForest model.
val rf = new RandomForestRegressor()
  .setLabelCol("Label_FM")
  .setFeaturesCol("indexedFeatures")
  .setSeed(12345) 
  .setMaxDepth(30)
  .setMaxBins(1024)
  .setNumTrees(20)
//  .setMinInstancesPerNode()
//  .setSubsamplingRate()

// Chain indexer and forest in a Pipeline.
val pipeline = new Pipeline()
  .setStages(Array(Imputer1,assembler,featureIndexer,rf))

val evaluator = new RegressionEvaluator()
  .setLabelCol("Label_FM")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

// Train model. This also runs the indexer.
val model = pipeline.fit(trainingData)

// Make predictions on validation set
val predictions = model.transform(valData)

// Select (prediction, true label) and compute test error.
val rmse = evaluator.evaluate(predictions);
println("Root Mean Squared Error (RMSE) on validation data = " + rmse)
// report the trees and store in rfModel
val rfModel = model.stages(3).asInstanceOf[RandomForestRegressionModel].featureImportances;

//convert prediction and real value to array of double
val forec = predictions.select("prediction").collect().map(_(0).toString.toDouble);
val actual = predictions.select("Label_FM").collect().map(_(0).toString.toDouble);
val smape_val = SMAPE(actual,forec);
println("sMAPE on validation data = " + smape_val)

//-----------Make predictions on TEST set----------------------------------------
// Change MonthDF to the time horizon of interst
val predictions2 = model.transform(FourMonthDFTest)

// Select prediction, true label and compute test error.
val evaluator2 = new RegressionEvaluator()
  .setLabelCol("Label_FM")
  .setPredictionCol("prediction")
  .setMetricName("rmse")
val rmse2 = evaluator2.evaluate(predictions2);
println("Root Mean Squared Error (RMSE) on test data = " + rmse2)
val rfModel2 = model.stages(3).asInstanceOf[RandomForestRegressionModel];
//println("Learned regression forest model:\n" + rfModel.toDebugString)

//convert prediction and real value to array of double
val forec2 = predictions2.select("prediction").collect().map(_(0).toString.toDouble);
val actual2 = predictions2.select("Label_FM").collect().map(_(0).toString.toDouble);
// Calculate SMAPE
val smape_val2 = SMAPE(actual2,forec2);
println("sMAPE on test data = " + smape_val2)



// COMMAND ----------

// Commented because it take at least an hour to run.
//// define the paramter search for best maxBins between 128 to 1024
//val paramGrid = new ParamGridBuilder()
//  .addGrid(rf.maxBins, Array(128,256,512,1024))
//  .build()

// create cross validator with 3 fold
//val cv = new CrossValidator()
//  .setEstimator(pipeline)
//  .setEvaluator(evaluator)
//  .setEstimatorParamMaps(paramGrid)
//  .setNumFolds(3)

// Train a CV model. 
// val cvModel = cv.fit(MonthDF)

//-----------Make predictions on TEST set using k-fold validation----------------------------------------
//val cvpredictions = cvModel.transform(MonthDFTest)
//// Calculate Cross validation RMSE
//val cvrmse = evaluator.evaluate(cvpredictions);
//println("Root Mean Squared Error (RMSE) on validation data using k-fold validation = " + cvrmse)

////convert prediction and real value to array of double
//val cvforec = cvpredictions.select("prediction").collect().map(_(0).toString.toDouble);
//val cvactual = cvpredictions.select("Label_FM").collect().map(_(0).toString.toDouble);
// Calculate SMAPE
//val cvsmape_val = SMAPE(cvactual,cvforec);
//println("sMAPE on test data using k-fold validation = " + cvsmape_val)



// COMMAND ----------

val rfModel = model.stages(3).asInstanceOf[RandomForestRegressionModel].featureImportances;
println(rfModel)

// COMMAND ----------

max(predictions2("prediction"))
