package com.sethah.spark.sparkopt

import com.sethah.spark.sparkopt.ml.optim.loss._
import com.sethah.spark.sparkopt.ml.optim.minimizers.{ConsensusADMM, IterativeMinimizer, LBFGS}
import com.sethah.spark.sparkopt.ml.{BaseAlgorithm, MLUtils}
import org.apache.spark.ml.ModelFactory
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.ml.classification.{BinaryLogisticRegressionTrainingSummary, LogisticRegressionModel, LogisticRegression => SparkLogisticRegression}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.regression.{FamilyAndLink, GLMWrapper, LinearRegressionModel, LinearRegression => SparkLinearRegression}
import org.apache.spark.ml.linalg._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.InstanceWrapper

object MyApp {

  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setAppName("test logistic regression").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)
    org.apache.log4j.Logger.getRootLogger.setLevel(org.apache.log4j.Level.WARN)
    try {
      val spark = SparkSession.builder().getOrCreate()
      import spark.sqlContext.implicits._
      val path = "/Users/shendrickson/LogisticRegressionSuite/binary"
      val df = spark.read.parquet(path)

      val ss = new StandardScaler()
        .setInputCol("features")
        .setOutputCol("featuresStd")
        .setWithStd(true)
      val ssModel = ss.fit(df)
      val df2 = ssModel.transform(df)
      val (model, sparkModel) = logisticRegressionExample(df, true, "features")
//      val (model, sparkModel) = linearRegressionExample(df2, true, "featuresStd")
//      val (model, sparkModel) = glrRegressionExample(df2, true, "featuresStd")
//      println(model.evaluate(df2).meanSquaredError, sparkModel.summary.meanSquaredError)

      println(model.coefficients)
      println(model.intercept)
      println(sparkModel.coefficients)
      println(sparkModel.intercept)
      val accuracy = model.transform(df2).filter("label == prediction").count()
      val accuracy2 = sparkModel.transform(df2).filter("label == prediction").count()
      println(accuracy, accuracy2)
    } finally {
      sc.stop()
    }
  }

  def logisticRegressionExample(
      df: Dataset[_],
      fitIntercept: Boolean,
      featuresCol: String = "features"): (LogisticRegressionModel, LogisticRegressionModel) = {
    val uid = "logreg"
    val (summarizer, labelSummarizer) = MLUtils.getClassificationSummaries(df)
    val numFeatures = summarizer.mean.size
    val fitIntercept = true
    val numFeaturesPlusIntercept = if (fitIntercept) numFeatures + 1 else numFeatures
    val initialCoefficients = Vectors.zeros(numFeaturesPlusIntercept)
    val instanceFunc = (inst: InstanceWrapper.tpe) => BinomialLoss(inst, fitIntercept)
    val minimizer = new LBFGS()

    val regParam = 0.1
    val l2Reg = new L2Regularization((idx: Int) =>
      if (idx >= 0 && idx < numFeatures) regParam else 0.0
    )
//    val partitionMinimizer = new LBFGS().setMaxIter(50)
//    val minimizer = new ConsensusADMM(partitionMinimizer)
//      .setMaxIter(50)
    val base = new BaseAlgorithm(uid, instanceFunc)
      .setFeaturesCol("features")
      .setInitialParams(initialCoefficients)
      .setMinimizer(minimizer)
      .setL2Reg(l2Reg)
    val baseModel = base.fit(df)
    val model = ModelFactory.createBinaryLogisticRegression(uid, baseModel, fitIntercept)
    val sparkLogReg = new SparkLogisticRegression()
      .setFeaturesCol(featuresCol)
    .setRegParam(regParam)
    .setStandardization(false)
    val sparkModel = sparkLogReg.fit(df)
    (model, sparkModel)
  }

  def linearRegressionExample(
     df: Dataset[_],
     fitIntercept: Boolean,
     featuresCol: String = "features"): (LinearRegressionModel, LinearRegressionModel) = {
    val uid = "linreg"
    // TODO
    val (summarizer, labelSummarizer) = MLUtils.getClassificationSummaries(df)
    val numFeatures = summarizer.mean.size
    val fitIntercept = true
    val numFeaturesPlusIntercept = if (fitIntercept) numFeatures + 1 else numFeatures
    val initialCoefficients = Vectors.zeros(numFeaturesPlusIntercept)
    val instanceFunc = (inst: InstanceWrapper.tpe) => SquaredLoss(inst, fitIntercept)
    val minimizer = new LBFGS()
//    val partitionMinimizer = new LBFGS().setMaxIter(50)
//    val minimizer = new ConsensusADMM(partitionMinimizer)
//      .setMaxIter(50)
    val base = new BaseAlgorithm(uid, instanceFunc)
      .setInitialParams(initialCoefficients)
      .setMinimizer(minimizer)
    val baseModel = base.fit(df)
    val model = ModelFactory.createLinearRegression(uid, baseModel, fitIntercept)
    val sparkLinReg = new SparkLinearRegression()
      .setSolver("l-bfgs")
      .setFeaturesCol(featuresCol)
    val sparkModel = sparkLinReg.fit(df)
    (model, sparkModel)
  }

  def glrRegressionExample(
      df: Dataset[_],
      fitIntercept: Boolean,
      featuresCol: String = "features"): (LogisticRegressionModel, LogisticRegressionModel) = {
    val uid = "logreg"
    val numFeatures = df.select(featuresCol).first().getAs[Vector](0).size
    val fitIntercept = true
    val numFeaturesPlusIntercept = if (fitIntercept) numFeatures + 1 else numFeatures
    val initialCoefficients = Vectors.zeros(numFeaturesPlusIntercept)
    val family = GLMWrapper.binomial
    val link = GLMWrapper.logit
    val familyAndLink = new FamilyAndLink(family, link)
    val logLike = (y: Double, mu: Double, w: Double) => {
      (y * math.log(mu) + (1 - y) * math.log(1 - mu)) * w
    }
    val instanceFunc = (inst: InstanceWrapper.tpe) => GLMLoss(inst, familyAndLink, fitIntercept,
      logLike)
    val minimizer = new LBFGS()
    val base = new BaseAlgorithm(uid, instanceFunc)
      .setInitialParams(initialCoefficients)
      .setMinimizer(minimizer)
    val baseModel = base.fit(df)
    val model = ModelFactory.createBinaryLogisticRegression(uid, baseModel, fitIntercept)
    val sparkLogReg = new SparkLogisticRegression()
      .setFeaturesCol(featuresCol)
    val sparkModel = sparkLogReg.fit(df)
    (model, sparkModel)
  }

}
