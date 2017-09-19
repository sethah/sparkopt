package com.sethah.spark.sparkopt.ml

import org.apache.spark.ml.InstanceWrapper
import org.apache.spark.ml.linalg._
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.ml.MultiClassSummarizer
import org.apache.spark.ml.InstanceWrapper
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.linalg.{DenseVector => OldDenseVector, Vector => OldVector}

object MLUtils {

  def getClassificationSummaries(
      dataset: Dataset[_],
      labelCol: String = "label",
      featuresCol: String = "features",
      weightCol: String = "",
      aggDepth: Int = 2): (MultivariateOnlineSummarizer, MultiClassSummarizer) = {
    val w = if (weightCol.isEmpty) lit(1.0) else col(weightCol)
    val instances: RDD[InstanceWrapper.tpe] =
      dataset.select(col(labelCol), w, col(featuresCol)).rdd.map {
        case Row(label: Double, weight: Double, features: Vector) =>
          InstanceWrapper.Instance(label, weight, features)
      }
    val seqOp = (c: (MultivariateOnlineSummarizer, MultiClassSummarizer),
                 instance: InstanceWrapper.tpe) =>
      (c._1.add(toMLlibVector(instance.features)), c._2.add(instance.label, instance.weight))

    val combOp = (c1: (MultivariateOnlineSummarizer, MultiClassSummarizer),
                  c2: (MultivariateOnlineSummarizer, MultiClassSummarizer)) =>
      (c1._1.merge(c2._1), c1._2.merge(c2._2))

    instances.treeAggregate(
      new MultivariateOnlineSummarizer, new MultiClassSummarizer
    )(seqOp, combOp, aggDepth)
  }

  def toMLlibVector(x: Vector): OldVector = {
    new OldDenseVector(x.toArray)
  }
}

