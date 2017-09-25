package org.apache.spark.ml

import org.apache.spark.ml.linalg._
import org.apache.spark.ml.feature.Instance

object InstanceWrapper {

  type Instance = org.apache.spark.ml.feature.Instance

  def create(label: Double, weight: Double, features: Vector): Instance = {
    Instance(label, weight, features)
  }

}
