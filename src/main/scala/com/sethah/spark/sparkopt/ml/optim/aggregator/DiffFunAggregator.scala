/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.sethah.spark.sparkopt.ml.optim.aggregator

import com.sethah.spark.sparkopt.ml.optim.loss.DiffFun
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.linalg.BLASWrapper.{instance => BLAS}

/**
 * This class provides a way to sum up gradients computed over one or more differentiable functions.
 * @param coef Vector of coefficients used to evaluate each differentiable function.
 */
private[ml] class DiffFunAggregator(coef: Vector) extends Serializable {

  lazy val coefficients: Vector = coef

  protected var weightSum: Double = 0.0
  protected var lossSum: Double = 0.0

  /** The dimension of the gradient array. */
  protected val dim: Int = coefficients.size

  /** Array of gradient values that are mutated when new instances are added to the aggregator. */
  protected lazy val gradientSumArray: Array[Double] = Array.ofDim[Double](dim)

  /** Add a single data point to this aggregator. */
  def add(instance: DiffFun[Vector]): this.type = {
    val loss = instance.computeInPlace(coefficients, Vectors.dense(gradientSumArray))
    weightSum += instance.weight
    lossSum += loss
    this
  }

  /** Merge two aggregators. The `this` object will be modified in place and returned. */
  def merge(other: DiffFunAggregator): this.type = {
    require(dim == other.dim, s"Dimensions mismatch when merging with another " +
      s"${getClass.getSimpleName}. Expecting $dim but got ${other.dim}.")

    if (other.weightSum != 0) {
      weightSum += other.weightSum
      lossSum += other.lossSum

      var i = 0
      val localThisGradientSumArray = this.gradientSumArray
      val localOtherGradientSumArray = other.gradientSumArray
      while (i < dim) {
        localThisGradientSumArray(i) += localOtherGradientSumArray(i)
        i += 1
      }
    }
    this
  }

  /** The current weighted averaged gradient. */
  def gradient: Vector = {
    require(weightSum > 0.0, s"The effective number of instances should be " +
      s"greater than 0.0, but was $weightSum.")
    val result = Vectors.dense(gradientSumArray.clone())
    BLAS.scal(1.0 / weightSum, result)
    result
  }

  /** Weighted count of instances in this aggregator. */
  def weight: Double = weightSum

  /** The current loss value of this aggregator. */
  def loss: Double = {
    require(weightSum > 0.0, s"The effective number of instances should be " +
      s"greater than 0.0, but was $weightSum.")
    lossSum / weightSum
  }

}


