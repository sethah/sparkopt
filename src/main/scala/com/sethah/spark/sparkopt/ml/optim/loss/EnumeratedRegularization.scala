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
package com.sethah.spark.sparkopt.ml.optim.loss

import org.apache.spark.ml.linalg._


trait EnumeratedRegularization[T]
  extends (T => Double) with Serializable {

  def regFunc: Int => Double

  def apply(x: T): Double

}

class L1Regularization(override val regFunc: Int => Double)
  extends EnumeratedRegularization[Vector] {

  override def apply(x: Vector): Double = {
    var sum = 0.0
    x.foreachActive((index, value) => sum += math.abs(value) * regFunc(index))
    sum
  }

}

class L2Regularization(override val regFunc: Int => Double)
  extends EnumeratedRegularization[Vector] with DiffFun[Vector] {

  override def doCompute(coefficients: Vector): (Double, Vector) = {
    val grad = Vectors.zeros(coefficients.size)
    val loss = doComputeInPlace(coefficients, grad)
    (loss, grad)
  }

  override def doComputeInPlace(coefficients: Vector, grad: Vector): Double = {
    val gradArray = grad.toArray
    coefficients match {
      case dv: DenseVector =>
        var sum = 0.0
        dv.values.indices.foreach { j =>
          val coef = coefficients(j)
          val reg = regFunc(j)
          sum += reg * coef * coef
          gradArray(j) += reg * coef
        }
        0.5 * sum
      case _: SparseVector =>
        throw new IllegalArgumentException("Sparse coefficients are not currently supported.")
    }
  }
}
