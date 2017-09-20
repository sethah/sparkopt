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
package com.sethah.spark.sparkopt.ml.optim.minimizers

import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.{CachedDiffFunction, DiffFunction, LineSearch, OWLQN => BreezeOWLQN}
import com.sethah.spark.sparkopt.ml.optim.loss.{DiffFun, HasRegularization, L1Regularization}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{BLAS, DenseVector, Vector, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable


trait OWLQNParams extends Params {

  /**
   * Param for maximum number of iterations (&gt;= 0).
   * @group param
   */
  final val maxIter: IntParam = new IntParam(this, "maxIter", "maximum number of iterations (>= 0)",
    ParamValidators.gtEq(0))

  /** @group getParam */
  final def getMaxIter: Int = $(maxIter)

  /**
   * Param for the convergence tolerance for iterative algorithms (&gt;= 0).
   * @group param
   */
  final val tol: DoubleParam = new DoubleParam(this, "tol", "the convergence tolerance for " +
    "iterative algorithms (>= 0)", ParamValidators.gtEq(0))

  /** @group getParam */
  final def getTol: Double = $(tol)


}

class OWLQN(override val uid: String) extends IterativeMinimizer[Vector,
  DiffFun[Vector], BreezeWrapperState[Vector]] with LBFGSParams
  with Logging {

  def this() = this(Identifiable.randomUID("lbfgs"))

  private type State = BreezeWrapperState[Vector]

  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 100)

  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1e-6)

  override def copy(extra: ParamMap): LBFGS = {
    new LBFGS(uid)
  }

  def initialState(lossFunction: DiffFun[Vector], initialParams: Vector): State = {
    val (firstLoss, _) = lossFunction.compute(initialParams)
    BreezeWrapperState(Vectors.dense(Array.fill(initialParams.size)(Double.MaxValue)),
      initialParams, 0, firstLoss)
  }

  override def iterations(lossFunction: DiffFun[Vector],
                          initialParameters: Vector): Iterator[State] = {
    val start = initialState(lossFunction, initialParameters)
    val breezeLoss = new DiffFunction[BDV[Double]] {
      override def valueAt(x: BDV[Double]): Double = {
        lossFunction.apply(new DenseVector(x.data))
      }
      override def gradientAt(x: BDV[Double]): BDV[Double] = {
        val sv = lossFunction.grad(new DenseVector(x.data))
        new BDV[Double](sv.toArray)
      }
      override def calculate(x: BDV[Double]): (Double, BDV[Double]) = {
        val (f, grad) = lossFunction.compute(new DenseVector(x.data))
        (f, new BDV[Double](grad.toArray))
      }
    }

    // TODO: need a better way
    val l1Fun = lossFunction match {
      case reg: HasRegularization[Vector] =>
        reg.regularizers.filter(_.isInstanceOf[L1Regularization])
          .map(_.asInstanceOf[L1Regularization]) match {
          case l1 :: _ => l1.regFunc
          case Nil => (_: Int) => 0.0
        }
      case _ => (_: Int) => 0.0
    }
    val breezeOptimizer = new BreezeOWLQN[Int, BDV[Double]](getMaxIter, 10, l1Fun, getTol)
    breezeOptimizer
      .iterations(breezeLoss, new BDV[Double](start.params.toArray))
      .map { bstate =>
        BreezeWrapperState(Vectors.zeros(initialParameters.size),
          new DenseVector(bstate.x.data), bstate.iter + 1, bstate.adjustedValue)
      }
  }
}

