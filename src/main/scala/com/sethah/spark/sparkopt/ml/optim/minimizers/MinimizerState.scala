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

/**
 * Data structure holding pertinent information about the optimizer state.
 *
 * @tparam T The type of minimized function's domain.
 */
trait MinimizerState[+T] {

  /** The current value of the minimization parameters. */
  def params: T

}

trait IterativeMinimizerState[+T] extends MinimizerState[T] {

  /** The number of completed iterations. */
  def iter: Int

  /** The loss function value at this iteration. */
  def loss: Double

}

/**
 * An minimizer state implementation designed for minimizers that wrap Breeze.
 */
private[ml] case class BreezeWrapperState[+T](
    prev: T,
    params: T,
    iter: Int,
    loss: Double) extends IterativeMinimizerState[T]


trait ConvergenceCriterion[T] {
  def apply(state: MinimizerState[T])
}

