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
package com.sethah.spark.sparkopt.ml.optim

import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

object Implicits {

  trait Aggregable[M[_]] extends Serializable {

    def aggregate[A: ClassTag, B: ClassTag](ma: M[A], b: B)(add: (B, A) => B,
                                                            combine: (B, B) => B): B

  }

  implicit object RDDCanAggregate extends Aggregable[RDD] {
    override def aggregate[A: ClassTag, B: ClassTag](
                                                      ma: RDD[A], b: B)(add: (B, A) => B, combine: (B, B) => B): B = {
      ma.treeAggregate(b)(add, combine)
    }
  }

  implicit object IterableCanAggregate extends Aggregable[Iterable] {
    override def aggregate[A: ClassTag, B: ClassTag](fa: Iterable[A], b: B)(
      add: (B, A) => B, combine: (B, B) => B): B = {
      fa.foldLeft(b)(add)
    }
  }

  implicit object IteratorCanAggregate extends Aggregable[Iterator] {
    override def aggregate[A: ClassTag, B: ClassTag](fa: Iterator[A], b: B)(
      add: (B, A) => B, combine: (B, B) => B): B = {
      fa.foldLeft(b)(add)
    }
  }
}

