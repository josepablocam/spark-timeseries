/**
 * Copyright (c) 2015, Cloudera, Inc. All Rights Reserved.
 *
 * Cloudera, Inc. licenses this file to you under the Apache License,
 * Version 2.0 (the "License"). You may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * This software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for
 * the specific language governing permissions and limitations under the
 * License.
 */

package com.cloudera.finance.examples

import breeze.linalg._
import com.cloudera.finance.YahooParser
import com.cloudera.sparkts.DateTimeIndex._
import com.cloudera.sparkts.{TimeSeriesKryoRegistrator, EasyPlot, TimeSeries}
import com.cloudera.sparkts.UnivariateTimeSeries._
import com.cloudera.sparkts.TimeSeriesRDD._
import com.cloudera.sparkts.TimeSeriesStatisticalTests._

import com.github.nscala_time.time.Imports._
import org.apache.commons.math3.analysis.MultivariateFunction
import org.apache.commons.math3.optim.nonlinear.scalar.{ObjectiveFunction, GoalType}
import org.apache.commons.math3.optim.univariate.BrentOptimizer

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Matrix => SparkMatrix, Vector => SparkVector}


import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression

import scala.util.{Failure, Success, Try}
import scala.collection.Iterable


object PortfolioOptimizationExample {
  def main(args: Array[String]): Unit = {
    val inputDir = "/Users/josecambronero/Projects/spark-timeseries/src/main/resources/data"

    val conf = new SparkConf().setMaster("local").setAppName("portoptim")
    TimeSeriesKryoRegistrator.registerKryoClasses(conf)
    val sc = new SparkContext(conf)

    // Load and parse the data, you might
    // want to make sure you deleted all files with 0 data (wget still creates a file
    // for them), from command line: find data/ -size 0 | xargs -I {} rm -rf {}
    val seriesByFile: RDD[TimeSeries] = YahooParser.yahooFiles(inputDir, sc)


    // Merge the series from individual files into a TimeSeriesRDD and just take closes
    val start = seriesByFile.map(_.index.first).takeOrdered(1).head
    val end = seriesByFile.map(_.index.last).top(1).head
    val dtIndex = uniform(start, end, 1.businessDays)
    val tsRdd = timeSeriesRDD(dtIndex, seriesByFile).filter(_._1.endsWith("csvClose"))
    println(s"Num time series: ${tsRdd.count()}")

    // Only look at close prices during 2015
    val startDate = nextBusinessDay(new DateTime("2015-1-1"))
    val endDate = nextBusinessDay(new DateTime("2015-6-6"))
    val recentRdd = tsRdd.slice(startDate, endDate)

    // take only part of the universe
    val nAssets = 200
    val reducedRdd = recentRdd.takeNSeries(nAssets)

    // Impute missing data with spline interpolation
    // fill forward and then backward for any remaining missing values
    // anything that still has NaNs we'll just drop...
    val filledRdd = reducedRdd.
      fill("linear").
      fill("previous").
      fill("next").
      filter(x => !x._2.toArray.exists(_.isNaN))


    // Calculate returns and fill first value with nearest....
    // Note that in the real world we would adjust prices for splits etc before
    // calculating returns
    val returnRdd = filledRdd.price2ret()

    // Convert returns into a matrix to distribute covariance calculation and average
    // returns calculation. We use toRowMatrix as we have no plans on directly
    // indexing into our structure

    val retMatrixRdd = returnRdd.toRowMatrix()
    val avgReturnsPerAsset = sparkVectortoBreeze(retMatrixRdd.computeColumnSummaryStatistics().mean)


    // let's plot out the average returns
    EasyPlot.ezplot(avgReturnsPerAsset)
    /// Woah! one of these guys spiked...Not reasonable
    val List(cIx) = which(avgReturnsPerAsset.toArray, (x: Double) => x == max(avgReturnsPerAsset))

    val culpritTicker = returnRdd.keys.collect()(cIx)
    // Let's check out prices without any of our imputation
    val culpritPrices = recentRdd.lookup(culpritTicker).head
    // Clearly there was some kind of jump in these prices, let's check the date of the largest
    // increase
    EasyPlot.ezplot(culpritPrices)
    val culpritPxDeltas = diff(culpritPrices.toDenseVector).toArray
    val List(dateIxMinus1) = which(culpritPxDeltas, (x: Double) => x == culpritPxDeltas.max)
    val culpritDate = recentRdd.index.dateTimeAtLoc(dateIxMinus1 + 1)
    // If we look up this date, we can see that
    // http://ir.amsc.com/releasedetail.cfm?ReleaseID=903185
    // AMSC had a reverse stock split that date: where the total # of stock goes down
    // and the prices increase s.t. the market capitalization of the company remains the same
    // in the real world we would adjust for this kind of things....in blog world
    // we're just gonna go ahead and drop this stock

    val cleanReturnRdd = filledRdd.filter(_._1 != culpritTicker).price2ret()
    val cleanRetMatrixRdd = cleanReturnRdd.toRowMatrix()
    val cleanAvgAssetRet = sparkVectortoBreeze(cleanRetMatrixRdd.computeColumnSummaryStatistics().mean)
    val covMatrix = sparkMatrixtoBreeze(cleanRetMatrixRdd.computeCovariance())

    // distributed
    val frontierPoints = markowitzFrontier(sc, covMatrix, cleanAvgAssetRet, 1000)
    // graph it
    EasyPlot.ezplot(frontierPoints.map(_._2), frontierPoints.map(_._1), '.')

    // find the portfolio with the minimum variance and show ER and allocations
    val (er, mr, w) = markowitzOptimal(
      covMatrix,
      cleanAvgAssetRet,
      frontierPoints.map(x=> (x._1, x._2))
    )

  }


  // http://www.maths.usyd.edu.au/u/alpapani/riskManagement/lecture4.pdf
  // returns expected return, variance, and weights to achieve that variance
  def markowitzSolveFrontierPoint(
    cov: DenseMatrix[Double],
    invertedMatrix: DenseMatrix[Double],
    eR: Double): (Double, Double, DenseVector[Double]) = {
    // find optimal portfolio given expected return
    // add 2 for the lagrangian related variables
    val n = invertedMatrix.rows
    val rhs = DenseMatrix.zeros[Double](n + 2, 1)
    rhs(-2, 0) = 0.0
    rhs(-1, 0) = eR
    // explicitly state type...since it seems intellij not picking it up despite compilation
    val solution: DenseMatrix[Double] = invertedMatrix * rhs
    val weights = solution(0 to -3, ::) // drop last 2 elements
    val variance = risk(cov, weights)
    (eR, variance, weights.toDenseVector)
  }

  def assembleLinearSystem(
    cov: DenseMatrix[Double],
    returns: Vector[Double]): DenseMatrix[Double] = {
    val n = returns.length
    val onesAndReturns = new DenseMatrix(n, 2, Array.fill(n)(1.0) ++ returns.toArray)
    val zeros = DenseMatrix.zeros[Double](2, 2)
    DenseMatrix.vertcat(
     DenseMatrix.horzcat(cov, onesAndReturns * -1.0),
     DenseMatrix.horzcat(onesAndReturns.t, zeros)
   )
  }

  def safeInversion(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    Try(inv(m)) match {
      case Success(im) => im
      case Failure(e) => pinv(m)
    }
  }

  def markowitzFrontier(
      sc: SparkContext,
      cov: DenseMatrix[Double],
      returns: DenseVector[Double],
      n: Int): Array[(Double, Double, DenseVector[Double])] = {
    val expectedReturns = sc.parallelize(markowitzPolyPoints(n))
    val lhs = assembleLinearSystem(cov, returns)
    val invertedLHS = safeInversion(lhs)
    expectedReturns.map { r =>
      markowitzSolveFrontierPoint(cov, invertedLHS, r)
    }.collect()
  }

  def markowitzOptimal(
      cov: DenseMatrix[Double],
      returns: DenseVector[Double],
      obs: Array[(Double, Double)]
    ): (Double, Double, DenseVector[Double]) = {
    // we'll fit a polynomial of degree 2 to the curve
    // fit points, get maximum expected return, and calculate weights
    // return all 3
    val curveFitter = new OLSMultipleLinearRegression()
    val X = obs.map(x => Array(x._1, x._1 * x._1))
    val y = obs.map(_._2)
    curveFitter.newSampleData(y, X)
    val coeffs = curveFitter.estimateRegressionParameters()
    // For now, we are fitting a simple parabola, so just solve
    val optimalReturn = -1 * coeffs(1) / (2 * coeffs(2))

//    val frontier = new ObjectiveFunction(new MultivariateFunction {
//      def value(params: Array[Double]): Double = {
//        coeffs.zipWithIndex.map { case (b, p) => b * math.pow(params(0), p) }.sum
//      }
//    })
//
//    val optimizer = new BrentOptimizer(1e-6, 1e-6)
//    val goal = GoalType.MINIMIZE
//    val optimal = optimizer.optimize(frontier, goal)
//    val optimalReturn = optimal.getPoint
    markowitzSolveFrontierPoint(
      cov,
      safeInversion(assembleLinearSystem(cov, returns)),
      optimalReturn
    )
  }

  // We follow the approach at http://blog.quantopian.com/markowitz-portfolio-optimization-2/
  // And show how to
  def markowitzPolyPoints(n: Int): Array[Double] = {
    (0 to n).map(x => math.pow(10.0, 5 * x / n.toDouble - 1.0)).toArray
  }

  // http://www.mif.vu.lt/portfolio/files/optimisation1.pdf
//  def greedyHeuristic(cov: DenseMatrix[Double], returns: Vector[Double], portSize: Int):
//  Vector[Double] = {
//    val weights = new DenseVector(Array.fill(returns.length)(0))
//    val credits = 1 to portSize
//    for (credit <- credits) {
//      /// add 1 to each weight and recalc the sharpeRatio, whichever one
//      // increased the ratio by most, wins and adds 1 to ratio
//    }
//
//    weights :/ portSize
//  }

  def risk(cov: DenseMatrix[Double], weights: DenseMatrix[Double]): Double = {
    // writing out types to avoid having intellij incorrectly highlight issues
    val wTC: DenseMatrix[Double] = weights.t * cov
    val wTCw: DenseMatrix[Double] = wTC * weights
    wTCw(0, 0)
  }

  def ret(perAssetExpectedReturns: Vector[Double], weights: Vector[Double]): Double = {
    perAssetExpectedReturns.dot(weights)
  }

//  def sharpeRatio(
//    cov: DenseMatrix[Double],
//    perAssetExpectedReturns: DenseVector[Double],
//    weights: DenseVector[Double]): Double = {
//    ret(perAssetExpectedReturns, weights) / risk(cov, weights.toDenseMatrix.t)
//  }

  def sparkMatrixtoBreeze(m: SparkMatrix): DenseMatrix[Double] = {
    val data = m.toArray
    new DenseMatrix(m.numRows, m.numCols, data)
  }

  def sparkVectortoBreeze(v: SparkVector): DenseVector[Double] = {
    new DenseVector(v.toArray)
  }

  def which[A, T](c: A, f: T => Boolean)(implicit ev: A => Iterable[T]): List[Int] = {
    c.zipWithIndex.filter(x => f(x._1)).map(_._2).toList
  }
}



