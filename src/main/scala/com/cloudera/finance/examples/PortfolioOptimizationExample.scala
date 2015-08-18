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
import com.cloudera.sparkts.TimeSeriesRDD._

import com.github.nscala_time.time.Imports._

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Matrix => SparkMatrix, Vector => SparkVector, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix

import org.apache.commons.math3.random.MersenneTwister

import scala.util.{Failure, Success, Try}
import scala.collection.Iterable

object PortfolioOptimizationExample {
  def main(args: Array[String]): Unit = {
    val inputDir = "/Users/josecambronero/Projects/spark-timeseries/src/main/resources/data"
    val sc = new SparkContext("local", "portfolioOptim")

    // Let's warm up with some simulated data
    val rand = new MersenneTwister(10L)
    val nRandAssets = 20

    // simulated returns
    val sampledRets = sc.parallelize(
      Array.fill(1000)(Vectors.dense(Array.fill(nRandAssets)(rand.nextGaussian))))
    val sampledRetMatrix = new RowMatrix(sampledRets)
    val avgSampledRets = sparkVectortoBreeze(sampledRetMatrix.computeColumnSummaryStatistics().mean)
    val sampledRetsCov = sparkMatrixtoBreeze(sampledRetMatrix.computeCovariance())

    // random portfolios
    val randPortfolios = Array.fill(1000)(Array.fill(nRandAssets)(math.random)).
      map(p => p.map(_ / p.sum)).
      map(p => new DenseMatrix(nRandAssets, 1, p))

    val randReturns = randPortfolios.map(x => ret(avgSampledRets, x.toDenseVector))
    val randSDs = randPortfolios.map(risk(sampledRetsCov, _))

    EasyPlot.ezplot(x = randSDs.map(math.pow(_, 2)), y = randReturns, '.')

    val randFrontierPoints = markowitzFrontier(sc,
      sampledRetsCov,
      avgSampledRets,
      0.0,
      max(avgSampledRets),
      10000)

    EasyPlot.ezplot(randFrontierPoints.map(_._2), randFrontierPoints.map(_._1), '.')
    val validRandPoints = randFrontierPoints.filter { p =>
      val w = p._3.toArray
      !w.exists(wi => wi < 0)
    }
    validRandPoints.length / randFrontierPoints.length.toDouble
    EasyPlot.ezplot(validRandPoints.map(_._2), validRandPoints.map(_._1), '.')

    // Real data example: daily price data from Yahoo finance
    // Load and parse the data, you might
    // want to make sure you deleted all files with 0 data (wget still creates a file
    // for them), from command line: find data/ -size 0 | xargs -I {} rm -rf {}
    val seriesByFile: RDD[TimeSeries] = YahooParser.yahooFiles(inputDir, sc)

    // Merge the series from individual files into a TimeSeriesRDD and just take closes
    val start = seriesByFile.map(_.index.first).takeOrdered(1).head
    val end = seriesByFile.map(_.index.last).top(1).head
    val dtIndex = uniform(start, end, 1.businessDays)
    val tsRdd = timeSeriesRDD(dtIndex, seriesByFile).filter(_._1.endsWith("csvClose"))

    // Only look at close prices during 2015
    val startDate = nextBusinessDay(new DateTime("2015-1-1"))
    val endDate = nextBusinessDay(new DateTime("2015-6-6"))
    val recentRdd = tsRdd.slice(startDate, endDate)

    // take only part of the universe for our portfolio (first X number of series)
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

    // Calculate returns as change in closing price day-over-day
    // Note that in the real world we would adjust prices for splits etc before
    // calculating returns
    val returnRdd = filledRdd.price2ret()

    // Convert returns into a matrix to distribute covariance calculation and average
    // returns calculation. We use toRowMatrix as we have no plans on directly
    // indexing into our structure
    val retMatrix = returnRdd.toRowMatrix()
    val avgReturnsPerAsset = sparkVectortoBreeze(retMatrix.computeColumnSummaryStatistics().mean)

    // let's plot out the average returns
    EasyPlot.ezplot(avgReturnsPerAsset)

    /// Woah! one of these tickers spiked...Not reasonable
    val List(pos) = which(avgReturnsPerAsset.toArray, (x: Double) => x == max(avgReturnsPerAsset))

    val culpritTicker = returnRdd.keys.collect()(pos)
    // Let's check out prices without any of our imputation
    val culpritPrices = recentRdd.lookup(culpritTicker).head
    EasyPlot.ezplot(culpritPrices)

    // Clearly was some kind of jump in prices, let's check the date of the largest increase
    val culpritPxDeltas = diff(culpritPrices.toDenseVector).toArray
    // Make sure to filter out NaN's before any min/max calcs
    val maxPxJump = culpritPxDeltas.filter(!_.isNaN).max
    val List(dateIxMinus1) = which(culpritPxDeltas, (x: Double) => x == maxPxJump)
    // add 1, since Breeze's diff drops 1 item
    val culpritDate = recentRdd.index.dateTimeAtLoc(dateIxMinus1 + 1)
    // If we look up this date, we can see that according to
    // http://ir.amsc.com/releasedetail.cfm?ReleaseID=903185
    // AMSC had a reverse stock split that date: where the total # of stock goes down
    // and the prices increase s.t. the market capitalization of the company remains the same
    // in the real world we would adjust for this kind of things....in blog world
    // we're just gonna go ahead and drop this stock

    val cleanRdd = filledRdd.filter(_._1 != culpritTicker).price2ret()
    val cleanMatrix = cleanRdd.toRowMatrix()
    val cleanAvgAssetRet = sparkVectortoBreeze(cleanMatrix.computeColumnSummaryStatistics().mean)
    val covMatrix = sparkMatrixtoBreeze(cleanMatrix.computeCovariance())

    val frontierPoints = markowitzFrontier(
        sc,
        covMatrix,
        cleanAvgAssetRet,
        0.0,
        0.04,
        100000
      )
    val frontierReturns = frontierPoints.map(_._1)
    val frontierSDs = frontierPoints.map(_._2)

    // graph it
    EasyPlot.ezplot(x = frontierSDs.map(x => math.pow(x, 2)), y = frontierReturns, '.')
  }


  // Some quick utility functions
  // Calculates the standard deviation of a portfolio (i.e. the risk)
  def risk(cov: DenseMatrix[Double], weights: DenseMatrix[Double]): Double = {
    // writing out types to avoid having intellij incorrectly highlight issues
    val wByCov: DenseMatrix[Double] = weights.t * cov
    val variance: DenseMatrix[Double] = wByCov * weights
    math.sqrt(variance(0, 0))
  }

  // Calculates the expected return of a portfolio
  def ret(perAssetExpectedReturns: Vector[Double], weights: Vector[Double]): Double = {
    perAssetExpectedReturns.dot(weights)
  }

  // Converts a spark local matrix to breeze matrix
  def sparkMatrixtoBreeze(m: SparkMatrix): DenseMatrix[Double] = {
    val data = m.toArray
    new DenseMatrix(m.numRows, m.numCols, data)
  }

  // Converts a spark local vector to a breeze vector
  def sparkVectortoBreeze(v: SparkVector): DenseVector[Double] = {
    new DenseVector(v.toArray)
  }

  // Returns the indices for the elements in a given collection that satisfy the predicate
  def which[A, T](c: A, f: T => Boolean)(implicit ev: A => Iterable[T]): List[Int] = {
    c.zipWithIndex.filter(x => f(x._1)).map(_._2).toList
  }

  // Markowitz-related functions
  /**
   * Solves the set of weights that result in the smallest possible variance for a given
   * expected return level, provided a covariance of returns and expected returns per asset.
   * I.e. this is the solution to:
   * min f(w) = w.t * C * w
   * given w.t * r = target_return
   * w.t * ones = 1
   * Note that the weights returned are NOT constrainted to be >=0 nor < 1. Users are encouraged
   * to filter results as desired (negative weights imply short selling, and weights >1 imply
   * leverage), see [[http://www.norstad.org/finance/portopt1.pdf]] for more information.
   * @param covMatrix a covariance matrix for returns
   * @param invertedMatrix the inverted matrix used to solve the system of equations, we provide
   *                       this as a parameter to avoid inversion for each point we want to solve
   * @param expectedReturn Expected return level to solve for
   * @return a triple of the form (expected return, variance, weights)
   */
  def markowitzSolveFrontierPoint(
    covMatrix: DenseMatrix[Double],
    invertedMatrix: DenseMatrix[Double],
    expectedReturn: Double): (Double, Double, DenseVector[Double]) = {
    val nAssets = invertedMatrix.rows - 2
    /// rhs of linear system
    val rhs = DenseMatrix.zeros[Double](nAssets + 2, 1)
    rhs(-2, 0) = 1.0
    rhs(-1, 0) = expectedReturn
    // explicitly state type...since it seems intellij not picking it up despite compilation
    val solution: DenseMatrix[Double] = invertedMatrix * rhs
    val weights = solution(0 to -3, ::) // drop last 2 elements (values for lagrangian multipliers)
    val sd = risk(covMatrix, weights)
    (expectedReturn, sd, weights.toDenseVector)
  }

  /**
   * Creates the constant part of the LHS of the system of equations to solve for the weights that
   * produce the minimal weights given a set of returns/covariance of returns/equality constraints
   * See [[http://www.maths.usyd.edu.au/u/alpapani/riskManagement/lecture4.pdf]] for more
   * information
   * @param covMatrix Covariance of returns
   * @param returns Expected return for each individual asset
   * @return
   */
  def assembleLinearSystem(
    covMatrix: DenseMatrix[Double],
    returns: DenseVector[Double]): DenseMatrix[Double] = {
    val n = returns.length
    val onesAndReturns = new DenseMatrix(n, 2, Array.fill(n)(1.0) ++ returns.toArray)
    val zeros = DenseMatrix.zeros[Double](2, 2)
    DenseMatrix.vertcat(
     DenseMatrix.horzcat(covMatrix, onesAndReturns * -1.0),
     DenseMatrix.horzcat(onesAndReturns.t, zeros)
   )
  }

  /**
   * Computes the inverse, and if fails, the pseudo-inverse of a matrix
   * For more information see [[http://www.scalanlp.org/api/breeze/index.html#breeze.linalg.pinv$]]
   * @param m
   * @return
   */
  def safeInversion(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    Try(inv(m)) match {
      case Success(im) => im
      case Failure(e) =>
        println("Inversion failed with: " + e.getMessage)
        println("Trying pseudo-inverse")
        pinv(m)
    }
  }

  /**
   * Computes the efficient frontier for a given set of assets along various expected return points
   * @param sc A SparkContext to distribute the linear system solving for each expected return
   * @param covMatrix A covariance matrix for the assets
   * @param returns A vector of expected return for each individual asset
   * @param s The starting expected return for frontier
   * @param e The ending expected return for frontier
   * @param len The number of points between starting and ending points of frontier (uniformly
   *           spaced)
   * @return An array containing triples of the form ( expected return, variance, weight allocation)
   */
  def markowitzFrontier(
      sc: SparkContext,
      covMatrix: DenseMatrix[Double],
      returns: DenseVector[Double],
      s: Double,
      e: Double,
      len: Int): Array[(Double, Double, DenseVector[Double])] = {
    val delta = (e - s) / len
    val lhs = assembleLinearSystem(covMatrix, returns)
    val invertedLHS = safeInversion(lhs)
    val expectedReturns = sc.parallelize((s to e by delta).toArray[Double])
    expectedReturns.map { r => markowitzSolveFrontierPoint(covMatrix, invertedLHS, r) }.collect()
  }
}



