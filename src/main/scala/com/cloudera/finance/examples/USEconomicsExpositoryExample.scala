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

import com.cloudera.sparkts.UnivariateTimeSeries
import com.cloudera.sparkts.UnivariateTimeSeries.differencesOfOrderD
import com.cloudera.sparkts.ARIMA
import com.cloudera.sparkts.EasyPlot._

import org.apache.commons.math3.random.MersenneTwister
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression
import com.cloudera.sparkts.TimeSeriesStatisticalTests._
import breeze.linalg._

import org.apache.spark.{SparkConf, SparkContext}
import org.joda.time.LocalDate

import scala.collection.immutable.ListMap


object USEconomicsExpositoryExample {
  // Illustrates the idea behind using ARIMA, we build upon this conceptually in
  // the USEconomicsExample, where we predict various economic indicators
  // as a function of a variety of regressors and then model remaining errors
  // as ARIMA
  def main(args: Array[String]): Unit = {

    val dateSorter = (x: (LocalDate, Double), y: (LocalDate, Double)) => x._1.compareTo(y._1) <= 0
    val claims = loadCSV("/Users/josecambronero/Documents/claims.csv").sortWith(dateSorter)
    val sp500 = loadCSV("/Users/josecambronero/Documents/sp500Closes.csv").sortWith(dateSorter)
    // Claims are reported with Saturday date, move back to Friday (week-ending on ...)
    val claimsMap = claims.map { case (dt, v) => (dt.minusDays(1), v) }.toMap
    // match up with SP values by looking up
    val matchedClaims = sp500.map { case (dt, sp) => claimsMap.getOrElse(dt, Double.NaN) }
    // interpolate missing values with a spline
    val interpClaims = UnivariateTimeSeries.fillSpline(matchedClaims)

    val data = (
      sp500.map(_._1),
      sp500.map(_._2),
      interpClaims
      ).zipped.toArray

    // roughly YoY change
    val yoyChanges = data.zip(data.drop(240)).map { case ((_, pSP, pC), (dt, sp, c)) =>
      (dt, sp / pSP - 1, c / pC - 1)
    }.filter(!_._3.isNaN)

    // http://www.nber.org/cycles.html states latest part of cycle started june 2009,
    // we advance a year to avoid capturing bounce from trough, which would result in
    // a high correlation between the 2 variable simply for macroeconomic reasons, not
    // any inherent relationship
    val latestCycle = yoyChanges.filter(_._1.compareTo(new LocalDate(2010, 6, 1)) >= 0)

    val justSP = new DenseVector(latestCycle.map(_._2))
    val justClaims = new DenseVector(latestCycle.map(_._3))

    val fig = plotnv(Seq(justClaims, justSP), '-')

    val simpleRegression = new OLSMultipleLinearRegression()
    simpleRegression.newSampleData(justSP.toArray, justClaims.toArray.map(Array(_)))

    println(s"adjR^2:${simpleRegression.calculateAdjustedRSquared()}")


    val residuals = simpleRegression.estimateResiduals()

    val changingVarianceResults = bptest(
      new DenseVector(residuals),
      new DenseMatrix(justClaims.length, 1,  justClaims.toArray)
    )
    println("Breusch-Pagan results: " + changingVarianceResults)

    val serialCorrResults = bgtest(
      new DenseVector(residuals),
      new DenseMatrix(justClaims.length, 1, justClaims.toArray),
      maxLag = 10)
    println("Breusch-Godfrey results: " + serialCorrResults)

    // Compare to results for white noise
    println("Demostrating results if residuals were white noise")
    val rand = new MersenneTwister(10L)
    val whiteNoise = Array.fill(residuals.length)(rand.nextGaussian)

    val refSerialCorrResults = bgtest(
      new DenseVector(whiteNoise),
      new DenseMatrix(justClaims.length, 1, justClaims.toArray),
      maxLag = 10)
    println("Breusch-Godfrey results for WN: " + refSerialCorrResults)

    val residualPlot = plot1(residuals, '-')
    residualPlot.saveas("/Users/josecambronero/Documents/residuals.png")

    acfPlot()

    acfPlot(differencesOfOrderD(new DenseVector(residuals),1).toArray, 10)
    pacfPlot(differencesOfOrderD(new DenseVector(residuals),1).toArray, 10)
    val model =


    // let's model our residuals as ARIMA (we'll see if the heteroskedasticity is an issue)
    val errorVector = new DenseVector(residuals)
    val errorModel = ARIMA.fitModel(1, 1, 1, errorVector)
    val forecasted = errorModel.forecast(errorVector, 100)
    plotnv(Seq(errorVector, forecasted), '-')


    // Transform our original regression
    val transSP = errorModel.removeTimeDependentEffects(justSP, justSP.copy)
    val transClaims = errorModel.removeTimeDependentEffects(justClaims, justClaims.copy)

    val newModel = new OLSMultipleLinearRegression()
    newModel.newSampleData(transSP.toArray.drop(2), transClaims.toArray.drop(2).map(Array(_)))
    println(s"adjR^2:${newModel.calculateAdjustedRSquared()}")



  }
  def loadCSV(file: String): Array[(LocalDate, Double)] = {
    val text = scala.io.Source.fromFile(file).getLines().toArray
    // we skip labels
    text.tail.map { line =>
      val tokens = line.split(",")
      val date = new LocalDate(tokens.head)
      val measures = tokens(1).toDouble
      (date, measures)
    }
  }

  def differenced(x: Array[Double], offset: Int): Array[Double] = {
    val n = x.length
    val newArray = Array.fill(n - offset)(0.0)
    if (offset == 0) {
      x.clone()
    } else {
      var i = 0
      while (i < n - offset) {
        newArray(i) = x(i + offset) - x(i)
        i += 1
      }
      newArray
    }
  }
}
