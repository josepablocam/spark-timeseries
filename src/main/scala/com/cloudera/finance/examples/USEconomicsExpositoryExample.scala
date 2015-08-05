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

import org.joda.time.LocalDate

object USEconomicsExpositoryExample {
  // Illustrates the idea behind using ARIMA, we build upon this conceptually in
  // by creating a simplistic regression and modeling error terms as ARIMA
  def main(args: Array[String]): Unit = {
    // Dummy hardcoded paths for now...
    val DATAPATH = "/Users/josecambronero/Documents/"
    val SAVEPATH = "/Users/josecambronero/Projects/summer_presentation/"

    val dateSorter = (x: (LocalDate, Double), y: (LocalDate, Double)) => x._1.compareTo(y._1) <= 0
    val claims = loadCSV(DATAPATH + "claims.csv").sortWith(dateSorter)
    val sp500 = loadCSV(DATAPATH + "sp500Closes.csv").sortWith(dateSorter)
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

    val bothTSPlot = ezplot(
      Seq(justClaims, justSP),
      '-', Array("Claims YoY Chg.", "SP500 YoY Chg.")
    )
    bothTSPlot.subplot(0).legend_=(true)
    bothTSPlot.saveas(SAVEPATH + "claims_and_sp500.png")

    val simpleRegression = new OLSMultipleLinearRegression()
    simpleRegression.newSampleData(justSP.toArray, justClaims.toArray.map(Array(_)))

    println(s"adjR^2:${simpleRegression.calculateAdjustedRSquared()}")

    val residuals = new DenseVector(simpleRegression.estimateResiduals())

    val changingVarianceResults = bptest(
      residuals,
      new DenseMatrix(justClaims.length, 1,  justClaims.toArray)
    )
    println("Breusch-Pagan results: " + changingVarianceResults)

    val serialCorrResults = bgtest(
      residuals,
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

    val residualPlot = ezplot(residuals, '-')
    residualPlot.saveas(SAVEPATH + "simple_residuals.png")

    val residualACFPlot = acfPlot(residuals.toArray, 10)
    residualACFPlot.saveas(SAVEPATH + "simple_residuals_ACF.png")

    val diffedResidualACFPlot = acfPlot(differencesOfOrderD(residuals,1).toArray, 10)
    val diffedResidualPACFPlot = pacfPlot(differencesOfOrderD(residuals,1).toArray, 10)
    diffedResidualACFPlot.saveas(SAVEPATH + "diffed_residuals_ACF.png")
    diffedResidualPACFPlot.saveas(SAVEPATH + "diffed_residuals_PACF.png")


    // let's model our residuals as ARIMA (we'll see if the heteroskedasticity is an issue)
    val errorModel = ARIMA.fitModel(1, 1, 1, residuals)
    val forecasted = errorModel.forecast(residuals, 240)
    println("Model coefficients:" + errorModel.coefficients.mkString(","))


    val extendedResiduals = DenseVector.vertcat(residuals,
      new DenseVector(Array.fill(240)(Double.NaN))
    )
    val forecastedPlot = ezplot(Seq(extendedResiduals , forecasted), '-')
    forecastedPlot.saveas(SAVEPATH + "forecasted_ARIMA.png")

    // Transform our original regression using 1 iteration of Cochrane-Orcutt
    val transSP = errorModel.removeTimeDependentEffects(justSP, justSP.copy)
    val transClaims = errorModel.removeTimeDependentEffects(justClaims, justClaims.copy)

    val transModel = new OLSMultipleLinearRegression()
    transModel.newSampleData(transSP.toArray.drop(2), transClaims.toArray.drop(2).map(Array(_)))

    println(s"Transformed Model Parameters:${
      transModel.estimateRegressionParameters().mkString(",")
    }")

    val transResiduals = transModel.estimateResiduals()
    val transResidualPlot = ezplot(transResiduals, '-')
    transResidualPlot.saveas(SAVEPATH + "trans_residuals.png")
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
}
