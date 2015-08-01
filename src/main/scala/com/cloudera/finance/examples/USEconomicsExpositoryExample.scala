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
import com.cloudera.sparkts.ARIMA
import com.cloudera.sparkts.EasyPlot._

import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression

import breeze.linalg._

import org.apache.spark.{SparkConf, SparkContext}
import org.joda.time.LocalDate


object USEconomicsExpositoryExample {
  // Illustrates the idea behind using ARIMA, we build upon this conceptually in
  // the USEconomicsExample, where we predict various economic indicators
  // as a function of a variety of regressors and then model remaining errors
  // as ARIMA
  def main(args: Array[String]): Unit = {

    val claims = loadCSV("/Users/josecambronero/Documents/claims.csv")
    val sp500Opens = loadCSV("/Users/josecambronero/Documents/sp500Opens.csv").sortWith {
      (x, y) => x._1.compareTo(y._1) <= 0
    }.toMap

    val sp500Closes = loadCSV("/Users/josecambronero/Documents/sp500Closes.csv").sortWith {
      (x, y) => x._1.compareTo(y._1) <= 0
    }.toMap

    // difference between the open on and the prior day's close
    val sp500Changes = sp500Opens.map{ case (dt, open) =>
      val close = sp500Closes.getOrElse(dt.minusDays(1), Double.NaN)
      (dt, -1 + open/close)
    }

    // The dates reported for claims are the "week-ending" date for that period
    // we want release dates. They are released the thursday of the coming week, so
    // we add 5 days to each date
    val dayOffset = 5
    val claimReleases = claims.map { case (dt, v) => (dt.plusDays(dayOffset), v) }
    val claimChanges = claimReleases.zip(claimReleases.drop(1)).map{ case (prior, current) =>
      (current._1, -1 + current._2 / prior._2)
    }
    // look up the % change between open and close price for SP500 on the day of the release
    // for dates that aren't found in SP (e.g. release was on holiday, so no market open)
    // we simply exclude...very small portion of observations
    val dataOnRelease = claimChanges.map { case (dt, claimsDelta) =>
      val spDelta = sp500Changes.getOrElse(dt, Double.NaN)
      (dt, claimsDelta, spDelta)
    }.filter(!_._3.isNaN).filter(_._1.getYear >= 2014)

    val justClaims = new DenseVector(dataOnRelease.map(_._2))
    val justSP =  new DenseVector(dataOnRelease.map(_._3))

    ezplot(Seq(justClaims, justSP))
    ezplot(justClaims)
    ezplot(justSP)

    // Simple regression log(unemployment)_t = log(claims/1e5)_{t-1} + error
    val regression = new OLSMultipleLinearRegression()
    regression.newSampleData(justSP.toArray, justClaims.toArray.map(Array(_)))

    println(s"adjR^2:${regression.calculateAdjustedRSquared()}")


    // claims is weekly, but unemployment is monthly
    // We will look up the unemployment number associated with the first
    // day of that month, and then fill in with NAs repeated dates, and interpolate
    val startOfMonth = claims.map { case (dt, claim) =>
      new LocalDate(dt.getYear, dt.getMonthOfYear, 1)
    }
    val unEmploymentMap = unemployment.clone().toMap
    val unemployExtended = startOfMonth.tail.scanLeft((startOfMonth.head, false)) {
      case ((priorDt, b), dt) => (dt, priorDt == dt)
    }.map{ case (dt, repeat) =>
      if (repeat) Double.NaN else unEmploymentMap.getOrElse(dt, Double.NaN)
    }

    // interpolate with cubic spline
    val interpUnemploy = UnivariateTimeSeries.fillSpline(unemployExtended)

    //apply logs to both
    val claimsLog = claims.map(_._2 / 1e5) //claims.map(x => math.log(x._2 / 1e5))
    val unEmployLog = interpUnemploy //interpUnemploy.map(math.log)

    // Lag claims 1 period and remove any missing information
    //val data = claimsLog.zip(unEmployLog.drop(1)).filter { case (x, y) => !x.isNaN && ! y.isNaN }
    val data = claimsLog.zip(unEmployLog).filter { case (x, y) => !x.isNaN && ! y.isNaN }
    val y = data.map(_._2)
    val x = data.map(_._1)

    // Simple regression log(unemployment)_t = log(claims/1e5)_{t-1} + error
    //val regression = new OLSMultipleLinearRegression()
    //regression.newSampleData(y, x.map(Array(_)))

    //println(s"adjR^2:${regression.calculateAdjustedRSquared()}")

    //val errors = regression.estimateResiduals()
    // clearly not white noise. Our parameter estimates are thus inefficent
   // ezplot(errors)

    // visualize PACF and ACF
    acfPlot(errors, 10)
    pacfPlot(errors, 10)

    // ACF plot shows that series is likely to be unit root, non-stationary, we should try
    // differencing
    val diffedY = differenced(y, 2)
    val diffedX = differenced(x, 2)


    val diffedRegression = new OLSMultipleLinearRegression()
    diffedRegression.newSampleData(diffedY, diffedX.map(Array(_)))
    val diffedErrors = diffedRegression.estimateResiduals()
    acfPlot(diffedErrors, 10) // much better, but clearly there is serial correlation
    pacfPlot(diffedErrors, 10)

    val errorModel = ARIMA.fitModel((2, 1, 2), new DenseVector(diffedErrors), method = "css-CGD")
    val n = diffedY.length
    val adjustedY = errorModel.removeTimeDependentEffects(
      new DenseVector(diffedY),
      new DenseVector(Array.fill(n)(0.0))
    )

    val adjustedX = errorModel.removeTimeDependentEffects(
      new DenseVector(diffedX),
      new DenseVector(Array.fill(n)(0.0))
    )

    // run new regression: this procedure is known as Cochrane-Orcutt... it's the
    // "poor" man's ARIMAX
    val transRegression = new OLSMultipleLinearRegression()
    transRegression.newSampleData(adjustedY.toArray, adjustedX.toArray.map(Array(_)))
    val newErrors = transRegression.estimateResiduals()

    ezplot(newErrors)

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
