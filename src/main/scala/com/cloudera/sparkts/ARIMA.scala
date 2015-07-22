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

package com.cloudera.sparkts

import breeze.linalg._

import org.apache.commons.math3.analysis.MultivariateFunction
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression
import org.apache.commons.math3.optim.{SimpleBounds, MaxEval, MaxIter, InitialGuess}
import org.apache.commons.math3.optim.nonlinear.scalar.{GoalType, ObjectiveFunction}
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer
import org.apache.commons.math.random.MersenneTwister



object ARIMA {
  /**
   * Fits an a non seasonal ARIMA model to the given time series.
   * http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/tutorials/xegbohtmlnode39.html
   */
  def fitModel(
      order:(Int, Int, Int),
      ts: Vector[Double],
      includeIntercept: Boolean = true,
      method: String = "css")
    : ARIMAModel = {
    val (p, d, q) = order
    val diffedTs = differences(ts, d).toArray.drop(d)

    if (p > 0 && q == 0) {
      val arModel = Autoregression.fitModel(new DenseVector(diffedTs), p)
      return new ARIMAModel(order, includeIntercept, Array(arModel.c) ++ arModel.coefficients)
    }

    val initParams = HannanRisannenInit(diffedTs, order, includeIntercept)
    val initCoeffs = fitWithCSS(diffedTs, order, includeIntercept, initParams)

    method match {
      case "css" => new ARIMAModel(order, includeIntercept, initCoeffs)
      case "ml" => throw new UnsupportedOperationException()
    }
  }

  /**
   * Fit an ARIMA model using conditional sum of squares, currently optimizes using unbounded
   * BOBYQA.
   * @param diffedY time series we wish to fit to, already differenced if necessary
   * @param includeIntercept does the model include an intercept
   * @param initParams initial parameter guesses
   * @return
   */
  private def fitWithCSS(
      diffedY: Array[Double],
      order: (Int, Int, Int),
      includeIntercept: Boolean,
      initParams: Array[Double]
      )
    : Array[Double]= {

    // We set up starting/ending trust radius using default suggested in
    // http://cran.r-project.org/web/packages/minqa/minqa.pdf
    // While # of interpolation points as mentioned common in
    // Source: http://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf
    val (p, d, q) = order
    val radiusStart = math.min(0.96, 0.2 * initParams.map(math.abs).max)
    val radiusEnd = radiusStart * 1e-6
    val dimension = p + q + (if (includeIntercept) 1 else 0)
    val interpPoints = dimension * 2 + 1

    val optimizer = new BOBYQAOptimizer(interpPoints, radiusStart, radiusEnd)
    val objFunction = new ObjectiveFunction(new MultivariateFunction() {
      def value(params: Array[Double]): Double = {
        new ARIMAModel(order, includeIntercept, params).logLikelihoodCSSARMA(diffedY)
      }
    })

    val initialGuess = new InitialGuess(initParams)
    val maxIter = new MaxIter(10000)
    val maxEval = new MaxEval(10000)
    // TODO: Enforce stationarity and invertibility for AR and MA terms
    val bounds = SimpleBounds.unbounded(dimension)
    val goal = GoalType.MAXIMIZE
    val optimal = optimizer.optimize(objFunction, goal, bounds, maxIter, maxEval,
      initialGuess)
    optimal.getPoint
  }

  // TODO: implement MLE parameter estimates with Kalman filter.
  private def fitWithML(diffedY: Array[Double],
      arTerms: Array[Array[Double]],
      maTerms: Array[Double],
      initCoeffs: Array[Double])
    : Array[Double] = {
    throw new UnsupportedOperationException()
  }

  /**
   * initialize ARMA estimates using the Hannan Risannen algorithm
   * Source: http://personal-homepages.mis.mpg.de/olbrich/script_chapter2.pdf
   */
  private def HannanRisannenInit(
      diffedY: Array[Double],
      order:(Int, Int, Int),
      includeIntercept: Boolean)
    : Array[Double] = {
    val (p, d, q) = order
    val addToLag = 1
    val m = math.max(p, q) + addToLag // m > max(p, q)
    // higher order AR(m) model
    val arModel = Autoregression.fitModel(new DenseVector(diffedY), m)
    val arTerms1 = Lag.lagMatTrimBoth(diffedY, m, false)
    val yTrunc = diffedY.drop(m)
    val estimated = arTerms1.zip(
      Array.fill(yTrunc.length)(arModel.coefficients)
      ).map { case (v, b) => v.zip(b).map { case (yi, bi) => yi * bi}.sum + arModel.c }
    // errors estimated from AR(m)
    val errors = yTrunc.zip(estimated).map { case (y, yhat) => y - yhat }
    // secondary regression, regresses X_t on AR and MA terms
    val arTerms2 = Lag.lagMatTrimBoth(yTrunc, p, false).drop(math.max(q - p, 0))
    val errorTerms = Lag.lagMatTrimBoth(errors, q, false).drop(math.max(p - q, 0))
    val allTerms = arTerms2.zip(errorTerms).map { case (ar, ma) => ar ++ ma }
    val regression = new OLSMultipleLinearRegression()
    regression.setNoIntercept(!includeIntercept)
    regression.newSampleData(yTrunc.drop(m - addToLag), allTerms)
    val params = regression.estimateRegressionParameters()
    params
  }

  /**
   * Calculate a differenced array of a given order
   * @param ts Array of doubles to difference
   * @param order The difference order (e.g. x means y(0) = ts(x) - ts(0), etc)
   * @return A differenced array, where the first `order` elements are differenced with 0 (and
   *         thus are just their original values)
   */
  def differences(ts: Vector[Double], order: Int): Vector[Double] = {
    diffOps(ts, order, _ - _)
  }

  def invDifferences(ts: Vector[Double], order: Int): Vector[Double] = {
    diffOps(ts, order, _ + _)
  }

  def diffOps(ts: Vector[Double], order: Int, op: (Double, Double) => Double): Vector[Double] = {
    if (order == 0) {
      ts
    } else {
      val n = ts.length
      val diffedTs = new DenseVector(Array.fill(n)(0.0))
      var i = 0

      while (i < n) {
        // elements prior to `order` are copied over without modification
        diffedTs(i) = if (i < order) ts(i) else op(ts(i), ts(i - order))
        i += 1
      }
      diffedTs
    }
  }
}

class ARIMAModel(
    val order:(Int, Int, Int), // order of autoregression, differencing, and moving average
    val hasIntercept: Boolean = true,
    val coefficients: Array[Double], // coefficients: intercept, AR coeffs, ma coeffs
    val seed: Long = 10L // seed for random number generation
    ) extends TimeSeriesModel {

  val rand = new MersenneTwister(seed)
  /**
   * loglikelihood based on conditional sum of squares
   * Source: http://www.nuffield.ox.ac.uk/economics/papers/1997/w6/ma.pdf
   * @param y time series
   * @return loglikehood
   */
  def logLikelihoodCSS(y: Vector[Double]): Double = {
    val d = order._2
    val diffedY = ARIMA.differences(y, d).toArray.drop(d)
    logLikelihoodCSSARMA(diffedY)
  }

  def logLikelihoodCSSARMA(diffedY: Array[Double]): Double = {
    val n = diffedY.length
    val yHat = new DenseVector(Array.fill(n)(0.0))
    val yVect = new DenseVector(diffedY)
    iterateARMA(yVect,  yHat, _ + _, goldStandard = yVect, initErrors = null)

    val (p, d, q) = order
    val maxLag = math.max(p, q)
    // drop first maxLag terms, since we can't estimate residuals there, since no
    // AR(n) terms available
    val css = diffedY.zip(yHat.toArray).drop(maxLag).map { case (obs, pred) =>
      math.pow(obs - pred, 2)
    }.sum
    val sigma2 = css / n
    (-n / 2) * math.log(2 * math.Pi * sigma2) - css / (2 * sigma2)
  }

  /**
   * Updates the error vector in place for a new (more recent) error
   * The newest error is placed in position 0, while older errors "fall off the end"
   * @param errs array of errors of length q in ARIMA(p, d, q), holds errors for t-1 through t-q
   * @param newError the error at time t
   * @return
   */
  def updateMAErrors(errs: Array[Double], newError: Double): Unit= {
    val n = errs.length
    var i = 0
    while (i < n - 1) {
      errs(i + 1) = errs(i)
      i += 1
    }
    if (n > 0) {
      errs(0) = newError
    }
  }

  /**
   * {@inheritDoc}
   */
  def removeTimeDependentEffects(ts: Vector[Double], destTs: Vector[Double]): Vector[Double] = {
    val (p, d, q) = order
    val maxLag = math.max(p, q)
    val diffedTs = new DenseVector(ARIMA.differences(ts, d).toArray.drop(d))
    val changes =  new DenseVector(Array.fill(diffedTs.length)(0.0))
    //copy values in ts into destTs first
    changes := diffedTs
    // Subtract AR terms taken from diffedTs. Errors drawn from gaussian
    iterateARMA(diffedTs, changes, _ - _, goldStandard = null, initErrors = null)
    //copy first d elements into destTs directly
    destTs(0 until d) := ts(0 until d)
    //copy remainder changes
    destTs(d until destTs.length) := changes
    destTs := ARIMA.differences(destTs, d)
  }

  /**
   * {@inheritDoc}
   */
  def addTimeDependentEffects(ts: Vector[Double], destTs: Vector[Double]): Vector[Double] = {
    val (p, d, q) = order
    val maxLag = math.max(p, q)
    val diffedTs = new DenseVector(ARIMA.differences(ts, d).toArray.drop(d)) // difference
    // copy values
    destTs(0 to -1) := diffedTs(0 to -1)
    // Note that destTs goes in both ts and dest in call below, so that AR recursion is done
    // correctly. Errors drawn from gaussian distribution
    iterateARMA(destTs, destTs, _ + _, goldStandard = null, initErrors = null)
    val result = new DenseVector(Array.fill(ts.length)(0.0))
    result(0 until d) := ts(0 until d) // take first d terms
    result(d to -1) := destTs
    destTs :=  ARIMA.invDifferences(destTs, d) // add up as necessary
  }

  /**
   * Perform operations with the AR and MA terms, based on the time series `ts` and the errors
   * based off of `goldStandard`, combined with elements from the series  `dest`. Weights for terms
   * are taken from the current model configuration.
   * So for example: iterateARMA(series1, series_of_zeros,  _ + _ , goldStandard = series1,
   * initErrors = null)
   * calculates the 1-step ahead forecasts for series1 assuming current coefficients, and initial
   * MA errors of 0.
   * @param ts Time series to use for AR terms
   * @param dest Time series holding initial values at each index
   * @param op Operation to perform between values in dest, and various combinations of ts, errors
   *           and intercept terms
   * @param goldStandard The time series to which to compare 1-step ahead forecasts to obtain
   *                     moving average errors. Default set to null, in which case errors are
   *                     drawn from a gaussian distribution.
   * @param initErrors Initialization for first q errors. If none provided (i.e. remains null, as
   *                   per default), then initializes to all zeros
   *
   * @return dest series
   */

  def iterateARMA(
      ts: Vector[Double],
      dest: Vector[Double],
      op: (Double, Double) => Double,
      goldStandard: Vector[Double] = null,
      initErrors: Array[Double] = null
      )
    : Vector[Double] = {
    val (p, d, q) = order
    val maTerms = if (initErrors == null) Array.fill(q)(0.0) else initErrors
    val intercept = if (hasIntercept) 1 else 0
    var i = math.max(p, q) // maximum lag
    var j = 0
    val n = ts.length
    var error = 0.0

    while (i < n) {
      j = 0
      // intercept
      dest(i) = op(dest(i), intercept * coefficients(j))
      // autoregressive terms
      while (j < p && i - j - 1 >= 0) {
        dest(i) = op(dest(i), ts(i - j - 1) * coefficients(intercept + j))
        j += 1
      }
      // moving average terms
      j = 0
      while (j < q) {
        dest(i) = op(dest(i), maTerms(j) * coefficients(intercept + p + j))
        j += 1
      }

      error = if (goldStandard == null) rand.nextGaussian() else goldStandard(i) - dest(i)
      updateMAErrors(maTerms, error)
      i += 1
    }
    dest
  }


  def sample(n: Int): Vector[Double] = {
    val vec = new DenseVector(Array.fill[Double](n)(rand.nextGaussian()))
    addTimeDependentEffects(vec, vec)
  }


  def forecast(ts: Vector[Double], nFuture: Int): Vector[Double] = {
    val (p, d, q) = order
    val maxLag = math.max(p, q)
    // difference timeseries as necessary for model
    val diffedTs = new DenseVector(ARIMA.differences(ts, d).toArray.drop(d))
    val nDiffed = diffedTs.length

    val hist = new DenseVector(Array.fill(nDiffed)(0.0))

    // fit historical values
    iterateARMA(diffedTs, hist, _ + _,  goldStandard = diffedTs, initErrors = null)

    // Last set of errors, to be used in forecast if MA terms included
    val maTerms = (for (i <- nDiffed - maxLag until nDiffed) yield diffedTs(i) - hist(i)).toArray

    // include maxLag to copy over last maxLag values, to use in iterateARMA
    val forward = new DenseVector(Array.fill(nFuture + maxLag)(0.0))
    forward(0 until maxLag) := hist(nDiffed- maxLag until nDiffed)
    // use self as ts to take AR from same series, use self as goldStandard to induce future errors
    // of zero
    iterateARMA(forward, forward, _ + _, goldStandard = forward, initErrors = maTerms)

    //results
    val results = new DenseVector(Array.fill(ts.length + nFuture)(0.0))
    //copy first d elements prior to differencing
    results(0 until d) := ts(0 until d)
    // copy max of p/q element posts differencing, those values in hist are 0'ed out
    results(d until d + maxLag) := diffedTs(0 until maxLag)
    // copy historical values, after first p/q
    results(d + maxLag until nDiffed) := hist(maxLag until nDiffed)
    // copy forward, after dropping first maxLag terms, since these were just repeated
    // for convenience in iterateARMA
    results(nDiffed to -1) := forward(maxLag to -1)
    // add up if there was any differencing
    ARIMA.invDifferences(results, d)
  }
}
