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

import org.apache.commons.math3.analysis.{MultivariateFunction, MultivariateVectorFunction}
import org.apache.commons.math3.optim.{MaxEval, MaxIter, InitialGuess, SimpleValueChecker}
import org.apache.commons.math3.optim.nonlinear.scalar.{GoalType, ObjectiveFunction, ObjectiveFunctionGradient}
import org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer
import org.apache.commons.math3.random.RandomGenerator

object GARCH {
  /**
   * Fits a GARCH(1, 1) model to the given time series.
   *
   * @param ts The time series to fit the model to.
   * @return The model.
   */
  def fitModel(ts: Vector[Double]): GARCHModel = {
    val optimizer = new NonLinearConjugateGradientOptimizer(
      NonLinearConjugateGradientOptimizer.Formula.FLETCHER_REEVES,
      new SimpleValueChecker(1e-6, 1e-6))
    val gradient = new ObjectiveFunctionGradient(new MultivariateVectorFunction() {
      def value(params: Array[Double]): Array[Double] = {
        new GARCHModel(params(0), params(1), params(2)).gradient(ts).toArray
      }
    })
    val objectiveFunction = new ObjectiveFunction(new MultivariateFunction() {
      def value(params: Array[Double]): Double = {
        new GARCHModel(params(0), params(1), params(2)).logLikelihood(ts)
      }
    })
    val initialGuess = new InitialGuess(Array(.2, .2, .2)) // TODO: make this smarter
    val maxIter = new MaxIter(10000)
    val maxEval = new MaxEval(10000)
    val optimal = optimizer.optimize(objectiveFunction, gradient, initialGuess, maxIter, maxEval)
    val params = optimal.getPoint
    new GARCHModel(params(0), params(1), params(2))
  }
}

object ARGARCH {
  /**
   * Fits an AR(1) + GARCH(1, 1) model to the given time series.
   *
   * @param ts The time series to fit the model to.
   * @return The model.
   */
  def fitModel(ts: Vector[Double]): ARGARCHModel = {
    val arModel = Autoregression.fitModel(ts)
    val residuals = arModel.removeTimeDependentEffects(ts, DenseVector.zeros[Double](ts.length))
    val garchModel = GARCH.fitModel(residuals)
    new ARGARCHModel(arModel.c, arModel.coefficients(0), garchModel.omega, garchModel.alpha,
      garchModel.beta)
  }
}

class GARCHModel(
    val omega: Double,
    val alpha: Double,
    val beta: Double) extends TimeSeriesModel {

  /**
   * Returns the log likelihood of the parameters on the given time series.
   *
   * Based on http://www.unc.edu/~jbhill/Bollerslev_GARCH_1986.pdf
   */
  def logLikelihood(ts: Vector[Double]): Double = {
    var sum = 0.0
    iterateWithHAndEta(ts) { (i, h, eta, prevH, prevEta) =>
      sum += -.5 * math.log(h) - .5 * eta * eta / h
    }
    sum + -.5 * math.log(2 * math.Pi) * (ts.length - 1)
  }

  /**
   * Find the gradient of the log likelihood with respect to the given time series.
   *
   * Based on http://www.unc.edu/~jbhill/Bollerslev_GARCH_1986.pdf
   * @return an 3-element array containing the gradient for the alpha, beta, and omega parameters.
   */
  private[sparkts] def gradient(ts: Vector[Double]): Array[Double] = {
    var omegaGradient = 0.0
    var alphaGradient = 0.0
    var betaGradient = 0.0
    var omegaDhdtheta = 0.0
    var alphaDhdtheta = 0.0
    var betaDhdtheta = 0.0

    iterateWithHAndEta(ts) { (i, h, eta, prevH, prevEta) =>
      omegaDhdtheta = 1 + beta * omegaDhdtheta
      alphaDhdtheta = prevEta * prevEta + beta * alphaDhdtheta
      betaDhdtheta = prevH + beta * betaDhdtheta

      val multiplier = (eta * eta / (h * h)) - (1 / h)
      omegaGradient += multiplier * omegaDhdtheta
      alphaGradient += multiplier * alphaDhdtheta
      betaGradient += multiplier * betaDhdtheta
    }
    Array(alphaGradient * .5, betaGradient * .5, omegaGradient * .5)
  }

  private def iterateWithHAndEta(ts: Vector[Double])
                                (fn: (Int, Double, Double, Double, Double) => Unit): Unit = {
    var prevH = omega / (1 - alpha - beta)
    var i = 1
    while (i < ts.length) {
      val h = omega + alpha * ts(i - 1) * ts(i - 1) + beta * prevH

      fn(i, h, ts(i), prevH, ts(i - 1))

      prevH = h
      i += 1
    }
  }

  /**
   * {@inheritDoc}
   */
  def removeTimeDependentEffects(ts: Vector[Double], dest: Vector[Double]): Vector[Double] = {
    var prevEta = ts(0)
    var prevVariance = omega / (1.0 - alpha - beta)
    dest(0) = prevEta / math.sqrt(prevVariance)
    for (i <- 1 until ts.length) {
      val variance = omega + alpha * prevEta * prevEta + beta * prevVariance
      val eta = ts(i)
      dest(i) = eta / math.sqrt(variance)

      prevEta = eta
      prevVariance = variance
    }
    dest
  }

  /**
   * {@inheritDoc}
   */
  override def addTimeDependentEffects(ts: Vector[Double], dest: Vector[Double]): Vector[Double] = {
    var prevVariance = omega / (1.0 - alpha - beta)
    var prevEta = ts(0) * math.sqrt(prevVariance)
    dest(0) = prevEta
    for (i <- 1 until ts.length) {
      val variance = omega + alpha * prevEta * prevEta + beta * prevVariance
      val standardizedEta = ts(i)
      val eta = standardizedEta * math.sqrt(variance)
      dest(i) = eta

      prevEta = eta
      prevVariance = variance
    }
    dest
  }

  private def sampleWithVariances(n: Int, rand: RandomGenerator): (Array[Double], Array[Double]) = {
    val ts = new Array[Double](n)
    val variances = new Array[Double](n)
    variances(0) = omega / (1 - alpha - beta)
    var eta = math.sqrt(variances(0)) * rand.nextGaussian()
    for (i <- 1 until n) {
      variances(i) = omega + beta * variances(i-1) + alpha * eta * eta
      eta = math.sqrt(variances(i)) * rand.nextGaussian()
      ts(i) = eta
    }

    (ts, variances)
  }

  /**
   * Samples a random time series of a given length with the properties of the model.
   *
   * @param n The length of the time series to sample.
   * @param rand The random generator used to generate the observations.
   * @return The samples time series.
   */
  def sample(n: Int, rand: RandomGenerator): Array[Double] = sampleWithVariances(n, rand)._1
}

/**
 * A GARCH(1, 1) + AR(1) model, where
 *   y(i) = c + phi * y(i - 1) + eta(i),
 * and h(i), the variance of eta(i), is given by
 *   h(i) = omega + alpha * eta(i) ** 2 + beta * h(i - 1) ** 2
 *
 * @param c The constant term.
 * @param phi The autoregressive term.
 * @param omega The constant term in the variance.
 */
class ARGARCHModel(
    val c: Double,
    val phi: Double,
    val omega: Double,
    val alpha: Double,
    val beta: Double) extends TimeSeriesModel {

  /**
   * {@inheritDoc}
   */
  override def removeTimeDependentEffects(ts: Vector[Double], dest: Vector[Double])
    : Vector[Double] = {
    var prevEta = ts(0) - c
    var prevVariance = omega / (1.0 - alpha - beta)
    dest(0) = prevEta / math.sqrt(prevVariance)
    for (i <- 1 until ts.length) {
      val variance = omega + alpha * prevEta * prevEta + beta * prevVariance
      val eta = ts(i) - c - phi * ts(i - 1)
      dest(i) = eta / math.sqrt(variance)

      prevEta = eta
      prevVariance = variance
    }
    dest
  }

  /**
   * {@inheritDoc}
   */
  override def addTimeDependentEffects(ts: Vector[Double], dest: Vector[Double]): Vector[Double] = {
    var prevVariance = omega / (1.0 - alpha - beta)
    var prevEta = ts(0) * math.sqrt(prevVariance)
    dest(0) = c + prevEta
    for (i <- 1 until ts.length) {
      val variance = omega + alpha * prevEta * prevEta + beta * prevVariance
      val standardizedEta = ts(i)
      val eta = standardizedEta * math.sqrt(variance)
      dest(i) = c + phi * dest(i - 1) + eta

      prevEta = eta
      prevVariance = variance
    }
    dest
  }

  private def sampleWithVariances(n: Int, rand: RandomGenerator): (Array[Double], Array[Double]) = {
    val ts = new Array[Double](n)
    val variances = new Array[Double](n)
    variances(0) = omega / (1 - alpha - beta)
    var eta = math.sqrt(variances(0)) * rand.nextGaussian()
    for (i <- 1 until n) {
      variances(i) = omega + beta * variances(i-1) + alpha * eta * eta
      eta = math.sqrt(variances(i)) * rand.nextGaussian()
      ts(i) = c + phi * ts(i - 1) + eta
    }

    (ts, variances)
  }

  /**
   * Samples a random time series of a given length with the properties of the model.
   *
   * @param n The length of the time series to sample.
   * @param rand The random generator used to generate the observations.
   * @return The samples time series.
   */
  def sample(n: Int, rand: RandomGenerator): Array[Double] = sampleWithVariances(n, rand)._1
}

object EGARCH {
  /**
   * Fits an EGARCH(1, 1) model to the given series of residuals
   *
   * @param ts The time series to fit the model to.
   * @return The model.
   */
  def fitModel(ts: Vector[Double]): EGARCHModel = {
    val optimizer = new NonLinearConjugateGradientOptimizer(
      NonLinearConjugateGradientOptimizer.Formula.FLETCHER_REEVES,
      new SimpleValueChecker(1e-6, 1e-6))
    val gradient = new ObjectiveFunctionGradient(new MultivariateVectorFunction() {
      def value(params: Array[Double]): Array[Double] = {
        new EGARCHModel(params(0), params(1), params(2), params(3)).gradient(ts)
      }
    })
    val objectiveFunction = new ObjectiveFunction(new MultivariateFunction() {
      def value(params: Array[Double]): Double = {
        new EGARCHModel(params(0), params(1), params(2), params(3)).logLikelihood(ts)
      }
    })
    val mean = sum(ts)/ ts.length
    val initialGuess = new InitialGuess(Array(0.2, 0.2, 0.2, 0.2)) // TODO: make this smarter
    val maxIter = new MaxIter(10000)
    val maxEval = new MaxEval(10000)
    val goal = GoalType.MAXIMIZE
    val optimal = optimizer.optimize(objectiveFunction, goal, gradient, initialGuess, maxIter,
      maxEval)
    val params = optimal.getPoint
    new EGARCHModel(params(0), params(1), params(2), params(3))
  }
}

//http://www.samsi.info/sites/default/files/Nelson_1991.pdf
class EGARCHModel(
    val alpha: Double,
    val theta: Double,
    val gamma: Double,
    val beta: Double) extends TimeSeriesModel {
  /**
   * Returns the log likelihood of the parameters for the given series
   * Based on http://swopec.hhs.se/hastef/papers/hastef0564.pdf and
   * defined as
   * - 0.5 * sum(ln h_t) - 0.5 * (eps_t ** 2 / h_t) + C (some constant)
   * we return a value up to a constant, and maximize
   */
  def logLikelihood(ts: Vector[Double]): Double = {
    //accumulators
    var likelihood = 0.0

    //initialize
    val n = ts.length
    var h = variance(ts) //ht is initialized to the overall variance
    var lnh = math.log(h) // just log of h
    var z = ts(0) / math.sqrt(h) //first zt term, recall z_t = error_t / sqrt(h_t)
    var i = 1

    while (i < n) {
      lnh = alpha + theta * z + gamma * math.abs(z) + beta * lnh
      h = math.exp(lnh)
      z = ts(i) / math.sqrt(h)
      likelihood -= 0.5 * (lnh + math.pow(ts(i), 2) / h)
      i += 1
    }

    likelihood
  }

  /**
   * calculate gradient, source http://swopec.hhs.se/hastef/papers/hastef0564.pdf
   */
  def gradient(ts: Vector[Double]): Array[Double] = {
    val dim = 4
    //accumulators
    var likelihood = 0.0

    //initialize
    val n = ts.length
    var h = variance(ts) //ht is initialized to the overall variance
    var grad = Array.fill(dim)(0.0) // h_0 not a function of params
    var lnh = math.log(h) // just log of h
    var z = ts(0) / math.sqrt(h) //first zt term, recall z_t = error_t / sqrt(h_t)

    var X = Array(1.0, z, math.abs(z), lnh)
    var dlnHdB = Array.fill(dim)(0.0) // first set of derivatives is 0, since h_0 = var(ts)

    var i = 1

    while (i < n) {
      dlnHdB = vecOp(
        X,
        dlnHdB.map { x => 0.5 * (theta * z + gamma * math.abs(z)) * x + beta * x },
        _ - _)

      grad = vecOp(
        grad,
        dlnHdB.map { x => 0.5 * (math.pow(ts(i), 2) / math.sqrt(h) - 1) * x},
        _ + _)

      //update values
      lnh = alpha + theta * z + gamma * math.abs(z) + beta * lnh
      h = math.exp(lnh)
      z = ts(i) / math.sqrt(h)
      X = Array(1.0, z, math.abs(z), lnh)
      i += 1
    }
    grad
  }

  //vectorize an operation, doesn't check lengths, so be careful!
  def vecOp(x: Array[Double], y: Array[Double], op: (Double, Double) => Double)
    : Array[Double] = {
    x.zip(y).map { case (xi, yi) => op(xi, yi) }
  }

  /**
   * {@inheritDoc}
   */
  override def removeTimeDependentEffects(ts: Vector[Double], dest: Vector[Double])
    : Vector[Double] = {
    val n = ts.length
    var h = math.exp(alpha) // assumes z_-1, |z_-1|, and ln h_-1 are all zero
    var lnh = math.log(h)

    dest(0) = ts(0) / math.sqrt(h)

    var i = 1
    while (i < n) {
      lnh = alpha + theta * dest(i - 1) + gamma * math.abs(dest(i - 1)) + beta * lnh
      h = math.exp(lnh)
      dest(i) = ts(i) / math.sqrt(h)
      i += 1
    }
    dest
  }

  /**
   * {@inheritDoc}
   */
  override def addTimeDependentEffects(ts: Vector[Double], dest: Vector[Double]): Vector[Double] = {
    val n = ts.length
    var h = math.exp(alpha) // assumes z_-1, |z_-1|, and ln h_-1 are all zero
    var lnh = math.log(h) // just log of h

    dest(0) = ts(0) * math.sqrt(h)

    var i = 1
    while (i < n) {
      lnh = alpha + theta * ts(i - 1) + gamma * math.abs(ts(i - 1)) + beta * lnh
      h = math.exp(lnh)
      dest(i) = ts(i) * math.sqrt(h)
      i += 1
    }
    dest
  }

  def variance(ts: Vector[Double]): Double = {
    val n = ts.length
    val mean = sum(ts) / n
    sum(ts.map{ x => math.pow(x - mean, 2)}) / n
  }
}
