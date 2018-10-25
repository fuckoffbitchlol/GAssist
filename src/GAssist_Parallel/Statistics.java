/***********************************************************************

	This file is part of KEEL-software, the Data Mining tool for regression, 
	classification, clustering, pattern mining and so on.

	Copyright (C) 2004-2010
	
	F. Herrera (herrera@decsai.ugr.es)
    L. Sç–£chez (luciano@uniovi.es)
    J. Alcal?½Fdez (jalcala@decsai.ugr.es)
    S. Garcåƒ?(sglopez@ujaen.es)
    A. Fernç–£dez (alberto.fernandez@ujaen.es)
    J. Luengo (julianlm@decsai.ugr.es)

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see http://www.gnu.org/licenses/
  
**********************************************************************/

/**
 * <p>
 * @author Written by Jaume Bacardit (La Salle, Ramû¥¢ Llull University - Barcelona) 28/03/2004
 * @author Modified by Xavi Sol?½(La Salle, Ramû¥¢ Llull University - Barcelona) 23/12/2008
 * @version 1.1
 * @since JDK1.2
 * </p>
 */



package GAssist_Parallel;
import keel.Algorithms.Genetic_Rule_Learning.Globals.FileManagement;
import keel.Algorithms.Genetic_Rule_Learning.Globals.LogManager;

public class Statistics {
/**
 * <p>
 * Computes and stores several statistics about the learning process
 * </p>
 */
	
  public double[] bestAccuracy;
  public double[] bestFit;
  public double[] bestRules;
  public double[] bestAliveRules;

  public double[] bestGlobalAccuracy;
  public double[] bestGlobalFit;
  public double[] bestGlobalRules;
  public double[] bestGlobalAliveRules;


  public int iterationsSinceBest = 0;
  public double bestFitness;
  public double last10IterationsAccuracyAverage;

  public int countStatistics = 0;
  public int countGlobalStatistics = 0;

  public void resetBestStats() {
    iterationsSinceBest = 0;
  }

  public int getIterationsSinceBest() {
    return iterationsSinceBest;
  }

  public void bestOfIteration(double itBestFit) {
    if (iterationsSinceBest == 0) {
      bestFitness = itBestFit;
      iterationsSinceBest++;
    }
    else {
      boolean newBest = false;
      if (Parameters.useMDL) {
        if (itBestFit < bestFitness) {
          newBest = true;
        }
      }
      else {
        if (itBestFit > bestFitness) {
          newBest = true;
        }
      }

      if (newBest) {
        bestFitness = itBestFit;
        iterationsSinceBest = 1;
      }
      else {
        iterationsSinceBest++;
      }
    }

    int i = countStatistics - 9;
    if (i < 0) {
      i = 0;
    }
    int max = countStatistics + 1;
    int num = max - i;
    last10IterationsAccuracyAverage = 0;
    for (; i < max; i++) {
      last10IterationsAccuracyAverage += bestAccuracy[i];
    }
    last10IterationsAccuracyAverage /= (double) num;
  }

  public void initStatistics() {
    int numStatistics = Parameters.numIterations;

    bestAccuracy = new double[numStatistics];
    bestRules = new double[numStatistics];
    bestFit = new double[numStatistics];
    bestAliveRules = new double[numStatistics];
    
    bestGlobalAccuracy = new double[Parameters.numberOfStatistics];
    bestGlobalRules = new double[Parameters.numberOfStatistics];
    bestGlobalFit = new double[Parameters.numberOfStatistics];
    bestGlobalAliveRules = new double[Parameters.numberOfStatistics];
  }

  public void computeStatistics(Classifier[] _population) {
    Classifier best = PopulationWrapper.getBest(_population);
        
    bestAccuracy[countStatistics] = best.getAccuracy();
    bestRules[countStatistics] = best.getNumRules();
    bestAliveRules[countStatistics] = best.getNumAliveRules();
    bestOfIteration(best.getFitness());

    countStatistics++;
  }

  public void computeGlobalStatistics(Classifier[] _population, Classifier[] _best) {
	    Classifier best = PopulationWrapper.getGlobalBest(_population);
	    Classifier best2 = PopulationWrapper.getGlobalBest(_best);
	    
	    if (best2 != null) {
  	    if (best2.globalCompareToIndividual(best)) {
  	      best = best2;
  	    }
	    }
	    
	    bestGlobalAccuracy[countGlobalStatistics] = best.getGlobalAccuracy();
	    bestGlobalRules[countGlobalStatistics] = best.getGlobalNumRules();
	    bestGlobalAliveRules[countGlobalStatistics] = best.getGlobalNumAliveRules();
	    bestGlobalFit[countGlobalStatistics] = best.getGlobalFitness();
	    
	    countGlobalStatistics++;
	  }
  
}

