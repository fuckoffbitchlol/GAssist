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

/*
 * java
 *
 * This class contains all the parameters of the system
 */

package GAssist_Parallel;

import java.util.*;
import java.io.*;

public final class Parameters {
	// Parallel variables
	public static int parallelParts;
	
  public static boolean evaluateAfterMigration;

  public static int rotationInterval;
  public static boolean rotateDataSets;
  
  public static int migrationInterval;
  public static boolean migrateBestIndividual;
  public static int individualsToMigrate;

  public static int subPopulationSize;
  public static int fixedSubPopSize;
  
  public static int globalEvaluationGenerations;
  public static boolean onlyDeleteRulesOnGlobalEvaluation;

  public static boolean saveGlobalStatistics;
	public static int saveStatisticsEveryXGeneration;
  public static int numberOfStatistics;
  
  public static String windowingMethod;
	
  public static boolean initializeRulesLocally;
  public static boolean keepBestIndividual;
  public static String eliteStrategy;
  public static int elitist;
  
  public final static int NONE = 0;
  public final static int ONE_PER_CORE = 1;
  public final static int ONE_PER_SET = 2;
	
	// Thread specific variables
	public static boolean[] doRuleDeletionPerThread;
	public static boolean[] doHierarchicalSelectionPerThread;
	public static int[] ruleDeletionMinRulesPerThread;
	public static double[] probReinitializePerThread;
	public static double[] percentageOfLearningPerThread;
	
	
	/*****************************
	*   GAssist parameters       *
	*****************************/
	
	 public static String algorithmName;

  public static double confidenceThreshold;
  public static double inconsistencyThreshold;
  public static double alpha;
  public static double beta;
  public static double lambda;
  public static int numIntervals;
  public static boolean amevaR;
  public static String numIntrvls;
  public static int Neighborhood;    
  public static int WindowsSize;
  public static String DistanceFunction;
  public static int minIntervals;
  public static String mapType;
  public static int minSupport;
  public static double mergedThreshold;
  public static double scalingFactor;
  public static boolean useDiscrete;
  public static int frequencySize;
  public static boolean setConfig;

  public static int numClasses;
  public static int numAttributes;
  public static int numInstances;

  public static int popSize;
  public static int initialNumberOfRules;
  public static int ruleDeletionMinRules;
  public static double probCrossover;
  public static double probMutationInd;
  public static int tournamentSize;
  public static int numIterations;

  public static boolean useMDL;
  public static int iterationMDL;
  public static double initialTheoryLengthRatio;
  public static double weightRelaxFactor;
  public static double theoryWeight;

  public static double probOne;

  public static String trainInputFile;
  public static String train2InputFile;
  public static String testInputFile;
  public static String trainOutputFile;
  public static String testOutputFile;
  public static String logOutputFile;
  public static int seed;

  public static int iterationRuleDeletion;
  public static int iterationHierarchicalSelection;
  public static double hierarchicalSelectionThreshold;
  public static int sizePenaltyMinRules;

  public static int numStrata;

  public static String discretizer1;
  public static String discretizer2;
  public static String discretizer3;
  public static String discretizer4;
  public static String discretizer5;
  public static String discretizer6;
  public static String discretizer7;
  public static String discretizer8;
  public static String discretizer9;
  public static String discretizer10;
  public static int maxIntervals;
  public static double probSplit;
  public static double probMerge;
  public static double probReinitializeBegin;
  public static double probReinitializeEnd;
  public static boolean adiKR=false;
  public static int minimumValuesOfSameClassPerInterval;
  public static String processType;
        public static String defaultClass;
        public static String initMethod;
  public static double coefficient; 
	
  /*****************************
  *   GAssist parameters       *
  *****************************/
	
  public static void setVars() {

    seed = seed
        + trainOutputFile.hashCode()
        + parallelParts
        + elitist
        + fixedSubPopSize
        + popSize
        + eliteStrategy.hashCode()
        ;
    
    // Parameters
    if (parallelParts <= 1) {
      parallelParts = 1;
      migrateBestIndividual = false;
      individualsToMigrate = 0;
      saveGlobalStatistics = false;
      saveStatisticsEveryXGeneration = 0;
    }
    else {
      if (saveStatisticsEveryXGeneration > 0) {

        saveGlobalStatistics = true;
        numberOfStatistics = 1 + (int) Math.floor((double) numIterations / (double) saveStatisticsEveryXGeneration);
      }
      else {
        saveGlobalStatistics = false;
      }
      
      if (individualsToMigrate > 0) {
        migrateBestIndividual = true;
      }
      else {
        migrateBestIndividual = false;
        individualsToMigrate = 0;
      }

      if (rotationInterval > 0 && rotationInterval < numIterations) {
        rotateDataSets = true;
      } else {
        rotateDataSets = false;
        rotationInterval = 0;
      }
      
      if (migrationInterval > 0 && migrationInterval < numIterations) {
        migrateBestIndividual = true;
      } else {
        migrateBestIndividual = false;
        migrationInterval = 0;
      }
    }

    doRuleDeletionPerThread = new boolean[parallelParts];
    doHierarchicalSelectionPerThread = new boolean[parallelParts];
    ruleDeletionMinRulesPerThread = new int[parallelParts];
    probReinitializePerThread = new double[parallelParts];
    percentageOfLearningPerThread = new double[parallelParts];
    
    for (int i=0; i < parallelParts; i++) {
      doRuleDeletionPerThread[i] = false;
      doHierarchicalSelectionPerThread[i] = false;
      ruleDeletionMinRulesPerThread[i] = ruleDeletionMinRules;
      probReinitializePerThread[i] = 0.0;
      percentageOfLearningPerThread[i] = 0.0;
    }
    
    if (eliteStrategy.equalsIgnoreCase("none")) {
      elitist = NONE;
      keepBestIndividual = false;
    }
    else if (eliteStrategy.equalsIgnoreCase("onePerCore")) {
      elitist = ONE_PER_CORE;
      keepBestIndividual = true;
    }
    else if (eliteStrategy.equalsIgnoreCase("onePerSet")) {
      elitist = ONE_PER_CORE;
      keepBestIndividual = true;
    }
    else {
      elitist = ONE_PER_SET;
      keepBestIndividual = true;
    }
  }
  
}

