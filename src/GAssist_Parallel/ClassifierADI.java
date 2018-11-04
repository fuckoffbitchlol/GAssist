/***********************************************************************

	This file is part of KEEL-software, the Data Mining tool for regression, 
	classification, clustering, pattern mining and so on.

	Copyright (C) 2004-2010
	
	F. Herrera (herrera@decsai.ugr.es)
    L. S疣chez (luciano@uniovi.es)
    J. Alcal?�Fdez (jalcala@decsai.ugr.es)
    S. Garc�?(sglopez@ujaen.es)
    A. Fern疣dez (alberto.fernandez@ujaen.es)
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
 * @author Written by Jaume Bacardit (La Salle, Ram��� Llull University - Barcelona) 28/03/2004
 * @author Modified by Xavi Sol?�(La Salle, Ram��� Llull University - Barcelona) 23/12/2008
 * @version 1.1
 * @since JDK1.2
 * </p>
 */

package GAssist_Parallel;

import keel.Dataset.*;
import keel.Algorithms.Genetic_Rule_Learning.Globals.*;

public class ClassifierADI
    extends Classifier implements Cloneable {
/**
 * <p>
 * Contains the classifier for the ADI knowledge representation
 * </p>
 */
	
  // The chromosome
  int[] crm;
  int length;
  int defaultClass;

  public ClassifierADI() {
    isEvaluated = false;
  }

  public void initRandomClassifier(int strata) {
    Rand rn = ParallelGlobals.getRand();
    
    numRules = Parameters.initialNumberOfRules;
    int ruleSize = ParallelGlobals.getGlobals_ADI().ruleSize;
    crm = new int[numRules * ruleSize];
    int base = 0;
    length = 0;

    if (ParallelGlobals.getGlobals_DefaultC().defaultClassPolicy
        == ParallelGlobals.getGlobals_DefaultC().AUTO) {
      defaultClass = rn.getInteger(0, Parameters.numClasses - 1);
    }
    else {
      defaultClass = ParallelGlobals.getGlobals_DefaultC().defaultClass;
    }

    for (int i = 0; i < numRules; i++) {
      AdaptiveRule.constructor(rn, crm, base, defaultClass, strata);
      length += crm[base];
      base += ruleSize;
    }
    resetPerformance();
  }
  
  @Override
  public double computeTheoryLength(PerformanceAgent pa) {
    int base = 0;
    int ruleSize = ParallelGlobals.getGlobals_ADI().ruleSize;
    theoryLength = 0;
    for (int i = 0; i < numRules; i++) {
      if (pa.getActivationsOfRule(i) > 0) {
        theoryLength
            += AdaptiveRule.computeTheoryLength(crm, base);
      }
      base += ruleSize;
    }
    if (ParallelGlobals.getGlobals_DefaultC().enabled) {
      theoryLength += 0.00000001;
    }
    return theoryLength;
  }

  public double getLength() {
    return length * (double) numAliveRules / (double) numRules;
  }

  /**
   * This function classifies input instances. It returns a class
   * prediction of -1 if the input example cannot be classified
   */
  public int doMatch(InstanceWrapper ins) {
    int i;
    int base = 0;
    int ruleSize = ParallelGlobals.getGlobals_ADI().ruleSize;
    
    for (i = 0; i < numRules; i++) {
    	if (i != numRules - 1){
    		if (AdaptiveRule.doMatch(crm, base, ins)) {
    			positionRuleMatch = i;
    			return crm[base + 1];
    		}
    		base += ruleSize;
    	}
    	else{
//    		LogManager.println("tl: " + AdaptiveRule.computeTheoryLength(crm, base));
    		boolean fuckB = false;
    		for(int k = 0; k < Parameters.numAttributes; k++) {
    			if (AdaptiveAttribute.dumpShit(crm, base + 2, k)) {
    				fuckB = true;
    			}
    		}
    		if (!fuckB) {
    			return -1;
    		}
    		else {
    			if (AdaptiveRule.doMatch(crm, base, ins)) {
    				positionRuleMatch = i;
    				return crm[base + 1];
    			}	
    		}
    	}
    }
    if (ParallelGlobals.getGlobals_DefaultC().enabled) {
      positionRuleMatch = numRules;
      return defaultClass;
    }
    return -1;
  }

  public void printClassifier() {
    int ruleSize = Globals_ADI.ruleSize;
    String str;
    int base = 0;

    for (int i = 0; i < numRules; i++) {
      str = i + ":";
      str += AdaptiveRule.dumpPhenotype(crm, base);
      LogManager.println(str);
      base += ruleSize;
    }
    if (Globals_DefaultC.enabled) {
      LogManager.println(numRules + ":Default rule -> "
                         +
                         Attributes.getAttribute(Parameters.numAttributes).
                         getNominalValue(defaultClass));
    }
  }

  public int getNumRules() {
    if (ParallelGlobals.getGlobals_DefaultC().enabled) {
      return numRules + 1;
    }
    return numRules;
  }

  public Classifier[] crossoverClassifiers(Classifier _parent2) {
    ClassifierADI offspring1 = new ClassifierADI();
    ClassifierADI offspring2 = new ClassifierADI();
    ClassifierADI parent2 = (ClassifierADI) _parent2;
    
    Rand rn = ParallelGlobals.getRand();

    int ruleSize = ParallelGlobals.getGlobals_ADI().ruleSize;
    int ruleP1 = (int) rn.getInteger(0, numRules - 1);
    int ruleP2 = (int) rn.getInteger(0, parent2.numRules - 1);
    offspring1.numRules = ruleP1 + parent2.numRules - ruleP2;
    offspring2.numRules = ruleP2 + numRules - ruleP1;
    int cutPoint = rn.getInteger(0, Parameters.numAttributes);
    offspring1.defaultClass = offspring2.defaultClass = defaultClass;

    offspring1.crm = new int[ruleSize * offspring1.numRules];
    offspring2.crm = new int[ruleSize * offspring2.numRules];

    System.arraycopy(crm, 0, offspring1.crm, 0, ruleP1 * ruleSize);
    System.arraycopy(parent2.crm, 0, offspring2.crm, 0, ruleP2 * ruleSize);

    AdaptiveRule.crossover(crm, parent2.crm, offspring1.crm
                           , offspring2.crm, ruleP1 * ruleSize,
                           ruleP2 * ruleSize
                           , cutPoint);

    int base1 = (ruleP1 + 1) * ruleSize;
    int base2 = (ruleP2 + 1) * ruleSize;

    System.arraycopy(crm, base1, offspring2.crm, base2,
                     (numRules - ruleP1 - 1) * ruleSize);
    System.arraycopy(parent2.crm, base2, offspring1.crm, base1,
                     (parent2.numRules - ruleP2 - 1) * ruleSize);

    int base = 0;
    offspring1.length = 0;
    for (int i = 0; i < offspring1.numRules; i++) {
      offspring1.length += offspring1.crm[base];
      base += ruleSize;
    }
    base = 0;
    offspring2.length = 0;
    for (int i = 0; i < offspring2.numRules; i++) {
      offspring2.length += offspring2.crm[base];
      base += ruleSize;
    }

    Classifier[] ret = new Classifier[2];
    ret[0] = offspring1;
    ret[1] = offspring2;

    return ret;
  }

  public Classifier copy() {
    int ruleSize = ParallelGlobals.getGlobals_ADI().ruleSize;
    ClassifierADI ret = new ClassifierADI();

    ret.numRules = numRules;
    ret.theoryLength = theoryLength;
    ret.exceptionsLength = exceptionsLength;
    ret.length = length;
    ret.accuracy = accuracy;
    ret.fitness = fitness;
    ret.isEvaluated = isEvaluated;
    ret.numAliveRules = numAliveRules;
    ret.defaultClass = defaultClass;

    ret.crm = new int[numRules * ruleSize];
    System.arraycopy(crm, 0, ret.crm, 0, numRules * ruleSize);

    return ret;
  }

  public void doMutation() {
    Rand rn = ParallelGlobals.getRand();
    
    int whichRule = rn.getInteger(0, numRules - 1);
    int ruleSize = ParallelGlobals.getGlobals_ADI().ruleSize;
    int base = whichRule * ruleSize;

    AdaptiveRule.mutation(rn, crm, base, defaultClass);
    isEvaluated = false;
  }

  public void deleteRules(int[] whichRules) {
    if (numRules == 1 || whichRules.length == 0) {
      return;
    }

    int ruleSize = ParallelGlobals.getGlobals_ADI().ruleSize;
    int rulesToDelete = whichRules.length;
    if (whichRules[rulesToDelete - 1] == numRules) {
      rulesToDelete--;
    }

    int[] newCrm = new int[ruleSize * (numRules - rulesToDelete)];
    int countPruned = 0;
    int baseOrig = 0;
    int baseNew = 0;

    for (int i = 0; i < numRules; i++) {
      if (countPruned < rulesToDelete) {
        if (i != whichRules[countPruned]) {
          System.arraycopy(crm, baseOrig, newCrm, baseNew, ruleSize);
          baseNew += ruleSize;
        }
        else {
          countPruned++;
        }
      }
      else {
        System.arraycopy(crm, baseOrig, newCrm, baseNew, ruleSize);
        baseNew += ruleSize;
      }
      baseOrig += ruleSize;
    }
    numRules -= rulesToDelete;
    crm = newCrm;
  }

  public int numSpecialStages() {
    return 3;
  }

  public void doSpecialStage(int stage) {
    Rand rn = ParallelGlobals.getRand();
    
    if (stage == 0) {
      doSplit(rn);
    }
    else if (stage == 1) {
      doMerge(rn);
    }
    else if (stage == 2) {
      doReinitialize(rn);
    }
    else {
      LogManager.printErr("Unknown special stage !!");
      System.exit(1);
    }
  }

  public void doSplit(Rand rn) {
    int base = 0;
    for (int i = 0; i < numRules; i++) {
      length -= crm[base];
      if (AdaptiveRule.doSplit(rn, crm, base)) {
        isEvaluated = false;
      }
      length += crm[base];
      base += ParallelGlobals.getGlobals_ADI().ruleSize;
    }
  }

  public void doMerge(Rand rn) {
    int base = 0;
    for (int i = 0; i < numRules; i++) {
      length -= crm[base];
      if (AdaptiveRule.doMerge(rn, crm, base)) {
        isEvaluated = false;
      }
      length += crm[base];
      base += ParallelGlobals.getGlobals_ADI().ruleSize;
    }
  }

  public void doReinitialize(Rand rn) {
    int base = 0;
    for (int i = 0; i < numRules; i++) {
      length -= crm[base];
      if (AdaptiveRule.doReinitialize(rn, crm, base)) {
        isEvaluated = false;
      }
      length += crm[base];
      base += ParallelGlobals.getGlobals_ADI().ruleSize;
    }
  }

  public int getNiche() {
    if (ParallelGlobals.getGlobals_DefaultC().defaultClassPolicy != ParallelGlobals.getGlobals_DefaultC().AUTO) {
      return 0;
    }
    return defaultClass;
  }

  public int getNumNiches() {
    if (ParallelGlobals.getGlobals_DefaultC().defaultClassPolicy != ParallelGlobals.getGlobals_DefaultC().AUTO) {
      return 1;
    }
    return Parameters.numClasses;
  }

}

