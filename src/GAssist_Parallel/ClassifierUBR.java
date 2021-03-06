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

public class ClassifierUBR
    extends Classifier implements Cloneable {
/**
 * <p>
 * Classifier for the UBR intervalar knowledge representation
 * </p>
 */
	
  double[] crm;
  int defaultClass;

  public ClassifierUBR() {
    isEvaluated = false;
  }

  public void initRandomClassifier(int strata) {
    Rand rn = ParallelGlobals.getRand();
    
    numRules = Parameters.initialNumberOfRules;

    int ruleSize = ParallelGlobals.getGlobals_UBR().ruleSize;
    double probNominal = Parameters.probOne;
    int nC = Parameters.numClasses;
    crm = new double[numRules * ruleSize];
    int base = 0;

    if (ParallelGlobals.getGlobals_DefaultC().defaultClassPolicy == ParallelGlobals.getGlobals_DefaultC().AUTO) {
      defaultClass = rn.getInteger(0, Parameters.numClasses - 1);
    }
    else {
      defaultClass = ParallelGlobals.getGlobals_DefaultC().defaultClass;
    }

    for (int i = 0; i < numRules; i++) {
      InstanceWrapper ins = null;
      if (PopulationWrapper.smartInit) {
        if (ParallelGlobals.getGlobals_DefaultC().defaultClassPolicy != ParallelGlobals.getGlobals_DefaultC().DISABLED) {
          ins = PopulationWrapper.getInstanceInit(defaultClass, strata);
        }
        else {
          ins = PopulationWrapper.getInstanceInit(Parameters.numClasses, strata);
        }
      }

      for (int j = 0; j < Parameters.numAttributes; j++) {
        if (ParallelGlobals.getGlobals_UBR().types[j] == Attribute.NOMINAL) {
          int value;
          if (ins != null) {
            value = ins.getNominalValue(j);
          }
          else {
            value = -1;
          }
          for (int k = 0; k < ParallelGlobals.getGlobals_UBR().size[j]; k++) {
            if (k != value) {
              if (rn.getReal() < probNominal) {
                crm[base + k] = 1;
              }
              else {
                crm[base + k] = 0;
              }
            }
            else {
              crm[base + k] = 1;
            }
          }
        }
        else {
          if (rn.getReal() < 0.75) {
            crm[base] = 1;
          }
          else {
            crm[base] = 0;
          }

          if (rn.getReal() < 0.75) {
            crm[base + 1] = 1;
          }
          else {
            crm[base + 1] = 0;
          }

          double min, max;
          double size = (rn.getReal() * 0.25 + 0.5)
              * ParallelGlobals.getGlobals_UBR().sizeD[j];

          if (ins != null) {
            double minD = ParallelGlobals.getGlobals_UBR().minD[j];
            double maxD = ParallelGlobals.getGlobals_UBR().maxD[j];
            double val = ins.getRealValue(j);
            min = val - size / 2.0;
            max = val + size / 2.0;
            if (min < minD) {
              min = minD;
            }

            if (max > maxD) {
              max = maxD;
            }
          }
          else {
            min = rn.getReal()
                * (ParallelGlobals.getGlobals_UBR().sizeD[j]
                   - size)
                + ParallelGlobals.getGlobals_UBR().minD[j];
            max = min + size;
          }

          if (rn.getReal() < 0.5) {
            crm[base + 2] = min;
            crm[base + 3] = max;
          }
          else {
            crm[base + 3] = min;
            crm[base + 2] = max;
          }
        }
        base += ParallelGlobals.getGlobals_UBR().size[j];
      }

      if (ins != null) {
        crm[base] = ins.classOfInstance();
      }
      else {
        do {
          crm[base] = rn.getInteger(0, nC - 1);
        }
        while (ParallelGlobals.getGlobals_DefaultC().enabled &&
               crm[base] == defaultClass);
      }
      base++;
    }

    resetPerformance();
  }

  public double computeTheoryLength(PerformanceAgent pa) {
    int base = 0;
    int ruleSize = ParallelGlobals.getGlobals_UBR().ruleSize;
    theoryLength = 0;
    for (int i = 0; i < numRules; i++) {
      if (pa.getActivationsOfRule(i) > 0) {
        int base2 = base;
        for (int j = 0; j < Parameters.numAttributes; j++) {
          if (ParallelGlobals.getGlobals_UBR().types[j] == Attribute.REAL) {
            double min = crm[base2];
            double max = crm[base2 + 1];
            double vmin = crm[base2 + 2];
            double vmax = crm[base2 + 3];
            if (vmin > vmax) {
              double tmp = vmin;
              vmin = vmax;
              vmax = tmp;
            }
            double dmin = ParallelGlobals.getGlobals_UBR().minD[j];
            double dmax = ParallelGlobals.getGlobals_UBR().maxD[j];

            if (min == 1) {
              vmin = dmin;
            }
            if (max == 1) {
              vmax = dmax;
            }

            theoryLength += 5.0;
            if (dmax - dmin > 0) {
              theoryLength += 5.0 - (vmax - vmin) / (dmax - dmin) * 5.0;
            }
          }
          else {
            double countFalses = 0;
            int numValues = ParallelGlobals.getGlobals_UBR().size[j];
            for (int k = 0; k < numValues; k++) {
              if (crm[base2 + k] == 0) {
                countFalses++;
              }
            }
            theoryLength += numValues
                + countFalses;
          }
          base2 += ParallelGlobals.getGlobals_UBR().size[j];
        }
      }
      if (ParallelGlobals.getGlobals_DefaultC().enabled) {
        theoryLength += 0.00000001;
      }
      base += ruleSize;
    }
    return theoryLength;
  }

  public int doMatch(InstanceWrapper ins) {
    int nA = Parameters.numAttributes;
    boolean okMatch;
    int i, j;
    int base = 0;
    int ruleSize = ParallelGlobals.getGlobals_UBR().ruleSize;

    int[] valN = ins.getNominalValues();
    double[] valR = ins.getRealValues();

    for (i = 0; i < numRules; i++) {
      okMatch = true;
      int base2 = 0;
      for (j = 0; okMatch && j < nA; j++) {
        if (ParallelGlobals.getGlobals_UBR().types[j] == Attribute.NOMINAL) {
          if (crm[base + base2 + valN[j]] == 0) {
            okMatch = false;
          }
        }
        else {
          double min = crm[base + base2 + 2];
          double max = crm[base + base2 + 3];
          if (max < min) {
            double temp = max;
            max = min;
            min = temp;
          }
          if ( (crm[base + base2] == 0 && valR[j] < min) ||
              (crm[base + base2 + 1] == 0 && valR[j] > max)) {
            okMatch = false;
          }
        }
        base2 += ParallelGlobals.getGlobals_UBR().size[j];
      }

      if (okMatch) {
        positionRuleMatch = i;
        return (int) crm[base + base2];
      }
      base += ParallelGlobals.getGlobals_UBR().ruleSize;
    }
    if (ParallelGlobals.getGlobals_DefaultC().enabled) {
      positionRuleMatch = numRules;
      return defaultClass;
    }
    return -1;
  }

  public void printClassifier() {
    int nA = Parameters.numAttributes;
    int ruleSize = ParallelGlobals.getGlobals_UBR().ruleSize;
    String str;
    int base = 0;

    for (int i = 0; i < numRules; i++) {
      str = i + ":";
      for (int j = 0; j < nA; j++) {
        if (ParallelGlobals.getGlobals_UBR().types[j] == Attribute.NOMINAL) {
          for (int k = 0; k < ParallelGlobals.getGlobals_UBR().size[j]; k++) {
            str += (int) crm[base + k];
          }
        }
        else {
          double minD = ParallelGlobals.getGlobals_UBR().minD[j];
          double maxD = ParallelGlobals.getGlobals_UBR().maxD[j];
          double min = crm[base + 2];
          double max = crm[base + 3];

          if (max < min) {
            double temp = max;
            max = min;
            min = temp;
          }
          if (crm[base] == 1 || min <= minD) {
            min = minD;
          }
          if (crm[base + 1] == 1 || max >= maxD) {
            max = maxD;
          }
          str += "[" + min + "," + max + "]";
        }
        str += "|";
        base += ParallelGlobals.getGlobals_UBR().size[j];
      }
      str += (int) crm[base];
      LogManager.println(str);
      base++;
    }
    if (ParallelGlobals.getGlobals_DefaultC().enabled) {
      LogManager.println(numRules + ":Default rule -> " + defaultClass);
    }

  }

  public int getNumRules() {
    if (ParallelGlobals.getGlobals_DefaultC().enabled) {
      return numRules + 1;
    }
    return numRules;
  }

  public Classifier[] crossoverClassifiers(Classifier _parent2) {
    Rand rn = ParallelGlobals.getRand();
    
    ClassifierUBR offspring1 = new ClassifierUBR();
    ClassifierUBR offspring2 = new ClassifierUBR();
    ClassifierUBR parent2 = (ClassifierUBR) _parent2;

    int ruleSize = ParallelGlobals.getGlobals_UBR().ruleSize;
    int ruleP1 = (int) rn.getInteger(0, numRules - 1);
    int ruleP2 = (int) rn.getInteger(0, parent2.numRules - 1);
    offspring1.numRules = ruleP1 + parent2.numRules - ruleP2;
    offspring2.numRules = ruleP2 + numRules - ruleP1;
    int cutPoint = (int) rn.getInteger(0, ParallelGlobals.getGlobals_UBR().ruleSize);
    offspring1.defaultClass = offspring2.defaultClass = defaultClass;

    offspring1.crm = new double[ruleSize * offspring1.numRules];
    offspring2.crm = new double[ruleSize * offspring2.numRules];

    System.arraycopy(crm, 0, offspring1.crm, 0, ruleP1 * ruleSize);
    System.arraycopy(parent2.crm, 0, offspring2.crm, 0, ruleP2 * ruleSize);

    int base1 = ruleP1 * ruleSize;
    int base2 = ruleP2 * ruleSize;

    System.arraycopy(crm, base1, offspring1.crm, base1, cutPoint);
    System.arraycopy(parent2.crm, base2, offspring2.crm, base2, cutPoint);
    System.arraycopy(crm, base1 + cutPoint, offspring2.crm, base2 + cutPoint,
                     ruleSize - cutPoint);
    System.arraycopy(parent2.crm, base2 + cutPoint, offspring1.crm,
                     base1 + cutPoint, ruleSize - cutPoint);

    base1 += ruleSize;
    base2 += ruleSize;
    System.arraycopy(crm, base1, offspring2.crm, base2,
                     (numRules - ruleP1 - 1) * ruleSize);
    System.arraycopy(parent2.crm, base2, offspring1.crm, base1,
                     (parent2.numRules - ruleP2 - 1) * ruleSize);

    Classifier[] ret = new Classifier[2];
    ret[0] = offspring1;
    ret[1] = offspring2;

    return ret;
  }

  public Classifier copy() {
    int ruleSize = ParallelGlobals.getGlobals_UBR().ruleSize;
    ClassifierUBR ret = new ClassifierUBR();

    ret.numRules = numRules;
    ret.theoryLength = theoryLength;
    ret.exceptionsLength = ret.exceptionsLength;
    ret.crm = new double[numRules * ruleSize];
    System.arraycopy(crm, 0, ret.crm, 0, numRules * ruleSize);
    ret.defaultClass = defaultClass;

    ret.setAccuracy(accuracy);
    ret.setFitness(fitness);
    ret.isEvaluated = isEvaluated;
    ret.setNumAliveRules(numAliveRules);
    return ret;
  }

  private double mutationOffset(double value, double offMin, double offMax) {
    Rand rn = ParallelGlobals.getRand();
    
    double newVal;
    if (rn.getReal() < 0.5) {
      newVal = value + rn.getReal() * offMax;
    }
    else {
      newVal = value - rn.getReal() * offMin;
    }
    return newVal;
  }

  public void doMutation() {
    Rand rn = ParallelGlobals.getRand();
    
    int whichRule = (int) rn.getInteger(0, numRules - 1);
    int ruleSize = ParallelGlobals.getGlobals_UBR().ruleSize;
    int base = whichRule * ruleSize;
    int pos;

    if (ParallelGlobals.getGlobals_DefaultC().numClasses > 1 && rn.getReal() < 0.1) {
      pos = ruleSize - 1;
      double oldValue = crm[base + pos];
      double newValue;
      do {
        newValue = rn.getInteger(0, Parameters.numClasses - 1);
      }
      while (newValue == oldValue ||
             (ParallelGlobals.getGlobals_DefaultC().enabled && (int) newValue == defaultClass));
      crm[base + pos] = newValue;
    }
    else {
      int attr = (int) rn.getInteger(0, Parameters.numAttributes - 1);
      if (ParallelGlobals.getGlobals_UBR().types[attr] == Attribute.NOMINAL) {
        int val = (int) rn.getInteger(0, ParallelGlobals.getGlobals_UBR().size[attr] - 1);
        pos = base + ParallelGlobals.getGlobals_UBR().offset[attr] + val;
        if (crm[pos] == 1) {
          crm[pos] = 0;
        }
        else {
          crm[pos] = 1;
        }
      }
      else {
        int val = (int) rn.getInteger(0, 3);
        pos = base + ParallelGlobals.getGlobals_UBR().offset[attr] + val;
        if (val < 2) {
          if (crm[pos] == 1) {
            crm[pos] = 0;
          }
          else {
            crm[pos] = 1;
          }
        }
        else {
          double minOff, maxOff;
          minOff = maxOff = 0.7 * ParallelGlobals.getGlobals_UBR().sizeD[attr];
          if (crm[pos] - minOff < ParallelGlobals.getGlobals_UBR().minD[attr]) {
            minOff = crm[pos] - ParallelGlobals.getGlobals_UBR().minD[attr];
          }
          if (crm[pos] + maxOff > ParallelGlobals.getGlobals_UBR().maxD[attr]) {
            maxOff = ParallelGlobals.getGlobals_UBR().maxD[attr] - crm[pos];
          }
          crm[pos] = mutationOffset(crm[pos], minOff, maxOff);
        }
      }
    }

    isEvaluated = false;
  }

  public void deleteRules(int[] whichRules) {
    if (numRules == 1 || whichRules.length == 0) {
      return;
    }

    int ruleSize = ParallelGlobals.getGlobals_UBR().ruleSize;
    int rulesToDelete = whichRules.length;
    if (whichRules[rulesToDelete - 1] == numRules) {
      rulesToDelete--;
    }

    double[] newCrm = new double[ruleSize * (numRules - rulesToDelete)];
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

  public double getLength() {
    return numAliveRules;
  }

  public int numSpecialStages() {
    return 0;
  }

  public void doSpecialStage(int stage) {}

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

