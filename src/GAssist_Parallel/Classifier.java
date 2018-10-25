/***********************************************************************

	This file is part of KEEL-software, the Data Mining tool for regression, 
	classification, clustering, pattern mining and so on.

	Copyright (C) 2004-2010
	
	F. Herrera (herrera@decsai.ugr.es)
    L. S鐤hez (luciano@uniovi.es)
    J. Alcal?紽dez (jalcala@decsai.ugr.es)
    S. Garc鍍?(sglopez@ujaen.es)
    A. Fern鐤ez (alberto.fernandez@ujaen.es)
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
 * @author Written by Jaume Bacardit (La Salle, Ram� Llull University - Barcelona) 28/03/2004
 * @author Modified by Xavi Sol?�(La Salle, Ram� Llull University - Barcelona) 23/12/2008
 * @version 1.1
 * @since JDK1.2
 * </p>
 */


package GAssist_Parallel;
import keel.Algorithms.Genetic_Rule_Learning.Globals.*;

abstract public class Classifier {
/**
 * <p>
 * Base class for all classifiers (knowledge representations)
 * </p>
 */

  protected boolean isEvaluated;

  protected boolean bloatControlDone;

  protected double accuracy;

  protected double fitness;

  protected double exceptionsLength;

  protected double theoryLength;

  protected int numAliveRules;

  protected int positionRuleMatch;

  protected int numRules;
  
  protected int globalNumAliveRules;
  protected int globalNumRules;
  protected double globalLength;
  protected double globalFitness;
  protected double globalAccuracy;

  public abstract void initRandomClassifier(int strata);

  public abstract int doMatch(InstanceWrapper ins);

  public abstract int getNumRules();
  
  public int getGlobalNumRules() { 
    return globalNumRules;
  }

  public abstract void deleteRules(int[] whichRules);

  public abstract Classifier[] crossoverClassifiers(Classifier _parent2);

  public abstract void doMutation();

  public abstract Classifier copy();

  public abstract void printClassifier();

  public boolean getIsEvaluated() {
    return isEvaluated;
  }

  public void setIsEvaluated(boolean _isEvaluated) {
    isEvaluated = _isEvaluated;
  }

  double getAccuracy() {
    return accuracy;
  }

  public void setAccuracy(double _accuracy) {
    accuracy = _accuracy;
  }

  public double getFitness() {
    return fitness;
  }
  
  public double getGlobalFitness() {
    return globalFitness;
  }
  
  public double getGlobalAccuracy() {
    return globalAccuracy;
  }

  public void setFitness(double _fitness) {
    fitness = _fitness;
  }

  public double getExceptionsLength() {
    return exceptionsLength;
  }

  public void setExceptionsLength(double _exceptionsLength) {
    exceptionsLength = _exceptionsLength;
  }

  public int getNumAliveRules() {
    return numAliveRules;
  }
  
  public int getGlobalNumAliveRules() {
    return globalNumAliveRules;
  }

  public void setNumAliveRules(int _numAliveRules) {
    numAliveRules = _numAliveRules;
  }
  
  public void resetPerformance() {
    accuracy = 0;
    fitness = 0;
    numAliveRules = 0;
    isEvaluated = false;
  }

  public void computePerformance(PerformanceAgent pa) {
    accuracy = pa.getAccuracy();
    fitness = pa.getFitness(this, pa);
    numAliveRules = pa.getNumAliveRules();
    isEvaluated = true;
  }
  
  public void computeGlobalPerformance(PerformanceAgent pa) {
    globalAccuracy = pa.getAccuracy();
    globalFitness = pa.getFitness(this, pa);
    globalNumAliveRules = pa.getNumAliveRules();
    globalNumRules = getNumRules();
    globalLength = getLength();
  }
  
  public double getGlobalLength() {
    return globalLength;
  }

  /**
   *	 positionRuleMatch contains the position within the classifier
   *	 (e.g. the rule) that matched the last classified input
   *	 instance
   */
  public int getPositionRuleMatch() {
    return positionRuleMatch;
  }

  public void setPositionRuleMatch(int _positionRuleMatch) {
    positionRuleMatch = _positionRuleMatch;
  }

  public abstract double getLength();

  public double getTheoryLength() {
    return theoryLength;
  }

  public abstract double computeTheoryLength(PerformanceAgent pa);

  /**
   * This function returns true if this individual is better than
   * the the individual passed as a parameter. This comparison can
   * be based on accuracy or a combination of accuracy and size
   */
  public boolean compareToIndividual(Classifier ind) {
    double l1 = getLength();
    double l2 = ind.getLength();
    double f1 = getFitness();
    double f2 = ind.getFitness();
    
    Rand rn = ParallelGlobals.getRand();
    int threadNo = ParallelGlobals.getThreadNo();

    if (Parameters.doHierarchicalSelectionPerThread[threadNo]) {
      if (Math.abs(f1 - f2) <= Parameters.hierarchicalSelectionThreshold) {
        if (l1 < l2) {
          return true;
        }
        if (l1 > l2) {
          return false;
        }
      }
    }

    if (Parameters.useMDL == false) {
      if (f1 > f2) {
        return true;
      }
      if (f1 < f2) {
        return false;
      }
      if (rn.getReal() < 0.5) {
        return true;
      }
      return false;
    }

    if (f1 < f2) {
      return true;
    }
    if (f1 > f2) {
      return false;
    }
    if (rn.getReal() < 0.5) {
      return true;
    }
    return false;
  }
  
  public boolean globalCompareToIndividual(Classifier ind) {
    
    if (ind == null) {
      return true;
    }
    
    Rand rn = ParallelGlobals.getRand();
    
    double l1 = getLength();
    double l2 = ind.getLength();
    double f1 = getFitness();
    double f2 = ind.getFitness();
    
    double e1 = getAccuracy();
    double e2 = ind.getAccuracy();

    /*
    if (Parameters.useMDL == false) {
      if (f1 > f2) {
        return true;
      }
      if (f1 < f2) {
        return false;
      }
      if (rn.getReal() < 0.5) {
        return true;
      }
      return false;
    }

    if (f1 < f2) {
      return true;
    }
    if (f1 > f2) {
      return false;
    }
    if (rn.getReal() < 0.5) {
      return true;
    }
    */
    
    if (e1 > e2) {
      return true;
    }
    else if (e1 < e2) {
      return false;
    }
    
    if (l1 < l2) {
      return true;
    }
    else if (l1 > l2) {
      return false;
    }
    if (rn.getReal() < 0.5) {
      return true;
    }
    
    return false;
    
  }
  
  public boolean ESCompareToIndividual(Classifier ind) {
    
    if (ind == null) {
      return true;
    }
    
    Rand rn = ParallelGlobals.getRand();
    
    double l1 = getGlobalLength();
    double l2 = ind.getGlobalLength();
    double f1 = getGlobalFitness();
    double f2 = ind.getGlobalFitness();
    
    double e1 = getGlobalAccuracy();
    double e2 = ind.getGlobalAccuracy();

    /*
    if (Parameters.useMDL == false) {
      if (f1 > f2) {
        return true;
      }
      if (f1 < f2) {
        return false;
      }
      if (rn.getReal() < 0.5) {
        return true;
      }
      return false;
    }

    if (f1 < f2) {
      return true;
    }
    if (f1 > f2) {
      return false;
    }
    if (rn.getReal() < 0.5) {
      return true;
    }
    */
    
    if (e1 > e2) {
      return true;
    }
    else if (e1 < e2) {
      return false;
    }
    
    if (l1 < l2) {
      return true;
    }
    else if (l1 > l2) {
      return false;
    }
    if (rn.getReal() < 0.5) {
      return true;
    }
    
    return false;
    
  }


  public abstract int getNiche();

  public abstract int getNumNiches();

  public abstract int numSpecialStages();

  public abstract void doSpecialStage(int stage);

}

