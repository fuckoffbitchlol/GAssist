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

public class AdaptiveRule {
  public static void constructor(Rand rn, int[] crm, int base, int defaultClass, int strata) {
    InstanceWrapper ins = null;
    if (PopulationWrapper.smartInit) {
      if (ParallelGlobals.getGlobals_DefaultC().defaultClassPolicy != ParallelGlobals.getGlobals_DefaultC().DISABLED) {
        ins = PopulationWrapper.getInstanceInit(defaultClass, strata);
      }
      else {
        ins = PopulationWrapper.getInstanceInit(Parameters.numClasses, strata);
      }
    }

    int base2 = base + 2;
    crm[base] = 1;
    for (int i = 0; i < Parameters.numAttributes; i++) {
      AdaptiveAttribute.constructor(rn, crm, base2, i, ins);
      crm[base] += crm[base2];
      base2 += ParallelGlobals.getGlobals_ADI().size[i];
    }

    if (ins != null) {
      crm[base + 1] = ins.classOfInstance();
    }
    else {
      do {
        crm[base + 1] = rn.getInteger(0, Parameters.numClasses - 1);
      }
      while (ParallelGlobals.getGlobals_DefaultC().enabled && crm[base + 1] == defaultClass);
    }
  }

  public static double computeTheoryLength(int[] crm, int base) {
    int base2 = base + 2;
    double length = 0;
    for (int i = 0; i < Parameters.numAttributes; i++) {
      if (ParallelGlobals.getGlobals_ADI().types[i] == Attribute.REAL) {
        double intervalCount = 0;
        int previousValue = crm[base2 + 3];
        int numInt = crm[base2];
        if (previousValue == 1) {
          intervalCount++;
        }
        for (int j = base2 + 5, k = 1; k < numInt; k++, j += 2) {
          if (crm[j] != previousValue) {
            intervalCount++;
          }
          previousValue = crm[j];
        }
        if (previousValue == 0) {
          intervalCount--;
        }
        length += numInt + intervalCount;
      }
      else {
        double countFalses = 0;
        int numValues = ParallelGlobals.getGlobals_ADI().size[i];
        for (int j = 2, pos = base2 + j; j < numValues; j++, pos++) {
          if (crm[pos] == 0) {
            countFalses++;
          }
        }
        length += numValues - 2.0 + countFalses;
      }
      base2 += ParallelGlobals.getGlobals_ADI().size[i];
    }
    return length;
  }

  public static boolean doMatch(int[] crm, int base, InstanceWrapper ins) {
    int base2 = base + 2;
    int[][] discValues = ins.getDiscretizedValues();
    int[] nominalValues = ins.getNominalValues();
    for (int i = 0; i < Parameters.numAttributes; i++) {
      if (ParallelGlobals.getGlobals_ADI().types[i] == Attribute.REAL) {
        int value = discValues[i][crm[base2 + 1]];
        if (!AdaptiveAttribute.doMatchReal(crm, base2, value)) {
          return false;
        }
      }
      else {
        int value = nominalValues[i];
        if (!AdaptiveAttribute.doMatchNominal(crm, base2, value)) {
          return false;
        }
      }
      base2 += ParallelGlobals.getGlobals_ADI().size[i];
    }
    return true;
  }

  public static String dumpPhenotype(int[] crm, int base) {
    int base2 = base + 2;
    String str = "";
    for (int i = 0; i < Parameters.numAttributes; i++) {
      String temp
          = AdaptiveAttribute.dumpPhenotype(crm, base2, i);
      if (temp.length() > 0) {
        str += temp + "|";
      }
      base2 += ParallelGlobals.getGlobals_ADI().size[i];
    }
    int cl = crm[base + 1];
    String name = Attributes.getAttribute(Parameters.numAttributes).
        getNominalValue(cl);
    str += name;
    return str;
  }

  public static void crossover(int[] p1, int[] p2, int[] s1, int[] s2
                               , int base1, int base2, int cutPoint) {
    int baseP1 = base1 + 2;
    int baseP2 = base2 + 2;

    s1[base1] = 1;
    s2[base2] = 1;

    for (int i = 0; i < cutPoint && i < Parameters.numAttributes; i++) {
      int inc = ParallelGlobals.getGlobals_ADI().size[i];
      System.arraycopy(p1, baseP1, s1, baseP1, inc);
      System.arraycopy(p2, baseP2, s2, baseP2, inc);
      s1[base1] += p1[baseP1];
      s2[base2] += p2[baseP2];
      baseP1 += inc;
      baseP2 += inc;
    }
    for (int i = cutPoint; i < Parameters.numAttributes; i++) {
      int inc = ParallelGlobals.getGlobals_ADI().size[i];
      System.arraycopy(p1, baseP1, s2, baseP2, inc);
      System.arraycopy(p2, baseP2, s1, baseP1, inc);
      s1[base1] += p2[baseP2];
      s2[base2] += p1[baseP1];
      baseP1 += inc;
      baseP2 += inc;
    }

    if (cutPoint == Parameters.numAttributes) {
      s1[base1 + 1] = p1[base1 + 1];
      s2[base2 + 1] = p2[base2 + 1];
    }
    else {
      s1[base1 + 1] = p2[base2 + 1];
      s2[base2 + 1] = p1[base1 + 1];
    }

  }

  public static void mutation(Rand rn, int[] crm, int base, int defaultClass) {
    if (ParallelGlobals.getGlobals_DefaultC().numClasses > 1 && rn.getReal() < 0.10) {
      int newClass;
      int oldClass = crm[base + 1];
      do {
        newClass = rn.getInteger(0, Parameters.numClasses - 1);
      }
      while (newClass == oldClass || (ParallelGlobals.getGlobals_DefaultC().enabled
                                      && newClass == defaultClass));
      crm[base + 1] = newClass;
    }
    else {
      int attribute = rn.getInteger(0, Parameters.numAttributes - 1);
      int base2 = base + 2 + ParallelGlobals.getGlobals_ADI().offset[attribute];
      AdaptiveAttribute.mutation(rn, crm, base2, attribute);
    }
  }

  public static boolean doSplit(Rand rn, int[] crm, int base) {
    int base2 = base + 2;
    boolean modif = false;

    for (int i = 0; i < Parameters.numAttributes; i++) {
      if (rn.getReal() < Parameters.probSplit) {
        modif = true;
        int pos = rn.getInteger(0, crm[base2] - 1);
        crm[base] += AdaptiveAttribute.doSplit(rn, crm, base2, i, pos);
      }
      base2 += ParallelGlobals.getGlobals_ADI().size[i];
    }
    return modif;
  }

  public static boolean doMerge(Rand rn, int[] crm, int base) {
    int base2 = base + 2;
    boolean modif = false;

    for (int i = 0; i < Parameters.numAttributes; i++) {
      if (rn.getReal() < Parameters.probMerge) {
        modif = true;
        int pos = rn.getInteger(0, crm[base2] - 1);
        crm[base] += AdaptiveAttribute.doMerge(rn, crm, base2, i, pos);
      }
      base2 += ParallelGlobals.getGlobals_ADI().size[i];
    }
    return modif;
  }

  public static boolean doReinitialize(Rand rn, int[] crm, int base) {
    int base2 = base + 2;
    boolean modif = false;
    
    int threadNo = ParallelGlobals.getThreadNo();

    if (Parameters.probReinitializePerThread[threadNo] == 0) {
      return modif;
    }

    for (int i = 0; i < Parameters.numAttributes; i++) {
      if (rn.getReal() < Parameters.probReinitializePerThread[threadNo]) {
        modif = true;
        crm[base] -= crm[base2];
        AdaptiveAttribute.doReinitialize(rn, crm, base2, i);
        crm[base] += crm[base2];
      }
      base2 += ParallelGlobals.getGlobals_ADI().size[i];
    }
    return modif;
  }

}

