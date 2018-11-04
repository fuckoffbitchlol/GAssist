/***********************************************************************

	This file is part of KEEL-software, the Data Mining tool for regression, 
	classification, clustering, pattern mining and so on.

	Copyright (C) 2004-2010
	
	F. Herrera (herrera@decsai.ugr.es)
    L. S閻ゎ柀hez (luciano@uniovi.es)
    J. Alcal?绱絛ez (jalcala@decsai.ugr.es)
    S. Garc閸�?(sglopez@ujaen.es)
    A. Fern閻ゎ柂ez (alberto.fernandez@ujaen.es)
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
 * @author Written by Jaume Bacardit (La Salle, Ram顛掞拷 Llull University - Barcelona) 28/03/2004
 * @author Modified by Xavi Sol?锟�(La Salle, Ram顛掞拷 Llull University - Barcelona) 23/12/2008
 * @version 1.1
 * @since JDK1.2
 * </p>
 */


package GAssist_Parallel;
import java.util.*;

import javax.swing.event.TreeWillExpandListener;

import keel.Algorithms.Genetic_Rule_Learning.Globals.*;



public class Windowing {
/**
 * <p>
 * Manages the subset of training instances
 * that is used at each iteration to perform the fitness computations
 * </p>
 */
	
  InstanceWrapper[] is;
  InstanceWrapper[][] strata;
  InstanceWrapper[] allData;
  // out of bag data use for out of bag estimation
  InstanceWrapper[] ofbData;
  
  int [][] patErrorCnt;
  int [][] patternCnt;
  int [][] curPatNum;
  
  int numStrata;
  int currentIteration;
  boolean lastIteration;

  public Windowing(Rand rn, InstanceWrapper[] _is) {
    is = _is;
    numStrata = Parameters.parallelParts;
    strata = new InstanceWrapper[numStrata][];
    currentIteration = 0;
    lastIteration = false;
    
    // create pattern number of error count
    patErrorCnt = new int [numStrata][is.length];
    // create list to count pattern in each thread
    patternCnt = new int[numStrata][is.length];
    curPatNum = new int[numStrata][is.length];
    // each thread have one pattern
    for (int i = 0; i < numStrata; i++) {
		for (int j = 0; j < is.length; j++) {
			patternCnt[i][j] = 1;
			curPatNum[i][j] = j;
		}
	}
    
    createStrata(rn);
  }
  
  public int[][] getPatternCnt() {
	  return patternCnt;
  }
  
  private void createStrata(Rand rn) {
    if (Parameters.windowingMethod.equalsIgnoreCase("dob-scv")) {
      createStrataDefault(rn);
    }
    else {
//      createStrataDefault(rn);
//       Ensemble version
       createStrataEnsembleBag(rn);
    }
  }
  
  private void createStrataDobSCV(Rand rn) {
    Vector[] tempStrata = WindowingDobSCV.createStrata(is, numStrata, rn);
    
    //TODO: is this strata the parallel parts ?
    for (int i = 0; i < numStrata; i++) {
      int num = tempStrata[i].size();
      strata[i] = new InstanceWrapper[num];
      for (int j = 0; j < num; j++) {
        strata[i][j] = (InstanceWrapper) tempStrata[i].elementAt(j);
      }
    }
  }

  private void createStrataDefault(Rand rn) {
    Vector[] tempStrata = new Vector[numStrata];
    Vector[] instancesOfClass = new Vector[Parameters.numClasses];

    for (int i = 0; i < numStrata; i++) {
      tempStrata[i] = new Vector();
    }
    for (int i = 0; i < Parameters.numClasses; i++) {
      instancesOfClass[i] = new Vector();
    }

    int numInstances = is.length;
    for (int i = 0; i < numInstances; i++) {
      int cl = is[i].classOfInstance();
      instancesOfClass[cl].addElement(is[i]);
    }

    for (int i = 0; i < Parameters.numClasses; i++) {
      int stratum = 0;
      int count = instancesOfClass[i].size();
      while (count >= numStrata) {
        int pos = rn.getInteger(0, count - 1);
        tempStrata[stratum].addElement(instancesOfClass[i].elementAt(pos));
        instancesOfClass[i].removeElementAt(pos);
        stratum = (stratum + 1) % numStrata;
        count--;
      }
      while (count > 0) {
        stratum = rn.getInteger(0, numStrata - 1);
        tempStrata[stratum].addElement(instancesOfClass[i].elementAt(0));
        instancesOfClass[i].removeElementAt(0);
        count--;
      }
    }

    for (int i = 0; i < numStrata; i++) {
      int num = tempStrata[i].size();
      strata[i] = new InstanceWrapper[num];
      for (int j = 0; j < num; j++) {
        strata[i][j] = (InstanceWrapper) tempStrata[i].elementAt(j);
      }
    }
  }

  private void createStrataEnsembleBag(Rand rn) {

	Vector[] tempStrata = new Vector[numStrata];
    Vector[] instancesOfClass = new Vector[Parameters.numClasses];

    for (int i = 0; i < numStrata; i++) {
      tempStrata[i] = new Vector();
    }
    for (int i = 0; i < Parameters.numClasses; i++) {
      instancesOfClass[i] = new Vector();
    }

    int numInstances = is.length;
    
    for(int i = 0; i < numStrata; i++) {
    	int cnt = 0;
    	while(cnt < numInstances) {
    		int k = rn.getInteger(0, numInstances-1);
    		tempStrata[i].addElement(is[k]);
    		cnt ++;
    	}
    }

    for (int i = 0; i < numStrata; i++) {
      int num = tempStrata[i].size();
      strata[i] = new InstanceWrapper[num];
      for (int j = 0; j < num; j++) {
        strata[i][j] = (InstanceWrapper) tempStrata[i].elementAt(j);
      }
    }
  }

   private void createStrataEnsemble() {
    
    // just copy default data to each part
    for (int i = 0; i < numStrata; i++) {
      int num = is.length;
      strata[i] = new InstanceWrapper[num];
      for (int j = 0; j < num; j++) {
        strata[i][j] = is[j];
      }
    }
  }
  
  
  public InstanceWrapper[] getInstances(int strataToUse) {
    
	  return strata[strataToUse];
  }
  
  // migrateData version 0.1 only migrate one pattern per generation
  public void migrateData() {
	
    Rand rn = new Rand();
    Random r = new Random();
    rn.initRand(Parameters.seed + r.nextInt());

    // care the migration direction
    int migrateNum = (int) (is.length * 0.01);
    
    while (migrateNum > 0){
    
    int [] noPattern = new int[numStrata];
    
    int insClas = 0;
    
    int [] max_p = new int[numStrata];
    
    for (int s = 0; s < numStrata; s++) {
    int max_v = 0;
    	for (int j = 0; j < is.length; j++ ) {
    		if (patErrorCnt[s][j] > max_v) {
    			max_v = patErrorCnt[s][j];
    			max_p[s] = j;
    		}
    	}
    }
    
    
    // assign which pattern to exchange for each strata
//    int num = strata[0].length;
//    noPattern[0] = rn.getInteger(0, num - 1);
//    insClas = strata[0][noPattern[0]].classOfInstance();
//    
//    for (int i = 1; i < numStrata; i++) {
//      num = strata[i].length;
//      noPattern[i] = rn.getInteger(0, num - 1); 
//      while( strata[i][noPattern[i]].classOfInstance() != insClas) {
//        noPattern[i] = rn.getInteger(0, num - 1); 
//      }
//    }
    
    for (int i = 0; i < numStrata; i ++) {
    	noPattern[i] = max_p[i];
    	patErrorCnt[i][max_p[i]] = 0;
    }
    
    // exchange patterns here
    for (int i = 0; i < numStrata - 1; i++) {
      int cur = noPattern[i];
      int pre = 0;
      
      if (i == 0) {
        pre = noPattern[numStrata - 1];
      }
      else {
        pre = noPattern[i - 1];
      }
      
      InstanceWrapper temp;
      int tmpNo = 0;
      
      if( i == 0) {
        temp = strata[i][cur];
        strata[i][cur] = strata[numStrata - 1][pre]; 
        strata[numStrata - 1][pre] = temp;
        
        tmpNo = curPatNum[i][cur];
        curPatNum[i][cur] = curPatNum[numStrata - 1][pre];
        curPatNum[numStrata - 1][pre] = tmpNo; 
        
        // modify the pattern count
        patternCnt[i][curPatNum[i][cur]] -= 1;
        patternCnt[i][curPatNum[numStrata - 1][pre]] += 1;
        patternCnt[numStrata - 1][curPatNum[i][cur]] += 1;
        patternCnt[numStrata - 1][curPatNum[numStrata -1][pre]] -= 1; 
//        LogManager.println("cur instance");
//        LogManager.println(strata[i][cur]+"abc");
//        LogManager.println("pre instance");
//        LogManager.println(strata[i][pre]+"abc");
      }
      else {
        temp = strata[i][cur];
        strata[i][cur] = strata[i - 1][pre]; 
        strata[i - 1][pre] = temp;
        
        tmpNo = curPatNum[i][cur];
        curPatNum[i][cur] = curPatNum[i - 1][pre];
        curPatNum[i - 1][pre] = tmpNo; 
        
        patternCnt[i][curPatNum[i][cur]] -= 1;
        patternCnt[i][curPatNum[i-1][pre]] += 1;
        patternCnt[i - 1][curPatNum[i][cur]] += 1;
        patternCnt[i - 1][curPatNum[i-1][pre]] -= 1; 

      }
    }
    migrateNum -= 1;
    }
  }
 
  public int numVersions() {
    if (lastIteration) {
      return 1;
    }
    return numStrata;
  }

  public int getCurrentVersion() {
    if (lastIteration) {
      return 0;
    }
    return currentIteration % numStrata;
  }

  private static <T> T[] concatAll(T[] first, T[]... rest) {
	  int totalLength = first.length;
	  for (T[] array : rest) {
	    totalLength += array.length;
	  }
	  T[] result = Arrays.copyOf(first, totalLength);
	  int offset = first.length;
	  for (T[] array : rest) {
	    System.arraycopy(array, 0, result, offset, array.length);
	    offset += array.length;
	  }
	  return result;
	}
}

