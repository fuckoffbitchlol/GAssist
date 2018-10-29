package GAssist_Parallel;

import java.io.*;
import java.nio.charset.Charset;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.Random;
import java.util.Vector;

import keel.Algorithms.Genetic_Rule_Learning.Globals.LogManager;

public final class WindowingDobSCV {
  
  private static Comparator<InstanceWrapper> cmp = new Comparator<InstanceWrapper>() {
    public int compare(InstanceWrapper o1, InstanceWrapper o2){
        return Double.compare(o1.getDistance(), o2.getDistance());
    }
  };
  
  // Run the program
  @SuppressWarnings("unchecked")
  public static Vector<InstanceWrapper>[] createStrata(InstanceWrapper[] is, int numStrata, Rand rn) {
    Vector<InstanceWrapper>[] tempStrata = new Vector[numStrata];
    Vector<InstanceWrapper>[] instancesOfClass = new Vector[Parameters.numClasses];
    
    int[] numPatternsPerStrata = new int[numStrata];
    
    for (int i = 0; i < numStrata; i++) {
      tempStrata[i] = new Vector<InstanceWrapper>();
    }
    for (int i = 0; i < Parameters.numClasses; i++) {
      instancesOfClass[i] = new Vector<InstanceWrapper>();
    }

    System.out.print("Dividing data through DOB-SCV");
    
    int curStrata = 0;
    
    int numInstances = is.length;
    for (int i = 0; i < numInstances; i++) {
      int cl = is[i].classOfInstance();
      instancesOfClass[cl].addElement(is[i]);
    }
    
    outputAverageDistance(instancesOfClass);
    
    for (int i = 0; i < Parameters.numClasses; i++) {
      System.out.print(".");
      
      int patternCount = instancesOfClass[i].size();
      
      while (patternCount > 0) {
       
        int rand = patternCount > 1 ? rn.getInteger(0, patternCount - 1) : 0;
        
        InstanceWrapper ins = instancesOfClass[i].elementAt(rand);
        
        instancesOfClass[i].removeElementAt(rand);
        patternCount--;

        Collections.shuffle(instancesOfClass[i], new Random(rn.getInteger(0, Integer.MAX_VALUE)));
        
        for (int j=0; j < patternCount; j++) {
          InstanceWrapper tmpIns = (InstanceWrapper) instancesOfClass[i].elementAt(j);
          double distance = calculateDistance(ins, tmpIns);
          
          tmpIns.setDistance(distance);
        }

        Collections.sort(instancesOfClass[i], cmp);

        tempStrata[curStrata].addElement(ins);
        numPatternsPerStrata[curStrata]++;
        
        for (int j = 1; j < numStrata; j++) {
          if (patternCount > 0) {
            
            int nextStrata = curStrata + j;
            
            if (nextStrata >= numStrata) {
              nextStrata -= numStrata;
            }
            
            InstanceWrapper tmpIns = (InstanceWrapper) instancesOfClass[i].elementAt(0);
            instancesOfClass[i].removeElementAt(0);

            tempStrata[nextStrata].addElement(tmpIns);
            numPatternsPerStrata[nextStrata]++;
            
            patternCount--;
          }
          else {            
            curStrata = curStrata + j;
            
            if (curStrata >= numStrata) {
              curStrata -= numStrata;
            }
            
            break;
          }
        }
      }
    }
    
    System.out.println(" Completed.");
    
    LogManager.println("Patterns per class in DOB-SCV:");

    for (int i = 0; i < numStrata; i++) {
      LogManager.println("Strata #" + Integer.toString(i+1) + ": " + Integer.toString(numPatternsPerStrata[i]));
    }
    
    LogManager.println("");
    
    return tempStrata;
  }
  
  public static double calculateDistance(InstanceWrapper x1, InstanceWrapper x2) {
    double sum = 0.0;
    
    for (int i=0; i < Parameters.numAttributes; i++) {
      sum += Math.pow(x1.getRealValue(i) - x2.getRealValue(i), 2.0);

      
    }
    
    return sum;
  }

  public static void outputAverageDistance(Vector<InstanceWrapper>[] instancesOfClass) {
    double sum = 0.0;
    double avg;
    int cnt;
    
    for (int i = 0; i < Parameters.numClasses; i++) {
      cnt = 0;
      sum = 0.0;
      avg = 0.0;
      
      int patternCount = instancesOfClass[i].size();

      for (int j = 0; j < patternCount; j++) {

        for (int k = j + 1; k < patternCount; k++) {
          cnt++;
          sum += Math.sqrt(calculateDistance(  instancesOfClass[i].elementAt(j), instancesOfClass[i].elementAt(k) ));
        }
      }
      
      avg = sum / cnt;
      
      LogManager.println("Class " + Integer.toString(i + 1) + " Average Distance: " + Double.toString(avg));
    }
  }
  
}
