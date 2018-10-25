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

import java.lang.reflect.Array;
import java.util.ArrayList;

import keel.Dataset.*;
import keel.Algorithms.Genetic_Rule_Learning.Globals.*;

public class PopulationWrapper {
/**
 * <p>
 * Contains methods that manipulate the population in various
 * ways: classifying the training set for the fitness computations, checking
 * if there are improved solutions in the population and performing the
 * test stage (generating the output files)
 * </p>
 */
	
  static public InstanceSet is;
  static public InstanceSet isTest;
  static public Windowing ilas;
  static public InstanceWrapper[] allInstances;

  static public int[][] instancesByClass;
  static public int[][][] instancesByWindowClass;
  static public Sampling[] samplesOfClasses;
  static public Sampling[][] samplesOfWindowClasses;
  static public boolean smartInit;
  static public boolean cwInit;

  
  // get training data here, and initialize the set
  public static void initInstancesEvaluation() {
    Rand rn = new Rand();
    rn.initRand(0);
    
    is = new InstanceSet();
    try {
      is.readSet(Parameters.trainInputFile, true);
    }
    catch (Exception e) {
      LogManager.printErr(e.toString());
      System.exit(1);
    }
    
    checkDataset();
    replaceMissing(is);
    if (Parameters.adiKR) {
      DiscretizationManager.init();
    }
    allInstances = createWrapperInstances(is);
    ilas = new Windowing(rn, allInstances);

    if (Parameters.initMethod != null) {
      if (Parameters.initMethod.equalsIgnoreCase("smart")) {
        smartInit = true;
        cwInit = false;
      }
      else if (Parameters.initMethod.equalsIgnoreCase("cwInit")) {
        smartInit = true;
        cwInit = true;
      }
      else {
        smartInit = false;
        cwInit = false;
      }
    }
    else {
      smartInit = false;
      cwInit = false;
    }
    
    // New Smart init
    if (Parameters.initializeRulesLocally && smartInit) {
      int nc = Parameters.numClasses;
      int classCounts[] = new int[nc];
      instancesByWindowClass = new int[Parameters.parallelParts][][];
      samplesOfWindowClasses = new Sampling[Parameters.parallelParts][];

      for (int i = 0; i < Parameters.parallelParts; i ++) {
        instancesByWindowClass[i] = new int[Parameters.numClasses][];
        samplesOfWindowClasses[i] = new Sampling[nc];
        
        for (int j = 0; j < nc; j++) {
          int num = numInstancesOfClass(j, i);
          instancesByWindowClass[i][j] = new int[num];
          samplesOfWindowClasses[i][j] = new Sampling( num );
          classCounts[j] = 0;
        }
        
        InstanceWrapper[] set = ilas.getInstances(i);
        
        for (int j = 0; j < set.length; j++) {
          int cl = set[j].classOfInstance();
          instancesByWindowClass[i][cl][classCounts[cl]++] = j;
        }        
        
      }
    }
    // Old Smart init
    else if (smartInit) {
      int nc = Parameters.numClasses;
      int classCounts[] = new int[nc];
      instancesByClass = new int[nc][];
      samplesOfClasses = new Sampling[nc];


      for (int i = 0; i < nc; i++) {
        int num = numInstancesOfClass(i);
        instancesByClass[i] = new int[num];
        samplesOfClasses[i] = new Sampling(num);
        classCounts[i] = 0;
      }
      for (int i = 0; i < allInstances.length; i++) {
        int cl = allInstances[i].classOfInstance();
        instancesByClass[cl][classCounts[cl]++] = i;
      }
    }
  }

  public static InstanceWrapper getInstanceInit(int forbiddenCL, int strata) {
    
    Rand rn = ParallelGlobals.getRand();
    
    // Create from local patterns
    if (Parameters.initializeRulesLocally && cwInit) {
      int cl;
      do {
        cl = rn.getInteger(0, Parameters.numClasses - 1);
      }
      while (cl == forbiddenCL
             || instancesByWindowClass[strata][cl].length == 0);
      int pos = samplesOfWindowClasses[strata][cl].getSample(rn);
      return ilas.getInstances(strata)[instancesByWindowClass[strata][cl][pos]];
    }
    // Create from global patterns
    else if (cwInit) {
      int cl;
      do {
        cl = rn.getInteger(0, Parameters.numClasses - 1);
      }
      while (cl == forbiddenCL
             || instancesByClass[cl].length == 0);
      int pos = samplesOfClasses[cl].getSample(rn);
      return allInstances[instancesByClass[cl][pos]];
    }

    int count[] = new int[Parameters.numClasses];
    int total = 0;
    for (int i = 0; i < count.length; i++) {
      if (i == forbiddenCL) {
        count[i] = 0;
      }
      else {
        count[i] = samplesOfClasses[i].numSamplesLeft();
        total += count[i];
      }
    }

    int pos = rn.getInteger(0, total - 1);
    int acum = 0;
    for (int i = 0; i < count.length; i++) {
      acum += count[i];
      if (pos < acum) {
        int inst = samplesOfClasses[i].getSample(rn);
        return allInstances[instancesByClass[i][inst]];
      }
    }

    LogManager.printErr("We should not be here !!!");
    System.exit(1);
    return null;
  }
  
  // @question what is this for ?
  public static InstanceWrapper[] createWrapperInstances(InstanceSet is) {
    InstanceWrapper[] iw
        = new InstanceWrapper[is.getNumInstances()];
    for (int i = 0; i < iw.length; i++) {
      iw[i] = new InstanceWrapper(is.getInstance(i));
    }
    return iw;
  }
  
  public static void doDataMigration() {
	  ilas.migrateData();
  }
  
  public static int[][] getPatternCnt(){
	 return ilas.patternCnt; 
  }
  
  public static void ClearnErrorCnt(int iteration) {
	  if (iteration % 100 != 0) {
		  for (int i = 0; i < Parameters.parallelParts; i++) {
		  ilas.patErrorCnt[0] = new int [ilas.is.length];
		  }
	  }
  }

  public static void evaluateClassifierES(Classifier ind, PerformanceAgent pa, int strataToUse, boolean deleteRules ) {
    int predicted, real;
    InstanceWrapper[] instances;
    		
    instances = ilas.getInstances(strataToUse);

    ind.resetPerformance();
    pa.resetPerformance(ind.getNumRules());
    for (int i = 0; i < instances.length; i++) {
      real = instances[i].classOfInstance();
      predicted = ind.doMatch(instances[i]);
      pa.addPredictionES(predicted, real
                                     , ind.getPositionRuleMatch(), ilas.patErrorCnt[strataToUse], i);
    }

    ind.computePerformance(pa);

    int threadNo = ParallelGlobals.getThreadNo();
    
    if (Parameters.doRuleDeletionPerThread[threadNo] && deleteRules) {
      ind.deleteRules(pa.controlBloatRuleDeletion());
    }
  }
  
 public static void evaluateClassifier(Classifier ind, PerformanceAgent pa, int strataToUse, boolean deleteRules) {
    int predicted, real;
    InstanceWrapper[] instances;
    		
    instances = ilas.getInstances(strataToUse);

    ind.resetPerformance();
    pa.resetPerformance(ind.getNumRules());
    for (int i = 0; i < instances.length; i++) {
      real = instances[i].classOfInstance();
      predicted = ind.doMatch(instances[i]);
      pa.addPrediction(predicted, real
                                     , ind.getPositionRuleMatch());
    }

    ind.computePerformance(pa);

    int threadNo = ParallelGlobals.getThreadNo();
    
    if (Parameters.doRuleDeletionPerThread[threadNo] && deleteRules) {
      ind.deleteRules(pa.controlBloatRuleDeletion());
    }
  }
  

  
  // evaluate classifier on test data for each generation
  public static void evaluateClassifierTest(Classifier ind, PerformanceAgent pa) {
	  int predicted, real;
	  InstanceWrapper[] instances;
	  InstanceSet testSet = new InstanceSet();
	  try {
            testSet.readSet(Parameters.testInputFile, false);
	  } catch(Exception e) {
            LogManager.printErr(e.toString());
            System.exit(1);
	  }
	  instances = createWrapperInstances(testSet);
			  
	  ind.resetPerformance();
	  pa.resetPerformance(ind.getNumRules());
	  for (int i = 0; i < instances.length; i++) {
		real = instances[i].classOfInstance();
		predicted = ind.doMatch(instances[i]);
		pa.addPrediction(predicted, real, ind.getPositionRuleMatch());
	  }

	  ind.computePerformance(pa);
  }
  public static void evaluateDiversityMeasure(Classifier [] ind, double [] fitness, PerformanceAgent pa) {
	  int real = 0;
	  int [] predicted = new int [Parameters.parallelParts];
	  int final_pred = 0;
	  double test_acc = 0;
	  
	  
	  InstanceWrapper[] instances;
	  InstanceSet testSet = new InstanceSet();
	  try {
            testSet.readSet(Parameters.trainInputFile, false);
	  } catch(Exception e) {
            LogManager.printErr(e.toString());
            System.exit(1);
	  }
	  instances = createWrapperInstances(testSet);
	  
	 // for k-statistics
	  
	  for (int i = 0; i < ind.length; i++) {
		  float dm = 0;
		  for (int j = i+1; j < ind.length; j++) {
			  float diffRes = 0;
			  float sameRes = 0;
			  float m = 0;
			 for (int k = 0; k < instances.length; k++) {
				 int predictClas, predictClas2;
				 predictClas = ind[i].doMatch(instances[k]);
				 predictClas2 = ind[j].doMatch(instances[k]);
				 if (predictClas != predictClas2) {
					 diffRes += 1.0;
				 }
				 else if(predictClas == predictClas2) {
					 sameRes += 1.0;
				 }
			 }
			 m = (float)instances.length;
			 dm = diffRes / m;
			 LogManager.println("dissimilar measure: cf:" + i + "-cf:"+j+" dm:" + dm);
		  }
	  }
	  
	  // k-statistic calculations
	  for (int i = 0; i < ind.length; i++) {

		  for (int j = i+1; j < ind.length; j++) {
			  float [] staticTable = new float[Parameters.numClasses*Parameters.numClasses];
			  float diffRes = 0;
			  float sameRes = 0;
			  // k-statistics contingency table, add q-statics
			  for (int k=0; k<instances.length; k++) {
				  int predictClas, predictClas2;
				  predictClas = ind[i].doMatch(instances[k]);
				  predictClas2 = ind[j].doMatch(instances[k]);
				  
				  if ((int)predictClas != -1 && (int)predictClas2 != -1) {

//				  LogManager.println("clas:"+ predictClas);
					  if (predictClas != predictClas2) {
//					  staticTable[(int)predictClas] += 1.0;
//					  staticTable[(int)predictClas2] += 1.0;
					  
					  // when disabled default class predictClas may be -1
//						  LogManager.println("1:" + predictClas + " 2:" + predictClas2);
						  diffRes += 1.0;
						  staticTable[(int)predictClas*Parameters.numClasses + predictClas2] += 1.0;
					  }
					  else if (predictClas == predictClas2) {
	//					  staticTable[(int)predictClas] += 1.0;
//						  LogManager.println("1:" + predictClas + " 2:" + predictClas2);
						  sameRes += 1.0;
						  staticTable[(int)predictClas*Parameters.numClasses + predictClas] += 1.0;
					  }
				  }
			  }
			  float m = (float)instances.length;
			  float p1 = sameRes / m;
			  float p2 = 0;
			  // for q-statics
			  float q1 = 1, q2=1;
			  
			  for (int l = 0; l<Parameters.numClasses; l++) {
				  float n1 = 0, n2 = 0; 
				  for (int l2 = 0; l2<Parameters.numClasses; l2++) {
					  if(l == l2) {
						  q1 *= staticTable[l*Parameters.numClasses + l];
					  }
					  else {
						  q2 *= staticTable[l2+(Parameters.numClasses)*l];
					  }
					  n1 += staticTable[l2+(Parameters.numClasses)*l]; 
					  n2 += staticTable[l+(Parameters.numClasses)*l2];
				  }
				  p2 += n1 * n2;
			  }
			  p2 = p2 / (m*m);
			  float kstatic = 0;
			  kstatic = (p1 - p2) / (1-p2);
			  float qstatic = 0;
			  qstatic = (q1-q2) / (q1+q2);
			  LogManager.println("dissimilar measure: cf:" + i + "-cf:"+j+" qstatic:" + qstatic);
			  LogManager.println("dissimilar measure: cf:" + i + "-cf:"+j+" kstatic:" + kstatic);
		  }
		  
		  
	  }
	  
  }


  // for ensemble model. evaluate on test data
  public static void evaluateClassifierES(Classifier [] ind, double [] fitness, PerformanceAgent pa) {
	  int real = 0;
	  int [] predicted = new int [Parameters.parallelParts];
	  int final_pred = 0;
	  double test_acc = 0;
	  
	  
	  InstanceWrapper[] instances;
	  InstanceSet testSet = new InstanceSet();
	  try {
            testSet.readSet(Parameters.testInputFile, false);
	  } catch(Exception e) {
            LogManager.printErr(e.toString());
            System.exit(1);
	  }
	  instances = createWrapperInstances(testSet);
	  
/* original gassist part*/
//	  for (int i = 0; i < Parameters.parallelParts; i++) {
//		  //ind[i].resetPerformance();
//		  //pa.resetPerformance(ind[i].getNumRules());
//	  }
//    
/* original gassist part*/

	  double max_fit = 0;
	  int best_cl = 0;
	  for (int i= 0; i< fitness.length; i++) {
		  if (fitness[i] > max_fit) {
			  max_fit = fitness[i];
			  best_cl = i;
		  }
	  }
	  
	  int tiedCount = 0;
	  // count instance number of each class
	  int [] numOfInsClas = new int[Parameters.numClasses];
	  for (int i = 0; i < instances.length; i++) {
		  int clas = instances[i].classOfInstance();
		  numOfInsClas[clas] += 1;
	  }

	  for (int i = 0; i < instances.length; i++) {
		  real = instances[i].classOfInstance(); 
		  int [] pred_clas = new int [Parameters.numClasses];

		  for (int s = 0; s < Parameters.parallelParts; s++) {
			  predicted[s] = ind[s].doMatch(instances[i]);
			  LogManager.println("classifier:" + s + " acc:" + fitness[s] + " clas:" + predicted[s]);
			  if ((int)predicted[s] != -1) {
				  pred_clas[predicted[s]] += 1;
			  }
		  }
		  
		  int max = -10000;
		  boolean isTied = false;
		  
		  for (int c = 0; c < Parameters.numClasses; c++) {
			  if (pred_clas[c] > max) {
				  final_pred = c;
				  max = pred_clas[c];
				  isTied = false;
			  }
			  else if (pred_clas[c] == max) {
				  isTied = true;
				  if(numOfInsClas[c] > numOfInsClas[final_pred]) {
					  final_pred = c;
				  }
//				  final_pred = predicted[best_cl];
			  }
		  }
		  if (isTied == true) {
			  tiedCount += 1;
		  }
		  for (int j = 0; j < Parameters.numClasses; j ++ ) {
			  LogManager.println("clas " + j + " " + pred_clas[j]);
		  }
		
		  LogManager.println("real is: " + real + " predict is: " + final_pred + " " + isTied);
		  if (final_pred == real) {
			  test_acc += 1.0;
		  }
	  }
	  double acc = test_acc / instances.length;
	  
	  LogManager.println("instance Num:" + instances.length + " tied:" + tiedCount);
	  LogManager.println("final Ensemble acc:" + acc);

	  

//pa.addPrediction(final_pred, real, ind[0].getPositionRuleMatch());
//	  ind.computePerformance(pa);
  }


  public static void doEvaluation(Classifier[] _population, PerformanceAgent pa, int strataToUse, boolean deleteRules) {
    int popSize = _population.length;

    for (int i = 0; i < popSize; i++) {
      if (!_population[i].getIsEvaluated()) {
        evaluateClassifierES(_population[i], pa, strataToUse, deleteRules);
      }
    }
  }
  
  public static void doEvaluationOnAll(Classifier[] _population, PerformanceAgent pa, int strataToUse) {
    int popSize = _population.length;
    
    for (int i = 0; i < popSize; i++) {
      _population[i].setIsEvaluated(false);
    }

    doEvaluation(_population, pa, strataToUse, true);
  }
   
  public static void doGlobalEvaluation(Classifier[] _population, PerformanceAgent pa) {
    int predicted, real;
    int popSize = _population.length;

    for (int i = 0; i < popSize; i++) {
      if (_population[i] != null) {
        Classifier ind = _population[i];
        
        pa.resetPerformance(ind.getNumRules());
        
        for (int j = 0; j < allInstances.length; j++) {
          real = allInstances[j].classOfInstance();
          predicted = ind.doMatch(allInstances[j]);
          pa.addPrediction(predicted, real
                          , ind.getPositionRuleMatch());
          }
    
        ind.computeGlobalPerformance(pa);
      }
    }
  }

  /**
   *  Obtains the best classifier of population.
   *  @param _population The population
   *  @return Best classifier of population.
   */
  public static Classifier getBest(Classifier[] _population) {
    int sizePop = _population.length;
    int posWinner = 0;

    for (int i = 1; i < sizePop; i++) {
      if (_population[i].compareToIndividual(_population[posWinner])) {
        posWinner = i;
      }
    }
    return _population[posWinner];
  }
  
  public static Classifier[] getBest(Classifier[] _population, int num) {
    int sizePop = _population.length;
    
    Classifier result[] = new Classifier[num];
    ArrayList<Integer> used = new ArrayList<Integer>();
    
    for (int n = 0; n < num; n++) {
      int posWinner = 0;
      for (int i = 1; i < sizePop; i++) {
        if (!used.contains(i)) {
          if (_population[i].compareToIndividual(_population[posWinner])) {
            posWinner = i;
          }
        }
      }
      
      used.add(posWinner);
      result[n] = _population[posWinner];
    }
    
    return result;
  }
  
  /**
   *  Obtains the globally best classifier of population.
   *  @param _population The population
   *  @return Best classifier of population.
   */  
  public static Classifier getGlobalBest(Classifier[] _population) {
	int sizePop = _population.length;
    int posWinner = 0;

    for (int i = 1; i < sizePop; i++) {
      if (_population[i] != null) {
        if (_population[i].globalCompareToIndividual(_population[posWinner])) {
          posWinner = i;
        }
      }
    }
    return _population[posWinner];
  }
  
  public static Classifier getFinalESBest(Classifier[] _population) {
	int sizePop = _population.length;
    int posWinner = 0;

    for (int i = 1; i < sizePop; i++) {
      if (_population[i] != null) {
        if (_population[i].ESCompareToIndividual(_population[posWinner])) {
          posWinner = i;
        }
      }
    }
    return _population[posWinner];

  }

  /**
   *  Obtains the worst classifier of population.
   *  @param _population The population
   *  @return Worst classifier of population.
   */
  public static int getWorst(Classifier[] _population) {
    int sizePop = _population.length;
    int posWorst = 0;
    
    for (int i = 1; i < sizePop; i++) {
      if (!_population[i].compareToIndividual(_population[posWorst])) {
        posWorst = i;
      }
    }
    return posWorst;
  }
  
  /**
   *  Obtains the worst classifier of population.
   *  @param _population The population
   *  @return Worst classifier of population.
   */
  public static int[] getWorst(Classifier[] _population, int num) {
    int sizePop = _population.length;
    int posWorst = 0;
    
    int[] result = new int[num];
    ArrayList<Integer> used = new ArrayList<Integer>();
    
    for (int n = 0; n < num; n++) {
      for (int i = 1; i < sizePop; i++) {
        if (!used.contains(i)) {
          if (!_population[i].compareToIndividual(_population[posWorst])) {
            posWorst = i;
          }
        }
      }
      
      used.add(posWorst);
      result[n] = posWorst;
    }
    return result;
  }
  
  public static void replaceWorstWithNewClassifiers(Classifier[] newClassifiers, Classifier[] _population,
      PerformanceAgent pa, int strataToUse) {
    int num = newClassifiers.length;
    
    int[] worst = getWorst(_population, num);
    
    for (int i=0; i < num; i++) {
      _population[worst[i]] = newClassifiers[i].copy();
      evaluateClassifier(_population[worst[i]], pa, strataToUse, true);
    }
  }

  public static void setModified(Classifier[] _population) {
    int sizePop = _population.length;

    for (int i = 0; i < sizePop; i++) {
      _population[i].setIsEvaluated(false);
    }
  }

  // test classifier on testData or trainData
  public static void testClassifier(Classifier ind, PerformanceAgent pa, String typeOfTest,
                                    String testInputFile, String testOutputFile) {
    InstanceSet testSet = new InstanceSet();
    try {
            testSet.readSet(testInputFile,false);
    } catch(Exception e) {
            LogManager.printErr(e.toString());
            System.exit(1);
    }
    replaceMissing(testSet);
    FileManagement fm = new FileManagement();
    InstanceWrapper []instances=createWrapperInstances(testSet);
    double numInstances=instances.length;
    int real, predicted;
    pa.resetPerformanceTest(ind.getNumRules());
    ind.resetPerformance();
    //Attribute att = Attributes.getAttribute(Parameters.numAttributes);
    Attribute att = Attributes.getOutputAttribute(0);

    try {
            fm.initWrite(testOutputFile);
            fm.writeLine(testSet.getHeader());

            for(int i=0;i<numInstances;i++) {
                    real=instances[i].classOfInstance();
                    predicted=ind.doMatch(instances[i]);
                    String pred;
                    if(predicted==-1) {
                            pred="unclassified";
                    } else {
                            pred=att.getNominalValue(predicted);
                    }
                    fm.writeLine(att.getNominalValue(real)+" "+pred+"\n");
                    pa.addPredictionTest(predicted,real
                            ,ind.getPositionRuleMatch());
            }
            fm.closeWrite();
    } catch(Exception e) {
            System.err.println("Error handling output test file "+testOutputFile);
            System.exit(1);
    }

    LogManager.println("\nStatistics on "+typeOfTest+" file");
    pa.dumpStats(typeOfTest);
    LogManager.println("");
  }

  static void checkDataset() {
    Attribute[] outputs = Attributes.getOutputAttributes();
    if (outputs.length != 1) {
      LogManager.printErr("Only datasets with one output are supported");
      System.exit(1);
    }
    if (outputs[0].getType() != Attribute.NOMINAL) {
      LogManager.printErr("Output attribute should be nominal");
      System.exit(1);
    }
    Parameters.numClasses = outputs[0].getNumNominalValues();
    Parameters.numAttributes = Attributes.getInputAttributes().length;
  }

  static void replaceMissing(InstanceSet is) {
    String[][] mostFreq = new String[Parameters.numAttributes][Parameters.
        numClasses];
    double[][] means = new double[Parameters.numAttributes][Parameters.
        numClasses];
    int[] types = new int[Parameters.numAttributes];
    Attribute[] inputs = Attributes.getInputAttributes();

    for (int i = 0; i < Parameters.numAttributes; i++) {
      types[i] = inputs[i].getType();
      if (types[i] == Attribute.NOMINAL) {
        for (int j = 0; j < Parameters.numClasses; j++) {
          mostFreq[i][j] = inputs[i].getMostFrequentValue(j);
        }
      }
      else {
        for (int j = 0; j < Parameters.numClasses; j++) {
          means[i][j] = inputs[i].getMeanValue(j);
        }
      }
    }

    Instance[] inst = is.getInstances();
    for (int i = 0; i < inst.length; i++) {
      if (inst[i].existsAnyMissingValue()) {
        int cl = inst[i].getOutputNominalValuesInt(0);
        boolean[] miss = inst[i].getInputMissingValues();
        for (int j = 0; j < Parameters.numAttributes; j++) {
          if (miss[j]) {
            if (types[j] == Attribute.NOMINAL) {
              inst[i].setInputNominalValue(j, mostFreq[j][cl]);
            }
            else {
              inst[i].setInputNumericValue(j, means[j][cl]);
            }
          }
        }
      }
    }
  }

  static int numInstancesOfClass(int clas){
    int count = 0;
    Instance [] inst = is.getInstances();
    for (int i = 0; i < inst.length; i++){
      if (inst[i].getOutputNominalValuesInt(0) == clas){
        count++;
      }
    }
    return count;
  }
  
  static int numInstancesOfClass(int clas, int strata){
    int count = 0;
    InstanceWrapper[] inst = ilas.getInstances(strata);
    for (int i = 0; i < inst.length; i++){
      if (inst[i].classOfInstance() == clas){
        count++;
      }
    }
    return count;
  }

}

