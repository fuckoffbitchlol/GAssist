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

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

import javax.print.attribute.standard.NumberUpSupported;

import keel.Algorithms.Genetic_Rule_Learning.Globals.*;

public class GA implements Runnable {

  Classifier[] population;
  Classifier[] bestIndiv;
  int numVersions;
  int threadNo;
  private volatile boolean[] finished;
  
  
  static public Windowing ilas;
  PerformanceAgent pa;
  Rand rn;
  Statistics st;
  Timers ti;
  
  int generationsToRun;
  int strataToUse;

  /** Creates a new instance of GA */
  public GA(int thread) {
    this.threadNo = thread;
  }
  
  public Statistics getStatistics() {
      return st;
  }

  /**
   *  Prepares GA for a new run.
   */
  public void initGA() {
    
    rn = new Rand();
    
    migratedIndividual = new Classifier[2][];
    migratedDataWindow = new int[2];
    neighbourIsDone = new boolean[2];
    finished = new boolean[Parameters.parallelParts];
    migratedDataWindow[0] = -1;
    migratedDataWindow[1] = -1;
    neighbourIsDone[0] = false;
    neighbourIsDone[1] = false;
    
    //Init population
    population = new Classifier[Parameters.subPopulationSize];
    
    if (Parameters.elitist == Parameters.ONE_PER_SET) {
      numVersions = Parameters.parallelParts;
    }
    else if (Parameters.elitist == Parameters.ONE_PER_CORE) {
      numVersions = 1;
    }
    else {
      numVersions = 0;
    }
    
    if (numVersions > 0) {
      bestIndiv = new Classifier[numVersions];
    }
    
    pa = new PerformanceAgent();
    st = new Statistics();
    ti = new Timers(); 
  }
  
  public void changeSeed(int val) {
    rn.initRand(Parameters.seed + threadNo * Parameters.parallelParts * 33 + val);
  }
  
  public void useNextStrata() {
    if (strataToUse == 0) {
      strataToUse = Parameters.parallelParts - 1;
    }
    else {
      strataToUse--;
    }
  }
  
  public void setStrataToUse(int strataToUse) {
		this.strataToUse = strataToUse;
  }

  /**
   *  Inits a new population.
   *  @param _population Population to init.
   */
  private void initPopulation(Classifier[] _population) {
    for (int i = 0; i < Parameters.subPopulationSize; i++) {
      _population[i] = ParallelGlobals.getFactory().newClassifier();
      _population[i].initRandomClassifier(strataToUse);
    }
  }

  public void checkBestIndividual() {
    Classifier best = PopulationWrapper.getBest( population);
    
    int currVer;
    
    if (Parameters.elitist == Parameters.ONE_PER_SET) {
      currVer = strataToUse;
    }
    else if (Parameters.elitist == Parameters.ONE_PER_CORE) {
      currVer = 0;
    }
    else {
      currVer = -1;
    }
    
    if (bestIndiv[currVer] == null) {
      bestIndiv[currVer] = best.copy();
    }
    else {

      if (Parameters.elitist == Parameters.ONE_PER_CORE) {
        PopulationWrapper.evaluateClassifier(bestIndiv[currVer], pa, strataToUse, false);
      }
      
      if (best.compareToIndividual(bestIndiv[currVer])) {
        bestIndiv[currVer] = best.copy();
      }
    }
  }
  
  private Classifier getBestIndividual() {
	  return PopulationWrapper.getBest(population);
  }
  
  private Classifier[] getBestIndividuals(int num) {
    return PopulationWrapper.getBest(population, num);
  }
  
  public Classifier getCopyOfBestIndividual() {
    return PopulationWrapper.getBest(population).copy();
  }
  
  public Classifier getGloballyBestIndividual() {
    return PopulationWrapper.getGlobalBest(population);
  }
  
  public Classifier getESBestIndividual() {
    return PopulationWrapper.getFinalESBest(population);
  }


  
  public void doGlobalEvaluation() {
      PopulationWrapper.doGlobalEvaluation(population, pa);
      PopulationWrapper.doGlobalEvaluation(bestIndiv, pa);
  }

  /**
   *  Executes a number of iterations of GA.
   */
  
  public void evaluateAll() {
    PopulationWrapper.doEvaluation(population, pa, strataToUse, false);
  }
  
  private Classifier[] offsprings;
  
  private GA nextThread;
  private GA previousThread;
  private GA [] threadList;
    
  private volatile Classifier[][] migratedIndividual;
  private volatile int[] migratedDataWindow;
  private volatile boolean[] neighbourIsDone;

  public volatile boolean migrationDone;
  public volatile boolean dataMigrationDone;
  public volatile boolean dataMigrationStart;
  public volatile boolean Migrated;
  
  
  public void parallelSettings(GA next, GA pre) {
    nextThread = next;
    previousThread = pre;
    
    migrationDone = false;
  }
  
  public void parallelSettingsMigration(GA [] thread) {
	  threadList = new GA[thread.length];
	  for (int i = 0; i < thread.length; i++) {
		  threadList[i] = thread[i];
	  }
	
	  dataMigrationDone = false;
	  dataMigrationStart = false;
	  Migrated = true;
  }
  
  public static final Classifier[] copyIndividual(Classifier[] individual) {
    Classifier[] newIndividual = null;
    
    if (individual != null) {
      newIndividual = new Classifier[individual.length];
      
      for (int i=0; i < individual.length; i++) {
        newIndividual[i] = individual[i].copy();
      }
    }
    
    return newIndividual;
  }
  
   // Migrate an individual from another population, overwriting the worst individual
  public void migrateIndividual(Classifier[] newIndividual) {
      if (this.migratedIndividual[0] == null) {
        this.migratedIndividual[0] = copyIndividual(newIndividual);
      }
      else {
        if (this.migratedIndividual[1] == null) {
          this.migratedIndividual[1] = copyIndividual(newIndividual);
        }
        else {
           System.err.println("MIGRATION ERROR (migrateIndividual)! This shouldn't happen!");
        }
      }
  }
  
  private void performMigration() {
    
    if (migratedIndividual[0] == null) {
      System.err.println("ERROR! Migrated individual is null!");
    }
    
    PopulationWrapper.replaceWorstWithNewClassifiers(migratedIndividual[0], population, pa, strataToUse);
    
    if (migratedIndividual[1] != null) {
      migratedIndividual[0] = migratedIndividual[1];
      migratedIndividual[1] = null;
    }
    else {
      migratedIndividual[0] = null;
      migratedIndividual[1] = null;
    }
  
    if (neighbourIsDone[1]) {
      neighbourIsDone[0] = true;
      neighbourIsDone[1] = false;
    }
    else {
      neighbourIsDone[0] = false;
      neighbourIsDone[1] = false;
    }
  }
  
  // Only used if only individual or only data is rotated
  private void neighbourDone() {
    if (!this.neighbourIsDone[0]) {
      this.neighbourIsDone[0] = true;
    }
    else {
      if (!this.neighbourIsDone[1]) {
        this.neighbourIsDone[1] = true;
      }
      else {
         System.err.println("MIGRATION ERROR (neighbourDone)! This shouldn't happen!");
      }
    }
  }
  
  
  private void initializeThread() {
    // Set the parallel globals
    ParallelGlobals.set(threadNo, rn, new Factory(), new Globals_ADI(), 
        new Globals_GABIL(), new Globals_DefaultC(), new Globals_MDL(), new Globals_UBR() 
    );
    
    rn.initRand(Parameters.seed + threadNo * 33);
    st.initStatistics();
    
    ParallelGlobals.getFactory().initialize();
  }
  
  @Override
  public void run() {
    
    initializeThread();

    initPopulation(population); 
    PopulationWrapper.doEvaluation(population, pa, strataToUse, true);
 
    int numIterations, interval = 0;

    boolean lastRun;
    
    if (Parameters.parallelParts > 1) {
      
      if (Parameters.rotateDataSets && Parameters.migrateBestIndividual) {
        if (Parameters.rotationInterval <= Parameters.migrationInterval) {
          interval = Parameters.rotationInterval < Parameters.numIterations ?
                          Parameters.rotationInterval : Parameters.numIterations;
        }
        else {
          interval = Parameters.migrationInterval < Parameters.numIterations ?
              Parameters.migrationInterval : Parameters.numIterations;
        }
      }
      else if (Parameters.rotateDataSets) {
          interval = Parameters.rotationInterval < Parameters.numIterations ?
              Parameters.rotationInterval : Parameters.numIterations;  
      }
      else if (Parameters.migrateBestIndividual) {
          interval = Parameters.migrationInterval < Parameters.numIterations ?
              Parameters.migrationInterval : Parameters.numIterations;    
      }
      else {
        interval = Parameters.numIterations;
      }
      
      
      if (interval < Parameters.numIterations) {
        numIterations = interval;
        lastRun = false;
      }
      else {
        numIterations = Parameters.numIterations;
        lastRun = true;
      }
    }
    else {
      numIterations = Parameters.numIterations;
      lastRun = true;
    }
    
    boolean running = true;
    boolean firstRun = true;
    
    boolean globallyEvaluated = false;
    
    int generationsRun = 0;
    int iteration = 0;
    
    while (running) {
      globallyEvaluated = false;
      
      if (Parameters.evaluateAfterMigration && !firstRun) {
        PopulationWrapper.doEvaluationOnAll(population, pa, strataToUse);
      }
      
      for (int i = 0; i < numIterations; i++) {
        
        boolean lastIteration = (iteration >= Parameters.numIterations - 1);
        
        if (iteration % 100 == 0) {
          if (threadNo == 0) {
            System.out.println("Generation " + iteration + " starting...");
          }
          
          if (rn.getReal() < 0.5) {
            changeSeed(iteration);
          }
        }
        
        Parameters.percentageOfLearningPerThread[threadNo] = (double) iteration
            / (double) Parameters.numIterations;
        
        boolean res = ti.runTimers(rn, iteration, population, st);
        
        PopulationWrapper.setModified(population);
        
        /*													   */
        /* output performance of each generation of each thread*/
        /*													   */
        File geneFile = new File("gene_"+threadNo+".txt");
        if (!geneFile.exists()) {
        	try {
        		geneFile.createNewFile();
        	}catch(IOException e) {
        		e.printStackTrace();
        	}
        }
        
                
        String tempstr = "";
        
        Classifier tempBest;
        tempBest = getBestIndividual();
        // output training accuracy for current strata data
        tempstr += threadNo + " : " + iteration + " gene " + tempBest.getAccuracy() +" " + 
        								tempBest.getNumRules()+" "+ tempBest.getTheoryLength() + "\n";
        // output test accuracy for test data
        PopulationWrapper.evaluateClassifierTest(tempBest, pa);
        tempstr += threadNo + " : " + iteration + " gene " + tempBest.getAccuracy() + "\n";
        								
        FileWriter filewriter;
        try {
        	filewriter = new FileWriter(geneFile, true);
        	filewriter.write(tempstr);
        	filewriter.close();
        }catch(IOException e) {
        	e.printStackTrace();
        }

        
        // GA cycle
        population = doTournamentSelection();
        offsprings = doCrossover();
        doMutation(offsprings);
        
        // pattern evaluation
        PopulationWrapper.doEvaluation(offsprings, pa, strataToUse, true);

        if (Parameters.keepBestIndividual) {
            checkBestIndividual();
        }
        
        population = replacementPolicy(offsprings, pa, lastIteration);

        if (Parameters.parallelParts == 1) {
          
          if (iteration % 100 == 0) {
            System.out.println("Generation " + iteration + " ...");
          }
        }
        else if (Parameters.saveGlobalStatistics) {
          int gen = iteration + 1;
          
          if ( gen == 1 || gen % Parameters.saveStatisticsEveryXGeneration == 0) {
            doGlobalEvaluation();
            st.computeGlobalStatistics(population, bestIndiv);
          
            globallyEvaluated = true;
          }
        }
        
        if (lastIteration && !globallyEvaluated) {
          doGlobalEvaluation();
        }

        st.computeStatistics(population);
        ti.runOutputTimers(iteration, population);
        
        

       iteration++;
        
      }
      
      if (iteration % 5000 == 0) {
        	
        if (threadNo == 0) {
    	  Migrated = false;
        }
            
        dataMigrationStart = false;
        dataMigrationDone = true;

	      try {
	    	  synchronized (this) {
	    		  boolean migration_data = false;
	    		  while((!migration_data && !threadList[0].Migrated) ||  !dataMigrationStart ) {
	    			  int count = 0;
	    			  for (int j = 0; j < threadList.length; j++) {
	    				  if (threadList[j].dataMigrationDone == true) {
	    					  count += 1;
	    				  }
	    			  }
	    			  if (count == Parameters.parallelParts) {
	    				  migration_data = true;
	    			  }
	    			  
	    			  if (threadNo == 0) {
	    				  PopulationWrapper.doDataMigration();
	    				  Migrated = true;
	    				  LogManager.println("migration happen on Ensemble " + iteration);
	    			  }
	    			  
	    			  if (migration_data && threadList[0].Migrated) {
	    				 for (int j = 0; j < threadList.length; j++) {
	    					 threadList[j].dataMigrationStart = true;
						} 
	    			  }else {
	    				  this.wait(10);
	    			  }
//	    			  System.out.println("wait on " + threadNo );
	    		  }
	    	  }
	      }catch(InterruptedException e) {
	    	System.err.println("interrupted exception");	  
	      }
        }
        
        PopulationWrapper.ClearnErrorCnt(iteration);
       
      if (threadNo == 0) {
      if (iteration % 5000 == 0) {
       
    	  String filename = "";
    	  int [][] patCnt;

    	  patCnt = PopulationWrapper.getPatternCnt();

    	  String tmpStr = "";
    	  for (int i = 0; i<3; i++) {
    		  for (int j = 0; j<patCnt[0].length;j++) {
    			  tmpStr += patCnt[i][j] + "\n";
    		  }	
    	  }

    		  File fn = new File("patCnt_"+"_"+iteration+".txt");
    		  if (!fn.exists()) {
    			  try {
    				  fn.createNewFile();
    			  }catch(IOException e) {
    				  LogManager.println("file error happen");
    				  e.printStackTrace();
    			  }
    		  }
    	  
    		  FileWriter fw;
    		  try {
    			  fw = new FileWriter(fn, true);
    			  fw.write(tmpStr);
    			  fw.close();
    		  }catch(IOException e) {
    			  LogManager.println("file error happen");
    			  e.printStackTrace();
    		  }
      }
      }

      
      /* 
       * synchronization start from here
       * gene -> 100
       * synchro previous thread, migrate best individual
       */
      
      if (Parameters.parallelParts > 1 && !lastRun) {
    	  /*
    	   * synchronization for data migration
    	   */
    
    	 if (Parameters.migrateBestIndividual) {
          if (iteration % Parameters.migrationInterval == 0) {  
            
              synchronized (nextThread) {
            	LogManager.println("thread:"+threadNo+ " is on" + iteration);
                if (Parameters.migrateBestIndividual) {
                	// individual to migrate set to one
                  Classifier[] best = getBestIndividuals(Parameters.individualsToMigrate);
                  nextThread.migrateIndividual( best );
                  nextThread.notify();
                }
              }
              
              synchronized (previousThread) {
                previousThread.neighbourDone();
                previousThread.notify();
              }
              
              try {
                synchronized ( this ) {
                  while (!migrationDone) {
                    // Check if the next migration is done
                    if ( (migratedIndividual[0] != null && neighbourIsDone[0] ) ) 
                    {
                      migrationDone = true;
                    }
                    
                    if (!migrationDone) {
                      this.wait(10);
                    }
                  }
                  
                  migrationDone = false;
      
                  performMigration();
                }
              }
              catch (InterruptedException e) {
                System.out.println("Interrupted exception!");
                System.out.println( e.getMessage() );
              }
          }
        }

        // rotate data set each generation
        if (Parameters.rotateDataSets) {
          if (iteration % Parameters.rotationInterval == 0) {  
//            useNextStrata();
          }
        }
        
        generationsRun += interval;

        firstRun = false;
        
        if (generationsRun + interval >= Parameters.numIterations) {
          
          if (generationsRun + interval > Parameters.numIterations) {
            numIterations = generationsRun + interval - Parameters.numIterations;
          }
          
          lastRun = true;
        }
      }
      else {
        running = false;
      }
    }
    
    System.out.println("Thread " + threadNo + ": Finished.");
  }
  
  public static void outputStatistics(Classifier best, PerformanceAgent pa) {
    LogManager.println("\nPhenotype: ");
    best.printClassifier();
    PopulationWrapper.testClassifier(best, pa, "training",Parameters.train2InputFile,Parameters.trainOutputFile);
    PopulationWrapper.testClassifier(best, pa, "test",Parameters.testInputFile,Parameters.testOutputFile);
        
  }

  Classifier[] doCrossover() {
    int i, j, k, countCross = 0;
    int numNiches = population[0].getNumNiches();
    ArrayList[] parents = new ArrayList[numNiches];
    Classifier parent1, parent2;
    Classifier[] offsprings = new Classifier[2];
    Classifier[] offspringPopulation = new Classifier[Parameters.subPopulationSize];

    for (i = 0; i < numNiches; i++) {
      parents[i] = new ArrayList();
      parents[i].ensureCapacity(Parameters.subPopulationSize);
    }

    for (i = 0; i < Parameters.subPopulationSize; i++) {
      int niche = population[i].getNiche();
      parents[niche].add(new Integer(i));
    }

    for (i = 0; i < numNiches; i++) {
      int size = parents[i].size();
      Sampling samp = new Sampling(size);
      int p1 = -1;
      for (j = 0; j < size; j++) {
        if (rn.getReal() < Parameters.probCrossover) {
          if (p1 == -1) {
            p1 = samp.getSample(rn);
          }
          else {
            int p2 = samp.getSample(rn);
            int pos1 = ( (Integer) parents[i].get(p1)).intValue();
            int pos2 = ( (Integer) parents[i].get(p2)).intValue();
            parent1 = population[pos1];
            parent2 = population[pos2];

            offsprings = parent1.crossoverClassifiers(parent2);
            offspringPopulation[countCross++] = offsprings[0];
            offspringPopulation[countCross++] = offsprings[1];
            p1 = -1;
          }
        }
        else {
          int pos = ( (Integer) parents[i].get(samp.getSample(rn))).intValue();
          offspringPopulation[countCross++] = population[pos].copy();
        }
      }
      if (p1 != -1) {
        int pos = ( (Integer) parents[i].get(p1)).intValue();
        offspringPopulation[countCross++] = population[pos].copy();
      }
    }

    return offspringPopulation;
  }

  private int selectNicheWOR(int[] quotas) {
    int num = quotas.length;
    if (num == 1) {
      return 0;
    }

    int total = 0, i;
    for (i = 0; i < num; i++) {
      total += quotas[i];
    }
    if (total == 0) {
      return rn.getInteger(0, num - 1);
    }
    int pos = rn.getInteger(0, total - 1);
    total = 0;
    for (i = 0; i < num; i++) {
      total += quotas[i];
      if (pos < total) {
        quotas[i]--;
        return i;
      }
    }

    LogManager.printErr("We should not be here");
    System.exit(1);
    return -1;
  }

  private void initPool(ArrayList pool, int whichNiche,
                        Classifier[] _population) {
    if (ParallelGlobals.getGlobals_DefaultC().nichingEnabled) {
      for (int i = 0; i < Parameters.subPopulationSize; i++) {
        if (_population[i].getNiche() == whichNiche) {
          pool.add(new Integer(i));
        }
      }
    }
    else {
      for (int i = 0; i < Parameters.subPopulationSize; i++) {
        pool.add(new Integer(i));
      }
    }
    
  }

  private int selectCandidateWOR(ArrayList pool, int whichNiche,
                                 Classifier[] _population) {
    if (pool.size() == 0) {
      initPool(pool, whichNiche, population);
      if (pool.size() == 0) {
        return rn.getInteger(0, Parameters.subPopulationSize - 1);
      }
    }

    int pos = rn.getInteger(0, pool.size() - 1);
    int elem = ( (Integer) pool.get(pos)).intValue();
    pool.remove(pos);
    return elem;
  }

  /**
   *  Does Tournament Selection without replacement.
   */
  public Classifier[] doTournamentSelection() {

    Classifier[] selectedPopulation;
    selectedPopulation = new Classifier[Parameters.subPopulationSize];
    int i, j, winner, candidate;
    int numNiches;
    if (ParallelGlobals.getGlobals_DefaultC().nichingEnabled) {
      numNiches = population[0].getNumNiches();
    }
    else {
      numNiches = 1;
    }

    ArrayList[] pools = new ArrayList[numNiches];
    for (i = 0; i < numNiches; i++) {
      pools[i] = new ArrayList();
    }
    int[] nicheCounters = new int[numNiches];
    int nicheQuota = Parameters.subPopulationSize / numNiches;
    for (i = 0; i < numNiches; i++) {
      nicheCounters[i] = nicheQuota;
    }

    for (i = 0; i < Parameters.subPopulationSize; i++) {
      // There can be only one
      int niche = selectNicheWOR(nicheCounters);
      winner = selectCandidateWOR(pools[niche], niche
                                  , population);
      for (j = 1; j < Parameters.tournamentSize; j++) {
        candidate = selectCandidateWOR(pools[niche]
                                       , niche, population);
        if (population[candidate].compareToIndividual(population[winner])) {
          winner = candidate;
        } 
      }
      selectedPopulation[i] = population[winner].copy();
    }
    
    return selectedPopulation;
  }

  public void doMutation(Classifier[] _population) {
    
    double probMut = Parameters.probMutationInd;

    for (int i = 0; i < Parameters.subPopulationSize; i++) {
      if (rn.getReal() < probMut) {
        _population[i].doMutation();
      }
    }

    doSpecialStages(_population);
  }

  void sortedInsert(ArrayList set, Classifier cl) {
    for (int i = 0, max = set.size(); i < max; i++) {
      if (cl.compareToIndividual( (Classifier) set.get(i))) {
        set.add(i, cl);
        return;
      }
    }
    set.add(cl);
  }

  public Classifier[] replacementPolicy(Classifier[] offspring, PerformanceAgent pa
                                        , boolean lastIteration) {
    int i;
    
    if (lastIteration) {
      for (i = 0; i < numVersions; i++) {
        if (bestIndiv[i] != null) {
          PopulationWrapper.evaluateClassifier(bestIndiv[i], pa, strataToUse, true);
        }
      }
      ArrayList set = new ArrayList();
      for (i = 0; i < Parameters.subPopulationSize; i++) {
        sortedInsert(set, offspring[i]);
      }
      for (i = 0; i < numVersions; i++) {
        if (bestIndiv[i] != null) {
          sortedInsert(set, bestIndiv[i]);
        }
      }

      for (i = 0; i < Parameters.subPopulationSize; i++) {
        offspring[i] = (Classifier) set.get(i);
      }
    }
    else {
      boolean previousVerUsed = false;
      
      if (Parameters.keepBestIndividual) {
        int currVer;
        
        if (Parameters.elitist == Parameters.ONE_PER_SET) {
          currVer = strataToUse;
        }
        else {
          currVer = 0;
        }
        
        if (bestIndiv[currVer] == null && currVer > 0) {
          previousVerUsed = true;
          currVer--;
        }
  
        if (bestIndiv[currVer] != null) {
          PopulationWrapper.evaluateClassifier(bestIndiv[currVer], pa, strataToUse, true);
          int worst = PopulationWrapper.getWorst(offspring);
          offspring[worst] = bestIndiv[currVer].copy();
        }       
        if (!previousVerUsed) {
          int prevVer;
          if (currVer == 0) {
            prevVer = numVersions - 1;
          }
          else {
            prevVer = currVer - 1;
          }
          if (bestIndiv[prevVer] != null) {
            PopulationWrapper.evaluateClassifier(bestIndiv[prevVer], pa, strataToUse, true);
            int worst = PopulationWrapper.getWorst(offspring);
            offspring[worst] = bestIndiv[prevVer].copy();
          }
        }
      }
    }
    
    return offspring;
  }

  public void doSpecialStages(Classifier[] population) {
    
    int numStages = population[0].numSpecialStages();

    for (int i = 0; i < numStages; i++) {
      for (int j = 0; j < population.length; j++) {
        population[j].doSpecialStage(i);
      }
    }
  }
  
}

