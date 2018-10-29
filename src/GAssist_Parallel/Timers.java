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

public class Timers {
/**
 * <p>
 * Manages timers: flags and parameters that are triggered at
 * certain iterations
 * </p>
 */

  Globals_DefaultC gdc;
  
  public Timers() {
    gdc = new Globals_DefaultC();
  }

  public boolean runTimers(Rand rn, int iteration, Classifier[] population, Statistics st) {
    ParallelGlobals.getGlobals_ADI().nextIteration();
    
    boolean res1 = ParallelGlobals.getGlobals_MDL().newIteration(rn, iteration, population, st);
    boolean res2 = timerBloat(iteration);

    if (res1 || res2) {
      return true;
    }
    return false;
  }

  public void runOutputTimers(int iteration
                                     , Classifier[] population) {
    gdc.checkNichingStatus(iteration, population);
  }

  boolean timerBloat(int _iteration) {
    int threadNo = ParallelGlobals.getThreadNo();
    
    Parameters.doRuleDeletionPerThread[threadNo] = (_iteration >= Parameters.iterationRuleDeletion);
    Parameters.doHierarchicalSelectionPerThread[threadNo] = (_iteration >=
                                          Parameters.iterationHierarchicalSelection);

    if (_iteration == Parameters.numIterations - 1) {
      Parameters.ruleDeletionMinRulesPerThread[threadNo] = 1;
      return true;
    }

    if (_iteration == Parameters.iterationRuleDeletion) {
      return true;
    }

    return false;
  }
}

