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

import keel.Algorithms.Genetic_Rule_Learning.Globals.*;

public class Globals_MDL {
	double theoryWeight;
	boolean activated = false;
	boolean fixedWeight = false;

	public boolean newIteration(Rand rn, int iteration, Classifier[]pop, Statistics st) {
		if (!Parameters.useMDL) {
			return false;
		}

		Classifier ind = PopulationWrapper.getBest(pop);

		boolean updateWeight = false;
		if (iteration == Parameters.iterationMDL) {
//			LogManager.println("Iteration " + iteration +
//					   " :MDL fitness activated");
			activated = true;
			double error = ind.getExceptionsLength();
			double theoryLength = ind.getTheoryLength();
			 theoryLength *= Parameters.numClasses;
			 theoryLength /= ind.getNumAliveRules();

			 theoryWeight =
			    (Parameters.initialTheoryLengthRatio /
			     (1.0 - Parameters.initialTheoryLengthRatio))
			    * (error / theoryLength);
			 updateWeight = true;
		}

		if (activated && !fixedWeight &&
		    (st.last10IterationsAccuracyAverage == 1.0 || Parameters.weightRelaxFactor == 1.0) ) {
			fixedWeight = true;
		}

		if (activated && !fixedWeight) {
			if (ind.getAccuracy() != 1.0) {
				if (st.getIterationsSinceBest() ==
				    10) {
					theoryWeight *=
					    Parameters.weightRelaxFactor;
					updateWeight = true;
				}
			}
		}

		if (updateWeight) {
			st.resetBestStats();
			return true;
		}

		return false;
	}

	public double mdlFitness(Classifier ind, PerformanceAgent pa) {
		double fit = 0;
		ind.computeTheoryLength(pa);
		if (activated) {
			fit = ind.getTheoryLength() * theoryWeight;
		}
		double exceptionsLength =
		    105.00 - pa.getAccuracy() * 100.0;
		ind.setExceptionsLength(exceptionsLength);
		fit += exceptionsLength;
		return fit;
	}
}

