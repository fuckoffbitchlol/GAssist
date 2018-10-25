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

/*
 * Rand.java
 *
 * Created on 29 de marzo de 2004, 23:31
 */

/**
 *
 */

package GAssist_Parallel;

import java.util.*;

import keel.Algorithms.Genetic_Rule_Learning.Globals.MTwister;

public class Rand {
    
    private MTwister random;
    
    /** Generates a new instance of Random */
    public void initRand(int seed) {
        random = new MTwister(seed);
    }
    
    /**
     *  Returns a random real between [0,1)
     */
    public double getReal() {
        return random.genrand_real2();
    }
    
    /**
     *  Returns a random long between [uLow,uHigh]
     */
    public int getInteger(int uLow, int uHigh) {
    	return (uLow + (int)(random.genrand_real2()*(uHigh + 1 - uLow)));
    }
    
}

