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

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import keel.Algorithms.Genetic_Rule_Learning.Globals.*;

public class Control {

  /** Creates a new instance of Control */
  public Control() {
  }

  /**
   * @param args the command line arguments
   */
  public static void main(String[] args) {
    long t1 = System.currentTimeMillis();

    ParserParameters.doParse(args[0]);
    Parameters.setVars();
    
    keel.Algorithms.Genetic_Rule_Learning.Globals.Rand.initRand();

    LogManager.initLogManager();

    try {
      outputParameters(args[0]);
    }
    catch (Exception e) { }
    
    System.out.println();
    
    // Run GAssist-Parallel
    Parallel pa = new Parallel();    
    pa.run();

    long t2 = System.currentTimeMillis();
    LogManager.println("Total time: " + ( (t2 - t1) / 1000.0));

    LogManager.closeLog();
  }
  
  public static void outputParameters(String file) throws IOException {

    BufferedReader reader = new BufferedReader( new FileReader (file));
    
    LogManager.println("Parameter Settings: \n");
    
    String line = null;

    while( ( line = reader.readLine() ) != null ) {
      LogManager.println( line );
    }

    LogManager.println("\n");
    
    reader.close();
  }

}

