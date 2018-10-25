package GAssist_Parallel;


public class ParallelGlobals {
  public static final ThreadLocal<Integer> threadNo = new ThreadLocal<Integer>();
  public static final ThreadLocal<Rand> localRand = new ThreadLocal<Rand>();
  public static final ThreadLocal<Factory> localFactory = new ThreadLocal<Factory>();
  public static final ThreadLocal<Globals_ADI> localGlobalsADI = new ThreadLocal<Globals_ADI>();
  public static final ThreadLocal<Globals_DefaultC> localGlobalsDefaultC = new ThreadLocal<Globals_DefaultC>();
  public static final ThreadLocal<Globals_GABIL> localGlobalsGABIL = new ThreadLocal<Globals_GABIL>();
  public static final ThreadLocal<Globals_MDL> localGlobalsMDL = new ThreadLocal<Globals_MDL>();
  public static final ThreadLocal<Globals_UBR> localGlobalsUBR = new ThreadLocal<Globals_UBR>();

  
  public static void set(int threadNo, Rand localRand, Factory localFactory, Globals_ADI localGlobalsADI, 
      Globals_GABIL localGlobalsGABIL, Globals_DefaultC localGlobalsDefaultC,
      Globals_MDL localGlobalsMDL, Globals_UBR localGlobalsUBR) {

    ParallelGlobals.threadNo.set(threadNo);
    ParallelGlobals.localRand.set(localRand);
    ParallelGlobals.localFactory.set(localFactory);
    ParallelGlobals.localGlobalsADI.set(localGlobalsADI);
    ParallelGlobals.localGlobalsDefaultC.set(localGlobalsDefaultC);
    ParallelGlobals.localGlobalsGABIL.set(localGlobalsGABIL);
    ParallelGlobals.localGlobalsMDL.set(localGlobalsMDL);
    ParallelGlobals.localGlobalsUBR.set(localGlobalsUBR);
  }
  
  public static void setRand(Rand localRand) {
    ParallelGlobals.localRand.set(localRand);
  }
  
  public static void unset() {
    
  }
  
  public static int getThreadNo() {
    return threadNo.get();
  }
  
  public static Rand getRand() {
    return localRand.get();
  }
  
  public static Factory getFactory() {
    return localFactory.get();
  }
  
  public static Globals_ADI getGlobals_ADI() {
    return localGlobalsADI.get();
  }
  
  public static Globals_DefaultC getGlobals_DefaultC() {
    return localGlobalsDefaultC.get();
  }
  
  public static Globals_GABIL getGlobals_GABIL() {
    return localGlobalsGABIL.get();
  }
  
  public static Globals_MDL getGlobals_MDL() {
    return localGlobalsMDL.get();
  }
  
  public static Globals_UBR getGlobals_UBR() {
    return localGlobalsUBR.get();
  }
}
