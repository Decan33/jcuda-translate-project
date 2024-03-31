package org.jcuda.kacygan.chapter3;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class JCudaCode {
    private static final Logger logger = LogManager.getLogger();
    public static void printf(String formatString, Object... params) {
        logger.printf(Level.DEBUG, formatString, params);
    }
}
