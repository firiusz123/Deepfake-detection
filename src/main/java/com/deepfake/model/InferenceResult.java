package com.deepfake.model;

/**
 * Holds the parsed JSON output from the Python inference script.
 */
public class InferenceResult {

    private String label;       // "REAL" or "FAKE"
    private double confidence;  // highest probability (0–100)
    private double realProb;    // probability of REAL (0–100)
    private double fakeProb;    // probability of FAKE (0–100)
    private String error;       // non-null if inference failed

    // ── Constructors ────────────────────────────────────────────────────

    public InferenceResult() {}

    public static InferenceResult ofError(String message) {
        InferenceResult r = new InferenceResult();
        r.error = message;
        return r;
    }

    // ── Getters / Setters ────────────────────────────────────────────────

    public String getLabel()            { return label; }
    public void   setLabel(String l)    { this.label = l; }

    public double getConfidence()             { return confidence; }
    public void   setConfidence(double c)     { this.confidence = c; }

    public double getRealProb()               { return realProb; }
    public void   setRealProb(double p)       { this.realProb = p; }

    public double getFakeProb()               { return fakeProb; }
    public void   setFakeProb(double p)       { this.fakeProb = p; }

    public String getError()            { return error; }
    public void   setError(String e)    { this.error = e; }

    public boolean hasError()           { return error != null && !error.isBlank(); }
}
