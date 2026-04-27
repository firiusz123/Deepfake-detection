package com.deepfake.service;

import com.deepfake.model.InferenceResult;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Runs the Python inference script as a subprocess and parses its JSON output.
 *
 * The script path and Python executable are configurable via application.properties
 * so the user can point them at the correct locations without touching Java code.
 */
@Service
public class InferenceService {

    /** Path to infer.py — resolved relative to the working directory or absolute. */
    @Value("${deepfake.infer-script:../infer.py}")
    private String inferScript;

    /** Python executable — override to "python3" or a venv path if needed. */
    @Value("${deepfake.python-cmd:python}")
    private String pythonCmd;

    /** Timeout per inference call in seconds. */
    @Value("${deepfake.timeout-seconds:60}")
    private int timeoutSeconds;

    private final ObjectMapper mapper = new ObjectMapper();

    // ── Public API ───────────────────────────────────────────────────────

    /**
     * Run inference on the given image file.
     *
     * @param imagePath  absolute path to the (already-saved) temp image
     * @return           parsed result, or an error result on failure
     */
    public InferenceResult predict(Path imagePath) {
        List<String> cmd = List.of(pythonCmd, inferScript, imagePath.toAbsolutePath().toString());

        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.redirectErrorStream(false);

        try {
            Process proc = pb.start();

            boolean finished = proc.waitFor(timeoutSeconds, TimeUnit.SECONDS);
            if (!finished) {
                proc.destroyForcibly();
                return InferenceResult.ofError("Inference timed out after " + timeoutSeconds + "s");
            }

            String stdout = new String(proc.getInputStream().readAllBytes(), StandardCharsets.UTF_8).trim();
            String stderr = new String(proc.getErrorStream().readAllBytes(), StandardCharsets.UTF_8).trim();

            if (proc.exitValue() != 0) {
                String msg = stderr.isBlank() ? "Python script exited with code " + proc.exitValue() : stderr;
                return InferenceResult.ofError(msg);
            }

            if (stdout.isBlank()) {
                return InferenceResult.ofError("Python script produced no output." +
                        (stderr.isBlank() ? "" : " STDERR: " + stderr));
            }

            // Parse the last line — in case Python prints progress text before JSON
            String jsonLine = lastJsonLine(stdout);
            return mapper.readValue(jsonLine, InferenceResult.class);

        } catch (IOException | InterruptedException e) {
            Thread.currentThread().interrupt();
            return InferenceResult.ofError("Failed to launch inference script: " + e.getMessage());
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    /** Returns the last line that looks like JSON from a multi-line stdout. */
    private String lastJsonLine(String output) {
        String[] lines = output.split("\\r?\\n");
        for (int i = lines.length - 1; i >= 0; i--) {
            String line = lines[i].trim();
            if (line.startsWith("{")) return line;
        }
        return output; // fall back to full output and let Jackson report the error
    }
}
