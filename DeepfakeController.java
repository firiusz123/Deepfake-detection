package com.deepfake.controller;

import com.deepfake.model.InferenceResult;
import com.deepfake.service.InferenceService;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.*;
import java.util.Map;
import java.util.UUID;

@Controller
public class DeepfakeController {

    private final InferenceService inferenceService;

    public DeepfakeController(InferenceService inferenceService) {
        this.inferenceService = inferenceService;
    }

    // ── Pages ────────────────────────────────────────────────────────────

    /** Serve the main page. */
    @GetMapping("/")
    public String index() {
        return "index";
    }

    // ── API ──────────────────────────────────────────────────────────────

    /**
     * POST /predict  multipart/form-data  field: "image"
     * Saves the upload to a temp file, runs inference, deletes the temp file,
     * and returns a JSON result.
     */
    @PostMapping(value = "/predict", consumes = MediaType.MULTIPART_FORM_DATA_VALUE,
                                     produces = MediaType.APPLICATION_JSON_VALUE)
    @ResponseBody
    public ResponseEntity<?> predict(@RequestParam("image") MultipartFile file) {

        if (file.isEmpty()) {
            return ResponseEntity.badRequest()
                    .body(Map.of("error", "No file received."));
        }

        String original = file.getOriginalFilename();
        String ext = (original != null && original.contains("."))
                ? original.substring(original.lastIndexOf('.'))
                : ".jpg";

        // Write to system temp dir
        Path tmp = null;
        try {
            tmp = Files.createTempFile("deepfake_" + UUID.randomUUID(), ext);
            file.transferTo(tmp);

            InferenceResult result = inferenceService.predict(tmp);
            return ResponseEntity.ok(result);

        } catch (IOException e) {
            return ResponseEntity.internalServerError()
                    .body(Map.of("error", "Failed to save upload: " + e.getMessage()));
        } finally {
            if (tmp != null) {
                try { Files.deleteIfExists(tmp); } catch (IOException ignored) {}
            }
        }
    }

    /** Health check — also reports Python connectivity. */
    @GetMapping(value = "/health", produces = MediaType.APPLICATION_JSON_VALUE)
    @ResponseBody
    public Map<String, String> health() {
        return Map.of("status", "ok", "app", "DeepTrace GUI");
    }
}
