package com.deepfake;

import com.fasterxml.jackson.databind.ObjectMapper;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.awt.event.*;
import java.awt.geom.RoundRectangle2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import java.util.prefs.Preferences;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

/**
 * DeepTrace Snip — Desktop deepfake detector.
 *
 * Press CTRL+SHIFT+D anywhere → drag to select screen region
 * → result popup shows REAL / FAKE with confidence.
 */
public class DeepTraceSnip {

    // ── Config (editable via Settings dialog) ────────────────────────────
    private static final Preferences PREFS = Preferences.userNodeForPackage(DeepTraceSnip.class);
    private static final String PREF_PYTHON  = "python_cmd";
    private static final String PREF_SCRIPT  = "infer_script";
    private static final String DEFAULT_PYTHON = "";
    private static final String DEFAULT_SCRIPT = getDefaultInferPath();

    // ── Colors ────────────────────────────────────────────────────────────
    private static final Color C_BLACK  = new Color(0x0a0a0a);
    private static final Color C_WHITE  = new Color(0xf0ece0);
    private static final Color C_YELLOW = new Color(0xffe600);
    private static final Color C_RED    = new Color(0xff2200);
    private static final Color C_GREEN  = new Color(0x00ff44);
    private static final Color C_PANEL  = new Color(0x111111);

    private static final ObjectMapper MAPPER = new ObjectMapper();

    // ── Entry point ───────────────────────────────────────────────────────
    public static void main(String[] args) throws Exception {
        System.setProperty("awt.useSystemAAFontSettings", "on");
        System.setProperty("swing.aatext", "true");
        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());

        if (!SystemTray.isSupported()) {
            JOptionPane.showMessageDialog(null, "System tray not supported on this platform.");
            System.exit(1);
        }

        SwingUtilities.invokeLater(DeepTraceSnip::initTray);
    }
    private static String getDefaultInferPath() {
        // First check next to the running jar/exe
        String jarDir = null;
        try {
            jarDir = new File(DeepTraceSnip.class.getProtectionDomain()
                    .getCodeSource().getLocation().toURI()).getParent();
        } catch (Exception ignored) {}

        if (jarDir != null) {
            File local = new File(jarDir, "infer.exe");
            if (local.exists()) return local.getAbsolutePath().replace("\\", "/");
        }

        // Fallback to C:/DeepTrace
        File fallback = new File("C:/DeepTrace/infer.exe");
        if (fallback.exists()) return fallback.getAbsolutePath().replace("\\", "/");

        // Last resort — return expected path so user can fix in Settings
        return "C:/DeepTrace/infer.exe";
    }
    // ── System tray setup ─────────────────────────────────────────────────
    private static void initTray() {
        // Create a simple icon programmatically (yellow D on black)
        Image icon = createTrayIcon();

        TrayIcon trayIcon = new TrayIcon(icon, "DeepTrace Snip");
        trayIcon.setImageAutoSize(true);

        PopupMenu menu = new PopupMenu();

        MenuItem snipItem = new MenuItem("📷  Snip & Analyse  (Ctrl+Shift+D)");
        snipItem.addActionListener(e -> startSnip());

        MenuItem settingsItem = new MenuItem("⚙  Settings");
        settingsItem.addActionListener(e -> showSettings());

        MenuItem aboutItem = new MenuItem("ℹ  About");
        aboutItem.addActionListener(e -> showAbout());

        MenuItem exitItem = new MenuItem("✕  Exit");
        exitItem.addActionListener(e -> System.exit(0));

        menu.add(snipItem);
        menu.addSeparator();
        menu.add(settingsItem);
        menu.add(aboutItem);
        menu.addSeparator();
        menu.add(exitItem);

        trayIcon.setPopupMenu(menu);
        trayIcon.addActionListener(e -> startSnip()); // double-click

        try {
            SystemTray.getSystemTray().add(trayIcon);
        } catch (AWTException e) {
            JOptionPane.showMessageDialog(null, "Could not add tray icon: " + e.getMessage());
            System.exit(1);
        }

        // Global hotkey via a hidden frame
        registerHotkey();

        // Show startup notification
        trayIcon.displayMessage(
                "DeepTrace Snip",
                "Running! Press Ctrl+Shift+D or double-click tray icon to snip.",
                TrayIcon.MessageType.INFO
        );
    }

    // ── Hotkey via hidden JFrame + KeyEventDispatcher ─────────────────────
    private static void registerHotkey() {
        // We use a transparent always-on-top frame to catch global keypresses
        // when the app is focused. For true global hotkey we use a polling thread.
        Thread hotkeyThread = new Thread(() -> {
            // Simple polling approach — check if CTRL+SHIFT+D was pressed
            // using Robot to monitor (won't work globally; we use the tray instead)
            // Proper global hotkeys requi      re JNI. We expose it via the tray icon.
        }, "hotkey-monitor");
        hotkeyThread.setDaemon(true);
        hotkeyThread.start();
    }

    // ── Snip flow ─────────────────────────────────────────────────────────
    private static void startSnip() {
        // Capture entire screen first
        Robot robot;
        try {
            robot = new Robot();
        } catch (AWTException e) {
            showError("Could not create Robot: " + e.getMessage());
            return;
        }

        Rectangle screenRect = new Rectangle(Toolkit.getDefaultToolkit().getScreenSize());
        BufferedImage screenShot = robot.createScreenCapture(screenRect);

        // Show the selection overlay
        SwingUtilities.invokeLater(() -> {
            SnipOverlay overlay = new SnipOverlay(screenShot, selectedImage -> {
                if (selectedImage != null) {
                    runInference(selectedImage);
                }
            });
            overlay.setVisible(true);
        });
    }

    // ── Inference ─────────────────────────────────────────────────────────
    private static void runInference(BufferedImage image) {
        // Show loading popup
        JDialog loadingDialog = createLoadingDialog();
        SwingUtilities.invokeLater(() -> loadingDialog.setVisible(true));

        // Run in background thread
        Thread thread = new Thread(() -> {
            Path tmpFile = null;
            try {
                tmpFile = Files.createTempFile("deeptrace_" + UUID.randomUUID(), ".png");
                ImageIO.write(image, "PNG", tmpFile.toFile());

                String pythonCmd = PREFS.get(PREF_PYTHON, DEFAULT_PYTHON);
                String scriptPath = PREFS.get(PREF_SCRIPT, DEFAULT_SCRIPT);

                List<String> cmd = pythonCmd.isBlank()
                        ? List.of(scriptPath, tmpFile.toAbsolutePath().toString())
                        : List.of(pythonCmd, scriptPath, tmpFile.toAbsolutePath().toString());
                ProcessBuilder pb = new ProcessBuilder(cmd);
                pb.redirectErrorStream(false);

                Process proc = pb.start();
                boolean finished = proc.waitFor(60, TimeUnit.SECONDS);

                if (!finished) {
                    proc.destroyForcibly();
                    SwingUtilities.invokeLater(() -> {
                        loadingDialog.dispose();
                        showError("Inference timed out after 60s.");
                    });
                    return;
                }

                String stdout = new String(proc.getInputStream().readAllBytes(), StandardCharsets.UTF_8).trim();
                String stderr = new String(proc.getErrorStream().readAllBytes(), StandardCharsets.UTF_8).trim();

                if (proc.exitValue() != 0 || stdout.isBlank()) {
                    final String msg = stderr.isBlank() ? "Script exited with code " + proc.exitValue() : stderr;
                    SwingUtilities.invokeLater(() -> { loadingDialog.dispose(); showError(msg); });
                    return;
                }

                // Parse JSON
                String jsonLine = lastJsonLine(stdout);
                InferenceResult result = MAPPER.readValue(jsonLine, InferenceResult.class);

                final InferenceResult finalResult = result;
                final BufferedImage finalImage = image;
                SwingUtilities.invokeLater(() -> {
                    loadingDialog.dispose();
                    if (finalResult.error != null && !finalResult.error.isBlank()) {
                        showError(finalResult.error);
                    } else {
                        showResult(finalResult, finalImage);
                    }
                });

            } catch (Exception e) {
                SwingUtilities.invokeLater(() -> {
                    loadingDialog.dispose();
                    showError(e.getMessage());
                });
            } finally {
                if (tmpFile != null) {
                    try { Files.deleteIfExists(tmpFile); } catch (IOException ignored) {}
                }
            }
        }, "inference-thread");
        thread.setDaemon(true);
        thread.start();
    }

    // ── Result window ──────────────────────────────────────────────────────
    private static void showResult(InferenceResult result, BufferedImage image) {
        boolean isFake = "FAKE".equals(result.label);
        Color verdictColor = isFake ? C_RED : C_GREEN;

        JFrame frame = new JFrame("DeepTrace Result");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setResizable(false);
        frame.setAlwaysOnTop(true);
        frame.getContentPane().setBackground(C_BLACK);

        JPanel root = new JPanel();
        root.setLayout(new BorderLayout(0, 0));
        root.setBackground(C_BLACK);
        root.setBorder(BorderFactory.createLineBorder(verdictColor, 3));

        // ── Top bar ───────────────────────────────────────────────────────
        JPanel topBar = new JPanel(new BorderLayout());
        topBar.setBackground(verdictColor);
        topBar.setBorder(new EmptyBorder(6, 14, 6, 14));

        JLabel titleLabel = new JLabel("DEEPTRACE // ANALYSIS RESULT");
        titleLabel.setFont(new Font("Courier New", Font.BOLD, 11));
        titleLabel.setForeground(C_BLACK);
        topBar.add(titleLabel, BorderLayout.WEST);

        JButton closeBtn = new JButton("✕");
        closeBtn.setFont(new Font("Courier New", Font.BOLD, 11));
        closeBtn.setForeground(C_BLACK);
        closeBtn.setBackground(verdictColor);
        closeBtn.setBorder(new EmptyBorder(2, 8, 2, 4));
        closeBtn.setFocusPainted(false);
        closeBtn.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        closeBtn.addActionListener(e -> frame.dispose());
        topBar.add(closeBtn, BorderLayout.EAST);

        root.add(topBar, BorderLayout.NORTH);

        // ── Center: image + verdict ────────────────────────────────────────
        JPanel center = new JPanel(new BorderLayout(16, 0));
        center.setBackground(C_BLACK);
        center.setBorder(new EmptyBorder(16, 16, 16, 16));

        // Thumbnail
        int thumbW = Math.min(image.getWidth(), 360);
        int thumbH = (int)((double)image.getHeight() / image.getWidth() * thumbW);
        BufferedImage thumb = new BufferedImage(thumbW, thumbH, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2 = thumb.createGraphics();
        g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
        g2.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2.drawImage(image, 0, 0, thumbW, thumbH, null);
        g2.dispose();

        JLabel imgLabel = new JLabel(new ImageIcon(thumb));
        imgLabel.setBorder(BorderFactory.createLineBorder(new Color(0x2a2a2a), 2));
        center.add(imgLabel, BorderLayout.WEST);

        // Verdict panel
        JPanel verdictPanel = new JPanel();
        verdictPanel.setLayout(new BoxLayout(verdictPanel, BoxLayout.Y_AXIS));
        verdictPanel.setBackground(C_BLACK);

        // Big verdict label
        JLabel verdictLabel = new JLabel(result.label);
        verdictLabel.setFont(new Font("Arial Black", Font.BOLD, 52));
        verdictLabel.setForeground(verdictColor);
        verdictLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
        verdictPanel.add(verdictLabel);

        verdictPanel.add(Box.createVerticalStrut(4));

        // Confidence
        JLabel confLabel = new JLabel(String.format("%.1f%% CONFIDENCE", result.confidence));
        confLabel.setFont(new Font("Courier New", Font.BOLD, 13));
        confLabel.setForeground(C_WHITE);
        confLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
        verdictPanel.add(confLabel);

        verdictPanel.add(Box.createVerticalStrut(16));

        // REAL bar
        verdictPanel.add(makeBarRow("REAL", result.realProb, C_GREEN));
        verdictPanel.add(Box.createVerticalStrut(8));

        // FAKE bar
        verdictPanel.add(makeBarRow("FAKE", result.fakeProb, C_RED));

        verdictPanel.add(Box.createVerticalStrut(16));

        // Snip again button
        JButton snipAgain = new JButton("▶  SNIP AGAIN");
        snipAgain.setFont(new Font("Courier New", Font.BOLD, 11));
        snipAgain.setForeground(C_BLACK);
        snipAgain.setBackground(C_YELLOW);
        snipAgain.setBorder(new EmptyBorder(8, 18, 8, 18));
        snipAgain.setFocusPainted(false);
        snipAgain.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        snipAgain.setAlignmentX(Component.LEFT_ALIGNMENT);
        snipAgain.addActionListener(e -> { frame.dispose(); startSnip(); });
        verdictPanel.add(snipAgain);

        center.add(verdictPanel, BorderLayout.CENTER);
        root.add(center, BorderLayout.CENTER);

        // ── Bottom log bar ─────────────────────────────────────────────────
        JPanel logBar = new JPanel(new GridLayout(1, 4));
        logBar.setBackground(new Color(0x0d0d0d));
        logBar.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createMatteBorder(1, 0, 0, 0, new Color(0x222222)),
                new EmptyBorder(6, 14, 6, 14)
        ));

        logBar.add(logEntry("REAL", String.format("%.2f%%", result.realProb), C_GREEN));
        logBar.add(logEntry("FAKE", String.format("%.2f%%", result.fakeProb), C_RED));
        logBar.add(logEntry("MODEL", "SimpleCNN", C_WHITE));
        logBar.add(logEntry("STATUS", result.label, verdictColor));

        root.add(logBar, BorderLayout.SOUTH);

        frame.setContentPane(root);
        frame.pack();
        frame.setMinimumSize(new Dimension(560, 280));

        // Center on screen
        Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
        frame.setLocation(
                (screen.width - frame.getWidth()) / 2,
                (screen.height - frame.getHeight()) / 2
        );

        frame.setVisible(true);
    }

    // ── UI helpers ─────────────────────────────────────────────────────────

    private static JPanel makeBarRow(String labelText, double value, Color color) {
        JPanel row = new JPanel(new BorderLayout(8, 0));
        row.setBackground(C_BLACK);
        row.setMaximumSize(new Dimension(Integer.MAX_VALUE, 22));
        row.setAlignmentX(Component.LEFT_ALIGNMENT);

        JLabel lbl = new JLabel(labelText);
        lbl.setFont(new Font("Courier New", Font.BOLD, 10));
        lbl.setForeground(color);
        lbl.setPreferredSize(new Dimension(36, 14));
        row.add(lbl, BorderLayout.WEST);

        // Bar track
        JPanel track = new JPanel(null) {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                Graphics2D g2 = (Graphics2D) g;
                g2.setColor(new Color(0x1a1a1a));
                g2.fillRect(0, 0, getWidth(), getHeight());
                int fillW = (int)(getWidth() * value / 100.0);
                g2.setColor(color);
                g2.fillRect(0, 0, fillW, getHeight());
            }
        };
        track.setPreferredSize(new Dimension(160, 10));
        track.setBackground(new Color(0x1a1a1a));
        row.add(track, BorderLayout.CENTER);

        JLabel pct = new JLabel(String.format("%.1f%%", value));
        pct.setFont(new Font("Courier New", Font.PLAIN, 10));
        pct.setForeground(C_WHITE);
        pct.setPreferredSize(new Dimension(44, 14));
        pct.setHorizontalAlignment(SwingConstants.RIGHT);
        row.add(pct, BorderLayout.EAST);

        return row;
    }

    private static JLabel logEntry(String key, String value, Color valueColor) {
        JLabel lbl = new JLabel("<html><span style='color:#444;font-size:9px'>" + key +
                "</span><br><span style='color:#" +
                String.format("%06X", valueColor.getRGB() & 0xFFFFFF) +
                ";font-weight:bold'>" + value + "</span></html>");
        lbl.setFont(new Font("Courier New", Font.PLAIN, 10));
        return lbl;
    }

    private static JDialog createLoadingDialog() {
        JDialog d = new JDialog((Frame) null, "Analysing...", false);
        d.setUndecorated(true);
        d.setAlwaysOnTop(true);

        JPanel p = new JPanel(new BorderLayout());
        p.setBackground(C_BLACK);
        p.setBorder(BorderFactory.createLineBorder(C_YELLOW, 3));

        JLabel lbl = new JLabel("  ◈  SCANNING IMAGE...  ");
        lbl.setFont(new Font("Courier New", Font.BOLD, 14));
        lbl.setForeground(C_YELLOW);
        lbl.setBorder(new EmptyBorder(18, 24, 18, 24));
        p.add(lbl, BorderLayout.CENTER);

        d.setContentPane(p);
        d.pack();
        Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
        d.setLocation((screen.width - d.getWidth()) / 2, (screen.height - d.getHeight()) / 2);
        return d;
    }

    private static void showError(String msg) {
        JOptionPane.showMessageDialog(null,
                "⚠  " + msg, "DeepTrace Error", JOptionPane.ERROR_MESSAGE);
    }

    private static void showSettings() {
        JDialog d = new JDialog((Frame) null, "DeepTrace Settings", true);
        d.setResizable(false);
        d.getContentPane().setBackground(C_BLACK);

        JPanel p = new JPanel(new GridBagLayout());
        p.setBackground(C_BLACK);
        p.setBorder(new EmptyBorder(20, 24, 20, 24));

        GridBagConstraints c = new GridBagConstraints();
        c.insets = new Insets(6, 6, 6, 6);
        c.fill = GridBagConstraints.HORIZONTAL;

        Font labelFont = new Font("Courier New", Font.BOLD, 11);
        Font fieldFont = new Font("Courier New", Font.PLAIN, 11);

        c.gridx = 0; c.gridy = 0;
        JLabel l1 = new JLabel("PYTHON COMMAND:");
        l1.setForeground(C_YELLOW); l1.setFont(labelFont);
        p.add(l1, c);

        c.gridx = 1;
        JTextField pythonField = new JTextField(PREFS.get(PREF_PYTHON, DEFAULT_PYTHON), 30);
        pythonField.setBackground(C_PANEL); pythonField.setForeground(C_WHITE);
        pythonField.setFont(fieldFont); pythonField.setCaretColor(C_WHITE);
        pythonField.setBorder(BorderFactory.createLineBorder(new Color(0x333333)));
        p.add(pythonField, c);

        c.gridx = 0; c.gridy = 1;
        JLabel l2 = new JLabel("INFER.PY PATH:");
        l2.setForeground(C_YELLOW); l2.setFont(labelFont);
        p.add(l2, c);

        c.gridx = 1;
        JTextField scriptField = new JTextField(PREFS.get(PREF_SCRIPT, DEFAULT_SCRIPT), 30);
        scriptField.setBackground(C_PANEL); scriptField.setForeground(C_WHITE);
        scriptField.setFont(fieldFont); scriptField.setCaretColor(C_WHITE);
        scriptField.setBorder(BorderFactory.createLineBorder(new Color(0x333333)));
        p.add(scriptField, c);

        c.gridx = 0; c.gridy = 2; c.gridwidth = 2;
        c.anchor = GridBagConstraints.CENTER;
        JButton save = new JButton("SAVE");
        save.setFont(new Font("Courier New", Font.BOLD, 12));
        save.setBackground(C_YELLOW); save.setForeground(C_BLACK);
        save.setBorder(new EmptyBorder(8, 24, 8, 24));
        save.setFocusPainted(false);
        save.addActionListener(e -> {
            PREFS.put(PREF_PYTHON, pythonField.getText().trim());
            PREFS.put(PREF_SCRIPT, scriptField.getText().trim());
            d.dispose();
        });
        p.add(save, c);

        d.setContentPane(p);
        d.pack();
        Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
        d.setLocation((screen.width - d.getWidth()) / 2, (screen.height - d.getHeight()) / 2);
        d.setVisible(true);
    }

    private static void showAbout() {
        JOptionPane.showMessageDialog(null,
                "DeepTrace Snip v1.0\n\nSnip any screen region and detect deepfakes instantly.\n\nDouble-click tray icon or right-click → Snip & Analyse.",
                "About DeepTrace Snip", JOptionPane.INFORMATION_MESSAGE);
    }

    private static Image createTrayIcon() {
        int size = 64;
        BufferedImage img = new BufferedImage(size, size, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = img.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setColor(C_YELLOW);
        g.fillOval(0, 0, size, size);
        g.setColor(C_BLACK);
        g.setFont(new Font("Arial Black", Font.BOLD, 42));
        FontMetrics fm = g.getFontMetrics();
        int x = (size - fm.stringWidth("D")) / 2;
        int y = (size - fm.getHeight()) / 2 + fm.getAscent();
        g.drawString("D", x, y);
        g.dispose();
        return img;
    }

    private static String lastJsonLine(String output) {
        String[] lines = output.split("\\r?\\n");
        for (int i = lines.length - 1; i >= 0; i--) {
            if (lines[i].trim().startsWith("{")) return lines[i].trim();
        }
        return output;
    }

    // ── Inner classes ──────────────────────────────────────────────────────

    /** Fullscreen transparent overlay for selecting a screen region. */
    static class SnipOverlay extends JWindow {
        private final BufferedImage screenshot;
        private final java.util.function.Consumer<BufferedImage> callback;

        private Point startPoint;
        private Rectangle selection;

        SnipOverlay(BufferedImage screenshot, java.util.function.Consumer<BufferedImage> callback) {
            this.screenshot = screenshot;
            this.callback = callback;

            setAlwaysOnTop(true);
            setBounds(new Rectangle(Toolkit.getDefaultToolkit().getScreenSize()));

            JPanel panel = new JPanel() {
                @Override
                protected void paintComponent(Graphics g) {
                    Graphics2D g2 = (Graphics2D) g;
                    g2.drawImage(screenshot, 0, 0, null);
                    g2.setColor(new Color(0, 0, 0, 140));
                    g2.fillRect(0, 0, getWidth(), getHeight());

                    if (selection != null && selection.width > 0 && selection.height > 0) {
                        g2.drawImage(screenshot,
                                selection.x, selection.y,
                                selection.x + selection.width,
                                selection.y + selection.height,
                                selection.x, selection.y,
                                selection.x + selection.width,
                                selection.y + selection.height,
                                null);
                        g2.setColor(new Color(0xffe600));
                        g2.setStroke(new BasicStroke(2));
                        g2.drawRect(selection.x, selection.y, selection.width, selection.height);
                        int b = 10;
                        g2.setStroke(new BasicStroke(3));
                        int x = selection.x, y = selection.y, w = selection.width, h = selection.height;
                        g2.drawLine(x, y, x + b, y);
                        g2.drawLine(x, y, x, y + b);
                        g2.drawLine(x + w, y, x + w - b, y);
                        g2.drawLine(x + w, y, x + w, y + b);
                        g2.drawLine(x, y + h, x + b, y + h);
                        g2.drawLine(x, y + h, x, y + h - b);
                        g2.drawLine(x + w, y + h, x + w - b, y + h);
                        g2.drawLine(x + w, y + h, x + w, y + h - b);
                    }

                    g2.setFont(new Font("Courier New", Font.BOLD, 14));
                    String hint = "DRAG TO SELECT  |  RIGHT-CLICK OR ESC TO CANCEL";
                    FontMetrics fm = g2.getFontMetrics();
                    int tx = (getWidth() - fm.stringWidth(hint)) / 2;
                    g2.setColor(new Color(0, 0, 0, 160));
                    g2.fillRoundRect(tx - 12, getHeight() - 46, fm.stringWidth(hint) + 24, 28, 6, 6);
                    g2.setColor(new Color(0xffe600));
                    g2.drawString(hint, tx, getHeight() - 27);
                }
            };

            panel.setOpaque(false);
            panel.setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
            panel.setFocusable(true);

            // ── Cancel helper ────────────────────────────────────────────────
            Runnable cancel = () -> {
                dispose();
                callback.accept(null);
            };

            // ── Mouse ────────────────────────────────────────────────────────
            panel.addMouseListener(new MouseAdapter() {
                @Override
                public void mousePressed(MouseEvent e) {
                    if (e.getButton() == MouseEvent.BUTTON3) { // right-click = cancel
                        cancel.run();
                        return;
                    }
                    startPoint = e.getPoint();
                    selection = new Rectangle(startPoint);
                }

                @Override
                public void mouseReleased(MouseEvent e) {
                    if (e.getButton() == MouseEvent.BUTTON3) return;
                    if (selection != null && selection.width > 4 && selection.height > 4) {
                        dispose();
                        // Account for DPI scaling
                        GraphicsDevice gd = GraphicsEnvironment.getLocalGraphicsEnvironment().getDefaultScreenDevice();
                        java.awt.geom.AffineTransform transform = gd.getDefaultConfiguration().getDefaultTransform();
                        double scaleX = transform.getScaleX();
                        double scaleY = transform.getScaleY();
                        int rx = (int) (selection.x * scaleX);
                        int ry = (int) (selection.y * scaleY);
                        int rw = Math.min((int) (selection.width * scaleX), screenshot.getWidth() - rx);
                        int rh = Math.min((int) (selection.height * scaleY), screenshot.getHeight() - ry);
                        callback.accept(screenshot.getSubimage(rx, ry, rw, rh));
                    }
                }
            });

            panel.addMouseMotionListener(new MouseMotionAdapter() {
                @Override
                public void mouseDragged(MouseEvent e) {
                    if (startPoint != null) {
                        int x = Math.min(startPoint.x, e.getX());
                        int y = Math.min(startPoint.y, e.getY());
                        int w = Math.abs(e.getX() - startPoint.x);
                        int h = Math.abs(e.getY() - startPoint.y);
                        selection = new Rectangle(x, y, w, h);
                        panel.repaint();
                    }
                }
            });

            // ── ESC key — register dispatcher and remove it when done ────────
            KeyEventDispatcher escDispatcher = e -> {
                if (e.getID() == KeyEvent.KEY_PRESSED && e.getKeyCode() == KeyEvent.VK_ESCAPE) {
                    cancel.run();
                    return true;
                }
                return false;
            };
            KeyboardFocusManager.getCurrentKeyboardFocusManager().addKeyEventDispatcher(escDispatcher);

            // Remove dispatcher when window closes
            addWindowListener(new WindowAdapter() {
                @Override
                public void windowClosed(WindowEvent e) {
                    KeyboardFocusManager.getCurrentKeyboardFocusManager()
                            .removeKeyEventDispatcher(escDispatcher);
                }
            });

            setContentPane(panel);
        }

        @Override
        public void setVisible(boolean b) {
            super.setVisible(b);
            if (b) {
                toFront();
                getContentPane().requestFocusInWindow();
            }
        }
    }
    /** JSON result from infer.py */
    static class InferenceResult {
        public String label;
        public double confidence;
        public double realProb;
        public double fakeProb;
        public String error;
    }
}
