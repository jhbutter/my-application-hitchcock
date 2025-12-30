package com.example.myapplication_hitchock;

import android.Manifest;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.YuvImage;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Range;
import android.util.Size;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;

import java.io.FileInputStream;
import java.io.OutputStream;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.camera2.interop.Camera2Interop;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.cardview.widget.CardView;
import androidx.core.content.ContextCompat;
import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.core.view.WindowInsetsControllerCompat;

import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import android.hardware.camera2.CaptureRequest;

public class VideoRecordActivity extends AppCompatActivity {

    private static final String TAG = "VideoRecordActivity";

    private ExecutorService cameraExecutor;
    private ExecutorService processExecutor;

    private PreviewView viewFinder;
    private Button videoCaptureButton;
    private ImageView fullscreenProcessedView;
    private ImageView processedImageView;
    private ImageView frozenPreviewImageView;
    private EditText ipAddressInput;
    private Button connectButton;
    private Button resetButton;
    private RectOverlayView rectOverlayView;

    private DollyZoomProcessor processor;
    private boolean isProcessing = false;
    private volatile boolean shouldInitTracking = false;
    private RectF pendingRoiNormalized = null;

    // Recording State
    private volatile boolean isRecording = false;
    private VideoEncoder videoEncoder;
    private File currentRecordingFile;

    // Permissions
    private final ActivityResultLauncher<String[]> activityResultLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestMultiplePermissions(), permissions -> {
                boolean permissionGranted = true;
                for (Map.Entry<String, Boolean> entry : permissions.entrySet()) {
                    if (entry.getKey().equals(Manifest.permission.CAMERA) || entry.getKey().equals(Manifest.permission.RECORD_AUDIO)) {
                        if (!entry.getValue()) {
                            permissionGranted = false;
                            break;
                        }
                    }
                }
                if (!permissionGranted) {
                    Toast.makeText(this, "Permission request denied", Toast.LENGTH_SHORT).show();
                } else {
                    startCamera();
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Initialize OpenCV
        if (OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV loaded successfully");
        } else {
            Log.e(TAG, "OpenCV initialization failed!");
            Toast.makeText(this, "OpenCV initialization failed!", Toast.LENGTH_LONG).show();
        }

        // Hide system UI
        WindowInsetsControllerCompat windowInsetsController =
                WindowCompat.getInsetsController(getWindow(), getWindow().getDecorView());
        windowInsetsController.setSystemBarsBehavior(
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE);
        windowInsetsController.hide(WindowInsetsCompat.Type.systemBars());

        setContentView(R.layout.activity_video_record);

        viewFinder = findViewById(R.id.viewFinder);
        videoCaptureButton = findViewById(R.id.video_capture_button);
        fullscreenProcessedView = findViewById(R.id.fullscreenProcessedView);
        processedImageView = findViewById(R.id.processedImageView);
        frozenPreviewImageView = findViewById(R.id.frozenPreviewImageView);
        ipAddressInput = findViewById(R.id.ipAddressInput);
        connectButton = findViewById(R.id.connectButton);
        resetButton = findViewById(R.id.resetButton);
        rectOverlayView = findViewById(R.id.rectOverlayView);

        // Hide Network UI elements
        ipAddressInput.setVisibility(View.GONE);
        connectButton.setVisibility(View.GONE);
        
        // Initialize Processor
        processor = new DollyZoomProcessor();
        
        TextView debugInfoText = findViewById(R.id.debugInfoText);
        processor.setOnDebugInfoListener(info -> {
            runOnUiThread(() -> {
                if (debugInfoText != null) {
                    debugInfoText.setText(info);
                }
            });
        });

        // Track Interval Control
        TextView intervalLabel = findViewById(R.id.intervalLabel);
        SeekBar intervalSeekBar = findViewById(R.id.intervalSeekBar);

        if (intervalLabel != null && intervalSeekBar != null) {
            // Initial state (match default in Processor or desired default)
            int defaultInterval = 4;
            intervalSeekBar.setProgress(defaultInterval - 1); // 0-based
            intervalLabel.setText("Track Interval: " + defaultInterval);
            processor.setTrackingInterval(defaultInterval);

            intervalSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                @Override
                public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                    int interval = progress + 1; // 1 to 20
                    intervalLabel.setText("Track Interval: " + interval);
                    if (processor != null) {
                        processor.setTrackingInterval(interval);
                    }
                }

                @Override
                public void onStartTrackingTouch(SeekBar seekBar) {}

                @Override
                public void onStopTrackingTouch(SeekBar seekBar) {}
            });
        }

        // Draggable Preview Card
        CardView previewCardView = findViewById(R.id.previewCardView);
        if (previewCardView != null) {
            previewCardView.setOnTouchListener(new View.OnTouchListener() {
                float dX, dY;
                float startX, startY;
                boolean isClick = false;

                @Override
                public boolean onTouch(View view, MotionEvent event) {
                    switch (event.getAction()) {
                        case MotionEvent.ACTION_DOWN:
                            dX = view.getX() - event.getRawX();
                            dY = view.getY() - event.getRawY();
                            startX = event.getRawX();
                            startY = event.getRawY();
                            isClick = true;
                            return true; // Consume to receive subsequent events
                        case MotionEvent.ACTION_MOVE:
                            if (Math.abs(event.getRawX() - startX) > 10 || Math.abs(event.getRawY() - startY) > 10) {
                                isClick = false;
                            }
                            view.animate()
                                    .x(event.getRawX() + dX)
                                    .y(event.getRawY() + dY)
                                    .setDuration(0)
                                    .start();
                            return true;
                        case MotionEvent.ACTION_UP:
                            if (isClick) {
                                view.performClick();
                            }
                            return true;
                        default:
                            return false;
                    }
                }
            });
        }

        rectOverlayView.setOnRectSelectedListener(new RectOverlayView.OnRectSelectedListener() {
            @Override
            public void onDrawingStarted() {
                if (viewFinder != null && viewFinder.getBitmap() != null) {
                    Bitmap bitmap = viewFinder.getBitmap();
                    frozenPreviewImageView.setImageBitmap(bitmap);
                    frozenPreviewImageView.setVisibility(View.VISIBLE);
                }
            }

            @Override
            public void onRectSelected(RectF rect) {
                // Store normalized ROI
                float viewWidth = viewFinder.getWidth();
                float viewHeight = viewFinder.getHeight();
                
                pendingRoiNormalized = new RectF(
                    rect.left / viewWidth,
                    rect.top / viewHeight,
                    rect.right / viewWidth,
                    rect.bottom / viewHeight
                );
                
                shouldInitTracking = true;
                
                frozenPreviewImageView.setVisibility(View.GONE);
                frozenPreviewImageView.setImageBitmap(null);
                
                processedImageView.setVisibility(View.VISIBLE);
                Toast.makeText(VideoRecordActivity.this, "Tracking Initialized!", Toast.LENGTH_SHORT).show();
            }
        });

        if (allPermissionsGranted()) {
            startCamera();
        } else {
            requestPermissions();
        }

        videoCaptureButton.setOnClickListener(v -> {
            if (!isRecording) {
                isRecording = true;
                videoCaptureButton.setText("Stop Recording");
                Toast.makeText(this, "Recording Started", Toast.LENGTH_SHORT).show();
            } else {
                isRecording = false;
                videoCaptureButton.setText("Start Recording");
                // Force stop recording immediately
                stopRecording();
            }
        });

        resetButton.setOnClickListener(v -> {
            shouldInitTracking = false;
            pendingRoiNormalized = null;
            
            if (processor != null) {
                processor.reset();
            }
            
            rectOverlayView.clear();
            frozenPreviewImageView.setVisibility(View.GONE);
            frozenPreviewImageView.setImageBitmap(null);
            processedImageView.setImageBitmap(null);
            processedImageView.setVisibility(View.GONE);
            
            // Hide fullscreen processed view to show raw camera again
            fullscreenProcessedView.setVisibility(View.GONE);
            fullscreenProcessedView.setImageBitmap(null);
        });

        cameraExecutor = Executors.newSingleThreadExecutor();
        processExecutor = Executors.newSingleThreadExecutor();
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(viewFinder.getSurfaceProvider());

                // Image Analysis for Processing
                ImageAnalysis.Builder builder = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                        .setTargetResolution(new Size(640, 480)); // VGA resolution for higher FPS

                Camera2Interop.Extender<ImageAnalysis> extender = new Camera2Interop.Extender<>(builder);
                extender.setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, new Range<>(60, 60));

                ImageAnalysis imageAnalysis = builder.build();
                imageAnalysis.setAnalyzer(cameraExecutor, this::analyzeImage);

                CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;

                try {
                    cameraProvider.unbindAll();
                    cameraProvider.bindToLifecycle(
                            this, cameraSelector, preview, imageAnalysis);

                } catch (Exception exc) {
                    Log.e(TAG, "Use case binding failed", exc);
                }

            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "CameraProvider future execution failed", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void analyzeImage(ImageProxy image) {
        if (isProcessing) {
            image.close();
            return;
        }
        isProcessing = true;

        try {
            // 1. Convert YUV to Mat directly (Much faster than JPEG conversion)
            byte[] nv21Data = yuv420ToNv21(image);
            
            Mat yuvMat = new Mat(image.getHeight() + image.getHeight() / 2, image.getWidth(), CvType.CV_8UC1);
            yuvMat.put(0, 0, nv21Data);
            
            Mat mat = new Mat();
            Imgproc.cvtColor(yuvMat, mat, Imgproc.COLOR_YUV2RGB_NV21);
            
            yuvMat.release();
            
            // Handle Rotation
            int rotation = image.getImageInfo().getRotationDegrees();
            if (rotation != 0) {
                Mat rotated = new Mat();
                if (rotation == 90) Core.rotate(mat, rotated, Core.ROTATE_90_CLOCKWISE);
                else if (rotation == 180) Core.rotate(mat, rotated, Core.ROTATE_180);
                else if (rotation == 270) Core.rotate(mat, rotated, Core.ROTATE_90_COUNTERCLOCKWISE);
                mat.release();
                mat = rotated;
            }
            
            // 2. Init Tracking if needed
            if (shouldInitTracking && pendingRoiNormalized != null) {
                int w = mat.width();
                int h = mat.height();
                
                android.graphics.RectF roi = new android.graphics.RectF(
                    pendingRoiNormalized.left * w,
                    pendingRoiNormalized.top * h,
                    pendingRoiNormalized.right * w,
                    pendingRoiNormalized.bottom * h
                );
                
                // Ensure ROI is valid
                if (roi.width() > 0 && roi.height() > 0) {
                    processor.init(mat, roi);
                    shouldInitTracking = false;
                }
            }
            
            // 3. Process
            Mat processedMat = processor.process(mat);
            
            // 4. Show Result
            Bitmap resultBitmap = Bitmap.createBitmap(processedMat.cols(), processedMat.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(processedMat, resultBitmap);
            
            boolean isTracking = processor.isInitialized();

            runOnUiThread(() -> {
                if (isTracking) {
                    // Tracking: Show processed image in the small preview card
                    if (processedImageView.getVisibility() != View.VISIBLE) {
                        processedImageView.setVisibility(View.VISIBLE);
                    }
                    processedImageView.setImageBitmap(resultBitmap);

                    // Ensure fullscreen processed view is GONE (so raw camera preview shows in background)
                    if (fullscreenProcessedView.getVisibility() == View.VISIBLE) {
                        fullscreenProcessedView.setVisibility(View.GONE);
                        fullscreenProcessedView.setImageBitmap(null);
                    }
                } else {
                    // Not tracking: Hide processed views
                    if (processedImageView.getVisibility() == View.VISIBLE) {
                        processedImageView.setVisibility(View.GONE);
                        processedImageView.setImageBitmap(null);
                    }
                    if (fullscreenProcessedView.getVisibility() == View.VISIBLE) {
                        fullscreenProcessedView.setVisibility(View.GONE);
                        fullscreenProcessedView.setImageBitmap(null);
                    }
                }
            });
            
            // 5. Record if needed
            handleRecording(processedMat);
            
            mat.release();
            if (processedMat != mat) processedMat.release();
            
        } catch (Exception e) {
            Log.e(TAG, "Error in analysis", e);
        } finally {
            image.close();
            isProcessing = false;
        }
    }

    private void stopRecording() {
        if (videoEncoder != null) {
            videoEncoder.release();
            videoEncoder = null;
            Log.d(TAG, "Recording stopped");
            String path = currentRecordingFile.getAbsolutePath();
            runOnUiThread(() -> Toast.makeText(this, "Saved to: " + path, Toast.LENGTH_LONG).show());
            
            // Export to Gallery for easier access
            exportToGallery(currentRecordingFile);
        }
    }

    private void handleRecording(Mat frame) {
        if (isRecording) {
            if (videoEncoder == null) {
                try {
                    File dir = getExternalFilesDir(Environment.DIRECTORY_MOVIES);
                    if (!dir.exists()) dir.mkdirs();
                    String fileName = "Hitchcock_" + new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date()) + ".mp4";
                    currentRecordingFile = new File(dir, fileName);
                    
                    videoEncoder = new VideoEncoder(frame.cols(), frame.rows(), currentRecordingFile);
                    Log.d(TAG, "Started recording to: " + currentRecordingFile.getAbsolutePath());
                } catch (IOException e) {
                    Log.e(TAG, "Failed to start recording", e);
                    isRecording = false;
                    runOnUiThread(() -> Toast.makeText(this, "Failed to start recording", Toast.LENGTH_SHORT).show());
                    return;
                }
            }
            
            // Convert to YUV for encoder
            Mat yuvMat = new Mat();
            // Input 'frame' is RGB (3 channels) now, not RGBA
            Imgproc.cvtColor(frame, yuvMat, Imgproc.COLOR_RGB2YUV_I420);
            
            byte[] yuvData = new byte[(int) (yuvMat.total() * yuvMat.channels())];
            yuvMat.get(0, 0, yuvData);
            
            if (videoEncoder.isSemiPlanar()) {
                // I420 to NV12
                byte[] nv12Data = new byte[yuvData.length];
                int width = frame.cols();
                int height = frame.rows();
                int ySize = width * height;
                int uSize = ySize / 4;
                
                // Copy Y
                System.arraycopy(yuvData, 0, nv12Data, 0, ySize);
                
                // Interleave U and V
                // I420: Y... U... V...
                // NV12: Y... UVUV...
                for (int i = 0; i < uSize; i++) {
                    nv12Data[ySize + 2 * i] = yuvData[ySize + i];       // U
                    nv12Data[ySize + 2 * i + 1] = yuvData[ySize + uSize + i]; // V
                }
                videoEncoder.encodeFrame(nv12Data);
            } else {
                // Planar (I420) - Direct copy
                videoEncoder.encodeFrame(yuvData);
            }
            
            yuvMat.release();
            
        } else {
            if (videoEncoder != null) {
                videoEncoder.release();
                videoEncoder = null;
                Log.d(TAG, "Recording stopped");
                String path = currentRecordingFile.getAbsolutePath();
                runOnUiThread(() -> Toast.makeText(this, "Saved to: " + path, Toast.LENGTH_LONG).show());
                
                // Export to Gallery for easier access
                exportToGallery(currentRecordingFile);
            }
        }
    }

    private void exportToGallery(File file) {
        if (!file.exists() || file.length() == 0) {
            Log.e(TAG, "Export failed: File does not exist or is empty");
            runOnUiThread(() -> Toast.makeText(this, "Save failed: Empty file", Toast.LENGTH_SHORT).show());
            return;
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            ContentValues values = new ContentValues();
            values.put(MediaStore.Video.Media.DISPLAY_NAME, file.getName());
            values.put(MediaStore.Video.Media.MIME_TYPE, "video/mp4");
            values.put(MediaStore.Video.Media.DATE_ADDED, System.currentTimeMillis() / 1000);
            values.put(MediaStore.Video.Media.DATE_TAKEN, System.currentTimeMillis());
            values.put(MediaStore.Video.Media.RELATIVE_PATH, Environment.DIRECTORY_MOVIES);

            ContentResolver resolver = getContentResolver();
            Uri uri = null;
            try {
                uri = resolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values);
            } catch (Exception e) {
                Log.e(TAG, "Failed to insert into MediaStore", e);
            }

            if (uri != null) {
                try (OutputStream out = resolver.openOutputStream(uri);
                     FileInputStream in = new FileInputStream(file)) {
                    
                    byte[] buffer = new byte[4096];
                    int len;
                    while ((len = in.read(buffer)) > 0) {
                        out.write(buffer, 0, len);
                    }
                    
                    runOnUiThread(() -> Toast.makeText(this, "Saved to Gallery!", Toast.LENGTH_SHORT).show());
                } catch (IOException e) {
                    Log.e(TAG, "Failed to export to gallery", e);
                    runOnUiThread(() -> Toast.makeText(this, "Export failed: " + e.getMessage(), Toast.LENGTH_SHORT).show());
                }
            } else {
                 Log.e(TAG, "Failed to create MediaStore entry");
                 runOnUiThread(() -> Toast.makeText(this, "Gallery Error: Could not create entry", Toast.LENGTH_SHORT).show());
            }
        } else {
            // For older Android versions, we just rely on the file being in getExternalFilesDir
            // Or we could use MediaScannerConnection
            runOnUiThread(() -> Toast.makeText(this, "Saved to: " + file.getAbsolutePath(), Toast.LENGTH_LONG).show());
        }
    }

    private byte[] yuv420ToNv21(ImageProxy image) {
        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        ImageProxy.PlaneProxy yPlane = planes[0];
        ImageProxy.PlaneProxy uPlane = planes[1];
        ImageProxy.PlaneProxy vPlane = planes[2];

        ByteBuffer yBuffer = yPlane.getBuffer();
        ByteBuffer uBuffer = uPlane.getBuffer();
        ByteBuffer vBuffer = vPlane.getBuffer();

        int width = image.getWidth();
        int height = image.getHeight();
        
        int ySize = width * height;
        int uvSize = width * height / 2;
        
        byte[] nv21 = new byte[ySize + uvSize];

        // Copy Y
        int yRowStride = yPlane.getRowStride();
        if (yRowStride == width) {
            yBuffer.get(nv21, 0, ySize);
        } else {
            for (int h = 0; h < height; h++) {
                yBuffer.position(h * yRowStride);
                yBuffer.get(nv21, h * width, width);
            }
        }

        // Copy UV (NV21: V then U)
        int uvRowStride = uPlane.getRowStride();
        int uvPixelStride = uPlane.getPixelStride();
        int uvWidth = width / 2;
        int uvHeight = height / 2;
        
        int offset = ySize;
        for (int h = 0; h < uvHeight; h++) {
            for (int w = 0; w < uvWidth; w++) {
                int bufferIndex = h * uvRowStride + w * uvPixelStride;
                // V
                nv21[offset++] = vBuffer.get(bufferIndex);
                // U
                nv21[offset++] = uBuffer.get(bufferIndex);
            }
        }
        
        return nv21;
    }

    private void requestPermissions() {
        activityResultLauncher.launch(REQUIRED_PERMISSIONS);
    }

    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(getBaseContext(), permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraExecutor.shutdown();
        processExecutor.shutdown();
        if (videoEncoder != null) {
            videoEncoder.release();
        }
    }

    private static final String[] REQUIRED_PERMISSIONS =
            new String[]{Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO};
}
