package com.example.myapplication_hitchock;

import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Tracker;
import org.opencv.tracking.TrackerCSRT;
import org.opencv.tracking.TrackerKCF;
import org.opencv.video.KalmanFilter;

import java.util.ArrayList;
import java.util.List;

public class DollyZoomProcessor {
    private static final String TAG = "DollyZoomProcessor";

    private Tracker tracker;
    private KalmanFilter kalman;
    private boolean isInitialized = false;
    private double initDiagonal;
    private double initCx, initCy;
    private int frameCount = 0;

    // Smoothing history
    private List<Double> scaleHistory = new ArrayList<>();
    private List<Double> cxHistory = new ArrayList<>();
    private List<Double> cyHistory = new ArrayList<>();
    private static final int WINDOW_SIZE = 5; // Reduced from 15 to 5 for faster response
    
    // Warmup frames to stabilize tracker
    private static final int WARMUP_FRAMES = 15;

    private int width, height;

    public DollyZoomProcessor() {
    }
    
    public void reset() {
        isInitialized = false;
        frameCount = 0;
        scaleHistory.clear();
        cxHistory.clear();
        cyHistory.clear();
    }

    // Downscale factor for tracker (1.0 = no downscale, 0.5 = 1/2 size)
    private static final double TRACKER_SCALE = 0.5; // Try 0.5 to improve CSRT performance

    public void init(Mat frame, android.graphics.RectF initialRect) {
        width = frame.cols();
        height = frame.rows();

        // Validate Initial Rect
        if (initialRect.width() < 10 || initialRect.height() < 10) {
            Log.e(TAG, "Initial Rect too small: " + initialRect);
            isInitialized = false;
            return;
        }

        // Initialize Tracker
        try {
            tracker = TrackerCSRT.create();
            Log.d(TAG, "Initialized CSRT Tracker");
        } catch (Exception e) {
            Log.e(TAG, "CSRT Tracker failed", e);
            tracker = TrackerKCF.create(); // Fallback
        }
        
        // Resize for tracker init
        Mat smallFrame = new Mat();
        Imgproc.resize(frame, smallFrame, new Size(), TRACKER_SCALE, TRACKER_SCALE, Imgproc.INTER_LINEAR);
        
        Rect roi = new Rect(
            (int)(initialRect.left * TRACKER_SCALE), 
            (int)(initialRect.top * TRACKER_SCALE), 
            (int)(initialRect.width() * TRACKER_SCALE), 
            (int)(initialRect.height() * TRACKER_SCALE)
        );
        
        // Ensure roi is within bounds
        roi.x = Math.max(0, roi.x);
        roi.y = Math.max(0, roi.y);
        roi.width = Math.min(smallFrame.cols() - roi.x, roi.width);
        roi.height = Math.min(smallFrame.rows() - roi.y, roi.height);
        
        tracker.init(smallFrame, roi);
        smallFrame.release();

        // Initialize Kalman Filter
        Rect2d roiDouble = new Rect2d(initialRect.left, initialRect.top, initialRect.width(), initialRect.height());
        initKalman(roiDouble);

        isInitialized = true;
        frameCount = 0;
        scaleHistory.clear();
        cxHistory.clear();
        cyHistory.clear();
        
        Log.d(TAG, "Init Diagonal: " + initDiagonal);
    }
    
    // ... initKalman ...
    private void initKalman(Rect2d rect) {
        kalman = new KalmanFilter(6, 3, 0, CvType.CV_32F);

        // Transition Matrix (A)
        // [1, 0, 0, 1, 0, 0]
        // [0, 1, 0, 0, 1, 0]
        // [0, 0, 1, 0, 0, 1] ...
        Mat transitionMatrix = new Mat(6, 6, CvType.CV_32F, new Scalar(0));
        transitionMatrix.put(0, 0, 1); transitionMatrix.put(0, 3, 1);
        transitionMatrix.put(1, 1, 1); transitionMatrix.put(1, 4, 1);
        transitionMatrix.put(2, 2, 1); transitionMatrix.put(2, 5, 1);
        transitionMatrix.put(3, 3, 1);
        transitionMatrix.put(4, 4, 1);
        transitionMatrix.put(5, 5, 1);
        kalman.set_transitionMatrix(transitionMatrix);

        // Measurement Matrix (H)
        Mat measurementMatrix = new Mat(3, 6, CvType.CV_32F, new Scalar(0));
        measurementMatrix.put(0, 0, 1);
        measurementMatrix.put(1, 1, 1);
        measurementMatrix.put(2, 2, 1);
        kalman.set_measurementMatrix(measurementMatrix);

        // Process Noise Covariance (Q) - Trust process/prediction more if low
        // Increased to 1e-1 to allow faster reaction to rapid zoom changes
        Mat processNoiseCov = Mat.eye(6, 6, CvType.CV_32F);
        Core.multiply(processNoiseCov, new Scalar(1e-1), processNoiseCov);
        kalman.set_processNoiseCov(processNoiseCov);

        // Measurement Noise Covariance (R) - Trust measurement more if low
        Mat measurementNoiseCov = Mat.eye(3, 3, CvType.CV_32F);
        Core.multiply(measurementNoiseCov, new Scalar(1e-3), measurementNoiseCov);
        kalman.set_measurementNoiseCov(measurementNoiseCov);

        // Initial State
        initCx = rect.x + rect.width / 2.0;
        initCy = rect.y + rect.height / 2.0;
        initDiagonal = Math.sqrt(rect.width * rect.width + rect.height * rect.height);

        Mat statePre = new Mat(6, 1, CvType.CV_32F, new Scalar(0));
        statePre.put(0, 0, 1.0); // Initial scale
        statePre.put(1, 0, initCx);
        statePre.put(2, 0, initCy);
        kalman.set_statePre(statePre);
        kalman.set_statePost(statePre);
    }

    public Mat process(Mat frame) {
        if (!isInitialized) {
            return frame;
        }

        // 1. Update Tracker on Downscaled Frame
        Mat smallFrame = new Mat();
        Imgproc.resize(frame, smallFrame, new Size(), TRACKER_SCALE, TRACKER_SCALE, Imgproc.INTER_LINEAR);
        
        Rect smallTrackRect = new Rect();
        boolean success = tracker.update(smallFrame, smallTrackRect);
        smallFrame.release();
        
        // Scale rect back up
        Rect trackRect = new Rect(
            (int)(smallTrackRect.x / TRACKER_SCALE),
            (int)(smallTrackRect.y / TRACKER_SCALE),
            (int)(smallTrackRect.width / TRACKER_SCALE),
            (int)(smallTrackRect.height / TRACKER_SCALE)
        );

        // 2. Kalman Predict
        Mat prediction = kalman.predict();
        double predScale = prediction.get(0, 0)[0];
        double predCx = prediction.get(1, 0)[0];
        double predCy = prediction.get(2, 0)[0];

        double kScale, kCx, kCy;

        // WARMUP LOGIC: Skip Kalman Update entirely
        if (frameCount < WARMUP_FRAMES) {
             kScale = 1.0;
             kCx = initCx;
             kCy = initCy;
             
             // Reset Kalman State to prevent drift
             Mat statePost = kalman.get_statePost();
             statePost.put(0, 0, 1.0);
             statePost.put(1, 0, initCx);
             statePost.put(2, 0, initCy);
             kalman.set_statePost(statePost);
        } else {
            if (success) {
                double currCx = trackRect.x + trackRect.width / 2.0;
                double currCy = trackRect.y + trackRect.height / 2.0;
                double currentDiagonal = Math.sqrt(trackRect.width * trackRect.width + trackRect.height * trackRect.height);
    
                double rawScale = 1.0;
                if (currentDiagonal > 10.0) { // Avoid div by small number
                    rawScale = initDiagonal / currentDiagonal;
                    // Clamp raw scale strictly
                    if (rawScale > 10.0) rawScale = 10.0; // Relaxed Clamp for stronger effect
                    if (rawScale < 0.5) rawScale = 0.5;
                }
    
                // Kalman Correct
                Mat measurement = new Mat(3, 1, CvType.CV_32F);
                measurement.put(0, 0, rawScale);
                measurement.put(1, 0, currCx);
                measurement.put(2, 0, currCy);
                
                Mat estimated = kalman.correct(measurement);
                kScale = estimated.get(0, 0)[0];
                kCx = estimated.get(1, 0)[0];
                kCy = estimated.get(2, 0)[0];
            } else {
                // Lost tracking, rely on prediction
                kScale = predScale;
                kCx = predCx;
                kCy = predCy;
            }
        }
        
        // 3. Sliding Window Smoothing
        if (kScale > 10.0) kScale = 10.0; // Final safety clamp
        
        scaleHistory.add(kScale);
        cxHistory.add(kCx);
        cyHistory.add(kCy);

        if (scaleHistory.size() > WINDOW_SIZE) {
            scaleHistory.remove(0);
            cxHistory.remove(0);
            cyHistory.remove(0);
        }

        double smoothScale = calculateMean(scaleHistory);
        double smoothCx = calculateMean(cxHistory);
        double smoothCy = calculateMean(cyHistory);

        // WARMUP LOGIC: Force return original frame
        if (frameCount < WARMUP_FRAMES) {
            frameCount++;
            // Draw debug info
            Imgproc.putText(frame, "WARMUP...", new org.opencv.core.Point(50, 50), 
                Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 0), 2);
            return frame;
        }
        
        // 4. Optimized Zoom (Crop & Resize) instead of WarpAffine
        // We want the object (at smoothCx, smoothCy) to be moved to (initCx, initCy)
        // And scaled by smoothScale.
        // NewX = Scale * (OldX - smoothCx) + initCx? No.
        // Affine Transform:
        // [ scale, 0,     Tx ]
        // [ 0,     scale, Ty ]
        // where Tx = initCx - scale * smoothCx
        //       Ty = initCy - scale * smoothCy
        
        // This is equivalent to:
        // Crop a ROI centered at (smoothCx, smoothCy) with size (Width/scale, Height/scale)
        // Then resize to (Width, Height).
        // Wait, if we want to shift the center to initCx, we need to shift the crop center?
        
        // Let's verify:
        // Center of cropped region in original image: (Cx_crop, Cy_crop)
        // After resize (scale S), this center moves to (Width/2, Height/2) relative to the new image?
        // No, resize maps the whole ROI to the whole Image.
        // So the point (Cx_crop, Cy_crop) becomes the center of the output image.
        // We want the object (smoothCx, smoothCy) to end up at (initCx, initCy).
        
        // If initCx is the screen center (W/2, H/2), then yes, we just crop around smoothCx.
        // If initCx is NOT center, "Crop & Resize" forces it to center.
        // HITCHCOCK ZOOM usually keeps the subject centered.
        // So forcing it to center (W/2, H/2) is actually better/stabilized!
        
        // So: Crop ROI centered at (smoothCx, smoothCy) with size (W/scale, H/scale).
        // Resize to (W, H).
        
        Mat result = new Mat();
        
        if (smoothScale > 1.0) {
            // Zoom In (Crop)
            double cropW = width / smoothScale;
            double cropH = height / smoothScale;
            
            int left = (int)(smoothCx - cropW / 2.0);
            int top = (int)(smoothCy - cropH / 2.0);
            
            // Boundary checks
            if (left < 0) left = 0;
            if (top < 0) top = 0;
            if (left + cropW > width) left = width - (int)cropW;
            if (top + cropH > height) top = height - (int)cropH;
            
            Rect cropRect = new Rect(left, top, (int)cropW, (int)cropH);
            
            // Fast Submat + Resize
            Mat cropMat = new Mat(frame, cropRect);
            Imgproc.resize(cropMat, result, new Size(width, height), 0, 0, Imgproc.INTER_LINEAR);
            // cropMat is just a header, but we should release if created manually? Submat doesn't need release but good practice.
            
        } else {
            // Zoom Out (Pad) - Rare for Hitchcock, but possible
            // We can just return original frame or do Affine if strictly needed.
            // For performance, let's just clamp scale to >= 1.0 or just do Affine for this rare case.
            // Let's use Affine for scale < 1.0 as it's cleaner to handle black borders.
            // But to keep it simple and fast, let's just clamp scale to 1.0 (no zoom out beyond original).
            
            if (smoothScale < 1.0) smoothScale = 1.0;
            
            // If scale is essentially 1.0, just return copy
             frame.copyTo(result);
        }

        // Draw debug info
        Imgproc.putText(result, String.format("Scale: %.2f", smoothScale), new org.opencv.core.Point(50, 50), 
            Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 255, 0), 2);
        
        frameCount++;
        return result;
    }

    private double calculateMean(List<Double> list) {
        if (list == null || list.isEmpty()) return 0;
        double sum = 0;
        for (Double d : list) sum += d;
        return sum / list.size();
    }
}
