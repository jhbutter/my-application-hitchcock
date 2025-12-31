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
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.video.Video;
import org.opencv.video.KalmanFilter;

import java.util.ArrayList;
import java.util.List;

public class DollyZoomProcessor {
    private static final String TAG = "DollyZoomProcessor";

    private static final int WARMUP_FRAMES = 15;
    private int width;
    private int height;

    private KalmanFilter kalman;
    private boolean isInitialized = false;
    private double initDiagonal;
    private double initCx, initCy;
    private int frameCount = 0;

    // Optical Flow Tracking Variables
    private Mat prevGray = new Mat();
    private MatOfPoint2f prevPoints = new MatOfPoint2f();
    private Rect trackRect = new Rect();
    private static final int MAX_CORNERS = 50;
    private static final double QUALITY_LEVEL = 0.01;
    private static final double MIN_DISTANCE = 5;

    // Smoothing
    private double smoothedScale = 1.0;
    private double smoothedCx = 0;
    private double smoothedCy = 0;
    
    // Smoothing Factors (Adjustable)
    // Default: 0.05 (Slow/Smooth)
    private double smoothAlphaScale = 0.05; 
    private double smoothAlphaPos = 0.03;   

    /**
     * Set smoothing factor (0.01 to 1.0)
     * Lower value = More Smooth (Slower response)
     * Higher value = Less Smooth (Faster response)
     * @param alpha Main smoothing alpha (applied to scale)
     */
    public void setSmoothingFactor(double alpha) {
        // Clamp to reasonable range [0.005, 1.0]
        if (alpha < 0.005) alpha = 0.005;
        if (alpha > 1.0) alpha = 1.0;
        
        this.smoothAlphaScale = alpha;
        // Position Smoothing Logic for "Lock" Stabilization:
        // To achieve "Visual Stabilization" (keeping the subject fixed in center),
        // the position tracking needs to be much more responsive (higher alpha) than the scale smoothing.
        // If position alpha is too low, the subject will "drift" or "float" when the camera moves.
        // We ensure a minimum responsiveness (0.5) to keep the lock tight.
        this.smoothAlphaPos = Math.max(alpha * 5.0, 0.5);
        
        if (this.smoothAlphaPos > 1.0) this.smoothAlphaPos = 1.0;
        
        Log.d(TAG, String.format("Smoothing set to: Scale=%.4f, Pos=%.4f", smoothAlphaScale, smoothAlphaPos));
    }
    
    // Average Radius Logic
    private double initAvgRadius = 0;

    public DollyZoomProcessor() {
    }
    
    public boolean isInitialized() {
        return isInitialized;
    }

    public void reset() {
        isInitialized = false;
        frameCount = 0;
        smoothedScale = 1.0;
        smoothedCx = 0;
        smoothedCy = 0;
        
        // Release Mats
        if (prevGray != null) prevGray.release();
        if (prevPoints != null) prevPoints.release();
        prevGray = new Mat();
        prevPoints = new MatOfPoint2f();
    }

    // Downscale factor for tracker (1.0 = no downscale, 0.5 = 1/2 size)
    // 0.5 (320x240) to reduce quantization error (jitter) compared to 0.25
    // private static final double TRACKER_SCALE = 0.5; // Unused for Optical Flow in current implementation
    
    // Tracking Interval (Frames to skip between tracker updates)
    // 1 = track every frame (no skip)
    // 4 = track 1 frame, predict 3 frames (Compensate for higher resolution)
    private int trackingInterval = 1;

    public void setTrackingInterval(int interval) {
        if (interval < 1) interval = 1;
        if (interval > 30) interval = 30; // Limit to reasonable max
        this.trackingInterval = interval;
        Log.d(TAG, "Tracking Interval set to: " + interval);
    }

    public void init(Mat frame, android.graphics.RectF initialRect) {
        width = frame.cols();
        height = frame.rows();

        // Validate Initial Rect
        if (initialRect.width() < 10 || initialRect.height() < 10) {
            Log.e(TAG, "Initial Rect too small: " + initialRect);
            isInitialized = false;
            return;
        }

        // 1. Initialize Optical Flow Tracking
        // Convert to Grayscale
        Imgproc.cvtColor(frame, prevGray, Imgproc.COLOR_RGB2GRAY);

        // Define ROI mask
        Mat mask = new Mat(prevGray.size(), CvType.CV_8UC1, new Scalar(0));
        Rect roi = new Rect(
            (int)initialRect.left, 
            (int)initialRect.top, 
            (int)initialRect.width(), 
            (int)initialRect.height()
        );
        // Ensure roi is within bounds
        roi.x = Math.max(0, roi.x);
        roi.y = Math.max(0, roi.y);
        roi.width = Math.min(width - roi.x, roi.width);
        roi.height = Math.min(height - roi.y, roi.height);
        
        // Draw ROI on mask (white on black)
        Imgproc.rectangle(mask, roi, new Scalar(255), -1);

        // Detect Features in ROI
        MatOfPoint corners = new MatOfPoint();
        Imgproc.goodFeaturesToTrack(prevGray, corners, MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE, mask);
        mask.release();

        if (corners.total() == 0) {
            Log.e(TAG, "No features found in ROI!");
            isInitialized = false;
            return;
        }

        // Convert to MatOfPoint2f
        prevPoints.fromArray(corners.toArray());
        corners.release();

        // Store initial rect
        trackRect = roi;
        Log.d(TAG, "Initialized Optical Flow with " + prevPoints.total() + " points.");

        // Initialize Kalman Filter
        Rect2d roiDouble = new Rect2d(initialRect.left, initialRect.top, initialRect.width(), initialRect.height());
        initKalman(roiDouble);

        // Initialize Smoothing Variables
        smoothedCx = initCx;
        smoothedCy = initCy;
        smoothedScale = 1.0;
        
        // Calculate Initial Radius (Centroid based)
        Point[] points = corners.toArray(); // corners is released above but we can use prevPoints
        Point[] pts = prevPoints.toArray();
        double sumDist = 0;
        for(Point p : pts) {
            double dx = p.x - initCx;
            double dy = p.y - initCy;
            sumDist += Math.sqrt(dx*dx + dy*dy);
        }
        initAvgRadius = (pts.length > 0) ? sumDist / pts.length : 1.0;
        if (initAvgRadius < 1.0) initAvgRadius = 1.0;

        isInitialized = true;
        frameCount = 0;
        
        Log.d(TAG, "Init Radius: " + initAvgRadius);
    }
    
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

        // Process Noise Covariance (Q) - RESPONSIVE STABILIZATION
        Mat processNoiseCov = Mat.eye(6, 6, CvType.CV_32F);
        // Scale: Low noise, smooth zoom
        processNoiseCov.put(0, 0, 1e-4);
        // Position: High noise allowed (We expect the subject to move in the frame)
        // Was 1e-6 (Stationary assumption), now 1e-2 to allow tracking fast motion
        processNoiseCov.put(1, 1, 1e-2);
        processNoiseCov.put(2, 2, 1e-2);
        // Velocities:
        processNoiseCov.put(3, 3, 1e-2);
        processNoiseCov.put(4, 4, 1e-2);
        processNoiseCov.put(5, 5, 1e-2);
        kalman.set_processNoiseCov(processNoiseCov);

        // Measurement Noise Covariance (R) - TRUST MEASUREMENT
        Mat measurementNoiseCov = Mat.eye(3, 3, CvType.CV_32F);
        // Reduce noise covariance to trust the Optical Flow result more.
        // Was 10.0 -> Now 1.0 (Standard trust)
        Core.multiply(measurementNoiseCov, new Scalar(1.0), measurementNoiseCov);
        kalman.set_measurementNoiseCov(measurementNoiseCov);

        // Initial State
        initCx = rect.x + rect.width / 2.0;
        initCy = rect.y + rect.height / 2.0;
        initDiagonal = Math.sqrt(rect.width * rect.width + rect.height * rect.height); // Still used for fallback if needed

        Mat statePre = new Mat(6, 1, CvType.CV_32F, new Scalar(0));
        statePre.put(0, 0, 1.0); // Initial scale
        statePre.put(1, 0, initCx);
        statePre.put(2, 0, initCy);
        kalman.set_statePre(statePre);
        kalman.set_statePost(statePre);
    }

    // Callback interface
    public interface OnDebugInfoListener {
        void onDebugInfo(String info);
    }
    
    public interface OnTrackedPointsListener {
        void onTrackedPoints(List<Point> points, Rect trackRect, int frameWidth, int frameHeight);
    }
    
    private OnDebugInfoListener debugListener;
    private OnTrackedPointsListener pointsListener;
    
    public void setOnDebugInfoListener(OnDebugInfoListener listener) {
        this.debugListener = listener;
    }
    
    public void setOnTrackedPointsListener(OnTrackedPointsListener listener) {
        this.pointsListener = listener;
    }

    // Helper for Median Calculation
    private double calculateMedian(List<Double> list) {
        if (list == null || list.isEmpty()) return 0;
        java.util.Collections.sort(list);
        int n = list.size();
        if (n % 2 == 0) {
            return (list.get(n/2 - 1) + list.get(n/2)) / 2.0;
        } else {
            return list.get(n/2);
        }
    }

    public Mat process(Mat frame) {
        if (!isInitialized) {
            return frame;
        }

        // 1. Kalman Predict (Always Step 1)
        Mat prediction = kalman.predict();
        double predScale = prediction.get(0, 0)[0];
        double predCx = prediction.get(1, 0)[0];
        double predCy = prediction.get(2, 0)[0];

        // Default to prediction
        double kScale = predScale;
        double kCx = predCx;
        double kCy = predCy;
        
        boolean trackerUpdated = false;

        // 2. Optical Flow Tracking (Every frame if possible, or skip logic)
        // Note: For OF, it's better to track every frame to handle motion.
        // We will respect trackingInterval only for Kalman updates if needed, 
        // but skipping frames for OF is risky. 
        // Strategy: Run OF every frame. It's fast.
        
        Mat currGray = new Mat();
        Imgproc.cvtColor(frame, currGray, Imgproc.COLOR_RGB2GRAY);

        // CHECK TRACKING INTERVAL
        // Only run heavy Optical Flow if within interval
        if (frameCount % trackingInterval == 0) {

            if (prevPoints.total() > 0) {
                MatOfPoint2f nextPoints = new MatOfPoint2f();
                MatOfByte status = new MatOfByte();
                MatOfFloat err = new MatOfFloat();

                // Calculate Optical Flow with larger window size for better tracking during fast motion
                // winSize: 31x31 (default 21x21), maxLevel: 3
                Video.calcOpticalFlowPyrLK(prevGray, currGray, prevPoints, nextPoints, status, err, new Size(31, 31), 3);

                // Filter valid points
                List<Point> prevList = prevPoints.toList();
                List<Point> nextList = nextPoints.toList();
                List<Byte> statusList = status.toList();
                List<Point> goodNewPoints = new ArrayList<>();

                for (int i = 0; i < statusList.size(); i++) {
                    if (statusList.get(i) == 1) {
                        goodNewPoints.add(nextList.get(i));
                    }
                }
                
                // Draw points removed - now sending to listener
                /*
                for(Point p : goodNewPoints) {
                    Imgproc.circle(frame, p, 3, new Scalar(0, 255, 0), -1);
                }
                */

                if (goodNewPoints.size() > 0) {
                    trackerUpdated = true;
                    
                    // Update prevPoints and prevGray
                    prevPoints.fromList(goodNewPoints);
                    prevGray.release();
                    prevGray = currGray.clone(); // Keep current as prev for next iter
                    
                    // Calculate Bounding Box
                    // We use bounding rect of points
                    MatOfPoint pointsMat = new MatOfPoint();
                    pointsMat.fromList(goodNewPoints);
                    trackRect = Imgproc.boundingRect(pointsMat);
                    pointsMat.release();
                    
                    // Notify Listener
                    /*
                    if (pointsListener != null) {
                        pointsListener.onTrackedPoints(goodNewPoints, trackRect, width, height);
                    }
                    */

                    // 3. Kalman Update
                    
                    // Robust Measurement Calculation:
                    // 1. Position: Use Median of tracked points (Robust to outliers)
                    List<Double> xCoords = new ArrayList<>();
                    List<Double> yCoords = new ArrayList<>();
                    for (Point p : goodNewPoints) {
                        xCoords.add(p.x);
                        yCoords.add(p.y);
                    }
                    double currCx = calculateMedian(xCoords);
                    double currCy = calculateMedian(yCoords);
                    
                    // 2. Scale: Use Weighted Average of Bounding Box and Average Radius
                    // Bounding Box is stable but can be insensitive if only internal points move.
                    // Average Radius is sensitive but noisy.
                    // Mix: 70% Bounding Box (Stability) + 30% Average Radius (Responsiveness)
                    
                    // A. Bounding Box Scale
                    double currentDiagonal = Math.sqrt(trackRect.width * trackRect.width + trackRect.height * trackRect.height);
                    double scaleBox = 1.0;
                    if (currentDiagonal > 10.0) {
                        scaleBox = initDiagonal / currentDiagonal;
                    }
                    
                    // B. Average Radius Scale
                    double currentSumDist = 0;
                    for (Point p : goodNewPoints) {
                        double dx = p.x - currCx;
                        double dy = p.y - currCy;
                        currentSumDist += Math.sqrt(dx*dx + dy*dy);
                    }
                    double currentAvgRadius = (goodNewPoints.size() > 0) ? currentSumDist / goodNewPoints.size() : 1.0;
                    if (currentAvgRadius < 1.0) currentAvgRadius = 1.0;
                    
                    double scaleRadius = 1.0;
                    if (initAvgRadius > 1.0) {
                        scaleRadius = initAvgRadius / currentAvgRadius;
                    }
                    
                    // C. Mix
                    double rawScale = 0.7 * scaleBox + 0.3 * scaleRadius;
                    
                    // Sanity Check
                    if (rawScale > 10.0 || rawScale < 0.1) {
                         // Limit
                         if (rawScale > 10.0) rawScale = 10.0;
                         if (rawScale < 0.1) rawScale = 0.1;
                    }
                    
                    // Clamping
                    if (rawScale > 10.0) rawScale = 10.0;
                    if (rawScale < 0.5) rawScale = 0.5;

                    // Correct Kalman
                    Mat measurement = new Mat(3, 1, CvType.CV_32F);
                    measurement.put(0, 0, rawScale);
                    measurement.put(1, 0, currCx);
                    measurement.put(2, 0, currCy);
                    
                    Mat estimated = kalman.correct(measurement);
                    kScale = estimated.get(0, 0)[0];
                    kCx = estimated.get(1, 0)[0];
                    kCy = estimated.get(2, 0)[0];

                } else {
                    Log.w(TAG, "All points lost!");
                    // Keep prediction
                    prevGray.release();
                    prevGray = currGray.clone(); // Still update gray to attempt recovery next frame
                }
                
                nextPoints.release();
                status.release();
                err.release();
            } else {
                // No points, just update gray
                prevGray.release();
                prevGray = currGray.clone();
            }
        } else {
            // SKIPPED FRAME (PREDICT ONLY)
            // Do NOT update prevGray, so next tracked frame compares against the old valid frame.
            // trackerUpdated remains false
        }
        
        currGray.release(); // We cloned it to prevGray if needed

        
        // 3. Low Pass Filter (Exponential Smoothing) instead of Sliding Window
        if (kScale > 10.0) kScale = 10.0; // Final safety clamp
        
        // Apply heavy smoothing for Scale
        smoothedScale = smoothedScale + smoothAlphaScale * (kScale - smoothedScale);

        // Adaptive Smoothing for Position
        // If the subject moves quickly (large difference between current and smoothed),
        // we must increase responsiveness to prevent "drift".
        double diffX = kCx - smoothedCx;
        double diffY = kCy - smoothedCy;
        double distSq = diffX*diffX + diffY*diffY;
        
        double dynamicAlphaPos = smoothAlphaPos;
        // Threshold: if distance > 10 pixels (100 squared), boost alpha
        if (distSq > 100.0) {
            // Boost alpha based on distance. Max out at 1.0
            // Example: dist=30px -> sq=900 -> boost=0.9
            double boost = Math.min(1.0, distSq / 1000.0); 
            dynamicAlphaPos = Math.max(smoothAlphaPos, boost);
            if (dynamicAlphaPos > 1.0) dynamicAlphaPos = 1.0;
        }

        smoothedCx = smoothedCx + dynamicAlphaPos * diffX;
        smoothedCy = smoothedCy + dynamicAlphaPos * diffY;

        // WARMUP LOGIC: Force return original frame
        if (frameCount < WARMUP_FRAMES) {
            frameCount++;
            // Draw debug info (Disabled)
            /*
            Imgproc.putText(frame, "WARMUP...", new org.opencv.core.Point(50, 50), 
                Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 0), 2);
            */
            return frame;
        }
        
        // 4. Optimized Zoom (Crop & Resize) instead of WarpAffine
        Mat result = new Mat();
        
        if (smoothedScale > 1.0) {
            // Zoom In (Crop)
            double cropW = width / smoothedScale;
            double cropH = height / smoothedScale;
            
            int left = (int)(smoothedCx - cropW / 2.0);
            int top = (int)(smoothedCy - cropH / 2.0);
            
            // Boundary checks
            if (left < 0) left = 0;
            if (top < 0) top = 0;
            if (left + cropW > width) left = width - (int)cropW;
            if (top + cropH > height) top = height - (int)cropH;
            
            Rect cropRect = new Rect(left, top, (int)cropW, (int)cropH);
            
            // Fast Submat + Resize
            Mat cropMat = new Mat(frame, cropRect);
            Imgproc.resize(cropMat, result, new Size(width, height), 0, 0, Imgproc.INTER_LINEAR);
            
        } else {
            if (smoothedScale < 1.0) smoothedScale = 1.0;
             frame.copyTo(result);
        }

        // Draw debug info
        String statusMode = trackerUpdated ? "TRACK" : "PREDICT";
        String debugInfo = String.format("Scale: %.2f [%s]", smoothedScale, statusMode);
        
        // Notify listener
        if (debugListener != null) {
            debugListener.onDebugInfo(debugInfo);
        }

        // Removed Imgproc.putText to avoid cropping issues and resolution dependency
        /*
        Imgproc.putText(result, debugInfo, new org.opencv.core.Point(50, 50), 
            Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, trackerUpdated ? new Scalar(0, 255, 0) : new Scalar(0, 255, 255), 2);
        */
        
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
