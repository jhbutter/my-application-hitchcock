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

    // Smoothing history
    private List<Double> scaleHistory = new ArrayList<>();
    private List<Double> cxHistory = new ArrayList<>();
    private List<Double> cyHistory = new ArrayList<>();
    private static final int WINDOW_SIZE = 15; // Increased to 15 for smoother output at 60fps

    // Warmup frames to stabilize tracker
    private static final int WARMUP_FRAMES = 15;

    private int width, height;

    public DollyZoomProcessor() {
    }
    
    public boolean isInitialized() {
        return isInitialized;
    }

    public void reset() {
        isInitialized = false;
        frameCount = 0;
        scaleHistory.clear();
        cxHistory.clear();
        cyHistory.clear();
        
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
    private int trackingInterval = 4;

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

        isInitialized = true;
        frameCount = 0;
        scaleHistory.clear();
        cxHistory.clear();
        cyHistory.clear();
        
        Log.d(TAG, "Init Diagonal: " + initDiagonal);
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

        // Process Noise Covariance (Q)
        Mat processNoiseCov = Mat.eye(6, 6, CvType.CV_32F);
        // Scale (Index 0): Allow some flexibility for zoom
        processNoiseCov.put(0, 0, 1e-4);
        // Position (Index 1, 2): Very stiff, assume center doesn't move much (Fixes jitter)
        processNoiseCov.put(1, 1, 1e-5);
        processNoiseCov.put(2, 2, 1e-5);
        // Velocities (Index 3-5):
        processNoiseCov.put(3, 3, 1e-3);
        processNoiseCov.put(4, 4, 1e-3);
        processNoiseCov.put(5, 5, 1e-3);
        kalman.set_processNoiseCov(processNoiseCov);

        // Measurement Noise Covariance (R)
        Mat measurementNoiseCov = Mat.eye(3, 3, CvType.CV_32F);
        // Trust measurement less to filter out tracker noise
        Core.multiply(measurementNoiseCov, new Scalar(100.0), measurementNoiseCov);
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

    // Callback interface
    public interface OnDebugInfoListener {
        void onDebugInfo(String info);
    }
    
    private OnDebugInfoListener debugListener;
    
    public void setOnDebugInfoListener(OnDebugInfoListener listener) {
        this.debugListener = listener;
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

        if (prevPoints.total() > 0) {
            MatOfPoint2f nextPoints = new MatOfPoint2f();
            MatOfByte status = new MatOfByte();
            MatOfFloat err = new MatOfFloat();

            // Calculate Optical Flow
            Video.calcOpticalFlowPyrLK(prevGray, currGray, prevPoints, nextPoints, status, err);

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
            
            // Draw points for debug
            for(Point p : goodNewPoints) {
                Imgproc.circle(frame, p, 3, new Scalar(0, 255, 0), -1);
            }

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

                // 3. Kalman Update
                double currCx = trackRect.x + trackRect.width / 2.0;
                double currCy = trackRect.y + trackRect.height / 2.0;
                double currentDiagonal = Math.sqrt(trackRect.width * trackRect.width + trackRect.height * trackRect.height);
    
                double rawScale = 1.0;
                if (currentDiagonal > 10.0) { 
                    rawScale = initDiagonal / currentDiagonal;
                    if (rawScale > 10.0) rawScale = 10.0;
                    if (rawScale < 0.5) rawScale = 0.5;
                }

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
                prevGray = currGray.clone(); // Still update gray
            }
            
            nextPoints.release();
            status.release();
            err.release();
        } else {
            prevGray.release();
            prevGray = currGray.clone();
        }
        
        currGray.release(); // We cloned it to prevGray if needed

        
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
            
        } else {
            if (smoothScale < 1.0) smoothScale = 1.0;
             frame.copyTo(result);
        }

        // Draw debug info
        String statusMode = trackerUpdated ? "TRACK" : "PREDICT";
        String debugInfo = String.format("Scale: %.2f [%s]", smoothScale, statusMode);
        
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
