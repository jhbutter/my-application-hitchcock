package com.example.myapplication_hitchock;

import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaCodecList;
import android.media.MediaFormat;
import android.media.MediaMuxer;
import android.util.Log;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;

public class VideoEncoder {
    private static final String TAG = "VideoEncoder";
    private static final String MIME_TYPE = "video/avc"; 
    private static final int FRAME_RATE = 30;
    private static final int IFRAME_INTERVAL = 5;

    private MediaCodec mEncoder;
    private MediaMuxer mMuxer;
    private int mTrackIndex;
    private boolean mMuxerStarted;
    private MediaCodec.BufferInfo mBufferInfo;
    private int mWidth;
    private int mHeight;
    private int mColorFormat;
    
    private long mStartTimeNs = 0;
    private long mLastFrameTimeNs = 0;

    public VideoEncoder(int width, int height, File outputFile) throws IOException {
        mWidth = width;
        mHeight = height;
        
        mBufferInfo = new MediaCodec.BufferInfo();
        
        MediaCodecInfo codecInfo = selectCodec(MIME_TYPE);
        if (codecInfo == null) {
            throw new IOException("Unable to find an appropriate codec for " + MIME_TYPE);
        }

        mColorFormat = selectColorFormat(codecInfo, MIME_TYPE);
        
        MediaFormat format = MediaFormat.createVideoFormat(MIME_TYPE, width, height);
        format.setInteger(MediaFormat.KEY_COLOR_FORMAT, mColorFormat);
        format.setInteger(MediaFormat.KEY_BIT_RATE, 4000000);
        format.setInteger(MediaFormat.KEY_FRAME_RATE, FRAME_RATE);
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, IFRAME_INTERVAL);

        mEncoder = MediaCodec.createEncoderByType(MIME_TYPE);
        mEncoder.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
        mEncoder.start();

        mMuxer = new MediaMuxer(outputFile.getAbsolutePath(), MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4);
        mTrackIndex = -1;
        mMuxerStarted = false;
        
        mStartTimeNs = System.nanoTime();
    }

    public int getColorFormat() {
        return mColorFormat;
    }

    public boolean isSemiPlanar() {
        return mColorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar;
    }
    
    public boolean isPlanar() {
        return mColorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Planar;
    }

    public void encodeFrame(byte[] yuvData) {
        drainEncoder(false);

        try {
            int inputBufferIndex = mEncoder.dequeueInputBuffer(10000); // 10ms timeout
            if (inputBufferIndex >= 0) {
                ByteBuffer inputBuffer = mEncoder.getInputBuffer(inputBufferIndex);
                inputBuffer.clear();
                inputBuffer.put(yuvData);
                
                long currentNs = System.nanoTime();
                long ptsUsec = (currentNs - mStartTimeNs) / 1000;
                
                // Ensure monotonic timestamps
                if (ptsUsec <= mLastFrameTimeNs) {
                    ptsUsec = mLastFrameTimeNs + 1000; // add 1ms
                }
                mLastFrameTimeNs = ptsUsec;

                mEncoder.queueInputBuffer(inputBufferIndex, 0, yuvData.length, ptsUsec, 0);
            } else {
                Log.w(TAG, "Input buffer not available");
            }
        } catch (Exception e) {
            Log.e(TAG, "Error encoding frame", e);
        }
    }

    public void drainEncoder(boolean endOfStream) {
        final int TIMEOUT_USEC = 10000;
        if (endOfStream) {
            try {
                mEncoder.signalEndOfInputStream();
            } catch (Exception e) {
                Log.w(TAG, "Error signalling end of stream", e);
            }
        }

        while (true) {
            int encoderStatus = mEncoder.dequeueOutputBuffer(mBufferInfo, TIMEOUT_USEC);
            if (encoderStatus == MediaCodec.INFO_TRY_AGAIN_LATER) {
                if (!endOfStream) break;
                else {
                    try { Thread.sleep(10); } catch (InterruptedException e) { break; }
                    // Break loop if taking too long to avoid ANR? 
                    // For now, just rely on standard behavior.
                }
            } else if (encoderStatus == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                if (mMuxerStarted) throw new RuntimeException("format changed twice");
                MediaFormat newFormat = mEncoder.getOutputFormat();
                mTrackIndex = mMuxer.addTrack(newFormat);
                mMuxer.start();
                mMuxerStarted = true;
            } else if (encoderStatus < 0) {
                // ignore
            } else {
                ByteBuffer encodedData = mEncoder.getOutputBuffer(encoderStatus);
                if (encodedData == null) throw new RuntimeException("encoderOutputBuffer " + encoderStatus + " was null");

                if ((mBufferInfo.flags & MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0) {
                    mBufferInfo.size = 0;
                }

                if (mBufferInfo.size != 0) {
                    if (!mMuxerStarted) throw new RuntimeException("muxer hasn't started");
                    encodedData.position(mBufferInfo.offset);
                    encodedData.limit(mBufferInfo.offset + mBufferInfo.size);
                    mMuxer.writeSampleData(mTrackIndex, encodedData, mBufferInfo);
                }

                mEncoder.releaseOutputBuffer(encoderStatus, false);

                if ((mBufferInfo.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                    break;
                }
            }
        }
    }

    public void release() {
        if (mEncoder != null) {
            try {
                drainEncoder(true);
            } catch (Exception e) {
                Log.e(TAG, "Error draining encoder", e);
            }
            try {
                mEncoder.stop();
                mEncoder.release();
            } catch (Exception e) {
                Log.e(TAG, "Error stopping encoder", e);
            }
            mEncoder = null;
        }
        if (mMuxer != null) {
            try {
                if (mMuxerStarted) {
                    mMuxer.stop();
                }
                mMuxer.release();
            } catch (Exception e) {
                 Log.e(TAG, "Error stopping muxer", e);
            }
            mMuxer = null;
        }
    }
    
    private static MediaCodecInfo selectCodec(String mimeType) {
        int numCodecs = MediaCodecList.getCodecCount();
        for (int i = 0; i < numCodecs; i++) {
            MediaCodecInfo codecInfo = MediaCodecList.getCodecInfoAt(i);
            if (!codecInfo.isEncoder()) continue;
            String[] types = codecInfo.getSupportedTypes();
            for (String type : types) {
                if (type.equalsIgnoreCase(mimeType)) return codecInfo;
            }
        }
        return null;
    }

    private static int selectColorFormat(MediaCodecInfo codecInfo, String mimeType) {
        MediaCodecInfo.CodecCapabilities capabilities = codecInfo.getCapabilitiesForType(mimeType);
        for (int i = 0; i < capabilities.colorFormats.length; i++) {
            int colorFormat = capabilities.colorFormats[i];
            if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar) { // NV12
                return colorFormat;
            }
            if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Planar) { // I420
                return colorFormat;
            }
        }
        return 0; // fail
    }
}
