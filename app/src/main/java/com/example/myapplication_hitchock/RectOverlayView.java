package com.example.myapplication_hitchock;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

import androidx.annotation.Nullable;

public class RectOverlayView extends View {

    private RectF rect = new RectF();
    private Paint borderPaint;
    private Paint fillPaint;
    private float startX, startY;
    private boolean isDrawing = false;
    private OnRectSelectedListener listener;

    public interface OnRectSelectedListener {
        void onDrawingStarted();
        void onRectSelected(RectF rect);
    }

    public void setOnRectSelectedListener(OnRectSelectedListener listener) {
        this.listener = listener;
    }

    public RectOverlayView(Context context) {
        super(context);
        init();
    }

    public RectOverlayView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        borderPaint = new Paint();
        borderPaint.setColor(Color.GREEN);
        borderPaint.setStyle(Paint.Style.STROKE);
        borderPaint.setStrokeWidth(5f);

        fillPaint = new Paint();
        fillPaint.setColor(Color.argb(50, 0, 255, 0));
        fillPaint.setStyle(Paint.Style.FILL);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (isDrawing || !rect.isEmpty()) {
            canvas.drawRect(rect, fillPaint);
            canvas.drawRect(rect, borderPaint);
        }
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                startX = x;
                startY = y;
                rect.set(x, y, x, y);
                isDrawing = true;
                invalidate();
                return true;

            case MotionEvent.ACTION_MOVE:
                float left = Math.min(startX, x);
                float top = Math.min(startY, y);
                float right = Math.max(startX, x);
                float bottom = Math.max(startY, y);
                rect.set(left, top, right, bottom);
                invalidate();
                return true;

            case MotionEvent.ACTION_UP:
                isDrawing = false;
                if (listener != null && rect.width() > 10 && rect.height() > 10) {
                    listener.onRectSelected(new RectF(rect));
                }
                // Optional: clear rect after selection or keep it? 
                // Let's keep it for a moment or until reset.
                // For now, let's clear it after a delay or just keep it green.
                invalidate();
                return true;
        }
        return super.onTouchEvent(event);
    }
    
    public void clear() {
        rect.setEmpty();
        invalidate();
    }
}
