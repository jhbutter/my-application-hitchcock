# Hitchcock Effect App

An Android application that implements the **Dolly Zoom (Vertigo) effect** locally on your device. By combining real-time object tracking with dynamic digital zooming, this app allows you to recreate the iconic cinematic effect made famous by Alfred Hitchcock.

## 🚀 Features

- **Real-time Object Tracking**: Utilizes OpenCV's tracking algorithms (CSRT/KCF) to lock onto a selected subject.
- **Automatic Dolly Zoom**: As you move the camera towards or away from the subject, the app automatically adjusts the digital zoom to keep the subject's size constant in the frame, creating the warping background effect.
- **Smooth Transitions**: Implements **Kalman Filtering** to stabilize tracking data and ensure smooth zoom operations.
- **Local Video Recording**: Record and save your Dolly Zoom shots directly to your device gallery.
- **Interactive UI**: Touch-to-select target for tracking.

## 🛠 Tech Stack

- **Language**: Java
- **Camera Engine**: [Android CameraX](https://developer.android.com/training/camerax)
- **Computer Vision**: [OpenCV for Android](https://opencv.org/android/)
- **Video Processing**: MediaCodec API
- **Algorithms**: 
  - CSRT/KCF Tracker for object tracking
  - Kalman Filter for motion smoothing

## 📋 Prerequisites

- **Android Studio**: Recommended latest version (Ladybug or newer).
- **JDK**: Java 17 or compatible.
- **Android Device**: Physical device recommended (CameraX features and performance on emulators may be limited).
- **OpenCV SDK**: The project depends on OpenCV. Ensure the OpenCV Android SDK is properly configured in your project if not already bundled.

## ⚙️ Setup & Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   ```

2. **Open in Android Studio**
   - Launch Android Studio.
   - Select "Open" and navigate to the project directory.

3. **Sync Gradle**
   - Wait for Android Studio to download dependencies and sync the project.

4. **Run the App**
   - Connect your Android device via USB or Wi-Fi debugging.
   - Click the **Run** button (green play icon).

## 📱 Usage

1. **Permissions**: On first launch, grant the necessary Camera and Audio permissions.
2. **Select Subject**: Point your camera at the subject you want to film.
3. **Start Tracking**: Tap the "Get Started" or Record button to enter the recording view, then select the object to track (or use the provided UI overlay to define the ROI).
4. **Perform Dolly Zoom**:
   - Physically move **backwards** (away from the subject) while the app automatically **zooms in**.
   - Or, move **forwards** (towards the subject) while the app **zooms out**.
5. **Record**: Press the capture button to start/stop recording your clip.

## 📂 Project Structure

- `app/src/main/java/com/example/myapplication_hitchock/`
  - `MainActivity.java`: Entry point of the application.
  - `VideoRecordActivity.java`: Handles camera preview, recording logic, and UI interactions.
  - `DollyZoomProcessor.java`: Core logic for tracking and calculating zoom factors.
  - `VideoEncoder.java`: Handles video encoding using MediaCodec.
  - `RectOverlayView.java`: Custom view for drawing the tracking rectangle overlay.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
