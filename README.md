# Micron_Level_Gap_measurement_system
##  Overview
This project presents an automated system for measuring weld gap width using digital image processing techniques. In industrial welding, maintaining a consistent and optimal gap between components is essential for ensuring strong and defect-free joints. Traditionally, this is done manually using calipers or feeler gauges, which is time-consuming and prone to human error.

This automated vision-based inspection pipeline replaces manual methods by capturing images of components and analyzing them to extract precise measurements without direct human intervention. It is highly suitable for integration into industrial production lines, such as automotive manufacturing.

 **Key Features**
  - Multi-Point Measurement: Instead of relying on a single measurement point, the system scans the image at multiple horizontal levels along the vertical axis (typically ~30 points) to capture variations across the entire length.
  - Advanced Image Preprocessing: Applies grayscale conversion, median filtering to remove impulsive noise, and bilateral filtering to smooth the image while maintaining sharp boundaries.
  - Robust Edge Detection: Employs gradient-based operators like Sobel and Scharr for accurate vertical edge detection, alongside a faster alternative using the Canny edge detector.
  - Real-World Calibration: Converts pixel-based measurements into real-world units (millimeters) using a known reference measurement, allowing direct comparison with industrial tolerance limits.
  - Graphical User Interface (GUI): Features a user-friendly interface that allows users to load images, capture real-time data using a camera, run analysis, and save reports without any Software technical knowledge.
  - Comprehensive Reporting: Generates visual outputs (annotated images, gap width profiles, histograms) and stores numerical results in structured Excel or CSV files for traceability.

 **Technology Stack**
  - Language: Python
  - Computer Vision: OpenCV
  - Data Processing: NumPy, SciPy (for Savitzky-Golay filters)
  - Data Visualization: Matplotlib
  - User Interface: Tkinter 

  **System Architecture Pipeline**
  - Camera / Image Acquisition: Capturing high-resolution images under controlled lighting.
  - Image Preprocessing: Resizing, grayscale conversion, and noise filtering.
  - Contrast Enhancement: Applying CLAHE.
  - Edge Detection: Identifying gap boundaries using Sobel/Scharr operators.
  - Gap Measurement: Finding the center line and measuring the distance between edges at multiple heights.
  - Data Storage: Saving results and generating visual reports.  
  
  
  **Calibration Support**
  - Convert pixel measurements to millimeters
  - User-defined actual gap input, which is calculated by capturing an image and checking the pixel distance of a known gap width spot and entering it in the software under the actual gap in the application screen
  - Has to be done only once after fixing the camera

**Visualization**
  - Gap detection overlay
  - Measurement profile graph
  - Histogram distribution
  - Coordinate reference map
  - Zoomed inspection view

---
# Results
Note: Due to Non-Disclosure Agreements (NDA), actual industrial images and raw manufacturing data cannot be shared in this repository. The examples below represent the format and capabilities of the system using mock data.
1. GUI Interface
<img width="1917" height="1014" alt="Screenshot 2026-04-22 121641" src="https://github.com/user-attachments/assets/4f86d108-9fb3-4273-a52f-e467e12d0d36"  /><br>
2. Validation Outputs <br>
<img width="585" height="286" alt="Screenshot 2026-05-13 111327" src="https://github.com/user-attachments/assets/308888a4-787b-4378-a264-2ac09c5d7464" /> <br>
3. Sample Output <br>
<img width="359" height="497" alt="gap_results(1)(1)" src="https://github.com/user-attachments/assets/c6d1a34a-fe63-4578-885b-585bdf6ed05a" /><br>
<img width="918" height="894" alt="Screenshot 2026-05-21 104410" src="https://github.com/user-attachments/assets/27e51e98-762f-4c05-b40e-a9d2220f62d8" /><br>

---

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/Logrooot/Micron_Level_Gap_measurement_system.git
cd Micron_Level_Gap_measurement_system


