# Project Title

## AI-Based Frame Interpolation and Video Generation System for WMS Services

### Project Description
This project focuses on developing an AI-powered system for frame interpolation and video generation. It generates smooth minute-by-minute animations by filling gaps between satellite images, ensuring continuous and high-quality visualizations. The system is designed to support applications in environmental monitoring, weather forecasting, and disaster management by providing real-time insights into dynamic phenomena.

### Project Overview
This group project is developed as part of the **MLOps** subject during the 5th semester at **Karunya Institute of Technology & Sciences**. The project was implemented by the following students:

- **Anson Saju George** - URK22CS7064  
- **Jeril Joseph** - URK22CS7086  
- **Shawn Jaison** - URK22CS7120  

The project involves integrating advanced AI algorithms with open-source WebGIS technologies to display satellite imagery in a user-friendly interface. By combining multiple satellite images and applying deep learning techniques, the system enables effective visualizations of dynamic phenomena, such as moving clouds.

---

### Features
- **AI-Powered Frame Interpolation:** Generates smooth animations by interpolating gaps between satellite images, enhancing temporal resolution.
- **Automated Video Generation:** Compiles interpolated frames into continuous videos, providing a coherent visualization of dynamic events.
- **WebGIS Integration:** Utilizes open-source technologies like OpenLayers for displaying videos on interactive browser-based maps, enhancing spatial context.
- **User-Friendly Interface:** Offers an intuitive interface that allows users to interact with videos, making complex satellite data more navigable and accessible.
- **Real-Time Monitoring:** Supports real-time monitoring of environmental changes, aiding in applications like weather forecasting and disaster management.

---

### Technologies Used
- **Programming Language:** Python  
- **Libraries:** 
  - **TensorFlow:** For developing and training deep learning models.  
  - **PyTorch:** To support various neural network architectures.  
  - **OpenCV:** For image processing tasks.  
  - **NumPy:** For numerical computations.  
- **Tools:** 
  - **Git:** Version control for tracking changes in the codebase.  
  - **Docker:** For containerization and easy deployment of applications.  
  - **Kubernetes (Optional):** For orchestration of containerized applications.  
  - **Jupyter Notebook:** For exploratory data analysis and visualization.(mostly .ipynb is used)
- **Platform:** Fedora OS with NVIDIA RTX 3060 GPU for model training and inference.
- ****          NVIDIA A100 GPU Core

---


### Workflow Overview
- **Satellite Image Capture:** Captures satellite images every 30 minutes to ensure up-to-date visualizations.
- **Image Pre-Processing:** Cleans and normalizes the captured images for better quality and processing.
- **AI-Powered Frame Interpolation Model:** Utilizes advanced AI algorithms, including CNNs like U-Net and Mask R-CNN, for effective interpolation.
- **Video Generation:** Compiles interpolated frames into seamless videos for smooth playback.
- **WebGIS Integration & Interactive Map Display:** Integrates with WebGIS for enhanced real-time spatial visualization.
- **User Interface (UI):** Provides video playback and interaction options for an engaging user experience.

---

### Future Work
- **Enhancements in AI Models:** Explore additional AI architectures, such as transformers, for improved interpolation accuracy.
- **Scalability Improvements:** Implement cloud-based solutions for scaling the processing capabilities to handle larger datasets.
- **User Feedback Integration:** Gather user feedback for continuous improvement of the interface and functionalities.

---

### Impact and Benefits
- **Enhanced Visualization:** Provides smooth, continuous videos of satellite imagery, improving the understanding of dynamic phenomena.
- **Accessibility:** Makes satellite imagery more interactive and accessible for developers and end-users.
- **Real-Time Monitoring:** Enables monitoring of environmental changes, aiding in timely interventions for disaster management.
- **Cost-Effective:** Utilizes open-source technologies, making the solution economically viable for various applications.
- **Scalability:** Capable of handling large volumes of data and adaptable for diverse use cases.

---

### Research and References
1. **DeepCloud: An Investigation of Geostationary Satellite Imagery Frame Interpolation for Improved Temporal Resolution:** Discusses a deep-learning methodology for enhancing the temporal resolution of multi-spectral weather products from geostationary satellites.  
   [Access the paper here](#)

2. **Temporal Interpolation of Geostationary Satellite Imagery With Optical Flow:** Presents a novel application of deep learning-based optical flow to enhance the temporal frequency of observations.  
   [Read more about it here](#)

3. **Satellite Imagery and AI - A New Era in Ocean Conservation:** Introduces specialized computer vision models for various types of satellite imagery, emphasizing the importance of reliable machine learning models.  
   [More details can be found here](#)

---

### Conclusion
This project aims to provide an innovative solution for visualizing satellite data through AI-based frame interpolation and video generation. By enhancing accessibility and usability, the system supports applications in environmental monitoring, weather forecasting, and disaster management.

