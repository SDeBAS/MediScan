<br />
<div align="center">
  <a href="https://github.com/SDeBAS/MediScan.git">
    <img src="static/images/logo.png" alt="Logo">
  </a>

  <h3 align="center">Digitizing Handwritten Medical Prescriptions</h3>

  <p align="center">
    Welcome to our project!
    <br />
    <a href="https://github.com/SDeBAS/MediScan.git"><strong>Explore the documentation</strong></a>
    <br/>
    <a href="https://drive.google.com/file/d/1oQ5JY3u107fGGAheen73WK0tgxWfVmIe/view?usp=sharing">View Demo</a>
  </p>
</div>


 <!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About the Project</a>
      <a href="#about-the-team">About the Team</a>
      <ul>
        <li><a href="#inspiration">Motivation Behind the Project</a></li>
        <li><a href=#social-impact>Outcome of the Project</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#intel-oneapi">Intel OneAPI</a>
      <ul>
        <li><a href="#use-of-onednn-in-our-project">Use of oneDNN and TensorFlow in our project</a></li>
      </ul>
    </li>
    <li><a href="#what-it-does">What it does</a></li>
    <li><a href="#how-we-built-it">How we built it</a></li>
    <li><a href="#what-we-learned">What we learned</a></li>
  </ol>
</details>



 <!-- ABOUT THE PROJECT -->
## 📚 Project Overview
<div align="center">
  <img src="static/images/ss1.png" width="750">
</div>
The "MediScan" project aims to transform healthcare by accurately converting handwritten prescriptions into digital formats. Leveraging advanced OCR and Artificial Intelligence (AI), enhances patient safety by reducing errors caused by manual interpretation. Developed collaboratively on GitHub, MediScan bridges traditional prescriptions with modern digital healthcare, ensuring efficient, secure, and precise medication management. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>


  <!-- ABOUT THE TEAM -->
## About The TEAM
<div align="center">
  <img align="right" src="https://github.com/rajput2107/rajput2107/blob/master/Assets/Developer.gif" width='200'/>
  <h2>Health Heroes</h2>
</div>
This is  Team Health Heros, and we are pleased to introduce our project on Medical Prescription Handwritten Text Recognition utilizing cutting-edge technologies such as Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL), on Intel one API server using Intel oneDAL and Intel extension for Tensorflow.👨‍💻. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Motivation -->
## ⚡ Motivation Behind Our Project
The motivation behind the project stems from the critical need to address the potential risks associated with :
1. Critical Need: The project is motivated by the necessity to mitigate risks linked to interpreting handwritten prescriptions in healthcare.
2. Error Prone: Handwritten prescriptions often result in misinterpretations, causing medication errors, adverse reactions, and patient harm.
3. Advanced Technology: Leveraging OCR and machine learning, the project seeks to convert handwritten prescriptions into accurate digital formats.
4. Reliable Solution: The project's goal is to provide a dependable method to transform prescriptions, enhancing precision and reducing errors.
5. Streamlined Verification: Medical professionals benefit from streamlined prescription verification, improving patient safety.
6. Reduced Likelihood of Errors: Through cutting-edge techniques, the project significantly diminishes the chance of errors with severe health consequences.

## ⚡ Outcome of our Project
1. Seamlessly bridge traditional handwritten prescriptions and modern digital healthcare systems.
2. Address concerns about potential errors in interpreting handwritten prescriptions due to medical complexity.
3. Utilize advanced optical character recognition (OCR) technology and machine learning to transcribe prescriptions accurately.
4. Transform handwritten prescriptions into a digital, machine-readable format.
5. Enable medical professionals to effortlessly access, verify, and process prescription information.
6. Minimize risks of medication errors, drug interactions, and dosage inaccuracies.

## ⚡ Social Impact of Our Project
The "MediScan" project holds several significant social impacts:

1. Enhanced Patient Safety: Reduces the potential for medication errors caused by misinterpretation of handwritten prescriptions, leading to improved patient well-being.

2. Accurate Treatment: Ensures that patients receive the correct medications, dosages, and instructions, enhancing the effectiveness of medical treatments.

3. Time Efficiency: Medical professionals can quickly access and verify prescription information, enabling them to allocate more time to patient care rather than deciphering handwriting.

4. Reduced Healthcare Costs: Minimizes the financial burden caused by medication errors, hospital readmissions, and adverse drug reactions.

5. Global Accessibility: Increases access to accurate prescription processing in remote or underserved areas, contributing to equitable healthcare distribution.

6. Improved Medical Record Keeping: Contributes to more accurate electronic health records, facilitating better treatment tracking and history for patients.

7. Professional Collaboration: Promotes smoother communication between healthcare providers, pharmacists, and patients, fostering a collaborative healthcare ecosystem.

8. Technology Adoption: Encourages the integration of modern technology in healthcare practices, setting a precedent for innovation in medical processes.

9. Medical Advancements: Generates data that could be used for medical research, enabling insights into prescribing trends, patient adherence, and treatment outcomes.

10. Patient Empowerment: Empowers patients with clear and legible prescriptions, enabling them to understand and follow their treatment plans accurately.


### Built With <img src="images/built.png" alt="png" width="30">

This section should list any major frameworks/libraries used to bootstrap your project. Here are a few examples.

* [![oneapi][oneapi]][oneapi-url]
  * [![onednn][onednn]][onednn-url]
* [![python][python]][python-url]
* [![jupyter][jupyter]][jupyter-url]
* [![tensorflow][tensorflow]][tensorflow-url]
* [![streamlit][streamlit]][streamlit-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Intel one api -->
## Intel oneAPI <img src="images/oneapi2.png" alt="png" width="30">
Intel OneAPI is a comprehensive development platform for building high-performance, cross-architecture applications. It provides a unified programming model, tools, and libraries that allow developers to optimize their applications for Intel CPUs, GPUs, FPGAs, and other hardware. Intel OneAPI includes support for popular programming languages like C++, Python, and Fortran, as well as frameworks for deep learning, high-performance computing, and data analytics. With Intel OneAPI, developers can build applications that can run on a variety of hardware platforms, from edge devices to data centers, and take advantage of the performance benefits of Intel architectures.

### Use of oneDNN and TensorFlow in our project
<img src="images/onednn.png" alt="png" width="700">
OneDNN provides highly optimized routines for various deep learning operations, including convolution, pooling, normalization, and activation functions. By using oneDNN, you can expect faster execution times and better performance on modern CPUs, especially those with Intel processors.

In this project <code>os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'</code> line sets an environment variable called <code>TF_ENABLE_ONEDNN_OPTS to '1'</code>. This enables the use of Intel's OneAPI Deep Neural Network Library (OneDNN) optimizations for TensorFlow on the system where this code is being run. OneDNN is a high-performance library for deep learning that is designed to optimize the performance of deep neural network computations on a variety of hardware platforms. By enabling OneDNN optimizations, this code may run faster on certain hardware architectures that are compatible with OneDNN. <strong>In this project, the Conv2D and Dense layers will be automatically optimized using oneDNN, which should result in faster training and inference times on compatible hardware.</strong>

The <code>tensorflow.keras</code> module is used to create a convolutional neural network (CNN) model for image classification. The model architecture consists of three convolutional blocks, each followed by a max pooling layer, and three fully connected layers with dropout for regularization.

Finally, the <code>model.compile</code> method is called to configure the optimizer, loss function, and evaluation metric for the model. The optimizer used is Adam, and the loss function used is sparse categorical cross-entropy. The model is also evaluated using the accuracy metric.

<!-- What it does -->
## What it does <img src="images/does.png" alt="png" width="30">
The garbage segregation project using AI/ML automates the process of identifying and sorting different types of waste. The system uses image recognition techniques to analyze images captured by a camera and machine learning algorithms to classify the waste into different categories. By accurately identifying and segregating waste, the system can reduce the workload on human workers, increase efficiency, and reduce environmental pollution. The project also promotes responsible waste disposal practices by making people more aware of the types of waste they generate and the proper ways to dispose of them.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


