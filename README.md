# Praescriptio
A final year full stack Data Science Project based on Predictive Model and Image processing. The features include:
- Predict 42 common diseases based on any 5 prominent symptoms among the 132 common symptoms given in the dataset. 
- Get to know if you are affected with Malaria using your red blood cell image, among two classes: Infected and Uninfected. 
- Classify the brain MRI reports into 4 classes: Glioma, Meningioma, Pituitary and No-Tumour. Later Segment the brain tumours.
- Classify the lungs X-Ray reports into 3 classes: Covid-19, Pneumonia and Normal.
# Implementation of Problem
System Specification used to develop the application
1. Laptop Configuration: Windows, core i5 10th generation, 8GB RAM, 1TB Hard disk.
2. Browser: Google Chrome
3. Code Editor: Visual Studio Code
4. Model Datasets: Kaggle Datasets, Excel Spreadsheets
5. Database: SQL Lite
6. Framework: Flask (Python)
7. Models training: Jupyter Notebook
8. Frontend: html, CSS, JavaScript, Bootstrap, Google Maps.
# Application’s Directory Structure
1. app.py: main flask application where server runs. 
2. Brain_Tumor_Classification.ipynb: Jupyter notebook to train Brain Tumor Model
3. Brain_Tumor_Segmentation.ipynb: Jupyter notebook to train Brain Tumor Model
4. Diseasepredictipn.py: Python file to train Common Diseases Model
5. Forms.py: Form for user registration, login and contact
6. Malaria-Detection.ipynb: Jupyter notebook to train Malaria Model
7. model_classifcation.h5: Saved weights of Brain MRI Model [Drive Link](https://drive.google.com/file/d/1D12RTGCWFYKQ-IcELjUUCth0obzqfu4m/view?usp=sharing)
8. model_segmentation.h5: Saved weights of Brain MRI Model [Drive Link](https://drive.google.com/file/d/1ox0f0TkTPpnCrzWrs_AAluTJDsxwIzpo/view?usp=sharing)
9. model111.h5: Saved weights of Pneumonia-Covid19 Model
10. my_model.h5: Saved weights of Malaria Model
11. Pneumonia-Detection.ipynb: Jupyter notebook to train Pneumonia-Covid19 Model
12. _pycache_: This is the folder where the interpreter compiles python code to byte code first (this 
is an oversimplification) and stores it.
13. Brain Dataset: Folder having both the classification and segmentation dataset of Brain Tumor.
14. dataset: Folder having the labeled dataset for common disease prediction.
15. static: Folder containing all the CSS, JS, Bootstrap components and Frontend Animations.
16. templates: Folder containing all the html files to run the application on web.
17. uploads: Folder containing all the uploaded images being given by end-users during testing the 
models.
# Home page for the Web Application 
The home page contain:
- Home
- Contact and Register
- Models
- About
- Hospitals 

https://user-images.githubusercontent.com/79920441/173737365-4b05999c-6510-47ae-b8f1-275b8ac80a67.mp4

# Disease Prediction System
The dataset to predict the common type of diseases is collected from Kaggle, a subsidiary of Google 
LLC, is an online community of data scientists and machine learning practitioners. This dataset will help 
applying Machine Learning to great use. Applying Knowledge to field of Medical Science and making 
the task of Physician easy is the main purpose of this dataset. This dataset has 132 parameters on which 
42 different types of diseases can be predicted. Complete Dataset consists of 2 files for Training and 
Testing. Each CSV file has 133 columns. 132 of these columns are symptoms that a person experiences 
and last column is the prognosis. These symptoms are mapped to 42 diseases we can classify these set of 
symptoms to. Training file has 1500 rows and Testing file has 42 rows. [Disease Dataset](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning?select=Testing.csv)

https://user-images.githubusercontent.com/79920441/173737369-1fab37d0-c132-464c-9f62-cf56c2874498.mp4

# Malaria Detection Using Deep Learning
The dataset for Malaria detection has been collected again from Kaggle’s website where the dataset 
contains two folder for cell images concerning stained Red Blood Cells. Infected Cell Images and 
Uninfected Cell Images are present respectively with the total of 27,558 images. [Malaria Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)

https://user-images.githubusercontent.com/79920441/173737372-188b0fe2-e964-4fd6-b6aa-95d1f7a8c046.mp4

# Brain Tumor Classification and Segmentation Using Deep Learning
The dataset for Brain Tumor Classification and Segmentation has been collected from Kaggle’s website 
where there are two different dataset for classification and segmentation, that in turn also has a 
collection of various datasets put together to train a model. The classification dataset contains 
7022 images of human brain MRI images which are classified into 4 classes: glioma - meningioma - no
tumor and pituitary. The segmentation dataset contains brain MR images together with manual FLAIR 
abnormality segmentation masks. The images were obtained from The Cancer Imaging Archive (TCIA).
They correspond to 110 patients included in The Cancer Genome Atlas (TCGA) lower-grade glioma 
collection with at least fluid-attenuated inversion recovery (FLAIR) sequence and genomic cluster data 
available. [MRI Classification Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) , [MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

https://user-images.githubusercontent.com/79920441/173737379-1b13e9ab-8e2e-4c9a-814b-dc3df643fba5.mp4

# Covid-19 and Pneumonia Detection Using Deep Learning
The dataset for Covid-19 and Pneumonia Detection has been collected again from Kaggle’s website 
where the dataset contains two folder for cell images concerning chest X-Ray. There is a collection and 
labeling of total 5,232 chest X-ray images from children, including 3,883 characterized as depicting 
pneumonia (2,538 bacterial and 1,345 viral) and 1,349 normal, from a total of 5,856 patients to train the 
AI system. The model was then tested with 234 normal images and 390 pneumonia images 
(242 bacterial and 148 viral) from 624 patients. [Pneumonia Detection Dataset](https://www.kaggle.com/code/shobhit18th/keras-nn-x-ray-predict-pneumonia-86-54/data)


https://user-images.githubusercontent.com/79920441/173737409-3e5812b0-a8de-42a8-b3a3-fdce8d5ff70e.mp4


