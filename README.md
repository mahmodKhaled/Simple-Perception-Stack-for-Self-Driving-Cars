# Simple Perception Stack for Self-Driving Cars

<p align="center">
  
In this project we are going to create a **_simple perception stack for self-driving cars_** (SDCs.) Although a typical perception stack for a self-driving car may contain different data sources from different sensors (ex.: cameras, lidar, radar, etc…), we’re only going to be focusing on video streams from cameras for simplicity. We’re mainly going to be analyzing the road ahead, detecting the lane lines, detecting other cars/agents on the road, and estimating some useful information that may help other SDCs stacks. The project is split into two phases. We’ll be going into each of them in the following parts.

</p>

<div align="center">
   
   ![GitHub last commit (branch)](https://img.shields.io/github/last-commit/mahmodKhaled/Simple-Perception-Stack-for-Self-Driving-Cars/main)
   ![Lines of code](https://img.shields.io/tokei/lines/github/mahmodKhaled/Simple-Perception-Stack-for-Self-Driving-Cars)
   ![GitHub repo size](https://img.shields.io/github/repo-size/mahmodKhaled/Simple-Perception-Stack-for-Self-Driving-Cars)
</div>

## Phase 1 - Lane Line detection 🌌
In this first phase, our goal is to write a **_software pipeline_** to identify the lane boundaries in a video from a front-facing camera on a car
### _Expected Output from Phase 1_
![expected output](https://user-images.githubusercontent.com/54672453/163658944-d04f1d58-98ae-4017-b196-ba660c7d4a1b.png)


## Phase 2 - Objects detection 🚗
In this stage we are locating and identifying the cars on the road
### _Expected output from phase 2_
![1](https://user-images.githubusercontent.com/54672453/170452052-d5eb1154-bbad-4010-aa7d-3f5aef81a17f.png)

### _Notes_
in this phase object detecion doesn't work with **_Debugging Mode_**

## Installation Guidelines ⚙️

The **_jupyter Notebook_** contains the code and Presentaion , we have written the actual codes plus some comments explaining what we were doing through out the project 

the **.py** file contains the file that will run in the **_CMD_**

### Files are : 📁
> test_image_detection.ipynb this is the notebook

> test_image_detection.py this is the python script that will run through **_CMD_**

### Parameters List : 📝
1. Input (Video or Image) path 
2. Output (Video or Image) path
3. Flag that indicates , the input whether it was an image or video
4. Flag that indicates the **Debugging Mode**

**_Third Parameter_** has only two values
> 1 that indicates the input is Video

> 0 that indicates the input is Image

**_Fourth Parameter_** has only two values
> 1 that indicates Debugging Mode Activated

> 0 tha indicates Debugging Mode Deactivated

### Parameters to the **_CMD_** Syntax : 📜
```
python test_image_detection.py #First_Parameter #Second_parameter #Third_parameter #Fourth_parameter
```
**Sample Example:**

```
python test_image_detection.py test_images/test1.jpg test_images 0 1
```

## Contributers 🤝

| **Name** | **ID** |
| --- | --- |
| Shereen Reda Sayed Mohamed | 1804990 |
| Mahmoud Khaled Abdelaal Aly | 1801004 |
| Maram Nabil Ibrahim Ali | 1803746 |
