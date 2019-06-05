# wearTECH

With the continuous evolution of technology, mobile devices are becoming more and more important in people's lives. In the same way, new needs related to the information provided by their users arise, making evident the need to develop systems that take advantage of their daily use. The recognition of personal activity based on the information provided by the last generation mobile devices can easily be considered as an useful tool for many purposes and future applications. This project presents the use of information provided from an iPhone 7 and a Myo armband device in different acquisition schemes, assessing conventional supervised classifiers to recognize personal activity by an identification of seven classes.

see YouTube video:

[![YouTube video](https://i.ytimg.com/vi/qYwHxlea-4E/hqdefault.jpg)](https://www.youtube.com/watch?v=qYwHxlea-4E&t=13s)

## Repository Folders
This repository contents:

```
/your_root        - path
|--data                 /Folder where you can find the project data
  |--iPhone             /Activity data from the iPhone 
  |--LearningData       /.sav files from the training
  |--Myo                /Activity data from the Myo
  |--project-Figures    /Figures from the project
  |--References         /Related work
  
|--Readme   / instructions for use the Activity recognition software 

|--src      / scripts for the Activity Recognition (A.R.)
  |--BestFeatures       /scripts for the A.R. after feature reduction
  |--FullFeatures       /scripts for the A.R. with Full Features
  |--Gauss-curves       / Curves for the feature reduction
  |--Graphics           / 2 and 3 bars graphics 
  |--Hierarchical       /scripts for the A.R. with hierarchical scheme
  |--Hierarchical-consolidate-stairs /scripts with upstairs and downstairs consolidated 
  |--HierarchicalRoutine-UnknownClass / scripts for the A.R. adding an unknown class 
  |--my_script.sh       /script to run the synchronized system to collect data
  |--myo_con.py         /script to recollect data from the myo armband
  |--posicioncelular.py /script that shows that the result of the prediction is independent of cellphone position in the pocket
  |--pruebasincronia.py /script that test the synchronized system

```

## Hardware elements
- [Myo Armband](https://support.getmyo.com/hc/en-us/articles/203398347-Getting-started-with-your-Myo-armband)
- [iPhone 7](https://www.apple.com/co/iphone-7/specs/)

## Software requirements
- [SensorLog App](https://itunes.apple.com/co/app/sensorlog/id388014573?mt=8) iOS
- [Ubuntu 16.04 Xenial Xerus](http://releases.ubuntu.com/16.04/)
- [ROS kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu)
- [ROS node for the Thalmic Labs Myo Armband](https://github.com/roboTJ101/ros_myo)
- Python >2.6 in this case I used [Spyder](https://docs.spyder-ide.org/installation.html#installing-with-anaconda-recommended) and [atom](https://flight-manual.atom.io/getting-started/sections/installing-atom/)

### Installing ROS node for the Myo
1. Open Terminal
2. Create a workspace
`$ mkdir -p ~/catkin_ws/src`
3. Open the workspace
`$ cd ~/catkin_ws/src`
4. Clone the repository.
`$ git clone https://github.com/roboTJ101/ros_myo.git`
5. Once it is completed, do the following:
`$ cd ~/catkin_ws`
`$ catkin_make`
6. Once the Node is installed, if you're using the new firmware for the Myo Armband go to ros_myo/scripts/myo-rawNode.py and in the 300 line you'll find this
 `elif attr == 0x23:`
    `typ, val, xdir = unpack('3B', pay)`

    just replace those 2 lines with
`elif attr == 0x23:`
    `typ, val, xdir, unknown1, unknown2, unknown3 = unpack('6B', pay)`

## Running Ros_myo
if you want to run the node you must follow the next steps
1. Open Terminal and run Roscore
`$ roscore`
2. Open a new window and do the following
`$ cd ~/catkin_ws`
`$ source ./devel/setup.bash`
`$ rosrun ros_myo myo-rawNode.py`
3. [Perform the sync gesture](https://support.getmyo.com/hc/en-us/articles/200755509-How-to-perform-the-sync-gesture)

**Note:** It's important to run roscore first and keep that 2 windows open while you're collecting data

### Collecting Data
Once you have ros_myo running, it's time to collecting data following the next steps:
1. Your computer and the cellphone must be connected at the same local network
2. Open a new Terminal window and go to `/src` folder
`$ cd/wearTECH/src`
3. Open the my_script.sh script
`$ sudo nano my_script.sh`
4. Update the IP address and the socket port
5. press "start" recording in the app SensorLog
6. Run the `my_script.sh`
`$ bash my_script.sh`
7. Press `ctrl+c` when you want to stop the data collection.
8. You will find a file called `mylog_test.csv` in your `/src` folder, this one corresponds to the iPhone data. And in your `/data` folder you'll find a `.txt` file with the date and time of the collection, this one corresponds to the Myo data.

**Note:** The `/data` folder contains two folders called iPhone and Myo. The content of these folders are the data collected, they're organized by activity, username and cellphone position in the pocket.

### How to start the Activity Recognition Test
1. Open your Python editor (spyder or atom)
2. From the `/src` folder choose the classification type
3. Run first the training script, and the learning Data are saved in the `/data` folder
4. Run the Validation script if you want to know the results of prediction for the offline tests (test 1) or run the routine script to know the routine prediction results( test 2)


## Authors:
**Universidad de Ibagué** - **Ingeniería Electrónica**
**Proyecto de Grado 2019/A**
- Maira J. Triana 
- [Harold F. Murcia](www.haroldmurcia.com)
***

