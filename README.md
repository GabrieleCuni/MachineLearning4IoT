# MachineLearning4IoT
Workshops and homeworks of the machine learning for IoT course

The course aims to introduce the problems related to the implementation of machine learning applications and algorithms on platforms other than high-performance servers available in the cloud.
The contents and the thus the skills acquired at the end of the course include both the hardware aspects of the problem (architectures of "edge" devices) and the software aspects (programming models, protocols and related APIs).
The skills acquired will allow a correct understanding of decentralized systems in which the flow of data is processed not only on servers, but rather locally on devices with reduced computational resources and energy.

GroupProject1 stores a Python script which is intended to run on a Raspberry Pi 4 and iteratively samples 1-second audio signals, process the MFCCs, and store the output on disk. The pre-processing latency must be <80ms on the referenced hardware.
More details are provided in the [Report1](https://github.com/GabrieleCuni/MachineLearning4IoT/blob/main/GroupProject1/Group18_Homework1.pdf).

The first exercise of the Group Project 2 trains and validates two Neural Networks able to infer temperature and humidity. The second one trains and validates three Neural Networks for keyword spotting on the original mini speech command dataset.
More details are provided in the [Report2](https://github.com/GabrieleCuni/MachineLearning4IoT/blob/main/GroupProject2/Group18_Homework2.pdf).

GroupProject3 stores two exercises: The first one is a Big/Little model for Keyword Spotting composed by a Big neural network running on notebook and a Little neural network running on Raspberry Pi. The second one is a cooperative inference microservice application made by one client which runs on the Raspberry Pi and multiple Neural Network servers which run on a notebook. More details are provided in the [Report3](https://github.com/GabrieleCuni/MachineLearning4IoT/blob/main/GroupProject3/Group18_Homework3.pdf).



