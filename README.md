## realtime-traffic-signs-detection

### Dataset
https://graphics.cs.msu.ru/ru/node/1266

Russian Traffic Sign Dataset (RTSD) is a dataset for training and testing of traffic sign recognition algorithms. It contains:\
179138 labelled frames,\
156 sign classes,\
15630 physical signs,\
104358 images of signs.\

### Model
https://github.com/AlexeyAB/darknet

I used yolo4-tiny for good speed detection.\
Train script train is in <b>trafficSigns_train_YOLOV4.ipynb</b>\
Before train need to create .txt files for darknet framework (<b>Create txt files for train.ipynb</b>)\
Trainning took 70 hours in Colab Pro.\
Resulting mAP (mean Average Precision) is 65%.




