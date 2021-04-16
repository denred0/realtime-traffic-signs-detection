### Realtime Traffic Signs Detection

### Result
https://www.youtube.com/watch?v=CGpu7aXEGmk<br>
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/CGpu7aXEGmk/0.jpg)](https://www.youtube.com/watch?v=CGpu7aXEGmk)


### Dataset
https://graphics.cs.msu.ru/ru/node/1266

Russian Traffic Sign Dataset (RTSD) is a dataset for training and testing of traffic sign recognition algorithms. It contains:\
179138 labelled frames,\
156 sign classes,\
15630 physical signs,\
104358 images of signs.

### Model
https://github.com/AlexeyAB/darknet

I used <b>yolo4-tiny</b> for good speed detection.\
Train script train is in <b>trafficSigns_train_YOLOV4.ipynb</b>\
Before train need to create .txt files for darknet framework (<b>Create txt files for train.ipynb</b>)\
Trainning took 70 hours in Colab Pro.\
Resulting mAP (mean Average Precision) is 65%.




