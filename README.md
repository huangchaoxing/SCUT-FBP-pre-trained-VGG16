# SCUT-FBP-pre-trained-VGG16
A face beauty predictor by training the VGG16

The code is based on this dataset:  https://arxiv.org/ftp/arxiv/papers/1511/1511.02459.pdf.

The SCUT-FBP dataset is proposed by the SCUT for the purpose of training a face beauty predictor.
By using the network, the computer can rate an Asian women's face beauty for a certain value between one and five.

This code is implemented on Tensorflow+Keras and a pre-trained VGG16 model is used to train the predictor.
If you want to test your own image, just place your image in the Rating_Collection and add its file dir to the dir list in
the test part.
The model and training process are included in the model.py ,which locates in the file of Rating collection.
The dataset is consisted of two files, the Data_collection and Rating collection.
