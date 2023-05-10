To run the the code you will need to install the following libraries:

tensorflow with keras
pytorch
pillow
matplotlib
numpy

1. To run the code you will need to download the kaggle datasets
2. Put the data directory in the same directory as all the code
3. Move the additional_test directory into that data directory

To run the 2-Layer MLP:
- Go in the code and uncomment greyscale if you want to test greyscale images
- Then change line 92 to -> images = ((images.view(-1, 350*350*1).to(torch.float32)).requires_grad_()).to(device)
- Otherwise leave everything as is to run color model
- run command python3 ignite_NN_basemodel.py

To run 3-layer ConvNet:
- run command python3 keras_ignite_NN.py

To run residual Convnet:
- run command python3 keras_res_NN.py

