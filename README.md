# foreground-segmentation-with-generative-adversarial-network
This is the foreground segmentation task using GAN with semi-supervised training.
# Setup Dataset
The data set is organized in the following way:  
./dataset/  
&emsp;/label/  
&emsp;&emsp;/category1/  
&emsp;&emsp;&emsp;/input/...  
&emsp;&emsp;&emsp;/groundtruth/...  
&emsp;&emsp;&emsp;/background/...  
&emsp;&emsp;/category2/...  
&emsp;/unlabel/  
&emsp;&emsp;/category1/  
&emsp;&emsp;&emsp;/input/...  
&emsp;&emsp;&emsp;/groundtruth/  
&emsp;&emsp;&emsp;/background/...  
&emsp;&emsp;/category2/...  
# Prepare Pretrained VGG16
Put 'vgg16-397923af.pth' in './save_models'.
# Train Model
python train.py --batchSize 4 --lr 0.0001 --cuda True
# Visualize Result
cd './utils'  
python visualize.py

