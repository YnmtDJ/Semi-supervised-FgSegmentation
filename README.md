# foreground-segmentation-with-generative-adversarial-network
This is the foreground segmentation task using GAN with semi-supervised training.
# Setup Dataset
The data set is organized in the following way:  
```
./dataset/  
    /label/  
        /category1/  
            /input/...  
            /groundtruth/...  
            /background/...  
        /category2/...  
    /unlabel/  
        /category1/  
            /input/...  
            /groundtruth/  
            /background/...  
        /category2/...  
```
# Prepare Pretrained VGG16
Put `vgg16-397923af.pth` in `./save_models`.  
# Train Model
`python train.py --batchSize 4 --lr 0.0001 --cuda True`  
# Visualize Result
`cd ./utils`  
`python visualize.py`  

