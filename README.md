# Reference 
https://blog.csdn.net/qianbin3200896/article/details/104181552  
# Data Directory  
<pre><code> +- VRDL_HW4
    +- models
        +- checkpoint_srresnet.pth.tar
        +- best_checkpoint_srresnet.pth.tar
    +- dataset
        +- testing_lr_images
            +- testing_lr_images
                +- 00.png
                +- 01.png
                ...
                +- 13.png
        +- training_hr_images
            +- training_hr_images
                +- 2092.png
                +- 8049.png
                +- 8143.png
                ...
    datasets.py
    eval.py
    model.py
    split_train_val.py
    test.py
    train.py
    utils.py
</code></pre>

# How to train 
1. Download the dataset: (I divide dataset into train and validation data)  
https://drive.google.com/file/d/1n796E-LuV1lxqtXJvYgGQIKCWLpN5deH/view?usp=sharing

<pre><code> python train.py
</code></pre>


# Use my model &  generate the answer.     (high-resolution image)  
Use the inference.ipynb on colab to generate the answer: https://colab.research.google.com/drive/1WwW7anJiJ9tDxJ93iKFvCDg9DRxixXYJ#scrollTo=4jL5aZH-2Jtu  
You only need to cleck all the cells and you can get the output images in high-resolution image folder  

My PSNR score: 27.4716 with 2100 epochs (training time 11hours 40mins)  
![image](https://user-images.githubusercontent.com/24381268/149338780-0f8c4389-1285-4726-9776-64c08e7eb25a.png)

