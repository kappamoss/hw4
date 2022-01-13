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


# Use my model &  generate the answer. (high-resolution image)  
Download my model: https://drive.google.com/file/d/1Ah4OHL_v_SiC6PtWQFnPOPd2obpOpHjA/view?usp=sharing  
and put into models folders.

