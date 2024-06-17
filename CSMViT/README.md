# CMViT

# 0.Preface

This repository holds the implementation code of the paper.

his repository provides code for "CSMViT: A Lightweight Transformer and CNN fusion 
network for lymph node pathological images diagnosis”.

# 1.Introduction

If you have any questions about our paper, feel free to contact us ([Contact](#5-Contact)). 

# 2. Usage

1. Train

   - In the first step, you can directly run the  `train.py` and just run it! 

     - Training dataset  

       ``` py
       ├── cell_data 
                  ├── 0   #Negative data
                  └── 1	  #Positive data
       ```

2. Test

   - When training is completed, the weights will be saved in `./weights/cmvit.pth`. 
   - Assign the Pathological images path、patch、patch coordinate save path, run the `predict_sys.py`, then the system will automatically detect the input pathological images
   -  The result is saved in the specified path

# 3.Contact

If you have any questions about our paper,  Feel free to email me(xyk68535@gmail.com) . 