# LRMtest
Simple Background Subtraction and Detection

# Installation and Execution Instructions
Use conda to create a new environment and activate it.
```
conda create -n lrm python=3.6 anaconda
conda activate lrm
```

Navigate to src in the parent folder, and run the following.
```
cd LRMtest/src
python main.py
```
Additional Tunable parameters:
```
'-i','--input',   default = '../data/challenge_clip.mkv', help='Path to source video'
'-o','--output',  default = '../data/result_clip.mp4',    help='Output file path and name'
'-f','--frames',  default = 1,                            help='Number of frames to use for background'
'-w','--width',   default = 450,                          help='Resize width keeping aspect ratio'
'-p','--rho',     default = 0.2,                          help='Background model averaging weight,i.e. bg(t) = p*I(t) + (1-p)*bg(t-1); range(0,1)'
'-t','--th',      default = 20,                           help='Threshold for blob detection; range(0,255)'
'-m','--minb',    default = 1000,                         help='Minimum blob size for detection; range(0, w*h)'

```

# Implementation Details
    - Rescalse to 450 
        Resize the frame to (w,h) -> (450, x) keeping aspect ratio.
    
    - Preprocess
        Gaussian and Median blur each frame to remove small artifacts and noise.

    - Background Subtraction 
        Compute a model of the background usnig the first 'f' frames.
        Implent on of the followgn two techniques to maintain a dynamic background model.
        
        - Weighted Running Average: 
            ```
            #Initialization
            bg(0) = I(0)
            
            #Updates 
            bg(t) = p * I(t) + (1-p) * bg(t-1)
            ```
            Here 'p' is the weight factor betwen 0 and 1.
        
        - Gaussian Mixture Model Estimate per pixel
            TBD

    - Detection 
        
        - Clean smaller artifacts that surface up post background subtraction 
          using erosion and dialtion techiques such as Opening and Closing.
        
        - Detect Blobs:
            Given the blob size range, (from minb to 2/3 of image size), find the connected 
            components using an 8-grid conenction. Each independent connected componet is a blob,
            and therfore an object candiate.

        - Get Bounding Rectangle:
            From the connected component mask, get the bounding rectangle using openCV.
            Rescale the bounding recatnagle to riginal frame size.
            Plot rectangle on the orignal frame.  
        
