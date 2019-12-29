# LRM
Background Subtraction and Detection

# Execution Instructions
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
    - Rescale to 450 
        Resize the frame to (w,h) -> (450, x) keeping aspect ratio.
    
    - Preprocess
        Gaussian and Median blur each frame to remove small artifacts and noise.

    - Background Subtraction 
        Compute a model of the background usnig the first 'f' frames.
        Implent on of the following two techniques to maintain a dynamic background model.
        
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
            Rescale the bounding rectangle to the original frame size.
            Plot rectangle on the orignal frame.  
        
# Observations and Comments

1. Background Subtraction:
    I started by implementing a simple Weighted Running Average method to compute the background mask. An alternative way was to implement the Gaussian Mixture Model estimation per pixel using EM, 
    The current video seems to do alright with a Weighted Running Average so I did not go ahead with GMMsbut it would work better for more complex/ dynamic scenes. 

2. Detection:
    Since the task only allows me to use numpy and opencv basic tools, not even standard convolution from scipy and certainly not any neural nets, I am limited to using standard statistical functions form open CV, so I went ahead with background subtraction, thresholding, erosion-dilation, and blob detection using connected components to give rough estimates of blobs in the foreground area. 

3. Future Considerations: 
    - New blobs once detected, could be tracked using optical flow, and also matched with new detections every few frames,      etc. in order to remove redundancy. 
    - The performance could be improved for detecting top view of human heads by template matching for circular head like      objects, for example using edge detection in the foreground area blobs. 
    - A neural-net trained to detect top-view humans would definitely perform the best. For example, lightweight networks      like YOLO or mobilNet finetuned for top view.
    - Traditional techniques like HOG features or Haar Cascade features trained to specifically detect top-view humans         would do better than what I have implemented. 

4. Current implementation failure cases:    
    - When groups of people walk close by, it detects them as a single blob, so not very useful for counting.    
    - Trailing shadows show up and expand blobs, this could be improved by ignoring shadows, some work in Fourier Domain        could help in this task. 
    - There are more false positives in the scene since this methodology has no understanding of Human body structures, it    just works with new pixels in the scene. As a result, the detection is solely based on contrasting pixels with          respect to the background and the thresholds we set. No feature extraction or shape matching is performed currently. 
