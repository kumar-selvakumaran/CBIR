# Content Based image Retrieval

- This project revolves around 2 main concepts namely, feature extraction and distance/similarity
computation and applies these concept to solve the problem of content-based image retrieval. 

- An array of feature computation methods ranging from colour-driven chromotacity histograms, to
texture-driven gradient and orientation histograms are used (HOGs). The computed features are evaluated
and compared with each other to find similar/distance images.

- The idea is to extract the most relevant and descriptive features and match them in the best possible way so that the matches are
highly semantically similar. Various combinations of factors such as choosing relevant regions of
interests, modifying classical processing techniques and pipelining them together with different
distance metrics are considered in order to generate optimal image pairs with high semantic/
aesthetic correlation.

## Key techniques:

- Histogram of Oriented gradients
- ResNet18 embeddings
- Strided histogram matching
- Laws Filters
- Morpholigcal operations (opening, closing, dilation, erosion)
- Distance metrics : L2 Norm, and Histogram Intersection.

## setup:

### key features:

- Windows 11, WSL, Docker, Flask, X11 screen

### requirements: 
- docker desktop version : v4.22.1
- docker version : 24.0.5
- (WSL) Ubuntu-22.04