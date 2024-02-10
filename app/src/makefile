# Compiler
CC = g++

# Project Name
PROJECT = app

DATA = ../data/

DEST = ../bin/

# OpenCV Libraries (pkg-config)
LIBS = `pkg-config --cflags --libs opencv4` 

BINDIR = ../bin

IMDIR = olympus/olympus

FEATURES = features.csv

# Compile and Link
# TASK = featureExtraction
# FEATURETYPE = Baseline

# Distance computaation requisites
TASK = distanceFinding
TARGET = pic.0571.jpg
DISTANCETYPE = EuclideanDistance
NUMIMAGES = 5

# $(PROJECT): $(TASK).cpp
# 	$(CC) -o $(DEST)$(TASK).exe $(TASK).cpp $(LIBS) 
# 	$(DEST)$(TASK).exe $(DATA)$(TASK).mp4 $(TASK)

#	FEATURE EXTRACTOR
# app: featureExtraction.cpp featureutils.cpp utils.cpp
# 	$(CC) $^ -o $(BINDIR)/$(TASK) $(LIBS) -I../include -DENABLE_PRECOMPILED_HEADERS=OFF
# 	$(BINDIR)/$(TASK) $(DATA)$(IMDIR)/ $(FEATURETYPE) $(DATA)$(FEATURES)

#	DISTANCE FINDER
# app: distanceFinding.cpp distanceutils.cpp utils.cpp
# 	$(CC) $^ -o $(BINDIR)/$(TASK) $(LIBS) -I../include -DENABLE_PRECOMPILED_HEADERS=OFF
# 	$(BINDIR)/$(TASK) $(DATA)$(FEATURES) $(DISTANCETYPE) $(DATA)$(IMDIR)/$(TARGET) $(NUMIMAGES)

app: playground.cpp distanceutils.cpp featureutils.cpp utils.cpp hog.cpp
	$(CC) $^ -o $(BINDIR)/playground $(LIBS) -I../include -DENABLE_PRECOMPILED_HEADERS=OFF
	$(BINDIR)/playground

# app: imgDisplay.cpp filter.cpp faceDetect.cpp
# 	$(CC) $^ -o $(BINDIR)/$(TASK) $(LIBS) -I../include -DENABLE_PRECOMPILED_HEADERS=OFF
# 	$(BINDIR)/$(TASK)


# app: switchboard.cpp filter.cpp
# 	$(CC) $^ -o $(BINDIR)/$(TASK) $(LIBS) -I../include -DENABLE_PRECOMPILED_HEADERS=OFF
# 	$(BINDIR)/$(TASK)

# app: timeBlur.cpp filter.cpp faceDetect.cpp
# 	$(CC) $^ -o $(BINDIR)/timeBlurr $(LIBS) -I../include -DENABLE_PRECOMPILED_HEADERS=OFF
# 	$(BINDIR)/timeBlurr ../data/cathedral.jpeg 

