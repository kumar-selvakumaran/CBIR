# Compiler
CC = g++

# Project Name
PROJECT = app

DATA = ../data/

DEST = ../bin/

# OpenCV Libraries (pkg-config)
LIBS = `pkg-config --cflags --libs opencv4` 

BINDIR = ../bin

# Compile and Link
TASK = featureExtraction

# image database directory
IMDIR = olympus/olympus

# targe image path
TARGET = pic.0145.jpg

#feature vector csv
FEATURES = features.csv

# TASK = imgDisplay


# $(PROJECT): $(TASK).cpp
# 	$(CC) -o $(DEST)$(TASK).exe $(TASK).cpp $(LIBS) 
# 	$(DEST)$(TASK).exe $(DATA)$(TASK).mp4 $(TASK)

# app: featureExtraction.cpp featureutils.cpp
# 	$(CC) $^ -o $(BINDIR)/$(TASK) $(LIBS) -I../include -DENABLE_PRECOMPILED_HEADERS=OFF
# 	$(BINDIR)/$(TASK) $(DATA)$(IMDIR)/ $(DATA)$(FEATURES)

app: playground.cpp distanceutils.cpp featureutils.cpp utils.cpp
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

