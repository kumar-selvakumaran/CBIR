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
FEATURETYPE = Baseline

# Distance computaation requisites
TASK = distanceFinding
TARGET = pic.0571.jpg
DISTANCETYPE = EuclideanDistance
NUMIMAGES = 5


#	FEATURE EXTRACTOR (without striding)
# app: featureExtraction.cpp featureutils.cpp utils.cpp
# 	$(CC) $^ -o $(BINDIR)/$(TASK) $(LIBS) -I../include -DENABLE_PRECOMPILED_HEADERS=OFF
# 	$(BINDIR)/$(TASK) $(DATA)$(IMDIR)/ $(FEATURETYPE) $(DATA)$(FEATURES)

#	FEATURE EXTRACTOR (with striding)
# app: featureExtraction.cpp featureutils.cpp utils.cpp
# 	$(CC) $^ -o $(BINDIR)/$(TASK) $(LIBS) -I../include -DENABLE_PRECOMPILED_HEADERS=OFF
# 	$(BINDIR)/$(TASK) $(DATA)$(IMDIR)/ $(FEATURETYPE) $(DATA)$(FEATURES)

#	DISTANCE FINDER
# app: distanceFinding.cpp distanceutils.cpp utils.cpp
# 	$(CC) $^ -o $(BINDIR)/$(TASK) $(LIBS) -I../include -DENABLE_PRECOMPILED_HEADERS=OFF
# 	$(BINDIR)/$(TASK) $(DATA)$(FEATURES) $(DISTANCETYPE) $(FEATURETYPE) $(DATA)$(IMDIR)/$(TARGET) $(NUMIMAGES)

# to run the whole pipeline
app: playground.cpp distanceutils.cpp featureutils.cpp utils.cpp hog.cpp
	$(CC) $^ -o $(BINDIR)/playground $(LIBS) -I../include -DENABLE_PRECOMPILED_HEADERS=OFF
	$(BINDIR)/playground

# to run the whole pipeline
# app: makeHist.cpp
# 	$(CC) $^ -o $(BINDIR)/makeHist $(LIBS) -I../include -DENABLE_PRECOMPILED_HEADERS=OFF
# 	$(BINDIR)/makeHist /app/data/sidewalk.jpg