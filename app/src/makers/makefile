# Compiler
CC = g++

# Project Name
PROJECT = app

DATA = ../data/

DEST = ../bin/

# OpenCV Libraries (pkg-config)
LIBS = `pkg-config --cflags --libs opencv4`

# Compile and Link
# TASK = test_video

# $(PROJECT): $(TASK).cpp
# 	$(CC) -o $(DEST)$(TASK).exe $(TASK).cpp $(LIBS)
# 	$(DEST)$(TASK).exe $(DATA)$(TASK).mp4 $(TASK)

# Compile and Link
# TASK = imgDisplay

# $(PROJECT): $(TASK).cpp
# 	$(CC) -o $(DEST)$(TASK).exe $(TASK).cpp $(LIBS)
# 	$(DEST)$(TASK).exe $(DATA)$(TASK).mp4 $(TASK)

# Compile and Link
# TASK = vidDisplay

# $(PROJECT): $(TASK).cpp
# 	$(CC) -o $(DEST)$(TASK).exe $(TASK).cpp $(LIBS)
# 	$(DEST)$(TASK).exe $(DATA)$(TASK).mp4 $(TASK)

# Compile and Link
TASK = playground

$(PROJECT): $(TASK).cpp
	$(CC) -o $(DEST)$(TASK).exe $(TASK).cpp $(LIBS)
	$(DEST)$(TASK).exe $(DATA)$(TASK).mp4 $(TASK)