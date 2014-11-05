CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

FIND_PACKAGE(OpenCV REQUIRED)

PROJECT(Projet552)

ADD_EXECUTABLE(facematching main.cpp faceDetection.h faceDetection.cpp)
TARGET_LINK_LIBRARIES(facematching ${OpenCV_LIBS})
