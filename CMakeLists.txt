CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

FIND_PACKAGE(OpenCV REQUIRED)

PROJECT(Projet552)

FIND_PACKAGE( Boost 1.40 COMPONENTS program_options system filesystem REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

ADD_EXECUTABLE(facematching main.cpp faceDetection.h faceDetection.cpp)
TARGET_LINK_LIBRARIES(facematching ${OpenCV_LIBS} ${Boost_LIBRARIES})
