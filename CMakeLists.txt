CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

FIND_PACKAGE(OpenCV REQUIRED)

PROJECT(Projet552)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_DEBUG_RUNTIME ON)
FIND_PACKAGE( Boost COMPONENTS program_options system filesystem REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

ADD_EXECUTABLE(facematching main.cpp faceDetection.h faceDetection.cpp featureDetection.h featureDetection.cpp bowMethod.cpp bowMethod.h tools.cpp tools.h getSiftKeypoints.cpp getSiftKeypoints.h)
TARGET_LINK_LIBRARIES(facematching ${OpenCV_LIBS} ${Boost_LIBRARIES})
