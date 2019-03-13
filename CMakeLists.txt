cmake_minimum_required(VERSION 3.13)
project(asr)

set(CMAKE_CXX_STANDARD 14)
set(C_DIR ${CMAKE_CURRENT_SOURCE_DIR})


add_executable(asr
        ${C_DIR}/main.cpp
        ${C_DIR}/core/c/model.cpp
        ${C_DIR}/core/c/readwav.cpp
        ${C_DIR}/core/c/reshape.cpp
        ${C_DIR}/core/c/constant.cpp
        ${C_DIR}/core/headers/constant.h
        ${C_DIR}/core/headers/model.h
        ${C_DIR}/core/headers/readwav.h
        ${C_DIR}/core/headers/reshape.h)

add_library(sound SHARED
        ${C_DIR}/main.cpp
        ${C_DIR}/core/c/model.cpp
        ${C_DIR}/core/c/readwav.cpp
        ${C_DIR}/core/c/reshape.cpp
        ${C_DIR}/core/c/constant.cpp
        ${C_DIR}/core/headers/constant.h
        ${C_DIR}/core/headers/model.h
        ${C_DIR}/core/headers/readwav.h
        ${C_DIR}/core/headers/reshape.h
        ${C_DIR}/core/headers/jni.h
        ${C_DIR}/core/headers/jni_md.h
        )

target_include_directories(sound PRIVATE ${C_DIR}/core/headers ${C_DIR}/core/c) # required all file in it
target_include_directories(asr PRIVATE ${C_DIR}/core/headers ${C_DIR}/core/c) # required all file in it