cmake_minimum_required(VERSION 3.12)
project(asr_c)
set(C_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 14)

add_executable(asr_c
        ${C_DIR}/main.cpp
        ${C_DIR}/core/c/model.cpp
        ${C_DIR}/core/c/readwav.cpp
        ${C_DIR}/core/c/reshape.cpp
        ${C_DIR}/core/c/constant.cpp
        ${C_DIR}/core/headers/constant.h
        ${C_DIR}/core/headers/model.h
        ${C_DIR}/core/headers/readwav.h
        ${C_DIR}/core/headers/reshape.h)

target_include_directories(asr_c PRIVATE
        ${C_DIR}/core/headers
        ${C_DIR}/core/c) # required all file in it