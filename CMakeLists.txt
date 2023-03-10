#Lets aim to be version 3.19 upwards
cmake_minimum_required (VERSION 3.19)

project (
    PROfit
    VERSION 0.1
    DESCRIPTION "A PROfessional PROfitable Fitter"
    LANGUAGES CXX)

include(FetchContent)


#Allow for nice support folders in IDEs like VScode
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


find_package(Doxygen)
if(Doxygen_FOUND)
    add_subdirectory(docs)
else()
    message(STATUS "Doxygen not found, not building docs")
endif()


##### Some Dependancies ######

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git 
    GIT_TAG        3.4
    )
FetchContent_MakeAvailable(eigen)


FetchContent_Declare(
    lbfgspp
    GIT_REPOSITORY https://github.com/yixuan/LBFGSpp.git 
    GIT_TAG        v0.2.0
    )
FetchContent_MakeAvailable(lbfgspp)


FetchContent_Declare(
    tinyxml2
    GIT_REPOSITORY https://github.com/leethomason/tinyxml2.git 
    GIT_TAG        9.0.0
    )
FetchContent_MakeAvailable(tinyxml2)


###### ROOT ROOT ##############33

FIND_PACKAGE(ROOT COMPONENTS MathCore MathMore)
  
if(ROOT_FOUND)
    message("Found ROOT using CMAKE FIND_PACKAGE")
    include(${ROOT_USE_FILE})
ELSE()
    message("ROOT not compiled using CMAKE, use root-config (BAD!)")
    FIND_PROGRAM(ROOT_CONFIG root-config)
    EXEC_PROGRAM(${ROOT_CONFIG} ARGS --cflags --glibs OUTPUT_VARIABLE ROOT_CXX_FLAGS)
    message("ROOT cxx flags set to: ${ROOT_CXX_FLAGS}")
ENDIF()





# The compiled library code lives inside here
add_subdirectory(src)

# The executable code will be placed here
add_subdirectory(bin)

