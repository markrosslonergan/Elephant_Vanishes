#Lets aim to be version 3.19 upwards
cmake_minimum_required (VERSION 3.19)

project (
    PROfit
    VERSION 0.1
    DESCRIPTION "A PROfessional PROfitable Fitter"
    LANGUAGES CXX)

include(FetchContent)
include(ExternalProject)
enable_language(C)

option(MPI_ENABLE "MPI_ENABLE" OFF)
message("MPI_ENABLE IS SET TO ${MPI_ENABLE}")

set(CMAKE_CXX_FLAGS "-g -O3 -Wall -Wextra")

#Allow for nice support folders in IDEs like VScode
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

find_package(GSL REQUIRED)

##### Some Dependencies ######

FetchContent_Declare(
  hdf5
  GIT_REPOSITORY https://github.com/HDFGroup/hdf5.git
  GIT_TAG        hdf5-1_14_0
  CMAKE_ARGS "-DBUILD_DOCS:BOOL=OFF"
)
FetchContent_MakeAvailable(hdf5)

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

FetchContent_Declare(
    SQUIDS    
    GIT_REPOSITORY https://github.com/jsalvado/SQuIDS/
    GIT_TAG        v1.3.0
    )
FetchContent_MakeAvailable(SQUIDS)

if(MPI_ENABLE)
    ExternalProject_Add(mfa
        GIT_REPOSITORY https://markrosslonergan@github.com/markrosslonergan/mfa4profit.git
        GIT_TAG 0e0cc41e027603668a9ca67b9a9195b3d6be50a3
        USES_TERMINAL_DOWNLOAD ON 
        UPDATE_COMMAND ""
        BUILD_IN_SOURCE 1
        CMAKE_FLAGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/mfa_build/
        SBNANAOBJ_INCLUDE_DIR
        INSTALL_COMMAND ""
        )
    ExternalProject_Get_Property(mfa install_dir)

    message("MFA installed to: ${install_dir}")

    set(MFA_INCLUDE_DIR ${install_dir}/src/mfa/include)
    set(DIY_INCLUDE_DIR ${MFA_INCLUDE_DIR}/diy/include)
    set(FMT_LIBRARY_PATH ${MFA_INCLUDE_DIR}/fmt)
    set(FMT_INCLUDE_DIR ${FMT_LIBRARY_PATH}/include)

    message("MFA include dir: ${MFA_INCLUDE_DIR}")
    message("DIY include dir: ${DIY_INCLUDE_DIR}")
    
    if                          (MFA_INCLUDE_DIR)
        include_directories     (SYSTEM ${MFA_INCLUDE_DIR} ${DIY_INCLUDE_DIR} ${FMT_INCLUDE_DIR})
        message                 ("MFA_INCLUDE_DIR = " ${MFA_INCLUDE_DIR})
        add_definitions         (-DMFA_NO_WEIGHTS)
        add_definitions         (-DMFA_SERIAL)
        message                 ("Weights are disabled for MFA")
    endif  ()

endif(MPI_ENABLE)

#This works, but only runs at make not cmake ..
#set_property(DIRECTORY PROPERTY EP_UPDATE_DISCONNECTED true)
#ExternalProject_Add(
#    sbnanaobj
#    GIT_REPOSITORY https://github.com/SBNSoftware/sbnanaobj.git 
#    GIT_TAG v09_20_05
#    CONFIGURE_COMMAND ""
#    BUILD_COMMAND ""
#    INSTALL_COMMAND ""
#    SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/sbnanaobj-src
#    INSTALL_DIR ${CMAKE_BINARY_DIR}/_deps/sbnanaobj-inc
#    GIT_SHALLOW ON
#)

#find_package(hdf5 REQUIRED COMPONENTS CXX C)

#this is a bit messier but works
execute_process(COMMAND git clone -b v09_20_05 https://github.com/SBNSoftware/sbnanaobj.git ${CMAKE_BINARY_DIR}/_deps/sbnanaobj-src )
set(SBNANAOBJ_INCLUDE_DIR "${CMAKE_BINARY_DIR}/_deps/sbnanaobj-src/")

find_package(Boost REQUIRED)
if(Boost_FOUND)
    include_directories (SYSTEM ${Boost_INCLUDE_DIR})
endif()

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


if(MPI_ENABLE)
    find_package(MPI REQUIRED)
    set(libraries ${libraries} ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
    message("MPI libraries: ${libraries}")
    message("MPI include path: ${MPI_INCLUDE_PATH}")
endif(MPI_ENABLE)


file(GLOB SRHeaders ${SBNANAOBJ_INCLUDE_DIR}/sbnanaobj/StandardRecord/*.h)
execute_process(COMMAND rootcling -f ${CMAKE_BINARY_DIR}/PROfit_dict.cxx -I${SBNANAOBJ_INCLUDE_DIR} -c ${SRHeaders} ${CMAKE_SOURCE_DIR}/dict/LinkDef.h)
add_library(PROfit_dict ${CMAKE_BINARY_DIR}/PROfit_dict.cxx)
target_include_directories(PROfit_dict PRIVATE ${SBNANAOBJ_INCLUDE_DIR})
target_link_directories(PROfit_dict PRIVATE ${CMAKE_BINARY_DIR})
target_link_libraries(PROfit_dict PRIVATE ${ROOT_LIBRARIES})

# The compiled library code lives inside here
add_subdirectory(src)

# The executable code will be placed here
add_subdirectory(bin)


