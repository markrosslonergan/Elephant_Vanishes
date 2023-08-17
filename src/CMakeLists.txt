file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/inc/*.h")

if(MPI_ENABLE)
    add_library(PROfitLib PROconfig.cxx PROspec.cxx PROmfa.cxx PROtocall.cxx PROcreate.cxx PROsyst.cxx ${CMAKE_BINARY_DIR}/PROfit_dict.cxx ${HEADER_LIST})
    target_include_directories(PROfitLib PUBLIC ${PROJECT_SOURCE_DIR}/inc ${install_dir}/src/mfa/include/ ${MPI_INCLUDE_PATH} ${DIY_INCLUDE_DIR} ${FMT_INCLUDE_DIR} ${SBNANAOBJ_INCLUDE_DIR})
    target_link_libraries(PROfitLib PRIVATE eigen tinyxml2 lbfgspp PROfit_dict ${ROOT_LIBRARIES} ${libraries})
    target_compile_features(PROfitLib PUBLIC cxx_std_14)
else ()
    add_library(PROfitLib PROconfig.cxx PROspec.cxx  PROtocall.cxx PROcreate.cxx PROsyst.cxx ${CMAKE_BINARY_DIR}/PROfit_dict.cxx ${HEADER_LIST})
    target_include_directories(PROfitLib PUBLIC ${PROJECT_SOURCE_DIR}/inc  ${SBNANAOBJ_INCLUDE_DIR} ${HDF5_INCLUDE_DIR})
    target_link_libraries(PROfitLib PRIVATE eigen tinyxml2 lbfgspp PROfit_dict ${ROOT_LIBRARIES} ${libraries})
    target_compile_features(PROfitLib PUBLIC cxx_std_14)
endif(MPI_ENABLE) 

# IDEs should put the headers in a nice place
source_group(
    TREE "${PROJECT_SOURCE_DIR}/inc"
    PREFIX "Header Files"
    FILES ${HEADER_LIST})


