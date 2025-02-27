add_executable(PROfit PROfit.cxx ${CMAKE_BINARY_DIR}/PROfit_dict.cxx )
target_include_directories(PROfit PRIVATE ${SBNANAOBJ_INCLUDE_DIR} ${HIGHFIVE_INCLUDE_DIR})
target_compile_features(PROfit PRIVATE cxx_std_17)
target_link_directories(PROfit_dict PRIVATE ${CMAKE_BINARY_DIR})
target_link_libraries(PROfit PRIVATE PROfitLib tinyxml2 eigen Boost::serialization lbfgspp PROfit_dict ${libraries})

add_executable(PROfc PROfc.cxx )
target_include_directories(PROfc PRIVATE ${HIGHFIVE_INCLUDE_DIR})
target_compile_features(PROfc PRIVATE cxx_std_17)
target_link_libraries(PROfc PRIVATE PROfitLib tinyxml2 eigen Boost::serialization lbfgspp ${libraries})

add_executable(PROsurf PROsurf.cxx )
target_include_directories(PROsurf PRIVATE ${HIGHFIVE_INCLUDE_DIR})
target_compile_features(PROsurf PRIVATE cxx_std_17)
target_link_libraries(PROsurf PRIVATE PROfitLib tinyxml2 eigen Boost::serialization lbfgspp ${libraries})

add_executable(PROmock PROmock.cxx )
target_include_directories(PROmock PRIVATE ${HIGHFIVE_INCLUDE_DIR})
target_compile_features(PROmock PRIVATE cxx_std_17)
target_link_libraries(PROmock PRIVATE PROfitLib tinyxml2 eigen Boost::serialization lbfgspp ${libraries})

add_executable(PROplot PROplot.cxx )
target_include_directories(PROplot PRIVATE ${HIGHFIVE_INCLUDE_DIR})
target_compile_features(PROplot PRIVATE cxx_std_17)
target_link_libraries(PROplot PRIVATE PROfitLib tinyxml2 eigen Boost::serialization lbfgspp ${libraries})


if(MPI_ENABLE)
    add_executable(PROmfa4u PROmfa4u.cxx)
    target_compile_features(PROmfa4u PRIVATE cxx_std_17)
    target_link_libraries(PROmfa4u PRIVATE PROfitLib tinyxml2 eigen Boost::serialization lbfgspp )
endif(MPI_ENABLE)

#add_executable(PROreader PROreader.cxx ${CMAKE_BINARY_DIR}/PROfit_dict.cxx )
#target_include_directories(PROreader PRIVATE ${SBNANAOBJ_INCLUDE_DIR} ${HIGHFIVE_INCLUDE_DIR})
#target_compile_features(PROreader PRIVATE cxx_std_14)
#target_link_directories(PROfit_dict PRIVATE ${CMAKE_BINARY_DIR})
#target_link_libraries(PROreader PRIVATE PROfitLib tinyxml2 eigen Boost::serialization lbfgspp PROfit_dict ${libraries})

    #add_executable(PROtransformer PROtransformer.cxx ${CMAKE_BINARY_DIR}/PROfit_dict.cxx )
    #message(${HIGHFIVE_INCLUDE_DIR})
    #    target_include_directories(PROtransformer PRIVATE ${SBNANAOBJ_INCLUDE_DIR} ${HDF5_INCLUDE_DIR} ${HIGHFIVE_INCLUDE_DIR})
    #    target_compile_features(PROtransformer PRIVATE cxx_std_14)
    #    target_link_directories(PROfit_dict PRIVATE ${CMAKE_BINARY_DIR})
    #    target_link_libraries(PROtransformer PRIVATE PROfitLib tinyxml2 eigen Boost::serialization lbfgspp PROfit_dict ${libraries})

add_executable(PROtest PROtest.cxx ${CMAKE_BINARY_DIR}/PROfit_dict.cxx )
target_include_directories(PROtest PRIVATE ${SBNANAOBJ_INCLUDE_DIR})
target_compile_features(PROtest PRIVATE cxx_std_17)
target_link_directories(PROfit_dict PRIVATE ${CMAKE_BINARY_DIR})
target_link_libraries(PROtest PRIVATE PROfitLib tinyxml2 eigen Boost::serialization lbfgspp PROfit_dict ${libraries})

install(TARGETS PROfit PROfc PROsurf PROmock PROplot PROtest
    LIBRARY DESTINATION ${CMAKE_BINARY_DIR})
