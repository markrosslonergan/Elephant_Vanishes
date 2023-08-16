add_executable(PROfit PROfit.cxx ${CMAKE_BINARY_DIR}/PROfit_dict.cxx )
target_include_directories(PROfit PRIVATE ${SBNANAOBJ_INCLUDE_DIR})
target_compile_features(PROfit PRIVATE cxx_std_14)
target_link_directories(PROfit_dict PRIVATE ${CMAKE_BINARY_DIR})
target_link_libraries(PROfit PRIVATE PROfitLib tinyxml2 eigen lbfgspp PROfit_dict ${libraries})


if(MPI_ENABLE)
    add_executable(PROmfa4u PROmfa4u.cxx ${CMAKE_BINARY_DIR}/PROfit_dict.cxx )
    target_include_directories(PROmfa4u PRIVATE ${SBNANAOBJ_INCLUDE_DIR})
    target_compile_features(PROmfa4u PRIVATE cxx_std_14)
    target_link_directories(PROfit_dict PRIVATE ${CMAKE_BINARY_DIR})
    target_link_libraries(PROmfa4u PRIVATE PROfitLib tinyxml2 eigen lbfgspp PROfit_dict ${libraries})
endif(MPI_ENABLE)




add_executable(PROreader PROreader.cxx ${CMAKE_BINARY_DIR}/PROfit_dict.cxx )
target_include_directories(PROreader PRIVATE ${SBNANAOBJ_INCLUDE_DIR})
target_compile_features(PROreader PRIVATE cxx_std_14)
target_link_directories(PROfit_dict PRIVATE ${CMAKE_BINARY_DIR})
target_link_libraries(PROreader PRIVATE PROfitLib tinyxml2 eigen lbfgspp PROfit_dict ${libraries})

add_executable(PROtest PROtest.cxx ${CMAKE_BINARY_DIR}/PROfit_dict.cxx )
target_include_directories(PROtest PRIVATE ${SBNANAOBJ_INCLUDE_DIR})
target_compile_features(PROtest PRIVATE cxx_std_14)
target_link_directories(PROfit_dict PRIVATE ${CMAKE_BINARY_DIR})
target_link_libraries(PROtest PRIVATE PROfitLib tinyxml2 eigen lbfgspp PROfit_dict ${libraries})


