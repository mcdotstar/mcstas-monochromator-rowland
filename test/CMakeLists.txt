set(LIB_SRC ${CMAKE_SOURCE_DIR}/mono-rowland-lib.c)
set(INC_DIRS ${CMAKE_SOURCE_DIR})

find_library(M_LIB m)
if(NOT M_LIB)
    set(M_LIB "")
endif()

function(add_unity_test name)
    add_executable(${name} ${name}.c ${LIB_SRC})
    target_include_directories(${name} PRIVATE ${INC_DIRS})
    target_link_libraries(${name} PRIVATE unity ${M_LIB})
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_unity_test(test_geometry)
