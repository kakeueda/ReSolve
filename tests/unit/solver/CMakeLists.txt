#[[

@brief Build iterative solver unit tests

]]

add_executable(runGCRODRTests.exe runGCRODRTests.cpp)
target_link_libraries(runGCRODRTests.exe PRIVATE ReSolve)

install(TARGETS runGCRODRTests.exe
        RUNTIME DESTINATION bin/resolve/tests/unit
)

add_test(NAME gcrodr_test COMMAND $<TARGET_FILE:runGCRODRTests.exe>)
