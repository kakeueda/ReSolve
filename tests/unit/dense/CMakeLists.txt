#[[

@brief Build small dense linear algebra unit tests

]]

add_executable(runSmallDenseLinearAlgebraTests.exe
               runSmallDenseLinearAlgebraTests.cpp
)
target_link_libraries(runSmallDenseLinearAlgebraTests.exe PRIVATE ReSolve)

install(TARGETS runSmallDenseLinearAlgebraTests.exe
        RUNTIME DESTINATION bin/resolve/tests/unit
)

add_test(NAME small_dense_linear_algebra_test
         COMMAND $<TARGET_FILE:runSmallDenseLinearAlgebraTests.exe>
)
