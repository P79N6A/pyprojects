cmake_minimum_required(VERSION 3.13)
project(test)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)
include_directories(test)
include_directories(test.xcodeproj)

add_executable(test
        test/algorithms/dp/fib.cpp
        test/algorithms/dp/serling.cpp
        test/algorithms/kmp.cpp
        test/om/1.cpp
        test/main.cpp
        test.xcodeproj/project.xcworkspace/xcshareddata/IDEWorkspaceChecks.plist
        test.xcodeproj/project.xcworkspace/xcuserdata/withheart.xcuserdatad/UserInterfaceState.xcuserstate
        test.xcodeproj/project.xcworkspace/contents.xcworkspacedata
        test.xcodeproj/xcuserdata/withheart.xcuserdatad/xcschemes/xcschememanagement.plist
        test.xcodeproj/project.pbxproj test/algorithms/sort/sorts.cpp test/algorithms/sort/sorts.h test/algorithms/offers/task_3_11.cpp test/algorithms/offers/task_3_11.h test/algorithms/binary/binary.cpp test/algorithms/binary/binary.h)
