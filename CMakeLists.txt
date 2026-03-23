cmake_minimum_required (VERSION 3.10)
project("bytetrack")

set(CMAKE_CXX_STANDARD 20)
# 设置为 ON 表示指定的 C++ 标准是必需的
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 设置为 OFF 表示禁用编译器特定的语言扩展
# 具体来说：
#   当设置为 OFF 时，编译器会使用纯标准的 C++ 标志，例如 -std=c++17
#   当设置为 ON（默认值）时，编译器会使用带扩展的标志，例如 -std=gnu++17
# 禁用扩展可以确保你的代码具有更好的可移植性，能够在不同的编译器之间兼容
set(CMAKE_CXX_EXTENSIONS OFF)

# 如果支持，请为 MSVC 编译器启用热重载。
if (POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

find_package(Eigen3 REQUIRED)

# 将源代码添加到此项目的可执行文件。
set(SOURCES
    main
    "main.cpp"
    "src/ByteTrack/BYTETracker.cpp"
    "src/ByteTrack/KalmanFilter.cpp"
    "src/ByteTrack/lapjv.cpp"
    "src/ByteTrack/Object.cpp"
    "src/ByteTrack/Rect.cpp"
    "src/ByteTrack/STrack.cpp"
)

# 添加可执行文件
add_executable(main ${SOURCES})

target_include_directories(main PRIVATE include)

target_link_libraries(main Eigen3::Eigen)
