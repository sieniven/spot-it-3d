cmake_minimum_required(VERSION 3.5)
project(mcmt_detect)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

################################################################################################
# find ament packages and libraries for ament and system dependencies
################################################################################################
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcutils REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(mcmt_msg REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

################################################################################################
# build
################################################################################################
include_directories(
  include
)

add_library(${PROJECT_NAME}_lib
  "src/Hungarian.cpp"
  "src/mcmt_detect_utils.cpp"
  "src/mcmt_params.cpp"
  "src/mcmt_processor.cpp"
  "src/mcmt_uvc_driver.cpp"
)

set(DEPENDENCIES
"rclcpp"
"rcutils"
"sensor_msgs"
"std_msgs"
"std_srvs"
"mcmt_msg"
"OpenCV"
"cv_bridge"
)

target_link_libraries(${PROJECT_NAME}_lib)
ament_target_dependencies(${PROJECT_NAME}_lib ${DEPENDENCIES})

set(PERCEPTION_EXECUTABLE "mcmt_perception")
set(PROCESSOR_EXECUTABLE "mcmt_processor")

add_executable(${PERCEPTION_EXECUTABLE} src/mcmt_perception_main.cpp)
target_link_libraries(${PERCEPTION_EXECUTABLE} ${PROJECT_NAME}_lib)
ament_target_dependencies(${PERCEPTION_EXECUTABLE} ${DEPENDENCIES})

add_executable(${PROCESSOR_EXECUTABLE} src/mcmt_processor_main.cpp)
target_link_libraries(${PROCESSOR_EXECUTABLE} ${PROJECT_NAME}_lib)
ament_target_dependencies(${PROCESSOR_EXECUTABLE} ${DEPENDENCIES})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

################################################################################################
# install
################################################################################################
install(TARGETS ${PERCEPTION_EXECUTABLE}
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS ${PROCESSOR_EXECUTABLE}
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_package()
