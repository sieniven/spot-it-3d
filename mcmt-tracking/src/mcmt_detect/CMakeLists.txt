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
	OpenCV_INCLUDE_DIR
)

add_library(${PROJECT_NAME}_lib
	"src/Hungarian.h"
	"src/Hungarian.cpp"
  "src/mcmt_detect_utils.cpp"
  "src/mcmt_detect_node.cpp"
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

set(PERCEPTION_1_EXECUTABLE "mcmt_perception_1")
set(PERCEPTION_2_EXECUTABLE "mcmt_perception_2")

add_executable(${PERCEPTION_1_EXECUTABLE} src/mcmt_perception_1.cpp)
target_link_libraries(${PERCEPTION_1_EXECUTABLE} ${PROJECT_NAME}_lib)
ament_target_dependencies(${PERCEPTION_1_EXECUTABLE} ${DEPENDENCIES})

add_executable(${PERCEPTION_2_EXECUTABLE} src/mcmt_perception_2.cpp)
target_link_libraries(${PERCEPTION_2_EXECUTABLE} ${PROJECT_NAME}_lib)
ament_target_dependencies(${PERCEPTION_2_EXECUTABLE} ${DEPENDENCIES})

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
install(TARGETS ${PERCEPTION_1_EXECUTABLE}
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS ${PERCEPTION_2_EXECUTABLE}
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_package()
