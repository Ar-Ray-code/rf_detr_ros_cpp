cmake_minimum_required(VERSION 3.16)
project(rf_detr_ros_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(OpenCV REQUIRED)

# OpenVINO support
option(RF_DETR_USE_OPENVINO "Enable OpenVINO backend" OFF)
if(RF_DETR_USE_OPENVINO)
  add_definitions(-DENABLE_OPENVINO)
endif()

include_directories(include)

set(SOURCES
  src/rf_detr_ros_cpp.cpp
)

# set(HEADERS
#   include/rf_detr_ros_cpp/rf_detr_ros_cpp.hpp
# )

include_directories(
  include
  ${CMAKE_CURRENT_BINARY_DIR}/include
)

ament_auto_add_library(${PROJECT_NAME} SHARED ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

rclcpp_components_register_nodes(${PROJECT_NAME}
  "rf_detr_ros_cpp::RfDetrNode"
)

# Install weights directory
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../weights/
#         DESTINATION share/${PROJECT_NAME}/weights)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
    launch
)