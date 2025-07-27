cmake_minimum_required(VERSION 3.16)
project(rf_detr_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(OpenCV REQUIRED)

# OpenVINO support
option(RF_DETR_USE_OPENVINO "Enable OpenVINO backend" OFF)
if(RF_DETR_USE_OPENVINO)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(OPENVINO REQUIRED openvino)
  add_definitions(-DENABLE_OPENVINO)
endif()

# Configure header
configure_file(
  include/rf_detr_cpp/config.h.in
  include/rf_detr_cpp/config.h
  @ONLY
)

include_directories(
  include
  ${CMAKE_CURRENT_BINARY_DIR}/include
)

set(SOURCES
  src/rf_detr_openvino.cpp
)

# set(HEADERS
#   include/rf_detr_cpp/core.hpp
#   include/rf_detr_cpp/rf_detr.hpp
#   include/rf_detr_cpp/rf_detr_openvino.hpp
#   include/rf_detr_cpp/coco_names.hpp
#   include/rf_detr_cpp/utils.hpp
# )

include_directories(
  include
  ${CMAKE_CURRENT_BINARY_DIR}/include
)

ament_auto_add_library(${PROJECT_NAME} SHARED ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

if(RF_DETR_USE_OPENVINO)
  target_link_libraries(${PROJECT_NAME} ${OPENVINO_LIBRARIES})
  target_include_directories(${PROJECT_NAME} PRIVATE ${OPENVINO_INCLUDE_DIRS})
  target_compile_options(${PROJECT_NAME} PRIVATE ${OPENVINO_CFLAGS_OTHER})
endif()

# Install configured header
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/rf_detr_cpp/config.h
        DESTINATION include/rf_detr_cpp)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()