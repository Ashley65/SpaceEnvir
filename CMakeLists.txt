cmake_minimum_required(VERSION 3.30)
project(space LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
# Find OpenGL, GLFW, and GLEW
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(Threads REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

set(SOURCES
        src/main.cu
        src/physics/Body.cu
        include/components/Body.cuh
        include/ecs/ECS.cuh
        include/ecs/EntityManager.cuh
        include/ecs/ComponentManager.cuh
        include/ecs/SystemManager.cuh
        src/ecs/ECSCoordinator.cu
        include/physics/physicsEngine.cuh
        src/physics/physicsEngine.cu
        "include/system/SpaceshipSystem.cuh"
        include/system/PhysicsSystem.cuh
        src/ecs/ecs.cu
)

add_executable(space ${SOURCES})
# Link libraries
target_link_libraries(space PRIVATE
        OpenGL::GL
        glfw
        GLEW::GLEW
        Threads::Threads
)

set_target_properties(space PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
