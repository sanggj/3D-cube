cmake_minimum_required(VERSION 3.16)

project(CubeChaos)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =========================================================
# EXECUTABLE
# =========================================================

add_executable(CubeChaos

    main.cpp

    # ---------------- GLAD
    external/glad/src/glad.c

    # ---------------- IMGUI CORE
    external/imgui/imgui.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/imgui_demo.cpp

    # ---------------- IMGUI BACKENDS
    external/imgui/backends/imgui_impl_glfw.cpp
    external/imgui/backends/imgui_impl_opengl3.cpp
)

# =========================================================
# INCLUDE DIRECTORIES
# =========================================================

target_include_directories(CubeChaos PRIVATE

    # ---------------- GLAD
    external/glad/include

    # ---------------- IMGUI
    external/imgui
    external/imgui/backends
)

# =========================================================
# OPENGL
# =========================================================

find_package(OpenGL REQUIRED)

# =========================================================
# GLFW
# =========================================================

add_subdirectory(external/glfw)

# =========================================================
# LINK
# =========================================================

target_link_libraries(CubeChaos PRIVATE

    glfw
    OpenGL::GL
)

# =========================================================
# WINDOWS
# =========================================================

if (WIN32)

    target_link_libraries(CubeChaos PRIVATE
        gdi32
    )

endif()