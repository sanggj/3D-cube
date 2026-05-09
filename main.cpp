#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "external/imgui/imgui.h"
#include "external/imgui/backends/imgui_impl_glfw.h"
#include "external/imgui/backends/imgui_impl_opengl3.h"

#include <vector>
#include <thread>
#include <random>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <new>
#include <chrono>

struct Mat4 {
    float m[16]{};
};

struct Instance {
    float x;
    float y;
    float z;
    float seed;
};

// =========================================================
// MATRIX
// =========================================================

static Mat4 identity() {

    Mat4 r{};

    r.m[0] = 1.0f;
    r.m[5] = 1.0f;
    r.m[10] = 1.0f;
    r.m[15] = 1.0f;

    return r;
}

static Mat4 mul(
    const Mat4& a,
    const Mat4& b
) {

    Mat4 r{};

    for (int c = 0; c < 4; ++c) {

        for (int row = 0; row < 4; ++row) {

            r.m[c * 4 + row] =

                a.m[0 * 4 + row] * b.m[c * 4 + 0] +
                a.m[1 * 4 + row] * b.m[c * 4 + 1] +
                a.m[2 * 4 + row] * b.m[c * 4 + 2] +
                a.m[3 * 4 + row] * b.m[c * 4 + 3];
        }
    }

    return r;
}

static Mat4 perspective(
    float fov,
    float aspect,
    float n,
    float f
) {

    float t =
        std::tan(
            fov *
            0.5f *
            3.1415926535f /
            180.0f
        );

    Mat4 r{};

    r.m[0] = 1.0f / (aspect * t);
    r.m[5] = 1.0f / t;

    r.m[10] = -(f + n) / (f - n);
    r.m[11] = -1.0f;

    r.m[14] =
        -(2.0f * f * n) /
        (f - n);

    return r;
}

static Mat4 translate(
    float x,
    float y,
    float z
) {

    Mat4 r = identity();

    r.m[12] = x;
    r.m[13] = y;
    r.m[14] = z;

    return r;
}

static Mat4 rotX(float a) {

    Mat4 r = identity();

    float c = std::cos(a);
    float s = std::sin(a);

    r.m[5] = c;
    r.m[9] = -s;

    r.m[6] = s;
    r.m[10] = c;

    return r;
}

static Mat4 rotY(float a) {

    Mat4 r = identity();

    float c = std::cos(a);
    float s = std::sin(a);

    r.m[0] = c;
    r.m[8] = s;

    r.m[2] = -s;
    r.m[10] = c;

    return r;
}

// =========================================================
// CONFIG
// =========================================================

static int W = 1280;
static int H = 720;

static int INST = 34356;

static float SPACING = 4.0f;

static float ROT_SENS = 0.01f;

static float ZOOM_BASE = 120.0f;
static float ZOOM_SPEED = 0.15f;

static float FOV = 70.0f;

static float NEAR_Z = 0.1f;
static float FAR_Z = 10000.0f;

static bool enableCulling = true;
static bool wireframe = false;

static bool randomSeedEveryRebuild = true;

static int aaSamples = 0;

// =========================================================
// CAMERA
// =========================================================

static double yaw = 0.0;
static double pitch = 0.0;

static bool drag = false;

static double lx = 0.0;
static double ly = 0.0;

static double zoomExp = 0.0;

// =========================================================
// FPS
// =========================================================

static double fps = 0.0;
static double fpsAcc = 0.0;

static int fpsCount = 0;

static double lastTime = 0.0;

// =========================================================

static bool rebuildInstances = false;

static char cubeCountBuffer[64] =
"34356";

static float getDist() {

    return
        ZOOM_BASE *
        std::exp(zoomExp);
}

// =========================================================
// SHADER
// =========================================================

static GLuint compileShader(
    GLenum type,
    const char* src
) {

    GLuint s =
        glCreateShader(type);

    glShaderSource(
        s,
        1,
        &src,
        nullptr
    );

    glCompileShader(s);

    GLint ok = 0;

    glGetShaderiv(
        s,
        GL_COMPILE_STATUS,
        &ok
    );

    if (!ok) {

        char log[8192];

        glGetShaderInfoLog(
            s,
            sizeof(log),
            nullptr,
            log
        );

        std::printf(
            "%s\n",
            log
        );

        std::exit(1);
    }

    return s;
}

static GLuint linkProgram(
    GLuint vs,
    GLuint fs
) {

    GLuint p =
        glCreateProgram();

    glAttachShader(p, vs);
    glAttachShader(p, fs);

    glLinkProgram(p);

    GLint ok = 0;

    glGetProgramiv(
        p,
        GL_LINK_STATUS,
        &ok
    );

    if (!ok) {

        char log[8192];

        glGetProgramInfoLog(
            p,
            sizeof(log),
            nullptr,
            log
        );

        std::printf(
            "%s\n",
            log
        );

        std::exit(1);
    }

    glDeleteShader(vs);
    glDeleteShader(fs);

    return p;
}

// =========================================================
// INSTANCE BUILD
// =========================================================

static std::vector<Instance>
buildInstances(int n) {

    std::vector<Instance> out(n);

    int side =
        (int)std::ceil(
            std::cbrt((double)n)
        );

    float half =
        (side - 1) * 0.5f;

    unsigned threads =
        std::max(
            1u,
            std::thread::hardware_concurrency()
        );

    int chunk =
        (n + (int)threads - 1) /
        (int)threads;

    uint32_t randomBase =
        randomSeedEveryRebuild
        ?
        (uint32_t)
        std::chrono::high_resolution_clock::now()
        .time_since_epoch()
        .count()
        :
        1337u;

    auto worker =
        [&](int begin, int end, uint32_t seedBase)
        {
            std::mt19937 rng(seedBase);

            std::uniform_real_distribution<float>
                jitter(-0.25f, 0.25f);

            std::uniform_real_distribution<float>
                seedDist(0.0f, 1000000.0f);

            for (
                int i = begin;
                i < end && i < n;
                ++i
                ) {

                int z =
                    i / (side * side);

                int rem =
                    i % (side * side);

                int y =
                    rem / side;

                int x =
                    rem % side;

                out[i].x =
                    (x - half) *
                    SPACING +
                    jitter(rng);

                out[i].y =
                    (y - half) *
                    SPACING +
                    jitter(rng);

                out[i].z =
                    (z - half) *
                    SPACING +
                    jitter(rng);

                out[i].seed =
                    seedDist(rng);
            }
        };

    std::vector<std::thread> pool;

    for (
        unsigned t = 0;
        t < threads;
        ++t
        ) {

        int begin =
            (int)t * chunk;

        int end =
            begin + chunk;

        pool.emplace_back(
            worker,
            begin,
            end,
            randomBase + t * 99991u
        );
    }

    for (auto& th : pool)
        th.join();

    return out;
}

// =========================================================
// MAIN
// =========================================================

int main() {

    if (!glfwInit())
        return 1;

    glfwWindowHint(
        GLFW_CONTEXT_VERSION_MAJOR,
        4
    );

    glfwWindowHint(
        GLFW_CONTEXT_VERSION_MINOR,
        6
    );

    glfwWindowHint(
        GLFW_OPENGL_PROFILE,
        GLFW_OPENGL_CORE_PROFILE
    );

    glfwWindowHint(
        GLFW_SAMPLES,
        aaSamples
    );

    GLFWwindow* win =
        glfwCreateWindow(
            W,
            H,
            "CubeChaos",
            nullptr,
            nullptr
        );

    if (!win)
        return 1;

    glfwMakeContextCurrent(win);

    glfwSwapInterval(0);

    if (
        !gladLoadGLLoader(
            (GLADloadproc)
            glfwGetProcAddress
        )
        ) {
        return 1;
    }

    // =====================================================
    // IMGUI
    // =====================================================

    IMGUI_CHECKVERSION();

    ImGui::CreateContext();

    ImGuiIO& io =
        ImGui::GetIO();

    io.ConfigFlags |=
        ImGuiConfigFlags_NavEnableKeyboard;

    ImGui_ImplGlfw_InitForOpenGL(
        win,
        false
    );

    ImGui_ImplOpenGL3_Init(
        "#version 460"
    );

    // =====================================================
    // OPENGL
    // =====================================================

    glEnable(GL_DEPTH_TEST);

    const char* vsSrc = R"GLSL(

#version 460 core

layout(location=0) in vec3 in_pos;
layout(location=1) in vec4 in_inst;

uniform mat4 mvp;
uniform float time;

out vec3 v_col;

float hash(float n){
    return fract(sin(n)*43758.5453);
}

vec3 hsv2rgb(vec3 c){

    vec4 K = vec4(
        1.0,
        2.0/3.0,
        1.0/3.0,
        3.0
    );

    vec3 p = abs(
        fract(c.xxx + K.xyz)
        * 6.0 - K.www
    );

    return c.z * mix(
        K.xxx,
        clamp(
            p - K.xxx,
            0.0,
            1.0
        ),
        c.y
    );
}

mat3 rot(vec3 a, float ang){

    a = normalize(a);

    float c = cos(ang);
    float s = sin(ang);

    float oc = 1.0 - c;

    return mat3(

        oc*a.x*a.x+c,
        oc*a.x*a.y-a.z*s,
        oc*a.z*a.x+a.y*s,

        oc*a.x*a.y+a.z*s,
        oc*a.y*a.y+c,
        oc*a.y*a.z-a.x*s,

        oc*a.z*a.x-a.y*s,
        oc*a.y*a.z+a.x*s,
        oc*a.z*a.z+c
    );
}

void main(){

    vec3 p = in_pos;

    float seed = in_inst.w;

    float sec = floor(time);

    float r1 = hash(seed + sec * 1.37);
    float r2 = hash(seed + sec * 2.91);
    float r3 = hash(seed + sec * 4.73);
    float r4 = hash(seed + sec * 7.11);

    vec3 axis = normalize(vec3(
        r1 * 2.0 - 1.0,
        r2 * 2.0 - 1.0,
        r3 * 2.0 - 1.0
    ));

    float dir =
        mix(
            -1.0,
            1.0,
            step(0.5,r4)
        );

    float speed =
        0.2 + r2 * 10.0;

    float jump =
        r3 * 6.2831853;

    float ang =
        time *
        speed *
        dir +
        jump;

    p = rot(axis, ang) * p;

    p += in_inst.xyz;

    float hue =
        fract(
            hash(seed) +
            time * 0.05 +
            r1 * 0.3
        );

    v_col =
        hsv2rgb(
            vec3(
                hue,
                0.9,
                1.0
            )
        );

    gl_Position =
        mvp *
        vec4(p,1.0);
}

)GLSL";

    const char* fsSrc = R"GLSL(

#version 460 core

in vec3 v_col;

out vec4 fragColor;

void main(){

    fragColor =
        vec4(v_col,1.0);
}

)GLSL";

    GLuint prog =
        linkProgram(
            compileShader(
                GL_VERTEX_SHADER,
                vsSrc
            ),
            compileShader(
                GL_FRAGMENT_SHADER,
                fsSrc
            )
        );

    float verts[] = {

        -1,-1,-1,
         1,-1,-1,
         1, 1,-1,
        -1, 1,-1,

        -1,-1, 1,
         1,-1, 1,
         1, 1, 1,
        -1, 1, 1
    };

    uint32_t inds[] = {

        0,1,2,0,2,3,
        4,5,6,4,6,7,
        0,1,5,0,5,4,
        2,3,7,2,7,6,
        1,2,6,1,6,5,
        0,3,7,0,7,4
    };

    GLuint vao;
    GLuint vbo;
    GLuint ebo;
    GLuint instVbo;

    glGenVertexArrays(1, &vao);

    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);

    glBindBuffer(
        GL_ARRAY_BUFFER,
        vbo
    );

    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(verts),
        verts,
        GL_DYNAMIC_DRAW
    );

    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        3 * sizeof(float),
        (void*)0
    );

    glEnableVertexAttribArray(0);

    glGenBuffers(1, &ebo);

    glBindBuffer(
        GL_ELEMENT_ARRAY_BUFFER,
        ebo
    );

    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER,
        sizeof(inds),
        inds,
        GL_DYNAMIC_DRAW
    );

    glGenBuffers(1, &instVbo);

    std::vector<Instance>
        instances =
        buildInstances(INST);

    glBindBuffer(
        GL_ARRAY_BUFFER,
        instVbo
    );

    glBufferData(
        GL_ARRAY_BUFFER,
        instances.size()
        * sizeof(Instance),
        instances.data(),
        GL_DYNAMIC_DRAW
    );

    glVertexAttribPointer(
        1,
        4,
        GL_FLOAT,
        GL_FALSE,
        sizeof(Instance),
        (void*)0
    );

    glEnableVertexAttribArray(1);

    glVertexAttribDivisor(1, 1);

    GLint mvpLoc =
        glGetUniformLocation(
            prog,
            "mvp"
        );

    GLint timeLoc =
        glGetUniformLocation(
            prog,
            "time"
        );

    // =====================================================
    // INPUT
    // =====================================================

    glfwSetMouseButtonCallback(
        win,
        [](GLFWwindow* window,
            int button,
            int action,
            int mods)
        {
            ImGui_ImplGlfw_MouseButtonCallback(
                window,
                button,
                action,
                mods
            );

            ImGuiIO& io =
                ImGui::GetIO();

            if (io.WantCaptureMouse)
                return;

            if (
                button ==
                GLFW_MOUSE_BUTTON_LEFT
                ) {

                drag =
                    (action ==
                        GLFW_PRESS);

                glfwGetCursorPos(
                    window,
                    &lx,
                    &ly
                );
            }
        }
    );

    glfwSetCursorPosCallback(
        win,
        [](GLFWwindow* window,
            double x,
            double y)
        {
            ImGui_ImplGlfw_CursorPosCallback(
                window,
                x,
                y
            );

            ImGuiIO& io =
                ImGui::GetIO();

            if (io.WantCaptureMouse) {

                lx = x;
                ly = y;

                return;
            }

            if (!drag)
                return;

            double dx =
                x - lx;

            double dy =
                y - ly;

            lx = x;
            ly = y;

            yaw += dx * ROT_SENS;

            pitch += dy * ROT_SENS;
        }
    );

    glfwSetScrollCallback(
        win,
        [](GLFWwindow* window,
            double xoff,
            double yoff)
        {
            ImGui_ImplGlfw_ScrollCallback(
                window,
                xoff,
                yoff
            );

            ImGuiIO& io =
                ImGui::GetIO();

            if (io.WantCaptureMouse)
                return;

            zoomExp -=
                yoff *
                ZOOM_SPEED;

            if (zoomExp < -40.0)
                zoomExp = -40.0;

            if (zoomExp > 40.0)
                zoomExp = 40.0;
        }
    );

    glfwSetKeyCallback(
        win,
        [](GLFWwindow* window,
            int key,
            int scancode,
            int action,
            int mods)
        {
            ImGui_ImplGlfw_KeyCallback(
                window,
                key,
                scancode,
                action,
                mods
            );
        }
    );

    glfwSetCharCallback(
        win,
        [](GLFWwindow* window,
            unsigned int c)
        {
            ImGui_ImplGlfw_CharCallback(
                window,
                c
            );
        }
    );

    // =====================================================
    // LOOP
    // =====================================================

    lastTime =
        glfwGetTime();

    while (
        !glfwWindowShouldClose(win)
        ) {

        glfwPollEvents();

        if (rebuildInstances) {

            if (INST < 1)
                INST = 1;

            try {

                std::vector<Instance>
                    newInstances =
                    buildInstances(INST);

                instances.swap(
                    newInstances
                );

                glBindBuffer(
                    GL_ARRAY_BUFFER,
                    instVbo
                );

                glBufferData(
                    GL_ARRAY_BUFFER,
                    instances.size() *
                    sizeof(Instance),
                    instances.data(),
                    GL_DYNAMIC_DRAW
                );

                std::snprintf(
                    cubeCountBuffer,
                    sizeof(cubeCountBuffer),
                    "%d",
                    INST
                );
            }
            catch (
                const std::bad_alloc&
                ) {

                std::printf(
                    "OUT OF MEMORY\n"
                );

                INST = 100000;

                instances =
                    buildInstances(INST);

                glBindBuffer(
                    GL_ARRAY_BUFFER,
                    instVbo
                );

                glBufferData(
                    GL_ARRAY_BUFFER,
                    instances.size() *
                    sizeof(Instance),
                    instances.data(),
                    GL_DYNAMIC_DRAW
                );
            }

            rebuildInstances = false;
        }

        double now =
            glfwGetTime();

        double dt =
            now - lastTime;

        lastTime = now;

        fpsAcc += dt;

        fpsCount++;

        if (fpsAcc >= 1.0) {

            fps =
                fpsCount /
                fpsAcc;

            fpsAcc = 0.0;

            fpsCount = 0;
        }

        // =================================================
        // IMGUI
        // =================================================

        ImGui_ImplOpenGL3_NewFrame();

        ImGui_ImplGlfw_NewFrame();

        ImGui::NewFrame();

        ImGui::SetNextWindowSize(
            ImVec2(430, 0),
            ImGuiCond_Once
        );

        ImGui::Begin(
            "Settings"
        );

        ImGui::PushItemWidth(220);

        if (
            ImGui::InputText(
                "Cube Count",
                cubeCountBuffer,
                sizeof(cubeCountBuffer),
                ImGuiInputTextFlags_CharsDecimal
            )
            ) {

            int newCount =
                std::atoi(
                    cubeCountBuffer
                );

            if (newCount < 1)
                newCount = 1;

            if (newCount != INST) {

                INST = newCount;

                rebuildInstances = true;
            }
        }

        if (
            ImGui::SliderFloat(
                "Spacing",
                &SPACING,
                1.0f,
                20.0f
            )
            ) {
            rebuildInstances = true;
        }

        ImGui::Checkbox(
            "Backface Culling",
            &enableCulling
        );

        ImGui::Checkbox(
            "Wireframe",
            &wireframe
        );

        ImGui::Checkbox(
            "Random Seed Rebuild",
            &randomSeedEveryRebuild
        );

        ImGui::Separator();

        ImGui::Text(
            "AA Samples: %d",
            aaSamples
        );

        if (ImGui::Button("AA OFF")) {
            aaSamples = 0;
        }

        ImGui::SameLine();

        if (ImGui::Button("2X")) {
            aaSamples = 2;
        }

        ImGui::SameLine();

        if (ImGui::Button("4X")) {
            aaSamples = 4;
        }

        ImGui::SameLine();

        if (ImGui::Button("8X")) {
            aaSamples = 8;
        }

        ImGui::Separator();

        ImGui::Text(
            "FPS: %.1f",
            fps
        );

        ImGui::Text(
            "Instances: %d",
            INST
        );

        ImGui::Text(
            "CPU Threads: %u",
            std::thread::hardware_concurrency()
        );

        ImGui::Text(
            "Change with caution!!"
        );

        ImGui::PopItemWidth();

        ImGui::End();

        // =================================================
        // RENDER
        // =================================================

        if (enableCulling)
            glEnable(GL_CULL_FACE);
        else
            glDisable(GL_CULL_FACE);

        if (wireframe)
            glPolygonMode(
                GL_FRONT_AND_BACK,
                GL_LINE
            );
        else
            glPolygonMode(
                GL_FRONT_AND_BACK,
                GL_FILL
            );

        int fbW;
        int fbH;

        glfwGetFramebufferSize(
            win,
            &fbW,
            &fbH
        );

        glViewport(
            0,
            0,
            fbW,
            fbH
        );

        glClearColor(
            0.02f,
            0.02f,
            0.03f,
            1.0f
        );

        glClear(
            GL_COLOR_BUFFER_BIT |
            GL_DEPTH_BUFFER_BIT
        );

        Mat4 P =
            perspective(
                FOV,
                (float)fbW /
                (float)fbH,
                NEAR_Z,
                FAR_Z
            );

        Mat4 V =
            translate(
                0.0f,
                0.0f,
                -getDist()
            );

        Mat4 M =
            mul(
                rotY((float)yaw),
                rotX((float)pitch)
            );

        Mat4 MVP =
            mul(
                P,
                mul(V, M)
            );

        glUseProgram(prog);

        glUniformMatrix4fv(
            mvpLoc,
            1,
            GL_FALSE,
            MVP.m
        );

        glUniform1f(
            timeLoc,
            (float)now
        );

        glBindVertexArray(vao);

        glDrawElementsInstanced(
            GL_TRIANGLES,
            36,
            GL_UNSIGNED_INT,
            nullptr,
            INST
        );

        // =================================================
        // IMGUI DRAW
        // =================================================

        ImGui::Render();

        ImGui_ImplOpenGL3_RenderDrawData(
            ImGui::GetDrawData()
        );

        char title[256];

        std::snprintf(
            title,
            sizeof(title),
            "FPS: %.1f | INST: %d",
            fps,
            INST
        );

        glfwSetWindowTitle(
            win,
            title
        );

        glfwSwapBuffers(win);
    }

    // =====================================================
    // CLEANUP
    // =====================================================

    ImGui_ImplOpenGL3_Shutdown();

    ImGui_ImplGlfw_Shutdown();

    ImGui::DestroyContext();

    glDeleteBuffers(1, &instVbo);
    glDeleteBuffers(1, &ebo);
    glDeleteBuffers(1, &vbo);

    glDeleteVertexArrays(1, &vao);

    glDeleteProgram(prog);

    glfwDestroyWindow(win);

    glfwTerminate();

    return 0;
}