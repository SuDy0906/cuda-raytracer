// =============================================================================
// cuda_raytracer.cu — Phases 2 + 3 + 4
//
// Phase 2: one CUDA thread per pixel, 2D grid, cudaMalloc/cudaMemcpy, error checks
// Phase 3: curand anti-aliasing, iterative ray bouncing, Lambertian/Metal/Glass
// Phase 4: scene in __constant__ memory, block-size benchmark (8×8, 16×16, 32×32)
//
// Build:
//   nvcc -std=c++17 -O3 --use_fast_math -arch=sm_75 \
//        -Isrc -o cuda_raytracer src/cuda_raytracer.cu -lcurand
//   (or via CMake — see CMakeLists.txt)
//
// Usage:
//   ./cuda_raytracer [--samples N] [--bounces N] [--width W] [--height H]
// =============================================================================

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "material.h"

// ── Compile-time limits ───────────────────────────────────────────────────────
#define MAX_SPHERES 64

// ── CUDA error-check macro ────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            std::fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n",                   \
                         __FILE__, __LINE__, cudaGetErrorString(_err));         \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// ── Constant memory — must hold a trivially-constructible type ───────────────
// CUDA 13 rejects __constant__ arrays whose element type has any user-defined
// constructor (even in a member like vec3). We mirror Sphere as a flat-float
// struct (CSphere) and reconstruct a real Sphere in device code.
struct CSphere {
    float cx, cy, cz;   // center
    float radius;
    // material
    int   mat_type;
    float albedo_r, albedo_g, albedo_b;
    float fuzz;
    float ior;
};

__constant__ CSphere d_spheres[MAX_SPHERES];
__constant__ int     d_num_spheres;

// Convert CSphere → Sphere for intersection tests
__device__ inline Sphere csphere_to_sphere(const CSphere& cs) {
    Sphere s = {};
    s.center = vec3(cs.cx, cs.cy, cs.cz);
    s.radius = cs.radius;
    s.mat    = Material::lambertian(vec3(cs.albedo_r, cs.albedo_g, cs.albedo_b)); // overwritten below
    s.mat.type   = static_cast<MaterialType>(cs.mat_type);
    s.mat.albedo = vec3(cs.albedo_r, cs.albedo_g, cs.albedo_b);
    s.mat.fuzz   = cs.fuzz;
    s.mat.ior    = cs.ior;
    return s;
}

// =============================================================================
// Camera (host-constructed, passed by value to the kernel)
// =============================================================================
struct Camera {
    vec3 origin;
    vec3 lower_left;
    vec3 horizontal;
    vec3 vertical;

    // Built on the host, the struct is trivially copyable into kernel args
    Camera() {}
    Camera(vec3 from, vec3 at, vec3 up, float vfov_deg, float aspect) {
        float theta = vfov_deg * 3.14159265f / 180.0f;
        float h     = tanf(theta / 2.0f);
        float vh    = 2.0f * h;
        float vw    = aspect * vh;

        vec3 w = unit_vector(from - at);
        vec3 u = unit_vector(cross(up, w));
        vec3 v = cross(w, u);

        origin     = from;
        horizontal = vw * u;
        vertical   = vh * v;
        lower_left = origin - horizontal / 2.0f - vertical / 2.0f - w;
    }

    __host__ __device__ Ray get_ray(float s, float t) const {
        return Ray(origin, lower_left + s * horizontal + t * vertical - origin);
    }
};

// =============================================================================
// Device helpers — random number generation via curand
// =============================================================================

// Random point inside the unit sphere (rejection sampling)
__device__ inline vec3 random_in_unit_sphere(curandState* st) {
    vec3 p;
    do {
        p = vec3(curand_uniform(st) * 2.0f - 1.0f,
                 curand_uniform(st) * 2.0f - 1.0f,
                 curand_uniform(st) * 2.0f - 1.0f);
    } while (p.length_squared() >= 1.0f);
    return p;
}

__device__ inline vec3 random_unit_vector(curandState* st) {
    return unit_vector(random_in_unit_sphere(st));
}

// =============================================================================
// Material scatter (device)
// Switch on MaterialType — no virtual dispatch, GPU-friendly
// =============================================================================
__device__ float schlick(float cosine, float ior) {
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 *= r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosine, 5.0f);
}

__device__ bool scatter(const Ray& r_in, const HitRecord& rec,
                        vec3& attenuation, Ray& scattered,
                        curandState* st) {
    switch (rec.mat.type) {

        case LAMBERTIAN: {
            vec3 dir = rec.normal + random_unit_vector(st);
            if (dir.near_zero()) dir = rec.normal;
            scattered   = Ray(rec.p, dir);
            attenuation = rec.mat.albedo;
            return true;
        }

        case METAL: {
            vec3 ref  = reflect(unit_vector(r_in.direction), rec.normal);
            scattered = Ray(rec.p, ref + rec.mat.fuzz * random_in_unit_sphere(st));
            attenuation = rec.mat.albedo;
            return dot(scattered.direction, rec.normal) > 0.0f;
        }

        case GLASS: {
            attenuation     = vec3(1.0f, 1.0f, 1.0f);
            float ratio     = rec.front_face ? (1.0f / rec.mat.ior) : rec.mat.ior;
            vec3  unit_dir  = unit_vector(r_in.direction);
            float cos_theta = fminf(dot(-unit_dir, rec.normal), 1.0f);
            float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
            vec3  dir;
            if (ratio * sin_theta > 1.0f || schlick(cos_theta, ratio) > curand_uniform(st))
                dir = reflect(unit_dir, rec.normal);
            else
                dir = refract(unit_dir, rec.normal, ratio);
            scattered = Ray(rec.p, dir);
            return true;
        }
    }
    return false;
}

// =============================================================================
// World hit — traverses constant-memory sphere list
// =============================================================================
__device__ bool world_hit(const Ray& r, float t_min, float t_max, HitRecord& rec) {
    HitRecord tmp;
    bool  hit_any = false;
    float closest = t_max;

    for (int i = 0; i < d_num_spheres; ++i) {
        Sphere s = csphere_to_sphere(d_spheres[i]);
        if (s.hit(r, t_min, closest, tmp)) {
            hit_any = true;
            closest = tmp.t;
            rec     = tmp;
        }
    }
    return hit_any;
}

// =============================================================================
// Ray colour — iterative (avoids GPU call-stack depth limits)
// =============================================================================
__device__ vec3 ray_color(Ray r, int max_bounces, curandState* st) {
    vec3 throughput(1.0f, 1.0f, 1.0f);

    for (int b = 0; b < max_bounces; ++b) {
        HitRecord rec;

        if (world_hit(r, 0.001f, 1e9f, rec)) {
            vec3 attenuation;
            Ray  scattered;
            if (scatter(r, rec, attenuation, scattered, st)) {
                throughput = throughput * attenuation;
                r          = scattered;
            } else {
                return vec3(0.0f, 0.0f, 0.0f);  // fully absorbed
            }
        } else {
            // Miss — sky gradient
            vec3  ud = unit_vector(r.direction);
            float t  = 0.5f * (ud.y + 1.0f);
            vec3  sky = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f)
                      + t           * vec3(0.5f, 0.7f, 1.0f);
            return throughput * sky;
        }
    }
    return vec3(0.0f, 0.0f, 0.0f);  // max bounces exceeded
}

// =============================================================================
// Kernel 1: initialise per-thread curand state
// =============================================================================
__global__ void init_curand_kernel(curandState* states,
                                   unsigned long long seed,
                                   int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    curand_init(seed, (unsigned long long)(y * w + x), 0, &states[y * w + x]);
}

// =============================================================================
// Kernel 2: render — one thread per pixel
// =============================================================================
__global__ void render_kernel(vec3* fb, int w, int h,
                              int samples, int max_bounces,
                              Camera cam, curandState* states) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int          idx = y * w + x;
    curandState* st  = &states[idx];

    vec3 color(0.0f, 0.0f, 0.0f);
    for (int s = 0; s < samples; ++s) {
        float u = (x + curand_uniform(st)) / float(w - 1);
        float v = (y + curand_uniform(st)) / float(h - 1);
        color  += ray_color(cam.get_ray(u, v), max_bounces, st);
    }
    color /= float(samples);

    // Gamma correction γ=2
    color.x = sqrtf(fmaxf(0.0f, color.x));
    color.y = sqrtf(fmaxf(0.0f, color.y));
    color.z = sqrtf(fmaxf(0.0f, color.z));

    fb[idx] = color;
}

// =============================================================================
// Host helpers
// =============================================================================

// Build the same scene as the CPU version — returns flat CSphere array
std::vector<CSphere> build_scene() {
    std::vector<CSphere> spheres;

    auto add = [&](float cx, float cy, float cz, float r, Material m) {
        CSphere cs;
        cs.cx = cx; cs.cy = cy; cs.cz = cz;
        cs.radius   = r;
        cs.mat_type = static_cast<int>(m.type);
        cs.albedo_r = m.albedo.x;
        cs.albedo_g = m.albedo.y;
        cs.albedo_b = m.albedo.z;
        cs.fuzz     = m.fuzz;
        cs.ior      = m.ior;
        spheres.push_back(cs);
    };

    add( 0,-1000, 0, 1000.0f, Material::lambertian(vec3(0.48f,0.48f,0.48f)));
    add( 0,    1, 0,   1.0f, Material::glass(1.5f));
    add(-4,    1, 0,   1.0f, Material::lambertian(vec3(0.4f,0.2f,0.1f)));
    add( 4,    1, 0,   1.0f, Material::metal(vec3(0.7f,0.6f,0.5f), 0.0f));

    struct Spec { float x,z,r,g,b; int t; float fuzz,ior; };
    static const Spec specs[] = {
        {-2,-2, 0.80f,0.30f,0.30f, 0, 0.00f,0.0f},
        { 2,-2, 0.10f,0.60f,0.90f, 1, 0.10f,0.0f},
        {-1, 2, 0.30f,0.70f,0.30f, 0, 0.00f,0.0f},
        { 1, 2, 0.90f,0.80f,0.10f, 1, 0.30f,0.0f},
        {-3, 1, 0.40f,0.40f,0.90f, 2, 0.00f,1.7f},
        { 3, 1, 0.90f,0.20f,0.20f, 0, 0.00f,0.0f},
        { 0,-3, 0.70f,0.70f,0.20f, 1, 0.00f,0.0f},
        {-2, 3, 0.20f,0.90f,0.80f, 0, 0.00f,0.0f},
        { 2, 3, 0.50f,0.50f,0.50f, 1, 0.50f,0.0f},
        { 0, 4, 1.00f,1.00f,1.00f, 2, 0.00f,1.5f},
        {-4, 3, 0.60f,0.30f,0.10f, 0, 0.00f,0.0f},
        { 4, 3, 0.20f,0.40f,0.80f, 1, 0.20f,0.0f},
    };
    for (const auto& s : specs) {
        Material m;
        switch (s.t) {
            case 0: m = Material::lambertian(vec3(s.r,s.g,s.b));      break;
            case 1: m = Material::metal(vec3(s.r,s.g,s.b), s.fuzz);   break;
            case 2: m = Material::glass(s.ior);                        break;
            default: m = Material::lambertian(vec3(1,0,1));            break;
        }
        add(s.x, 0.3f, s.z, 0.3f, m);
    }
    return spheres;
}

void write_ppm(const std::string& path, const vec3* fb, int w, int h) {
    std::ofstream f(path);
    f << "P3\n" << w << ' ' << h << "\n255\n";
    for (int j = h - 1; j >= 0; --j) {
        for (int i = 0; i < w; ++i) {
            const vec3& c = fb[j * w + i];
            f << int(255.99f * fmaxf(0.0f, fminf(1.0f, c.x))) << ' '
              << int(255.99f * fmaxf(0.0f, fminf(1.0f, c.y))) << ' '
              << int(255.99f * fmaxf(0.0f, fminf(1.0f, c.z))) << '\n';
        }
    }
    f.close();
    std::printf("  Saved: %s\n", path.c_str());
}

// Run one timed render with a given block size.
// Returns elapsed milliseconds.
double benchmark_block_size(int bx, int by,
                            int w, int h,
                            int samples, int max_bounces,
                            const Camera& cam,
                            curandState* d_states,
                            vec3*        d_fb,
                            const std::string& save_path = "") {
    dim3 blocks( (w + bx - 1) / bx, (h + by - 1) / by );
    dim3 threads(bx, by);

    // Re-seed curand (same seed → deterministic output per run)
    init_curand_kernel<<<blocks, threads>>>(d_states, 42ULL, w, h);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 1-sample warmup to pay JIT / driver overhead once
    render_kernel<<<blocks, threads>>>(d_fb, w, h, 1, max_bounces, cam, d_states);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed render
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));

    render_kernel<<<blocks, threads>>>(d_fb, w, h, samples, max_bounces, cam, d_states);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    // Optionally save the frame
    if (!save_path.empty()) {
        std::vector<vec3> h_fb(w * h);
        CUDA_CHECK(cudaMemcpy(h_fb.data(), d_fb,
                              (size_t)w * h * sizeof(vec3),
                              cudaMemcpyDeviceToHost));
        write_ppm(save_path, h_fb.data(), w, h);
    }

    return double(ms);
}

// =============================================================================
// main
// =============================================================================
int main(int argc, char** argv) {
    // ── Parse optional CLI args ───────────────────────────────────────────────
    int W       = 800;
    int H       = 600;
    int SAMPLES = 128;
    int BOUNCES = 10;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--samples") == 0 && i+1 < argc) SAMPLES = atoi(argv[++i]);
        if (strcmp(argv[i], "--bounces") == 0 && i+1 < argc) BOUNCES = atoi(argv[++i]);
        if (strcmp(argv[i], "--width")   == 0 && i+1 < argc) W       = atoi(argv[++i]);
        if (strcmp(argv[i], "--height")  == 0 && i+1 < argc) H       = atoi(argv[++i]);
    }

    std::filesystem::create_directories("renders");

    // ── GPU info ──────────────────────────────────────────────────────────────
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("=== CUDA Ray Tracer ===\n");
    std::printf("GPU        : %s  (SM %d.%d,  %d MPs,  %.0f MB VRAM)\n",
                prop.name, prop.major, prop.minor,
                prop.multiProcessorCount,
                prop.totalGlobalMem / 1048576.0);
    std::printf("Resolution : %d x %d\n", W, H);
    std::printf("Samples    : %d spp\n",  SAMPLES);
    std::printf("Max bounces: %d\n\n",    BOUNCES);

    // ── Build & upload scene ──────────────────────────────────────────────────
    auto h_spheres = build_scene();
    int  n         = int(h_spheres.size());
    if (n > MAX_SPHERES) {
        std::fprintf(stderr, "ERROR: scene has %d spheres (max %d)\n", n, MAX_SPHERES);
        return 1;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_spheres,     h_spheres.data(), (size_t)n * sizeof(CSphere)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_spheres, &n,               sizeof(int)));
    std::printf("Scene      : %d spheres  →  constant memory (CSphere)\n\n", n);

    // ── Camera ────────────────────────────────────────────────────────────────
    Camera cam(vec3(13,2,3), vec3(0,0,0), vec3(0,1,0), 20.0f, float(W)/float(H));

    // ── Allocate device buffers ───────────────────────────────────────────────
    vec3*        d_fb;
    curandState* d_states;
    CUDA_CHECK(cudaMalloc(&d_fb,     (size_t)W * H * sizeof(vec3)));
    CUDA_CHECK(cudaMalloc(&d_states, (size_t)W * H * sizeof(curandState)));

    // ── Phase 4: block-size benchmark ─────────────────────────────────────────
    const int BENCH_SAMPLES = 32;  // fewer samples for the benchmark runs
    struct BSResult { int bx, by; double ms; };
    std::vector<BSResult> bsr;
    double best_ms = 1e18;
    int best_bx = 16, best_by = 16;

    std::printf("--- Block-size benchmark (%d spp) ---\n", BENCH_SAMPLES);
    std::printf("%-14s  %12s\n", "Block size", "Time (ms)");
    std::printf("%s\n", std::string(28, '-').c_str());

    for (auto [bx, by] : std::vector<std::pair<int,int>>{{8,8},{16,16},{32,32}}) {
        double ms = benchmark_block_size(bx, by, W, H, BENCH_SAMPLES, BOUNCES,
                                         cam, d_states, d_fb);
        std::printf("  %dx%-10d  %12.1f\n", bx, by, ms);
        bsr.push_back({bx, by, ms});
        if (ms < best_ms) { best_ms = ms; best_bx = bx; best_by = by; }
    }
    std::printf("\nBest block size: %dx%d  (%.1f ms @ %d spp)\n\n",
                best_bx, best_by, best_ms, BENCH_SAMPLES);

    // ── Full quality render with optimal block size ───────────────────────────
    std::printf("--- Full render: %d spp, block %dx%d ---\n",
                SAMPLES, best_bx, best_by);

    double full_ms = benchmark_block_size(best_bx, best_by, W, H, SAMPLES, BOUNCES,
                                          cam, d_states, d_fb,
                                          "renders/gpu_render.ppm");

    std::printf("GPU render time: %.1f ms  (%.2f s)\n", full_ms, full_ms / 1000.0);
    std::printf("\n");
    std::printf("=== Speedup summary ===\n");
    std::printf("  GPU (%d spp): %.1f ms\n", SAMPLES, full_ms);
    std::printf("  CPU (%d spp): run cpu_raytracer for baseline time\n", 4);
    std::printf("\n");
    std::printf("Open renders/gpu_render.ppm with any PPM viewer.\n");

    CUDA_CHECK(cudaFree(d_fb));
    CUDA_CHECK(cudaFree(d_states));
    return 0;
}
