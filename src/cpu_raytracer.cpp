// =============================================================================
// cpu_raytracer.cpp — Phase 1: plain C++ ray tracer
//
// Features:
//   • Ray-sphere intersection (analytic)
//   • Lambertian diffuse, Metal reflection, Glass refraction (Schlick)
//   • Anti-aliasing: 4 samples per pixel
//   • Recursive ray bouncing: up to 10 depth
//   • Same scene as the CUDA version for a fair comparison
//   • Output: renders/baseline_render.ppm + CPU render time in ms
//
// Build:
//   g++ -std=c++17 -O3 -o cpu_raytracer src/cpu_raytracer.cpp -Isrc
//   (or via CMake — see CMakeLists.txt)
// =============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <string>
#include <cstdio>

#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "material.h"

// ── Render settings ───────────────────────────────────────────────────────────
static const int IMG_W      = 800;
static const int IMG_H      = 600;
static const int SAMPLES    = 4;     // SPP — kept low so the CPU baseline is fast
static const int MAX_BOUNCES = 10;

// ── CPU random helpers ────────────────────────────────────────────────────────
static std::mt19937 rng(42);
static std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

inline float randf()  { return dist01(rng); }
inline float randf(float lo, float hi) { return lo + (hi - lo) * randf(); }

vec3 random_in_unit_sphere() {
    vec3 p;
    do { p = vec3(randf(-1,1), randf(-1,1), randf(-1,1)); }
    while (p.length_squared() >= 1.0f);
    return p;
}
vec3 random_unit_vector() { return unit_vector(random_in_unit_sphere()); }

// ── Camera ────────────────────────────────────────────────────────────────────
struct Camera {
    vec3 origin, lower_left, horizontal, vertical;

    Camera(vec3 from, vec3 at, vec3 up, float vfov_deg, float aspect) {
        float theta  = vfov_deg * 3.14159265f / 180.0f;
        float h      = tanf(theta / 2.0f);
        float vh     = 2.0f * h;
        float vw     = aspect * vh;

        vec3 w = unit_vector(from - at);
        vec3 u = unit_vector(cross(up, w));
        vec3 v = cross(w, u);

        origin      = from;
        horizontal  = vw * u;
        vertical    = vh * v;
        lower_left  = origin - horizontal / 2.0f - vertical / 2.0f - w;
    }

    Ray get_ray(float s, float t) const {
        return Ray(origin, lower_left + s * horizontal + t * vertical - origin);
    }
};

// ── Scene (BVH-less linear list) ──────────────────────────────────────────────
bool world_hit(const std::vector<Sphere>& world, const Ray& r,
               float t_min, float t_max, HitRecord& rec) {
    HitRecord tmp;
    bool hit     = false;
    float closest = t_max;
    for (const auto& s : world) {
        if (s.hit(r, t_min, closest, tmp)) {
            hit     = true;
            closest = tmp.t;
            rec     = tmp;
        }
    }
    return hit;
}

// ── Material scatter (CPU version) ────────────────────────────────────────────
static float schlick(float cosine, float ior) {
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 *= r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosine, 5.0f);
}

bool scatter_ray(const Ray& r_in, const HitRecord& rec,
                 vec3& attenuation, Ray& scattered) {
    switch (rec.mat.type) {
        case LAMBERTIAN: {
            vec3 dir = rec.normal + random_unit_vector();
            if (dir.near_zero()) dir = rec.normal;
            scattered   = Ray(rec.p, dir);
            attenuation = rec.mat.albedo;
            return true;
        }
        case METAL: {
            vec3 ref = reflect(unit_vector(r_in.direction), rec.normal);
            scattered   = Ray(rec.p, ref + rec.mat.fuzz * random_in_unit_sphere());
            attenuation = rec.mat.albedo;
            return dot(scattered.direction, rec.normal) > 0.0f;
        }
        case GLASS: {
            attenuation    = vec3(1.0f, 1.0f, 1.0f);
            float ratio    = rec.front_face ? (1.0f / rec.mat.ior) : rec.mat.ior;
            vec3 unit_dir  = unit_vector(r_in.direction);
            float cos_th   = std::min(dot(-unit_dir, rec.normal), 1.0f);
            float sin_th   = sqrtf(1.0f - cos_th * cos_th);
            vec3 dir;
            if (ratio * sin_th > 1.0f || schlick(cos_th, ratio) > randf())
                dir = reflect(unit_dir, rec.normal);
            else
                dir = refract(unit_dir, rec.normal, ratio);
            scattered = Ray(rec.p, dir);
            return true;
        }
    }
    return false;
}

// ── Recursive ray colour ──────────────────────────────────────────────────────
vec3 ray_color(const Ray& r, const std::vector<Sphere>& world, int depth) {
    if (depth <= 0) return vec3(0.0f, 0.0f, 0.0f);

    HitRecord rec;
    if (world_hit(world, r, 0.001f, 1e9f, rec)) {
        vec3 attenuation;
        Ray  scattered;
        if (scatter_ray(r, rec, attenuation, scattered))
            return attenuation * ray_color(scattered, world, depth - 1);
        return vec3(0.0f, 0.0f, 0.0f);
    }

    // Sky gradient
    vec3 unit_dir = unit_vector(r.direction);
    float t = 0.5f * (unit_dir.y + 1.0f);
    return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
}

// ── Scene builder (identical to CUDA version) ─────────────────────────────────
std::vector<Sphere> build_scene() {
    std::vector<Sphere> spheres;

    auto add = [&](vec3 center, float radius, Material mat) {
        Sphere s = {}; s.center = center; s.radius = radius; s.mat = mat;
        spheres.push_back(s);
    };

    // Ground plane (huge sphere)
    add(vec3(0, -1000, 0), 1000.0f, Material::lambertian(vec3(0.48f, 0.48f, 0.48f)));

    // Three hero spheres
    add(vec3( 0, 1, 0), 1.0f, Material::glass(1.5f));
    add(vec3(-4, 1, 0), 1.0f, Material::lambertian(vec3(0.4f, 0.2f, 0.1f)));
    add(vec3( 4, 1, 0), 1.0f, Material::metal(vec3(0.7f, 0.6f, 0.5f), 0.0f));

    // Scatter of small spheres
    struct SmallSpec { float x, z, r, g, b; int type; float fuzz, ior; };
    static const SmallSpec specs[] = {
        {-2,-2, 0.80f,0.30f,0.30f, 0, 0.00f, 0.0f},
        { 2,-2, 0.10f,0.60f,0.90f, 1, 0.10f, 0.0f},
        {-1, 2, 0.30f,0.70f,0.30f, 0, 0.00f, 0.0f},
        { 1, 2, 0.90f,0.80f,0.10f, 1, 0.30f, 0.0f},
        {-3, 1, 0.40f,0.40f,0.90f, 2, 0.00f, 1.7f},
        { 3, 1, 0.90f,0.20f,0.20f, 0, 0.00f, 0.0f},
        { 0,-3, 0.70f,0.70f,0.20f, 1, 0.00f, 0.0f},
        {-2, 3, 0.20f,0.90f,0.80f, 0, 0.00f, 0.0f},
        { 2, 3, 0.50f,0.50f,0.50f, 1, 0.50f, 0.0f},
        { 0, 4, 1.00f,1.00f,1.00f, 2, 0.00f, 1.5f},
        {-4, 3, 0.60f,0.30f,0.10f, 0, 0.00f, 0.0f},
        { 4, 3, 0.20f,0.40f,0.80f, 1, 0.20f, 0.0f},
    };

    for (const auto& s : specs) {
        Material mat;
        switch (s.type) {
            case 0: mat = Material::lambertian(vec3(s.r, s.g, s.b)); break;
            case 1: mat = Material::metal(vec3(s.r, s.g, s.b), s.fuzz); break;
            case 2: mat = Material::glass(s.ior); break;
        }
        add(vec3(s.x, 0.3f, s.z), 0.3f, mat);
    }

    return spheres;
}

// ── PPM writer ────────────────────────────────────────────────────────────────
void write_ppm(const std::string& path, const std::vector<vec3>& fb, int w, int h) {
    std::ofstream f(path);
    f << "P3\n" << w << ' ' << h << "\n255\n";
    for (int j = h - 1; j >= 0; --j) {
        for (int i = 0; i < w; ++i) {
            const vec3& c = fb[j * w + i];
            f << int(255.99f * std::min(1.0f, std::max(0.0f, c.x))) << ' '
              << int(255.99f * std::min(1.0f, std::max(0.0f, c.y))) << ' '
              << int(255.99f * std::min(1.0f, std::max(0.0f, c.z))) << '\n';
        }
    }
    f.close();
    std::printf("  Saved: %s\n", path.c_str());
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main() {
    std::filesystem::create_directories("renders");

    const float aspect = float(IMG_W) / float(IMG_H);
    Camera cam(vec3(13,2,3), vec3(0,0,0), vec3(0,1,0), 20.0f, aspect);
    auto   world = build_scene();

    std::vector<vec3> fb(IMG_W * IMG_H);

    std::printf("=== CPU Ray Tracer ===\n");
    std::printf("Resolution : %d x %d\n", IMG_W, IMG_H);
    std::printf("Samples    : %d spp\n",  SAMPLES);
    std::printf("Max bounces: %d\n",      MAX_BOUNCES);
    std::printf("Spheres    : %zu\n\n",   world.size());

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int j = 0; j < IMG_H; ++j) {
        if (j % 60 == 0)
            std::printf("\r  Progress: %3d / %d scanlines", j, IMG_H);
        std::fflush(stdout);

        for (int i = 0; i < IMG_W; ++i) {
            vec3 color(0.0f, 0.0f, 0.0f);
            for (int s = 0; s < SAMPLES; ++s) {
                float u = (i + randf()) / float(IMG_W - 1);
                float v = (j + randf()) / float(IMG_H - 1);
                color += ray_color(cam.get_ray(u, v), world, MAX_BOUNCES);
            }
            color /= float(SAMPLES);
            // Gamma-correct (γ=2)
            color.x = sqrtf(std::max(0.0f, color.x));
            color.y = sqrtf(std::max(0.0f, color.y));
            color.z = sqrtf(std::max(0.0f, color.z));
            fb[j * IMG_W + i] = color;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::printf("\r  Progress: %d / %d scanlines (done)\n\n", IMG_H, IMG_H);
    std::printf("CPU render time: %.1f ms  (%.2f s)\n", ms, ms / 1000.0);

    write_ppm("renders/baseline_render.ppm", fb, IMG_W, IMG_H);

    std::printf("\nDone! Open renders/baseline_render.ppm with any PPM viewer\n");
    std::printf("(GIMP, IrfanView, vscode extension, etc.)\n");
    return 0;
}
