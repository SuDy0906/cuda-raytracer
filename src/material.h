// =============================================================================
// material.h — Material types: Lambertian diffuse, Metal, Glass
// Plain data struct — no virtual functions so it works on GPU constant memory
// =============================================================================
#pragma once
#include "vec3.h"

enum MaterialType : int {
    LAMBERTIAN = 0,
    METAL      = 1,
    GLASS      = 2
};

struct Material {
    MaterialType type;
    vec3         albedo;
    float        fuzz;   // metal roughness [0,1]
    float        ior;    // index of refraction (glass ~1.5)

    // ── Factories ─────────────────────────────────────────────────────────────
    HD static Material lambertian(const vec3& albedo) {
        Material m = {};
        m.type   = LAMBERTIAN;
        m.albedo = albedo;
        m.fuzz   = 0.0f;
        m.ior    = 1.0f;
        return m;
    }

    HD static Material metal(const vec3& albedo, float fuzz) {
        Material m = {};
        m.type   = METAL;
        m.albedo = albedo;
        m.fuzz   = (fuzz < 1.0f) ? fuzz : 1.0f;
        m.ior    = 1.0f;
        return m;
    }

    HD static Material glass(float ior) {
        Material m = {};
        m.type   = GLASS;
        m.albedo = vec3(1.0f, 1.0f, 1.0f);  // glass attenuates nothing
        m.fuzz   = 0.0f;
        m.ior    = ior;
        return m;
    }
};
