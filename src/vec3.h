// =============================================================================
// vec3.h — 3-component float vector, shared between CPU and GPU
// The HD macro expands to __host__ __device__ when compiled by nvcc,
// and to nothing when compiled by a regular C++ compiler.
// =============================================================================
#pragma once

#ifdef __CUDACC__
#  define HD __host__ __device__
#else
#  define HD
#endif

#include <cmath>

struct vec3 {
    float x, y, z;

    HD vec3() : x(0.0f), y(0.0f), z(0.0f) {}
    HD vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    HD vec3  operator-()                const { return vec3(-x, -y, -z); }
    HD vec3& operator+=(const vec3& v)        { x += v.x; y += v.y; z += v.z; return *this; }
    HD vec3& operator-=(const vec3& v)        { x -= v.x; y -= v.y; z -= v.z; return *this; }
    HD vec3& operator*=(float t)              { x *= t;   y *= t;   z *= t;   return *this; }
    HD vec3& operator/=(float t)              { return *this *= (1.0f / t); }

    HD float length()         const { return sqrtf(length_squared()); }
    HD float length_squared() const { return x*x + y*y + z*z; }

    HD bool near_zero() const {
        const float s = 1e-8f;
        return (fabsf(x) < s) && (fabsf(y) < s) && (fabsf(z) < s);
    }
};

// ── Arithmetic operators ──────────────────────────────────────────────────────
HD inline vec3 operator+(const vec3& a, const vec3& b) { return vec3(a.x+b.x, a.y+b.y, a.z+b.z); }
HD inline vec3 operator-(const vec3& a, const vec3& b) { return vec3(a.x-b.x, a.y-b.y, a.z-b.z); }
HD inline vec3 operator*(const vec3& a, const vec3& b) { return vec3(a.x*b.x, a.y*b.y, a.z*b.z); }  // element-wise
HD inline vec3 operator*(float t, const vec3& v)       { return vec3(t*v.x,   t*v.y,   t*v.z);   }
HD inline vec3 operator*(const vec3& v, float t)       { return t * v; }
HD inline vec3 operator/(const vec3& v, float t)       { return (1.0f / t) * v; }

// ── Math utilities ────────────────────────────────────────────────────────────
HD inline float dot(const vec3& a, const vec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

HD inline vec3 cross(const vec3& a, const vec3& b) {
    return vec3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}

HD inline vec3 unit_vector(const vec3& v) { return v / v.length(); }

// Specular reflection: v reflected about normal n
HD inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2.0f * dot(v, n) * n;
}

// Snell's law refraction
HD inline vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
    float cos_theta    = fminf(dot(-uv, n), 1.0f);
    vec3  r_out_perp   = etai_over_etat * (uv + cos_theta * n);
    vec3  r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

// ── iostream support (CPU only) ───────────────────────────────────────────────
#ifndef __CUDACC__
#  include <iostream>
inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.x << ' ' << v.y << ' ' << v.z;
}
#endif
