// =============================================================================
// sphere.h — Sphere geometry + HitRecord, usable on CPU and GPU
// =============================================================================
#pragma once
#include "vec3.h"
#include "ray.h"
#include "material.h"

// ── Hit record ────────────────────────────────────────────────────────────────
struct HitRecord {
    vec3     p;           // world-space hit point
    vec3     normal;      // always points against the incident ray
    float    t;           // ray parameter at intersection
    bool     front_face;  // did the ray hit the outside?
    Material mat;

    // Ensure the stored normal always opposes the ray direction
    HD void set_face_normal(const Ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction, outward_normal) < 0.0f;
        normal     = front_face ? outward_normal : -outward_normal;
    }
};

// ── Sphere ────────────────────────────────────────────────────────────────────
struct Sphere {
    vec3     center;
    float    radius;
    Material mat;

    // Returns true if the ray hits this sphere in [t_min, t_max].
    // Uses the quadratic formula with a factored half_b form for numerical stability.
    HD bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        vec3  oc     = r.origin - center;
        float a      = dot(r.direction, r.direction);
        float half_b = dot(oc, r.direction);
        float c      = dot(oc, oc) - radius * radius;
        float disc   = half_b * half_b - a * c;

        if (disc < 0.0f) return false;

        float sqrtd = sqrtf(disc);

        // Try the near root first, then the far root
        float root = (-half_b - sqrtd) / a;
        if (root < t_min || root > t_max) {
            root = (-half_b + sqrtd) / a;
            if (root < t_min || root > t_max)
                return false;
        }

        rec.t = root;
        rec.p = r.at(root);
        vec3 outward = (rec.p - center) / radius;
        rec.set_face_normal(r, outward);
        rec.mat = mat;
        return true;
    }
};
