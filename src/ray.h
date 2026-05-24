// =============================================================================
// ray.h — Ray: origin + direction, usable on CPU and GPU
// =============================================================================
#pragma once
#include "vec3.h"

struct Ray {
    vec3 origin;
    vec3 direction;

    HD Ray() {}
    HD Ray(const vec3& origin, const vec3& direction)
        : origin(origin), direction(direction) {}

    HD vec3 at(float t) const { return origin + t * direction; }
};
