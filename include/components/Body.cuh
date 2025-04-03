//
// Created by DevAccount on 04/03/2025.
//


#ifndef BODY_H
#define BODY_H

#include <cstdint>
#include <cuda_runtime.h>
#include <limits>

struct Body {
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
    double mass;
    double radius;
    uint8_t isSpaceship;  // use 0 or 1
    uint8_t isElastic;    // use 0 or 1
    bool isStatic;
};

struct celestialBody {
    Body base;               // Inherits basic physics properties
    enum Type {
        STAR,
        PLANET,
        MOON,
        ASTEROID,
        COMET,
    };
};

struct Spaceship {
    Body base;               // Inherits basic physics properties
    double fuel;              // Remaining fuel
    double thrust;            // Current thrust power
    double maxThrust;         // Maximum thrust power
    double dirX, dirY, dirZ;  // Orientation for thrust direction
    double angularVelocity;   // Rotational speed for turns

    // CUDA device functions for thrust calculations
    __device__ void applyThrust(float deltaTime);
    __device__ void rotate(float deltaTime, float rotX, float rotY, float rotZ);
};

struct Trajectory {
    float3* positions;        // Device pointer to position history
    float3* velocities;       // Device pointer to velocity history
    int maxPoints;            // Maximum number of points to store
    int currentSize;          // Current number of stored points
    float recordInterval;     // Time interval between recordings
    float timeSinceLastRecord; // Time accumulator

    __host__ Trajectory() : positions(nullptr), velocities(nullptr), maxPoints(0), currentSize(0),
                   recordInterval(0.0f), timeSinceLastRecord(0.0f) {}

    __host__ ~Trajectory() {
        if (positions) cudaFree(positions);
        if (velocities) cudaFree(velocities);
    }

    __host__ bool initialize(int maxPointCount, float interval);
    __device__ void recordPoint(const float3& position, const float3& velocity, float deltaTime);
};

// Add sensor data structure to support AI decision-making
struct SensorData {
    float3 nearestObstaclePosition;
    float nearestObstacleDistance;
    float3 targetPosition;
    float targetDistance;
    float fuelRemaining;
    uint8_t inDanger;  // 0 or 1

    __device__ __host__ SensorData() :
        nearestObstaclePosition{0.0, 0.0, 0.0},
        nearestObstacleDistance(std::numeric_limits<float>::max()),
        targetDistance(std::numeric_limits<float>::max()),
        targetPosition{0.0, 0.0, 0.0},

        fuelRemaining(0.0),
        inDanger(0) {}
};

// Command structure to facilitate AI control
struct SpaceshipCommand {
    float thrustLevel;  // 0.0 to 1.0
    float3 targetDirection;
    uint8_t emergencyStop;  // 0 or 1

    __device__ __host__ SpaceshipCommand() :
        thrustLevel(0.0f),
        targetDirection{0.0f, 0.0f, 1.0f},
        emergencyStop(0) {}
};

#endif //BODY_H