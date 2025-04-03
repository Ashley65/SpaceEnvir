//
// Created by DevAccount on 28/03/2025.
//
//
// Created by DevAccount on 28/03/2025.
//

#include <components/Body.cuh>
#include <math.h>

__device__ void Spaceship::applyThrust(float deltaTime) {
    // Calculate thrust force
    float thrustForce = thrust * deltaTime;

    // Apply thrust in the direction the spaceship is facing
    float dirMagnitude = sqrt(dirX*dirX + dirY*dirY + dirZ*dirZ);

    // Normalize direction vector if it's not already normalized
    if (dirMagnitude > 0.0001f) {
        float normDirX = dirX / dirMagnitude;
        float normDirY = dirY / dirMagnitude;
        float normDirZ = dirZ / dirMagnitude;

        // Update acceleration based on thrust (F=ma, so a=F/m)
        base.ax += (thrustForce * normDirX) / base.mass;
        base.ay += (thrustForce * normDirY) / base.mass;
        base.az += (thrustForce * normDirZ) / base.mass;

        // Decrease fuel based on thrust applied
        fuel -= thrust * deltaTime * 0.01f; // Fuel consumption rate
        if (fuel < 0.0f) {
            fuel = 0.0f;
            thrust = 0.0f; // No fuel, no thrust
        }
    }
}

__device__ void Spaceship::rotate(float deltaTime, float rotX, float rotY, float rotZ) {
    // Apply rotation around each axis
    // This is a simplified rotation model

    // Calculate rotation amounts
    float rotAmountX = rotX * angularVelocity * deltaTime;
    float rotAmountY = rotY * angularVelocity * deltaTime;
    float rotAmountZ = rotZ * angularVelocity * deltaTime;

    // Apply X rotation (pitch)
    float oldDirY = dirY;
    float oldDirZ = dirZ;
    dirY = oldDirY * cosf(rotAmountX) - oldDirZ * sinf(rotAmountX);
    dirZ = oldDirY * sinf(rotAmountX) + oldDirZ * cosf(rotAmountX);

    // Apply Y rotation (yaw)
    float oldDirX = dirX;
    oldDirZ = dirZ;
    dirX = oldDirX * cosf(rotAmountY) + oldDirZ * sinf(rotAmountY);
    dirZ = -oldDirX * sinf(rotAmountY) + oldDirZ * cosf(rotAmountY);

    // Apply Z rotation (roll)
    oldDirX = dirX;
    oldDirY = dirY;
    dirX = oldDirX * cosf(rotAmountZ) - oldDirY * sinf(rotAmountZ);
    dirY = oldDirX * sinf(rotAmountZ) + oldDirY * cosf(rotAmountZ);

    // Normalize direction vector
    float magnitude = sqrt(dirX*dirX + dirY*dirY + dirZ*dirZ);
    if (magnitude > 0.0001f) {
        dirX /= magnitude;
        dirY /= magnitude;
        dirZ /= magnitude;
    }
}

__host__ bool Trajectory::initialize(int maxPointCount, float interval) {
    // Allocate memory for trajectory points
    maxPoints = maxPointCount;
    recordInterval = interval;
    currentSize = 0;
    timeSinceLastRecord = 0.0f;

    // Allocate device memory for positions and velocities
    cudaError_t posError = cudaMalloc(&positions, maxPoints * sizeof(float3));
    cudaError_t velError = cudaMalloc(&velocities, maxPoints * sizeof(float3));

    // Return success if both allocations succeeded
    return (posError == cudaSuccess && velError == cudaSuccess);
}

__device__ void Trajectory::recordPoint(const float3& position, const float3& velocity, float deltaTime) {
    timeSinceLastRecord += deltaTime;

    // If it's time to record a new point
    if (timeSinceLastRecord >= recordInterval) {
        // Reset timer
        timeSinceLastRecord = 0.0f;

        // Record the point if there's space
        if (currentSize < maxPoints) {
            positions[currentSize] = position;
            velocities[currentSize] = velocity;
            currentSize++;
        }
        else {
            // If the buffer is full, shift everything back one spot
            for (int i = 0; i < maxPoints - 1; i++) {
                positions[i] = positions[i + 1];
                velocities[i] = velocities[i + 1];
            }

            // Add new point at the end
            positions[maxPoints - 1] = position;
            velocities[maxPoints - 1] = velocity;
        }
    }
}