//
// Created by DevAccount on 28/03/2025.
//

#ifndef PHYSICSENGINE_CUH
#define PHYSICSENGINE_CUH
#pragma once
#include <cuda_runtime.h>
#include <ecs/ECS.cuh>
#include <components/Body.cuh>
#include <vector>
#include <memory>

namespace engine {
    // constants for the physics engine
    constexpr double G = 6.67430e-11; // gravitational constant
    constexpr double TIME_STEP = 0.01; // time step for the simulation
    constexpr double SOFTENING = 1e-9; // softening factor for gravitational force calculation
    struct BoundingBox {
        double minX, minY, minZ;
        double maxX, maxY, maxZ;
    };
    struct Trajectory {
        float3* positions;
        float3* velocities;
        int maxPoints;
        int currentSize;
        float recordInterval;
        float timeSinceLastRecord;
    };


    struct OctreeNode {
        BoundingBox bounds;
        double centerOfMassX, centerOfMassY, centerOfMassZ;
        double totalMass;
        bool isLeaf;
        int bodyIndex;  // Only valid for leaf nodes
        OctreeNode* children[8];

        __device__ OctreeNode() : totalMass(0.0), isLeaf(true), bodyIndex(-1) {
            for (int i = 0; i < 8; i++) {
                children[i] = nullptr;
            }
        }
    };

    // CUDA kernel functions
    __global__ void buildOctreeKernel(Body* bodies, int numBodies, OctreeNode* rootNode, BoundingBox globalBounds);
    __global__ void computeBarnesHutForcesKernel(Body* bodies, int numBodies, OctreeNode* rootNode, double theta, double G);
    __global__ void computeGravitationalForcesKernel(Body* bodies, int numBodies, double dt, double G);
    __global__ void integrateRK45Kernel(Body* bodies, Body* k1, Body* k2, Body* k3, Body* k4, Body* k5, Body* k6,
                                    Body* temp, int numBodies, double dt,
                                    const double* a, const double* b, const double* c);
    __global__ void updateSpaceshipsKernel(Body* bodies, int numBodies, double dt);
    __global__ void updateTrajectoriesKernel(Body* bodies, Trajectory* trajectories, int numBodies, double dt);
    __global__ void calculateGlobalBoundsKernel(Body* bodies, int numBodies, OctreeNode* rootNode);
    __global__ void computeBarnesHutForcesNonRecursiveKernel(Body* bodies, int numBodies, OctreeNode* rootNode, double theta, double G);
    __device__ double atomicAddDouble(double* address, double val);
    __device__ OctreeNode* atomicCASptr(OctreeNode** address, OctreeNode* compare, OctreeNode* val);


    class PhysicsEngine {
        private:
            ecs::ECSCoordinator* coordinator;
            std::vector<ecs::EntityID> entities;

            // RK45 parameters
            struct RK45Parameters {
                const double a[6] = {0.0, 0.2, 0.3, 0.6, 1.0, 0.875};
                const double b[6][5] = {
                    {0.0, 0.0, 0.0, 0.0, 0.0},
                    {0.2, 0.0, 0.0, 0.0, 0.0},
                    {3.0/40.0, 9.0/40.0, 0.0, 0.0, 0.0},
                    {0.3, -0.9, 1.2, 0.0, 0.0},
                    {-11.0/54.0, 2.5, -70.0/27.0, 35.0/27.0, 0.0},
                    {1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0}
                };
                const double c[6] = {37.0/378.0, 0.0, 250.0/621.0, 125.0/594.0, 0.0, 512.0/1771.0};
                const double cStar[6] = {2825.0/27648.0, 0.0, 18575.0/48384.0, 13525.0/55296.0, 277.0/14336.0, 0.25};
            };
            RK45Parameters rk45Params;
            // Device memory for physics calculations
            Body* d_bodies;
            Body* d_k1;
            Body* d_k2;
            Body* d_k3;
            Body* d_k4;
            Body* d_k5;
            Body* d_k6;
            Body* d_temp;
            int numBodies;

            // Barnes-Hut Octree parameters (forward declaration)
            OctreeNode* d_octreeRoot;

            // Helper methods
            void allocateDeviceMemory();

            void copyBodiesToDevice();
            void copyBodiesFromDevice();

    public:

        explicit PhysicsEngine(ecs::ECSCoordinator* coordinator);
        ~PhysicsEngine();

        void initialise();
        void update(double deltaTime);
        void registerEntity(ecs::EntityID entity);
        void unregisterEntity(ecs::EntityID entity);

        // RK45 integration methods
        void integrateRK45(double deltaTime);
        void freeDeviceMemory();
        void cleanup() {
            freeDeviceMemory();
        }

        // Methods for calculating forces
        void calculateGravitationalForces(double deltaTime);
        void updateSpaceships(double deltaTime);
        void updateTrajectories(double deltaTime);
        void buildBarnesHutOctree();
        void computeBarnesHutForces();
    };



}

#endif //PHYSICSENGINE_CUH
