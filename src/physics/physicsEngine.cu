//
// Created by DevAccount on 29/03/2025.
//
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <physics/physicsEngine.cuh>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <components/Body.cuh>
namespace engine {

    PhysicsEngine::PhysicsEngine(ecs::ECSCoordinator* coordinator)
    : coordinator(coordinator), numBodies(0), d_bodies(nullptr), d_k1(nullptr), d_k2(nullptr),
      d_k3(nullptr), d_k4(nullptr), d_k5(nullptr), d_k6(nullptr), d_temp(nullptr), d_octreeRoot(nullptr) {
    }
    PhysicsEngine::~PhysicsEngine() {
        freeDeviceMemory();
    }

    void PhysicsEngine::initialise() {
        numBodies = entities.size();
        allocateDeviceMemory();
        copyBodiesToDevice();
    }
    void PhysicsEngine::allocateDeviceMemory() {
        if (numBodies > 0) {
            cudaMalloc(&d_bodies, numBodies * sizeof(Body));
            cudaMalloc(&d_k1, numBodies * sizeof(Body));
            cudaMalloc(&d_k2, numBodies * sizeof(Body));
            cudaMalloc(&d_k3, numBodies * sizeof(Body));
            cudaMalloc(&d_k4, numBodies * sizeof(Body));
            cudaMalloc(&d_k5, numBodies * sizeof(Body));
            cudaMalloc(&d_k6, numBodies * sizeof(Body));
            cudaMalloc(&d_temp, numBodies * sizeof(Body));
        }
    }
    void PhysicsEngine::freeDeviceMemory() {
        if (d_bodies) cudaFree(d_bodies);
        if (d_k1) cudaFree(d_k1);
        if (d_k2) cudaFree(d_k2);
        if (d_k3) cudaFree(d_k3);
        if (d_k4) cudaFree(d_k4);
        if (d_k5) cudaFree(d_k5);
        if (d_k6) cudaFree(d_k6);
        if (d_temp) cudaFree(d_temp);
        if (d_octreeRoot) cudaFree(d_octreeRoot);

        d_bodies = nullptr;
        d_k1 = d_k2 = d_k3 = d_k4 = d_k5 = d_k6 = d_temp = nullptr;
        d_octreeRoot = nullptr;
    }
    void PhysicsEngine::copyBodiesToDevice() {
        std::vector<Body> hostBodies;
        hostBodies.reserve(numBodies);

        for (auto entityID : entities) {
            hostBodies.push_back(coordinator->getComponent<Body>(entityID));
        }

        if (!hostBodies.empty()) {
            cudaMemcpy(d_bodies, hostBodies.data(), numBodies * sizeof(Body), cudaMemcpyHostToDevice);
        }
    }
    void PhysicsEngine::copyBodiesFromDevice() {
        std::vector<Body> hostBodies(numBodies);
        cudaMemcpy(hostBodies.data(), d_bodies, numBodies * sizeof(Body), cudaMemcpyDeviceToHost);

        for (size_t i = 0; i < entities.size(); ++i) {
            coordinator->addComponent<Body>(entities[i], hostBodies[i]);
        }

    }

    void PhysicsEngine::registerEntity(ecs::EntityID entity) {
        entities.push_back(entity);
    }

    void PhysicsEngine::unregisterEntity(ecs::EntityID entity) {
        auto it = std::find(entities.begin(), entities.end(), entity);
        if (it != entities.end()) {
            entities.erase(it);
        }
    }

    void PhysicsEngine::update(double deltaTime) {
        if (entities.empty()) return;

        // Re-initialize if the number of bodies changed
        if (numBodies != entities.size()) {
            freeDeviceMemory();
            initialise();
        }

        copyBodiesToDevice();
        integrateRK45(deltaTime);
        updateSpaceships(deltaTime);
        updateTrajectories(deltaTime);
        copyBodiesFromDevice();
    }

    void PhysicsEngine::integrateRK45(double deltaTime) {
        if (numBodies == 0) return;

        int blockSize = 256;
        int numBlocks = (numBodies + blockSize - 1) / blockSize;

        // Launch the RK45 integration kernel
        // (In a real implementation, we'd need to pass the RK45 parameters to the kernel)
        integrateRK45Kernel<<<numBlocks, blockSize>>>(d_bodies, d_k1, d_k2, d_k3, d_k4, d_k5, d_k6,
                                                     d_temp, numBodies, deltaTime, nullptr, nullptr, nullptr);

        cudaDeviceSynchronize();
    }

    void PhysicsEngine::calculateGravitationalForces(double deltaTime) {
        if (numBodies == 0) return;

        int blockSize = 256;
        int numBlocks = (numBodies + blockSize - 1) / blockSize;

        computeGravitationalForcesKernel<<<numBlocks, blockSize>>>(d_bodies, numBodies, deltaTime, G);

        cudaDeviceSynchronize();
    }

    void PhysicsEngine::updateSpaceships(double deltaTime) {
        if (numBodies == 0) return;

        int blockSize = 256;
        int numBlocks = (numBodies + blockSize - 1) / blockSize;

        updateSpaceshipsKernel<<<numBlocks, blockSize>>>(d_bodies, numBodies, deltaTime);

        cudaDeviceSynchronize();
    }

    void PhysicsEngine::updateTrajectories(double deltaTime) {
        // This would require storing trajectory data for each entity
        // For simplicity, we'll leave this as a stub
    }
    void PhysicsEngine::computeBarnesHutForces() {
        if (numBodies == 0) return;

        int blockSize = 256;
        int numBlocks = (numBodies + blockSize - 1) / blockSize;

        // Theta is the accuracy parameter for Barnes-Hut algorithm (0.5 is typical)
        double theta = 0.5;

        // Launch the non-recursive kernel
        computeBarnesHutForcesNonRecursiveKernel<<<numBlocks, blockSize>>>(d_bodies, numBodies, d_octreeRoot, theta, G);

        cudaDeviceSynchronize();
    }
    void PhysicsEngine::buildBarnesHutOctree() {
        if (numBodies == 0) return;

        // Allocate memory for the root node if not already allocated
        if (d_octreeRoot == nullptr) {
            cudaMalloc(&d_octreeRoot, sizeof(OctreeNode));
        }

        // Initialize the root node directly on the device
        cudaMemset(d_octreeRoot, 0, sizeof(OctreeNode));

        // Calculate global bounds in a separate kernel
        calculateGlobalBoundsKernel<<<1, 1>>>(d_bodies, numBodies, d_octreeRoot);
        cudaDeviceSynchronize();

        // Get the global bounds for building the octree
        BoundingBox globalBounds;
        cudaMemcpy(&globalBounds, &(d_octreeRoot->bounds), sizeof(BoundingBox), cudaMemcpyDeviceToHost);

        // Build the octree
        int blockSize = 256;
        int numBlocks = (numBodies + blockSize - 1) / blockSize;
        buildOctreeKernel<<<numBlocks, blockSize>>>(d_bodies, numBodies, d_octreeRoot, globalBounds);
        cudaDeviceSynchronize();
    }

    // CUDA kernel implementations
    __global__ void computeGravitationalForcesKernel(Body* bodies, int numBodies, double dt, double G) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numBodies) return;

        Body& body = bodies[idx];
        body.ax = body.ay = body.az = 0.0;

        for (int j = 0; j < numBodies; j++) {
            if (idx == j) continue;

            Body& other = bodies[j];

            // Calculate distance vector
            double dx = other.x - body.x;
            double dy = other.y - body.y;
            double dz = other.z - body.z;

            // Calculate squared distance with softening to prevent numerical instability
            double distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            double distSixth = distSqr * distSqr * distSqr;
            double invDist = 1.0 / sqrt(distSqr);

            // Calculate gravitational force
            double force = G * body.mass * other.mass * invDist * invDist * invDist;

            // Apply acceleration (F = ma, so a = F/m)
            body.ax += force * dx / body.mass;
            body.ay += force * dy / body.mass;
            body.az += force * dz / body.mass;
        }
    }

   __global__ void integrateRK45Kernel(Body* bodies, Body* k1, Body* k2, Body* k3, Body* k4, Body* k5, Body* k6,
                                   Body* temp, int numBodies, double dt,
                                   const double a[6], const double b[6][5], const double c[6]) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBodies) return;

    // Copy original body data
    Body orig = bodies[idx];

    // First stage - calculate k1
    k1[idx] = orig;
    k1[idx].vx = orig.vx;
    k1[idx].vy = orig.vy;
    k1[idx].vz = orig.vz;
    k1[idx].x = orig.ax;  // K derivatives represent: dx/dt = v, dv/dt = a
    k1[idx].y = orig.ay;
    k1[idx].z = orig.az;

    // Second stage - calculate k2
    temp[idx] = orig;
    temp[idx].x = orig.x + b[1][0] * k1[idx].vx * dt;
    temp[idx].y = orig.y + b[1][0] * k1[idx].vy * dt;
    temp[idx].z = orig.z + b[1][0] * k1[idx].vz * dt;
    temp[idx].vx = orig.vx + b[1][0] * k1[idx].x * dt;
    temp[idx].vy = orig.vy + b[1][0] * k1[idx].y * dt;
    temp[idx].vz = orig.vz + b[1][0] * k1[idx].z * dt;
    // Here we would calculate forces again but for simplicity use k1's acceleration
    k2[idx] = temp[idx];
    k2[idx].x = temp[idx].ax;
    k2[idx].y = temp[idx].ay;
    k2[idx].z = temp[idx].az;

    // Third stage - calculate k3
    temp[idx] = orig;
    temp[idx].x = orig.x + (b[2][0] * k1[idx].vx + b[2][1] * k2[idx].vx) * dt;
    temp[idx].y = orig.y + (b[2][0] * k1[idx].vy + b[2][1] * k2[idx].vy) * dt;
    temp[idx].z = orig.z + (b[2][0] * k1[idx].vz + b[2][1] * k2[idx].vz) * dt;
    temp[idx].vx = orig.vx + (b[2][0] * k1[idx].x + b[2][1] * k2[idx].x) * dt;
    temp[idx].vy = orig.vy + (b[2][0] * k1[idx].y + b[2][1] * k2[idx].y) * dt;
    temp[idx].vz = orig.vz + (b[2][0] * k1[idx].z + b[2][1] * k2[idx].z) * dt;
    // Recalculate forces here for accurate RK45
    k3[idx] = temp[idx];
    k3[idx].x = temp[idx].ax;
    k3[idx].y = temp[idx].ay;
    k3[idx].z = temp[idx].az;

    // Fourth stage - calculate k4
    temp[idx] = orig;
    temp[idx].x = orig.x + (b[3][0] * k1[idx].vx + b[3][1] * k2[idx].vx + b[3][2] * k3[idx].vx) * dt;
    temp[idx].y = orig.y + (b[3][0] * k1[idx].vy + b[3][1] * k2[idx].vy + b[3][2] * k3[idx].vy) * dt;
    temp[idx].z = orig.z + (b[3][0] * k1[idx].vz + b[3][1] * k2[idx].vz + b[3][2] * k3[idx].vz) * dt;
    temp[idx].vx = orig.vx + (b[3][0] * k1[idx].x + b[3][1] * k2[idx].x + b[3][2] * k3[idx].x) * dt;
    temp[idx].vy = orig.vy + (b[3][0] * k1[idx].y + b[3][1] * k2[idx].y + b[3][2] * k3[idx].y) * dt;
    temp[idx].vz = orig.vz + (b[3][0] * k1[idx].z + b[3][1] * k2[idx].z + b[3][2] * k3[idx].z) * dt;
    // Recalculate forces here
    k4[idx] = temp[idx];
    k4[idx].x = temp[idx].ax;
    k4[idx].y = temp[idx].ay;
    k4[idx].z = temp[idx].az;

    // Fifth stage - calculate k5
    temp[idx] = orig;
    temp[idx].x = orig.x + (b[4][0] * k1[idx].vx + b[4][1] * k2[idx].vx + b[4][2] * k3[idx].vx + b[4][3] * k4[idx].vx) * dt;
    temp[idx].y = orig.y + (b[4][0] * k1[idx].vy + b[4][1] * k2[idx].vy + b[4][2] * k3[idx].vy + b[4][3] * k4[idx].vy) * dt;
    temp[idx].z = orig.z + (b[4][0] * k1[idx].vz + b[4][1] * k2[idx].vz + b[4][2] * k3[idx].vz + b[4][3] * k4[idx].vz) * dt;
    temp[idx].vx = orig.vx + (b[4][0] * k1[idx].x + b[4][1] * k2[idx].x + b[4][2] * k3[idx].x + b[4][3] * k4[idx].x) * dt;
    temp[idx].vy = orig.vy + (b[4][0] * k1[idx].y + b[4][1] * k2[idx].y + b[4][2] * k3[idx].y + b[4][3] * k4[idx].y) * dt;
    temp[idx].vz = orig.vz + (b[4][0] * k1[idx].z + b[4][1] * k2[idx].z + b[4][2] * k3[idx].z + b[4][3] * k4[idx].z) * dt;
    // Recalculate forces here
    k5[idx] = temp[idx];
    k5[idx].x = temp[idx].ax;
    k5[idx].y = temp[idx].ay;
    k5[idx].z = temp[idx].az;

    // Sixth stage - calculate k6
    temp[idx] = orig;
    temp[idx].x = orig.x + (b[5][0] * k1[idx].vx + b[5][1] * k2[idx].vx + b[5][2] * k3[idx].vx +
                          b[5][3] * k4[idx].vx + b[5][4] * k5[idx].vx) * dt;
    temp[idx].y = orig.y + (b[5][0] * k1[idx].vy + b[5][1] * k2[idx].vy + b[5][2] * k3[idx].vy +
                          b[5][3] * k4[idx].vy + b[5][4] * k5[idx].vy) * dt;
    temp[idx].z = orig.z + (b[5][0] * k1[idx].vz + b[5][1] * k2[idx].vz + b[5][2] * k3[idx].vz +
                          b[5][3] * k4[idx].vz + b[5][4] * k5[idx].vz) * dt;
    temp[idx].vx = orig.vx + (b[5][0] * k1[idx].x + b[5][1] * k2[idx].x + b[5][2] * k3[idx].x +
                            b[5][3] * k4[idx].x + b[5][4] * k5[idx].x) * dt;
    temp[idx].vy = orig.vy + (b[5][0] * k1[idx].y + b[5][1] * k2[idx].y + b[5][2] * k3[idx].y +
                            b[5][3] * k4[idx].y + b[5][4] * k5[idx].y) * dt;
    temp[idx].vz = orig.vz + (b[5][0] * k1[idx].z + b[5][1] * k2[idx].z + b[5][2] * k3[idx].z +
                            b[5][3] * k4[idx].z + b[5][4] * k5[idx].z) * dt;
    // Recalculate forces here
    k6[idx] = temp[idx];
    k6[idx].x = temp[idx].ax;
    k6[idx].y = temp[idx].ay;
    k6[idx].z = temp[idx].az;

    // Final integration step - update position and velocity using weighted sum
    bodies[idx].x = orig.x + dt * (c[0] * k1[idx].vx + c[1] * k2[idx].vx + c[2] * k3[idx].vx +
                                 c[3] * k4[idx].vx + c[4] * k5[idx].vx + c[5] * k6[idx].vx);
    bodies[idx].y = orig.y + dt * (c[0] * k1[idx].vy + c[1] * k2[idx].vy + c[2] * k3[idx].vy +
                                 c[3] * k4[idx].vy + c[4] * k5[idx].vy + c[5] * k6[idx].vy);
    bodies[idx].z = orig.z + dt * (c[0] * k1[idx].vz + c[1] * k2[idx].vz + c[2] * k3[idx].vz +
                                 c[3] * k4[idx].vz + c[4] * k5[idx].vz + c[5] * k6[idx].vz);
    bodies[idx].vx = orig.vx + dt * (c[0] * k1[idx].x + c[1] * k2[idx].x + c[2] * k3[idx].x +
                                   c[3] * k4[idx].x + c[4] * k5[idx].x + c[5] * k6[idx].x);
    bodies[idx].vy = orig.vy + dt * (c[0] * k1[idx].y + c[1] * k2[idx].y + c[2] * k3[idx].y +
                                   c[3] * k4[idx].y + c[4] * k5[idx].y + c[5] * k6[idx].y);
    bodies[idx].vz = orig.vz + dt * (c[0] * k1[idx].z + c[1] * k2[idx].z + c[2] * k3[idx].z +
                                   c[3] * k4[idx].z + c[4] * k5[idx].z + c[5] * k6[idx].z);
}

    __global__ void updateSpaceshipsKernel(Body* bodies, int numBodies, double dt) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numBodies) return;

        // Only process bodies marked as spaceships
        if (bodies[idx].isSpaceship) {
            // This would apply thrust and handle spaceship-specific physics
            // In a real implementation, we'd need access to the Spaceship component
        }
    }

    __global__ void updateTrajectoriesKernel(Body* bodies, Trajectory* trajectories, int numBodies, double dt) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numBodies) return;

        Body& body = bodies[idx];
        Trajectory& traj = trajectories[idx];

        // Update time since last record
        traj.timeSinceLastRecord += dt;

        // If it's time to record a new point
        if (traj.timeSinceLastRecord >= traj.recordInterval) {
            // Reset timer
            traj.timeSinceLastRecord = 0.0f;

            // Record the point if there's space
            if (traj.currentSize < traj.maxPoints) {
                // Record position and velocity
                traj.positions[traj.currentSize] = make_float3(body.x, body.y, body.z);
                traj.velocities[traj.currentSize] = make_float3(body.vx, body.vy, body.vz);
                traj.currentSize++;
            }
            else {
                // If the buffer is full, shift everything back one spot
                for (int i = 0; i < traj.maxPoints - 1; i++) {
                    traj.positions[i] = traj.positions[i + 1];
                    traj.velocities[i] = traj.velocities[i + 1];
                }

                // Add new point at the end
                traj.positions[traj.maxPoints - 1] = make_float3(body.x, body.y, body.z);
                traj.velocities[traj.maxPoints - 1] = make_float3(body.vx, body.vy, body.vz);
            }
        }
    }

    __global__ void buildOctreeKernel(Body* bodies, int numBodies, OctreeNode* rootNode, BoundingBox globalBounds) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numBodies) return;

        Body& body = bodies[idx];
        OctreeNode* currentNode = rootNode;

        // Insert body into octree
        while (!currentNode->isLeaf) {
            // Calculate center of current node
            double centerX = (currentNode->bounds.minX + currentNode->bounds.maxX) / 2.0;
            double centerY = (currentNode->bounds.minY + currentNode->bounds.maxY) / 2.0;
            double centerZ = (currentNode->bounds.minZ + currentNode->bounds.maxZ) / 2.0;

            // Update center of mass and total mass using our custom atomicAdd
            atomicAddDouble(&currentNode->centerOfMassX, body.mass * body.x);
            atomicAddDouble(&currentNode->centerOfMassY, body.mass * body.y);
            atomicAddDouble(&currentNode->centerOfMassZ, body.mass * body.z);
            atomicAddDouble(&currentNode->totalMass, body.mass);

            // Determine which octant the body belongs to
            int octant = 0;
            if (body.x >= centerX) octant |= 1;
            if (body.y >= centerY) octant |= 2;
            if (body.z >= centerZ) octant |= 4;

            // Create child if it doesn't exist
            if (currentNode->children[octant] == nullptr) {
                OctreeNode* newNode;
                cudaMalloc(&newNode, sizeof(OctreeNode));

                // Initialize the new node
                newNode->isLeaf = true;
                newNode->bodyIndex = -1;
                newNode->totalMass = 0.0;
                newNode->centerOfMassX = newNode->centerOfMassY = newNode->centerOfMassZ = 0.0;
                for (int i = 0; i < 8; i++) {
                    newNode->children[i] = nullptr;
                }

                // Calculate bounds for the new node
                newNode->bounds.minX = (octant & 1) ? centerX : currentNode->bounds.minX;
                newNode->bounds.maxX = (octant & 1) ? currentNode->bounds.maxX : centerX;
                newNode->bounds.minY = (octant & 2) ? centerY : currentNode->bounds.minY;
                newNode->bounds.maxY = (octant & 2) ? currentNode->bounds.maxY : centerY;
                newNode->bounds.minZ = (octant & 4) ? centerZ : currentNode->bounds.minZ;
                newNode->bounds.maxZ = (octant & 4) ? currentNode->bounds.maxZ : centerZ;

                // Use atomic exchange to safely set the child pointer
                OctreeNode* expected = nullptr;
                OctreeNode* oldValue = atomicCASptr(&currentNode->children[octant], expected, newNode);

                // If oldValue is not null, another thread already created a child
                if (oldValue != nullptr) {
                    cudaFree(newNode);
                }
            }

            // Move to the child node
            currentNode = currentNode->children[octant];
        }

        // If current node is empty, add the body
        if (currentNode->bodyIndex == -1) {
            // Use atomicCAS to safely update the body index
            int expected = -1;
            int oldValue = atomicCAS(&currentNode->bodyIndex, expected, idx);

            // If we successfully claimed this node
            if (oldValue == -1) {
                atomicAddDouble(&currentNode->centerOfMassX, body.x * body.mass);
                atomicAddDouble(&currentNode->centerOfMassY, body.y * body.mass);
                atomicAddDouble(&currentNode->centerOfMassZ, body.z * body.mass);
                atomicAddDouble(&currentNode->totalMass, body.mass);
            }
            // Otherwise, we need to split the node
            else {
                // Mark this node as not a leaf
                bool wasLeaf = true;
                // Use atomicCAS to safely update isLeaf
                if (atomicCAS((int*)&currentNode->isLeaf, 1, 0) == 1) {
                    // We were the first to change this to a non-leaf node
                    // Reset the body index
                    atomicExch(&currentNode->bodyIndex, -1);

                    // Insert both bodies again - but we'll do this in a simplified way
                    // First, insert the body that was already in this node
                    Body& oldBody = bodies[oldValue];

                    // Calculate octant for old body
                    double centerX = (currentNode->bounds.minX + currentNode->bounds.maxX) / 2.0;
                    double centerY = (currentNode->bounds.minY + currentNode->bounds.maxY) / 2.0;
                    double centerZ = (currentNode->bounds.minZ + currentNode->bounds.maxZ) / 2.0;

                    int oldOctant = 0;
                    if (oldBody.x >= centerX) oldOctant |= 1;
                    if (oldBody.y >= centerY) oldOctant |= 2;
                    if (oldBody.z >= centerZ) oldOctant |= 4;

                    // Create a new node for the old body
                    OctreeNode* oldBodyNode;
                    cudaMalloc(&oldBodyNode, sizeof(OctreeNode));

                    // Initialize the new node
                    oldBodyNode->isLeaf = true;
                    oldBodyNode->bodyIndex = oldValue;
                    oldBodyNode->totalMass = oldBody.mass;
                    oldBodyNode->centerOfMassX = oldBody.x * oldBody.mass;
                    oldBodyNode->centerOfMassY = oldBody.y * oldBody.mass;
                    oldBodyNode->centerOfMassZ = oldBody.z * oldBody.mass;
                    for (int i = 0; i < 8; i++) {
                        oldBodyNode->children[i] = nullptr;
                    }

                    // Calculate bounds for the new node
                    oldBodyNode->bounds.minX = (oldOctant & 1) ? centerX : currentNode->bounds.minX;
                    oldBodyNode->bounds.maxX = (oldOctant & 1) ? currentNode->bounds.maxX : centerX;
                    oldBodyNode->bounds.minY = (oldOctant & 2) ? centerY : currentNode->bounds.minY;
                    oldBodyNode->bounds.maxY = (oldOctant & 2) ? currentNode->bounds.maxY : centerY;
                    oldBodyNode->bounds.minZ = (oldOctant & 4) ? centerZ : currentNode->bounds.minZ;
                    oldBodyNode->bounds.maxZ = (oldOctant & 4) ? currentNode->bounds.maxZ : centerZ;

                    // Set the child pointer
                    currentNode->children[oldOctant] = oldBodyNode;
                }

                // Continue insertion with current body on the next loop iteration
            }
        }
        // If current node already has a body, split it
        else {
            // The splitting logic is handled in the previous section
            // Set isLeaf to false to trigger the splitting process
            atomicExch((int*)&currentNode->isLeaf, 0);
        }
}

    // Barnes-Hut force calculation kernel
    __global__ void computeBarnesHutForcesKernel(Body* bodies, int numBodies, OctreeNode* rootNode, double theta, double G) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numBodies) return;

        Body& body = bodies[idx];
        body.ax = body.ay = body.az = 0.0;

        // Stack-based traversal to avoid recursion
        const int maxStackSize = 64;
        OctreeNode* stack[maxStackSize];
        int stackSize = 0;

        // Push root node to start
        stack[stackSize++] = rootNode;

        while (stackSize > 0) {
            // Pop a node from the stack
            OctreeNode* node = stack[--stackSize];

            if (node == nullptr) continue;

            // If node is a leaf and not our own body
            if (node->isLeaf && node->bodyIndex != idx && node->bodyIndex != -1) {
                Body& other = bodies[node->bodyIndex];

                // Calculate distance vector
                double dx = other.x - body.x;
                double dy = other.y - body.y;
                double dz = other.z - body.z;

                // Calculate squared distance with softening to prevent numerical instability
                double distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
                double invDist = 1.0 / sqrt(distSqr);
                double invDist3 = invDist * invDist * invDist;

                // Calculate gravitational force
                double forceMag = G * body.mass * other.mass * invDist3;

                // Apply acceleration (F = ma, so a = F/m)
                body.ax += forceMag * dx / body.mass;
                body.ay += forceMag * dy / body.mass;
                body.az += forceMag * dz / body.mass;
            }
            // If node is not a leaf
            else if (!node->isLeaf) {
                // Calculate distance to center of mass
                double dx = node->centerOfMassX / node->totalMass - body.x;
                double dy = node->centerOfMassY / node->totalMass - body.y;
                double dz = node->centerOfMassZ / node->totalMass - body.z;
                double distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;

                // Calculate node size
                double nodeSize = node->bounds.maxX - node->bounds.minX;
                double s_over_d = nodeSize / sqrt(distSqr);

                // If the node is far enough away, treat it as a point mass
                if (s_over_d < theta) {
                    double invDist = 1.0 / sqrt(distSqr);
                    double invDist3 = invDist * invDist * invDist;
                    double forceMag = G * body.mass * node->totalMass * invDist3;

                    body.ax += forceMag * dx / body.mass;
                    body.ay += forceMag * dy / body.mass;
                    body.az += forceMag * dz / body.mass;
                }
                // Otherwise, go deeper into the tree
                else {
                    // Push all children to the stack
                    for (int i = 0; i < 8; i++) {
                        if (node->children[i] != nullptr) {
                            stack[stackSize++] = node->children[i];
                        }
                    }
                }
            }
        }
    }
    __global__ void computeBarnesHutForcesNonRecursiveKernel(Body* bodies, int numBodies, OctreeNode* rootNode, double theta, double G) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numBodies) return;

        Body& body = bodies[idx];
        body.ax = body.ay = body.az = 0.0;

        // Stack-based traversal to avoid recursion
        const int maxStackSize = 64;  // Should be enough for most octrees
        OctreeNode* stack[maxStackSize];
        int stackSize = 0;

        // Push root node to start
        stack[stackSize++] = rootNode;

        while (stackSize > 0) {
            // Pop a node from the stack
            OctreeNode* node = stack[--stackSize];

            if (node == nullptr) continue;

            // If node is a leaf and not our own body
            if (node->isLeaf && node->bodyIndex != idx && node->bodyIndex != -1) {
                Body& other = bodies[node->bodyIndex];

                // Calculate distance vector
                double dx = other.x - body.x;
                double dy = other.y - body.y;
                double dz = other.z - body.z;

                // Calculate squared distance with softening to prevent numerical instability
                double distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
                double invDist = 1.0 / sqrt(distSqr);
                double invDist3 = invDist * invDist * invDist;

                // Calculate gravitational force
                double forceMag = G * body.mass * other.mass * invDist3;

                // Apply acceleration (F = ma, so a = F/m)
                body.ax += forceMag * dx / body.mass;
                body.ay += forceMag * dy / body.mass;
                body.az += forceMag * dz / body.mass;
            }
            // If node is not a leaf
            else if (!node->isLeaf) {
                // Calculate distance to center of mass
                double dx = node->centerOfMassX / node->totalMass - body.x;
                double dy = node->centerOfMassY / node->totalMass - body.y;
                double dz = node->centerOfMassZ / node->totalMass - body.z;
                double distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;

                // Calculate node size
                double nodeSize = node->bounds.maxX - node->bounds.minX;
                double s_over_d = nodeSize / sqrt(distSqr);

                // If the node is far enough away, treat it as a point mass
                if (s_over_d < theta) {
                    double invDist = 1.0 / sqrt(distSqr);
                    double invDist3 = invDist * invDist * invDist;
                    double forceMag = G * body.mass * node->totalMass * invDist3;

                    body.ax += forceMag * dx / body.mass;
                    body.ay += forceMag * dy / body.mass;
                    body.az += forceMag * dz / body.mass;
                }
                // Otherwise, go deeper into the tree
                else {
                    // Push all children to the stack (in reverse order)
                    for (int i = 7; i >= 0; i--) {
                        if (node->children[i] != nullptr && stackSize < maxStackSize) {
                            stack[stackSize++] = node->children[i];
                        }
                    }
                }
            }
        }
    }
    __device__ double atomicAddDouble(double* address, double val) {
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;

        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val + __longlong_as_double(assumed)));
        } while (assumed != old);

        return __longlong_as_double(old);
    }
    __device__ OctreeNode* atomicCASptr(OctreeNode** address, OctreeNode* compare, OctreeNode* val) {
        return (OctreeNode*)atomicCAS(
            (unsigned long long int*)address,
            (unsigned long long int)compare,
            (unsigned long long int)val
        );
    }

    // Kernel to calculate global bounds for the Barnes-Hut octree
    __global__ void calculateGlobalBoundsKernel(Body* bodies, int numBodies, OctreeNode* rootNode) {
        // This is a single thread operation to find the bounds of all bodies
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            // Initialize bounds to extreme values
            rootNode->bounds.minX = rootNode->bounds.minY = rootNode->bounds.minZ = 1e30;
            rootNode->bounds.maxX = rootNode->bounds.maxY = rootNode->bounds.maxZ = -1e30;

            // Find min and max for each coordinate
            for (int i = 0; i < numBodies; i++) {
                Body& body = bodies[i];

                // Update min bounds
                rootNode->bounds.minX = min(rootNode->bounds.minX, body.x);
                rootNode->bounds.minY = min(rootNode->bounds.minY, body.y);
                rootNode->bounds.minZ = min(rootNode->bounds.minZ, body.z);

                // Update max bounds
                rootNode->bounds.maxX = max(rootNode->bounds.maxX, body.x);
                rootNode->bounds.maxY = max(rootNode->bounds.maxY, body.y);
                rootNode->bounds.maxZ = max(rootNode->bounds.maxZ, body.z);
            }

            // Add some padding to ensure bodies at edges are included
            double padding = 0.01 * (rootNode->bounds.maxX - rootNode->bounds.minX);
            if (padding < 1e-10) padding = 1.0; // Minimum padding

            rootNode->bounds.minX -= padding;
            rootNode->bounds.minY -= padding;
            rootNode->bounds.minZ -= padding;
            rootNode->bounds.maxX += padding;
            rootNode->bounds.maxY += padding;
            rootNode->bounds.maxZ += padding;
        }
    }
}
