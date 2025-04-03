#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "ecs/ECS.cuh"
#include "components/Body.cuh"
#include "system/PhysicsSystem.cuh"
#include "system/SpaceshipSystem.cuh"

// Forward declarations for any rendering functions (not shown in provided code)
void initializeRenderer();
void renderScene(const std::vector<ecs::EntityID>& entities, ecs::ECSCoordinator* coordinator);
void cleanupRenderer();

// Helper functions for simulation setup
void createSolarSystem(ecs::ECSCoordinator* coordinator, std::vector<ecs::EntityID>& entities);
void createSpaceships(ecs::ECSCoordinator* coordinator, std::vector<ecs::EntityID>& entities, int count);

int main() {
    // Initialize CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA initialization failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    // Create the ECS coordinator
    ecs::ECSCoordinator coordinator;

    // Register components
    coordinator.registerComponent<Body>();
    coordinator.registerComponent<Spaceship>();

    // Register systems
    auto physicsSystem = coordinator.registerSystem<systems::PhysicsSystem>();
    auto spaceshipSystem = coordinator.registerSystem<systems::SpaceshipSystem>();

    // Set up system signatures
    {
        // Physics system needs only Body component
        ecs::Signature physicsSignature;
        physicsSignature.set(coordinator.getComponentID<Body>());
        coordinator.setSystemSignature<systems::PhysicsSystem>(physicsSignature);

        // Spaceship system needs both Body and Spaceship components
        ecs::Signature spaceshipSignature;
        spaceshipSignature.set(coordinator.getComponentID<Body>());
        spaceshipSignature.set(coordinator.getComponentID<Spaceship>());
        coordinator.setSystemSignature<systems::SpaceshipSystem>(spaceshipSignature);
    }

    // Set coordinator reference to spaceship system
    spaceshipSystem->setCoordinator(&coordinator);

    // Create entities
    std::vector<ecs::EntityID> entities;
    createSolarSystem(&coordinator, entities);
    createSpaceships(&coordinator, entities, 5);

    // Initialize physics system
    physicsSystem->init();

    // Initialize rendering (implementation not provided)
    initializeRenderer();

    // Simulation variables
    float deltaTime = 0.016f;  // ~60 fps
    bool running = true;
    auto lastFrameTime = std::chrono::high_resolution_clock::now();

    // Main simulation loop
    while (running) {
        // Calculate delta time
        auto currentTime = std::chrono::high_resolution_clock::now();
        deltaTime = std::chrono::duration<float>(currentTime - lastFrameTime).count();
        lastFrameTime = currentTime;

        // Cap delta time to avoid instability with very large steps
        if (deltaTime > 0.1f) deltaTime = 0.1f;

        // Update physics
        physicsSystem->update(deltaTime);

        // Update spaceships
        spaceshipSystem->update(deltaTime);

        // Render scene (implementation not provided)
        renderScene(entities, &coordinator);

        // Handle user input and check for exit conditions
        // (simplified - actual implementation would depend on your input system)

    }

    // Cleanup
    cleanupRenderer();

    // Cleanup CUDA
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceReset failed: " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    return 0;
}

// Implementation of helper functions

void createSolarSystem(ecs::ECSCoordinator* coordinator, std::vector<ecs::EntityID>& entities) {
    // Create the Sun
    {
        ecs::EntityID sun = coordinator->createEntity();
        entities.push_back(sun);

        Body sunBody;
        sunBody.mass = 1.989e30;  // Sun's mass in kg
        sunBody.radius = 696340e3;  // Sun's radius in meters
        sunBody.x = sunBody.y = sunBody.z = 0.0;  // At the center
        sunBody.vx = sunBody.vy = sunBody.vz = 0.0;  // Stationary
        sunBody.isStatic = true;  // Sun doesn't move

        coordinator->addComponent<Body>(sun, sunBody);
    }

    // Create Earth
    {
        ecs::EntityID earth = coordinator->createEntity();
        entities.push_back(earth);

        Body earthBody;
        earthBody.mass = 5.972e24;  // Earth's mass in kg
        earthBody.radius = 6371e3;  // Earth's radius in meters
        earthBody.x = 149.6e9;  // Earth's distance from Sun in meters
        earthBody.y = 0.0;
        earthBody.z = 0.0;
        earthBody.vx = 0.0;
        earthBody.vy = 29780.0;  // Earth's orbital velocity in m/s
        earthBody.vz = 0.0;

        coordinator->addComponent<Body>(earth, earthBody);
    }

    // Add more planets as needed...
}

void createSpaceships(ecs::ECSCoordinator* coordinator, std::vector<ecs::EntityID>& entities, int count) {
    for (int i = 0; i < count; i++) {
        ecs::EntityID ship = coordinator->createEntity();
        entities.push_back(ship);

        // Create ship body
        Body shipBody;
        shipBody.mass = 1000.0;  // 1000 kg
        shipBody.radius = 10.0;  // 10 meters
        shipBody.x = 149.6e9 + 10000e3 + i * 1000.0;  // Near Earth
        shipBody.y = i * 1000.0;
        shipBody.z = i * 500.0;
        shipBody.vx = 0.0;
        shipBody.vy = 29780.0 + i * 10.0;  // Slightly faster than Earth
        shipBody.vz = 0.0;
        shipBody.isSpaceship = true;

        // Create spaceship component
        Spaceship ship;
        ship.fuel = 1000.0;
        ship.maxThrust = 10000.0;
        ship.thrust = 0.0;
        ship.dirX = 1.0;
        ship.dirY = 0.0;
        ship.dirZ = 0.0;
        ship.angularVelocity = 0.1;

        coordinator->addComponent<Body>(ship, shipBody);
        coordinator->addComponent<Spaceship>(ship, ship);
    }
}

// Placeholder implementations for rendering functions
void initializeRenderer() {
    // Initialize rendering system
    std::cout << "Initializing renderer..." << std::endl;
}

void renderScene(const std::vector<ecs::EntityID>& entities, ecs::ECSCoordinator* coordinator) {
    // Render all entities
    // This would be replaced with actual rendering code
}

void cleanupRenderer() {
    // Clean up rendering resources
    std::cout << "Cleaning up renderer..." << std::endl;
}