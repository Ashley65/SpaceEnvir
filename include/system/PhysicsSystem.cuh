//
// Created by DevAccount on 02/04/2025.
//

#ifndef PHYSICSSYSTEM_CUH
#define PHYSICSSYSTEM_CUH


#pragma once

#include <ecs/ECS.cuh>
#include <physics/physicsEngine.cuh>
#include <components/Body.cuh>

#include "ecs/SystemManager.cuh"

namespace systems {

    class PhysicsSystem : public ecs::System {
        private:
            ecs::ECSCoordinator* coordinator;
            engine::PhysicsEngine physicsEngine;

        public:
            explicit PhysicsSystem(ecs::ECSCoordinator* coordinator)
            : coordinator(coordinator), physicsEngine(coordinator) {}

            void init() {
                // Register all existing entities with the physics engine
                for (auto entity : entities) {
                    physicsEngine.registerEntity(entity);
                }
                physicsEngine.initialise();
            }

            void update(float deltaTime) {
                physicsEngine.update(deltaTime);
            }

            void entityAdded(ecs::EntityID entity) {
                physicsEngine.registerEntity(entity);
            }

            void entityRemoved(ecs::EntityID entity) {
                physicsEngine.unregisterEntity(entity);
            }

            void resetSimulation() {
                // Reset physics state if needed
                physicsEngine.cleanup();
                physicsEngine.initialise();
            }
    };

}

#endif //PHYSICSSYSTEM_CUH
