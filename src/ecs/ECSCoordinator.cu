//
// Created by DevAccount on 28/03/2025.
//


#include "components/Body.cuh"
#include "ecs/ComponentManager.cuh"
#include "ecs/ECS.cuh"
#include "ecs/EntityManager.cuh"
#include "ecs/SystemManager.cuh"
#include "system/PhysicsSystem.cuh"
#include "system/SpaceshipSystem.cuh"

namespace ecs {
    ECSCoordinator::ECSCoordinator() {
        entityManager = std::make_unique<EntityManager>();
        componentManager = std::make_unique<ComponentManager>();
        systemManager = std::make_unique<SystemManager>();
    }
    ECSCoordinator::~ECSCoordinator() = default;
    EntityID ECSCoordinator::createEntity() const {
        return entityManager->createEntity();
    }
    void ECSCoordinator::destroyEntity(EntityID entity) const {
        assert(entity < MAX_ENTITIES && "Entity out of range!");
        entityManager->destroyEntity(entity);
        systemManager->entityDestroyed(entity);
    }

    // Instantiate registerComponent for needed types
    template void ECSCoordinator::registerComponent<Body>();
    template void ECSCoordinator::registerComponent<Spaceship>();

    // Component operations
    template void ECSCoordinator::addComponent<Body>(EntityID, Body);
    template void ECSCoordinator::addComponent<Spaceship>(EntityID, Spaceship);

    template void ECSCoordinator::removeComponent<Body>(EntityID) const;
    template void ECSCoordinator::removeComponent<Spaceship>(EntityID) const;

    template Body& ECSCoordinator::getComponent<Body>(EntityID);
    template Spaceship& ECSCoordinator::getComponent<Spaceship>(EntityID);

    template ComponentID ECSCoordinator::getComponentID<Body>();
    template ComponentID ECSCoordinator::getComponentID<Spaceship>();

    // System registration
    template std::shared_ptr<systems::PhysicsSystem> ECSCoordinator::registerSystem<systems::PhysicsSystem>();
    template std::shared_ptr<systems::SpaceshipSystem> ECSCoordinator::registerSystem<systems::SpaceshipSystem>();

    // System signature setting
    template void ECSCoordinator::setSystemSignature<systems::PhysicsSystem>(Signature) const;
    template void ECSCoordinator::setSystemSignature<systems::SpaceshipSystem>(Signature) const;


}
