//
// Created by DevAccount on 02/04/2025.
//


#include "ecs/ComponentManager.cuh"
#include "ecs/ECS.cuh"
#include "ecs/EntityManager.cuh"
#include "ecs/SystemManager.cuh"

namespace ecs {
    template<typename T>
   void ECSCoordinator::registerComponent() {
        // Assuming componentManager has a registerComponent method
        componentManager->registerComponent<T>();
    }

    template<typename T>
    void ECSCoordinator::addComponent(EntityID entity, T component) {
        componentManager->addComponent<T>(entity, component);

        // Update entity signature
        Signature signature = entityManager->getSignature(entity);
        signature.set(getComponentID<T>(), true);
        entityManager->setSignature(entity, signature);

        // Notify systems
        systemManager->entitySignatureChanged(entity, signature);
    }

    template<typename T>
    void ECSCoordinator::removeComponent(EntityID entity) const {
        componentManager->removeComponent<T>(entity);

        // Update entity signature
        Signature signature = entityManager->getSignature(entity);
        signature.set(getComponentID<T>(), false);
        entityManager->setSignature(entity, signature);

        // Notify systems
        systemManager->entitySignatureChanged(entity, signature);
    }

    template<typename T>
    T& ECSCoordinator::getComponent(EntityID entity) {

        return componentManager->getComponent<T>(entity);
    }

    template<typename T>
    ComponentID ECSCoordinator::getComponentID() {
        return componentManager->getComponentID<T>();
    }

    template<typename T>
    std::shared_ptr<T> ECSCoordinator::registerSystem() {
        return systemManager->registerSystem<T>();
    }

    template<typename T>
    void ECSCoordinator::setSystemSignature(Signature signature) const {
        systemManager->setSignature<T>(signature);
    }
}
