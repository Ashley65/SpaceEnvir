//
// Created by DevAccount on 28/03/2025.
//

#ifndef ECS_CUH
#define ECS_CUH

#pragma once
#include <vector>
#include <unordered_map>
#include <memory>
#include <bitset>
#include <array>
#include <cassert>
#include <cuda_runtime.h>




namespace ecs {
    using EntityID = std::uint32_t;
    using ComponentID = std::uint8_t;
    using Signature = std::bitset<64>; // Support up to 64 component types

    constexpr EntityID  MAX_ENTITIES = 10000;
    constexpr ComponentID Max_COMPONENTS = 64;


    // Forward declaration of the EntityManager class
    class EntityManager;
    class ComponentManager;
    class SystemManager;

    class ECSCoordinator {
        private:
            std::unique_ptr<EntityManager> entityManager;
            std::unique_ptr<ComponentManager> componentManager;
            std::unique_ptr<SystemManager> systemManager;
        public:
            ECSCoordinator();
            ~ECSCoordinator();

            // Entity methods
            EntityID createEntity() const;
            void destroyEntity(EntityID entity) const;

            // Component methods
            template<typename T>
            void registerComponent();

            template<typename T>
            void addComponent(EntityID entity, T component);

            template<typename T>
            void removeComponent(EntityID entity) const;

            template<typename T>
            T& getComponent(EntityID entity);

            template<typename T>
            ComponentID getComponentID();

            // System methods
            template<typename T>
            std::shared_ptr<T> registerSystem();

            template<typename T>
            void setSystemSignature(Signature signature) const;
    };


}

#endif //ECS_CUH
