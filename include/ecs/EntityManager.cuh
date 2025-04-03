//
// Created by DevAccount on 28/03/2025.
//

#ifndef ENTITYMANAGER_CUH
#define ENTITYMANAGER_CUH
#pragma once
#include "ECS.cuh"
#include <queue>

namespace ecs {
    class EntityManager {
        private:
            std::queue<EntityID> availableEntities;
            std::array<Signature, MAX_ENTITIES> signatures;
            uint32_t livingEntityCount = 0;
        public:
            EntityManager() {
                // Initialize available entity IDs
                for (EntityID entity = 0; entity < MAX_ENTITIES; ++entity) {
                    availableEntities.push(entity);
                }
            }

            EntityID createEntity() {
                assert(livingEntityCount < MAX_ENTITIES && "Too many entities!");

                // Get an ID from the queue
                EntityID id = availableEntities.front();
                availableEntities.pop();
                ++livingEntityCount;

                return id;
            }

            void destroyEntity(EntityID entity) {
                assert(entity < MAX_ENTITIES && "Entity out of range!");

                // Reset the entity's signature
                signatures[entity].reset();

                // Put the destroyed ID back into the queue
                availableEntities.push(entity);
                --livingEntityCount;
            }

            void setSignature(EntityID entity, Signature signature) {
                assert(entity < MAX_ENTITIES && "Entity out of range!");

                signatures[entity] = signature;
            }

            Signature getSignature(const EntityID entity) {
                assert(entity < MAX_ENTITIES && "Entity out of range!");

                return signatures[entity];
            }
    };
}

#endif //ENTITYMANAGER_CUH
