//
// Created by DevAccount on 28/03/2025.
//

#ifndef SYSTEMMANAGER_CUH
#define SYSTEMMANAGER_CUH
#pragma once

#include <ecs/ECS.cuh>
#include <memory>
#include <unordered_map>
#include <typeindex>

namespace ecs {

class System {
public:
    std::vector<EntityID> entities;
};

class SystemManager {
    private:
        std::unordered_map<std::type_index, std::shared_ptr<System>> systems;
        std::unordered_map<std::type_index, Signature> signatures;

    public:
        template<typename T>
        std::shared_ptr<T> registerSystem() {
            std::type_index typeIndex = std::type_index(typeid(T));
            assert(systems.find(typeIndex) == systems.end() && "System already registered!");

            auto system = std::make_shared<T>();
            systems.insert({typeIndex, system});
            return system;
        }

        template<typename T>
        void setSignature(Signature signature) {
            std::type_index typeIndex = std::type_index(typeid(T));
            assert(systems.find(typeIndex) != systems.end() && "System not registered!");

            signatures[typeIndex] = signature;
        }

        void entityDestroyed(EntityID entity) {
            for (auto const& pair : systems) {
                auto const& system = pair.second;
                system->entities.erase(
                    std::remove(system->entities.begin(), system->entities.end(), entity),
                    system->entities.end()
                );
            }
        }

        void entitySignatureChanged(EntityID entity, Signature entitySignature) {
            for (auto const& pair : systems) {
                auto const& type = pair.first;
                auto const& system = pair.second;
                auto const& systemSignature = signatures[type];

                // Entity signature matches system signature - insert into set
                if ((entitySignature & systemSignature) == systemSignature) {
                    system->entities.push_back(entity);
                }
                // Entity signature no longer matches system signature - erase from set
                else {
                    system->entities.erase(
                        std::remove(system->entities.begin(), system->entities.end(), entity),
                        system->entities.end()
                    );
                }
            }
        }
};

}

#endif //SYSTEMMANAGER_CUH
