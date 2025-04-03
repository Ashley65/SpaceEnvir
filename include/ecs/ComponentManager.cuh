 //
// Created by DevAccount on 28/03/2025.
//

#ifndef COMPONENTMANAGER_CUH
#define COMPONENTMANAGER_CUH
#pragma once
#include "ECS.cuh"
#include <unordered_map>
#include <memory>
#include <typeindex>

namespace ecs {

// Base component array class
class IComponentArray {
public:
    virtual ~IComponentArray() = default;
    virtual void entityDestroyed(EntityID entity) = 0;
};

// Specialised component array for each component type
template<typename T>
class ComponentArray : public IComponentArray {
private:
    std::array<T, MAX_ENTITIES> componentArray;
    std::unordered_map<EntityID, size_t> entityToIndexMap;
    std::unordered_map<size_t, EntityID> indexToEntityMap;
    size_t size = 0;

public:
    void insertData(EntityID entity, T component) {
        assert(entityToIndexMap.find(entity) == entityToIndexMap.end() && "Component already exists for this entity!");

        // Put the component at the end and update the maps
        size_t newIndex = size;
        entityToIndexMap[entity] = newIndex;
        indexToEntityMap[newIndex] = entity;
        componentArray[newIndex] = component;
        ++size;
    }

    void removeData(EntityID entity) {
        assert(entityToIndexMap.find(entity) != entityToIndexMap.end() && "Removing non-existent component!");

        // Copy the last element to the deleted position to maintain density
        size_t indexOfRemovedEntity = entityToIndexMap[entity];
        size_t indexOfLastElement = size - 1;
        componentArray[indexOfRemovedEntity] = componentArray[indexOfLastElement];

        // Update the maps
        EntityID entityOfLastElement = indexToEntityMap[indexOfLastElement];
        entityToIndexMap[entityOfLastElement] = indexOfRemovedEntity;
        indexToEntityMap[indexOfRemovedEntity] = entityOfLastElement;

        entityToIndexMap.erase(entity);
        indexToEntityMap.erase(indexOfLastElement);

        --size;
    }

    T& getData(EntityID entity) {
        assert(entityToIndexMap.find(entity) != entityToIndexMap.end() && "Retrieving non-existent component!");

        return componentArray[entityToIndexMap[entity]];
    }

    void entityDestroyed(EntityID entity) override {
        if (entityToIndexMap.find(entity) != entityToIndexMap.end()) {
            removeData(entity);
        }
    }
};

class ComponentManager {
private:
    std::unordered_map<std::type_index, ComponentID> componentTypes;
    std::unordered_map<std::type_index, std::shared_ptr<IComponentArray>> componentArrays;
    ComponentID nextComponentID = 0;

public:
    template<typename T>
    void registerComponent() {
        std::type_index typeIndex = std::type_index(typeid(T));
        assert(componentTypes.find(typeIndex) == componentTypes.end() && "Component type already registered!");

        componentTypes[typeIndex] = nextComponentID;
        componentArrays[typeIndex] = std::make_shared<ComponentArray<T>>();
        ++nextComponentID;
    }

    template<typename T>
    ComponentID getComponentID() {
        std::type_index typeIndex = std::type_index(typeid(T));
        assert(componentTypes.find(typeIndex) != componentTypes.end() && "Component not registered!");

        return componentTypes[typeIndex];
    }

    template<typename T>
    void addComponent(EntityID entity, T component) {
        std::type_index typeIndex = std::type_index(typeid(T));
        getComponentArray<T>()->insertData(entity, component);
    }

    template<typename T>
    void removeComponent(EntityID entity) {
        std::type_index typeIndex = std::type_index(typeid(T));
        getComponentArray<T>()->removeData(entity);
    }

    template<typename T>
    T& getComponent(EntityID entity) {
        return getComponentArray<T>()->getData(entity);
    }

    void entityDestroyed(EntityID entity) {
        for (auto const& pair : componentArrays) {
            pair.second->entityDestroyed(entity);
        }
    }

private:
    template<typename T>
    std::shared_ptr<ComponentArray<T>> getComponentArray() {
        std::type_index typeIndex = std::type_index(typeid(T));
        assert(componentTypes.find(typeIndex) != componentTypes.end() && "Component not registered!");

        return std::static_pointer_cast<ComponentArray<T>>(componentArrays[typeIndex]);
    }
};

}

#endif //COMPONENTMANAGER_CUH
