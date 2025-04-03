//
// Created by DevAccount on 02/04/2025.
//

#ifndef SPACESHIPSYSTEM_CUH
#define SPACESHIPSYSTEM_CUH

#include <ecs/ECS.cuh>
#include <components/Body.cuh>
#include <memory>

namespace systems {
    class SpaceshipSystem : public ecs::System {
    private:
        ecs::ECSCoordinator* coordinator{nullptr};

    public:
        SpaceshipSystem() = default;

        void setCoordinator(ecs::ECSCoordinator* coord) {
            coordinator = coord;
        }

        void update(float deltaTime) {
            for (auto entityID : entities) {
                auto& body = coordinator->getComponent<Body>(entityID);

                if (body.isSpaceship) {
                    // Get the spaceship component
                    auto& spaceship = coordinator->getComponent<Spaceship>(entityID);
                    updateSpaceshipPhysics(body, spaceship, deltaTime);
                    updateSpaceshipAI(body, spaceship, deltaTime);
                }
            }
        }

    private:
        void updateSpaceshipPhysics(Body& body, Spaceship& spaceship, double deltaTime) {
            // Apply thrust - can't call device function from host code
            // Instead implement thrust logic directly
            if (spaceship.fuel > 0 && spaceship.thrust > 0) {
                // Calculate thrust vector based on direction
                double thrustMagnitude = spaceship.thrust;

                // Apply thrust acceleration to the body
                body.ax += thrustMagnitude * spaceship.dirX;
                body.ay += thrustMagnitude * spaceship.dirY;
                body.az += thrustMagnitude * spaceship.dirZ;

                // Consume fuel
                double fuelConsumption = thrustMagnitude * deltaTime * 0.1; // Adjust fuel consumption rate
                spaceship.fuel = max(0.0, spaceship.fuel - fuelConsumption);
            }

            // Apply rotation - can't call device function from host code
            // Implement rotation logic directly
            if (spaceship.angularVelocity > 0) {
                // Simplified rotation around Z axis
                double rotationAngle = spaceship.angularVelocity * deltaTime;
                double cosAngle = cos(rotationAngle);
                double sinAngle = sin(rotationAngle);

                // Rotate direction vector
                double newDirX = spaceship.dirX * cosAngle - spaceship.dirY * sinAngle;
                double newDirY = spaceship.dirX * sinAngle + spaceship.dirY * cosAngle;

                spaceship.dirX = newDirX;
                spaceship.dirY = newDirY;

                // Normalize direction vector
                double length = sqrt(spaceship.dirX * spaceship.dirX +
                                   spaceship.dirY * spaceship.dirY +
                                   spaceship.dirZ * spaceship.dirZ);

                if (length > 0) {
                    spaceship.dirX /= length;
                    spaceship.dirY /= length;
                    spaceship.dirZ /= length;
                }
            }
        }

        void updateSpaceshipAI(Body& body, Spaceship& spaceship, double deltaTime) {
            // Simple AI behavior
            spaceship.thrust = spaceship.maxThrust * 0.5;
        }
    };
}

#endif //SPACESHIPSYSTEM_CUH