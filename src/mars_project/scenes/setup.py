"""Preview-only scene, lighting and constant-velocity command configuration."""


def configure_preview(cfg, appearance, forward_velocity=0.0, soil="mpm"):
    """Use a constant body-forward command and configure the visual backdrop."""
    cfg.observations.policy.enable_corruption = False
    cfg.events.push_robot = None
    # Both backdrops use fixed USD ground, not generated terrain levels.
    cfg.curriculum.terrain_levels = None
    cfg.commands.base_velocity.ranges.lin_vel_x = (forward_velocity, forward_velocity)
    if forward_velocity:
        # Otherwise the inherited standing fraction can zero the walking command.
        cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.debug_vis = False
    cfg.viewer.eye = (4.0, 4.0, 2.5)
    cfg.viewer.lookat = (0.0, 0.0, 0.9)
    if appearance == "grand-canyon":
        from mars_project.scenes.canyon import configure_canyon_scene

        configure_canyon_scene(cfg, soil=soil)
    if appearance == "earth":
        from mars_project.scenes.earth import configure_earth_scene

        configure_earth_scene(cfg)
