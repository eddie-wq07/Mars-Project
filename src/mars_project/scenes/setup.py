"""Preview-only scene, lighting and stationary-command configuration."""

def configure_preview(cfg, appearance):
    """Use a single stationary command and configure the visual backdrop."""
    cfg.observations.policy.enable_corruption = False
    cfg.events.push_robot = None
    cfg.commands.base_velocity.ranges.lin_vel_x = (0., 0.)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0., 0.)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0., 0.)
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.debug_vis = False
    cfg.viewer.eye = (4., 4., 2.5)
    cfg.viewer.lookat = (0., 0., 0.9)
    if appearance == 'grand-canyon':
        from mars_project.scenes.canyon import configure_canyon_scene
        configure_canyon_scene(cfg)
    if appearance == 'earth':
        from mars_project.scenes.earth import configure_earth_scene
        configure_earth_scene(cfg)
