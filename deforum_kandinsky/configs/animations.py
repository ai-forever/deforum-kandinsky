from .animation_args import DeforumAnimArgs
from .deforum_args import DeforumArgs

def update_anim_args(deforum_args, anim_args, **kwargs):
    for key, value in kwargs.items():
        if key in deforum_args: 
            if key in ["W", "H"]:
                value = value - value % 64
            deforum_args[key] = value
        elif key in anim_args:
            anim_args[key] = value
        else: 
            raise KeyError(str(key))
    return deforum_args, anim_args
            
animations = dict(	
	right=update_anim_args(
        DeforumArgs(), 
        DeforumAnimArgs(),
        animation_mode="3D",
        translation_x="0:(2.5)",
        max_frames=96,
        strength_schedule = "0:(0.12), 96:(0.15)",
    ),
    left=update_anim_args(
        DeforumArgs(), 
        DeforumAnimArgs(),
        animation_mode="3D",
        translation_x="0:(-2.5)",
        max_frames=96,
        strength_schedule = "0:(0.12), 96:(0.15)",
    ),
    up=update_anim_args(
        DeforumArgs(), 
        DeforumAnimArgs(),
        animation_mode="3D",
        translation_y="0:(2.5)",
        max_frames=96,
        strength_schedule = "0:(0.15), 96:(0.2)",
    ),
    down=update_anim_args(
        DeforumArgs(), 
        DeforumAnimArgs(),
        animation_mode="3D",
        translation_y="0:(-2.5)", 
        max_frames=96,
        strength_schedule = "0:(0.15), 96:(0.2)",
    ),
    zoomin=update_anim_args(
        DeforumArgs(), 
        DeforumAnimArgs(),
        animation_mode="2D",
        zoom="0:(1.018)",
        max_frames=96,
        strength_schedule = "0:(0.15), 96:(0.16)",
    ),
    zoomout=update_anim_args(
        DeforumArgs(), 
        DeforumAnimArgs(),
        zoom="0:(0.97)", 
        animation_mode="2D", 
        max_frames=96,
        strength_schedule = "0:(0.15), 96:(0.16)",
    ),
    spin_clockwise=update_anim_args(
        DeforumArgs(), 
        DeforumAnimArgs(),
        animation_mode = "2D",
        angle="0:(-3)", 
        max_frames=96,
        strength_schedule = "0:(0.15), 96:(0.16)",
    ),
    spin_counterclockwise=update_anim_args(
        DeforumArgs(), 
        DeforumAnimArgs(),
        animation_mode = "2D",
        angle="0:(3)", 
        max_frames=96,
        strength_schedule = "0:(0.15), 96:(0.16)",
    ),
    
    flipping_gamma = update_anim_args(
        DeforumArgs(), 
        DeforumAnimArgs(),
        animation_mode = "2D",
        flip_2d_perspective = True, 
        perspective_flip_gamma =  "0:(7.0)", 
        max_frames=96,
        strength_schedule = "0:(0.15), 96:(0.25)",
    ), 
    flipping_phi = update_anim_args(
        DeforumArgs(), 
        DeforumAnimArgs(),
        animation_mode = "2D",
        flip_2d_perspective = True, 
        perspective_flip_phi = "0:(7.0)", 
        max_frames=96,
        strength_schedule = "0:(0.15), 96:(0.25)",
    ), 
    flipping_theta = update_anim_args(
        DeforumArgs(), 
        DeforumAnimArgs(),
        animation_mode = "2D",
        flip_2d_perspective = True, 
        perspective_flip_theta =  "0:(7.0)",
        max_frames=96,
        strength_schedule = "0:(0.15), 96:(0.25)",
    ),
    perspective = update_anim_args(
        DeforumArgs(), 
        DeforumAnimArgs(),
        scale = 13,
        animation_mode = "3D",
        zoom = "0:(1)",
        translation_x = "0:(0.0), 1:(3.0), 89:(0.0), 90:(-3.0), 179:(0.0), 180:(-3.0), 269:(0.0), 270:(3.0), 359:(0.0), ",
        rotation_3d_y = "0:(0.0), 1:(-0.5), 89:(0.0), 90:(0.5), 179:(0.0), 180:(0.5), 269:(0.0), 270:(-0.5), 359:(0.0), ",
        strength_schedule = "0:(0.2), 360:(0.25)",
        use_depth_warping = True,
        midas_weight = 0.3,
        near_plane = 200,
        far_plane = 10000,
        fov = 90,
        diffusion_cadence = "4",
        max_frames = 360,
    ),
    panorama = update_anim_args(
        DeforumArgs(), 
        DeforumAnimArgs(),
        animation_mode="3D",
        rotation_3d_y="0:(0.0), 1:(2.0), 45:(0.0), 46:(-2.0), 90:(0.0), 91:(-2.0), 135:(0.0), 136:(2.0), 180:(0.0)", 
        diffusion_cadence="2",
        max_frames = 180,
        strength_schedule = "0:(0.15), 180:(0.16)",
    ),
    live=update_anim_args(
        DeforumArgs(), 
        DeforumAnimArgs(),
        max_frames=96,
        strength_schedule = "0:(0.15), 180:(0.16)",
        diffusion_cadence = 4,
    ),
    default=update_anim_args(
        DeforumArgs(), 
        DeforumAnimArgs(),
    )
)