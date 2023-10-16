from ..configs.animation_args import DeforumAnimArgs
from ..configs.deforum_args import DeforumArgs
from typing import List, Optional, Union
import json

class Script:
    def __init__(
            self, 
            animations:List[str], 
            durations:List[Union[int, float]], 
            accelerations:List[Union[int, float]],
            linear_transition=True,
            fps=24,
            **kwargs
            ):
        assert isinstance(animations, list)
        self.deforum_args = DeforumArgs()
        self.anim_args = DeforumAnimArgs()
        seconds_elapsed = 0
        for index, (animation, duration, acceleration) in enumerate(zip(animations, durations, accelerations)): 
            self.process_animation(
                animation, 
                start_frame=int((seconds_elapsed)*fps), 
                n_frames=int(duration*fps), 
                acceleration=acceleration,
                linear_transition=linear_transition
                )
            
            seconds_elapsed += duration
    
        if len(set(animations))==1 and animations[0]=="flipping_phi":
            self.anim_args["animation_mode"] = "2D"
        self._update_anim_args(**kwargs)

    def _animation_params_dict(self, acceleration=1): 
        assert acceleration > 0
        params_dict = {
            "right": dict(
                translation_x = acceleration,
            ),
            "left": dict(
                translation_x = -acceleration
            ),
            "up": dict(
                translation_y = acceleration
            ),
            "down": dict(
                translation_y = - acceleration
            ),
            "spin_clockwise": dict(
                rotation_3d_z = acceleration
            ),
            "spin_counterclockwise": dict(
                rotation_3d_z = -acceleration
            ),
            "zoomin": dict(
                translation_z = acceleration
            ),
            "zoomout": dict(
                translation_z = -acceleration
            ),
            "rotate_right": dict(
                rotation_3d_y = acceleration
            ),
            "rotate_left": dict(
                rotation_3d_y = -acceleration
            ),
            "rotate_up": dict(
                rotation_3d_x = acceleration
            ),
            "rotate_down": dict(
                rotation_3d_x = -acceleration
            ),
            "around_right": dict(
                translation_x = 2.5 * acceleration, 
                rotation_3d_y =-0.5 * acceleration,
            ),
            "around_left": dict(
                translation_x = -2.5 * acceleration, 
                rotation_3d_y = 0.5 * acceleration,
            ),
            "zoomin_sinus_x": dict(
                translation_z = acceleration, 
                translation_x = f"{acceleration}*1.5*sin(3.14*t/24))"
            ),
            "zoomout_sinus_y": dict(
                translation_z = -acceleration, 
                translation_y = f"{acceleration}*1.5*sin(3.14*t/24))"
            ),
            "right_sinus_y": dict(
                translation_x = acceleration, 
                translation_y = f"{acceleration}*1.5*sin(3.14*t/24))"
            ),
            "left_sinus_y": dict(
                translation_z = -acceleration, 
                translation_x = f"{acceleration}*1.5*sin(3.14*t/24))"
            ),
            "flipping_phi": dict( 
                perspective_flip_phi = 5.0 * acceleration,
            ),
            "live": dict(
                translation_x = "sin(3.14159*t/90)*0.05",
            ),
        }
        return params_dict

    def process_animation(self, animation:str, start_frame:int, n_frames:int=48, acceleration=1.0, linear_transition:bool=True): 
        assert isinstance(start_frame, int) and isinstance(n_frames, int)        

        if "spin" in animation:
            linear_transition = False
        
        # parse string to dictionary
        animations_params = self._animation_params_dict(acceleration)
        assert animation in animations_params, f"animation should be one of {list(animations_params.keys())}"
        
        for animations_param, value in animations_params[animation].items():
            current_param = dict([item.split(":") for item in self.anim_args[animations_param].split(",")])
            current_param = {int(k):v for k,v in current_param.items()}

            if linear_transition:
                current_param[start_frame] = value
                current_param[start_frame+n_frames] = 0.0
            else: 
                current_param[start_frame] = 0.0
                current_param[start_frame+1] = value
                current_param[start_frame+n_frames-1] = value
                current_param[start_frame+n_frames] = 0.0

            # convert dict back to string
            output_string = ""
            for key, value in sorted(current_param.items(), key=lambda a: a[0]): 
                value = f"({str(value).strip()})".replace("((", "(").replace("))", ")").strip()
                output_string += f" {key}: {value},"
        
            self.anim_args[animations_param] = output_string[1:-1]


    def _update_anim_args(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.deforum_args: 
                if key in ["W", "H"]:
                    value = value - value % 64
                self.deforum_args[key] = value
            elif key in self.anim_args:
                self.anim_args[key] = value
            else: 
                raise KeyError(str(key))
        return self.deforum_args, self.anim_args

    @property
    def args(self):
        return self.deforum_args, self.anim_args

    
# move_1 = Move("right") + Move("left") # no movement 
# move_2 = Move("up") + Move("right") # diagonal movement left$$bottom -> right$$top
# move_3 = Move("right") * 3 # movement to right 3 times faster
# move_4 = Move("zoomin") + move3.make_sin(period=1) # zoomin with horizontal movements specified by sine


# class Move:
#     def __init__(self, mode:str, duration:int, **kwargs) -> None:
#         pass
    
#     def __add__(self, animation:Animation) -> self:
#         '''Combine two animations into one.'''
#         pass
    
#     def __radd__(self, animation:Animation) -> self:
#         '''Combine two animations into one.'''
#         pass
    
#     def __mul__(self, scale:int|float) -> self:
#         '''Accelerate animation by a given scale'''
#         pass
    
#     def __rmul__(self, scale:int|float) -> self:
#         '''Accelerate animation by a given scale'''
#         pass
    
#     def __truediv__(self, scale:int|float) -> self:
#         '''Accelerate animation by a given scale'''
#         pass
        
#     def make_sin(self, period:int|float) -> self
#         '''Add sine trajectory with a given period
#         by following formula acceleration*sin(pi*t/period)
#         '''
#         pass
    
#     def params(self) -> dict:
#         pass
    
