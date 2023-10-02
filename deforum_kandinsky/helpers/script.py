from ..configs.animation_args import DeforumAnimArgs
from ..configs.deforum_args import DeforumArgs
from typing import List, Optional, Union
import json

class Script:
    def __init__(
            self, 
            animations:List[str], 
            durations:Union[List[int], int]=5, 
            linear_transition=True,
            fps=24,
            **kwargs
            ):
        assert isinstance(animations, list)
        self.deforum_args = DeforumArgs()
        self.anim_args = DeforumAnimArgs()

        if isinstance(durations, int): durations = [durations]*len(animations)
        seconds_elapsed = 0
        strength_schedule = ""
        for index, (animation, duration) in enumerate(zip(animations, durations)): 
            self.process_animation(
                animation, 
                start_frame=int((seconds_elapsed)*fps), 
                n_frames=int(duration*fps), 
                linear_transition=linear_transition
                )
            
            min_strength = 0.08
            if animation == "live":
                strength_schedule += f"{int(seconds_elapsed*fps)}:({min_strength}), "
                strength_schedule += f"{int(seconds_elapsed*fps)+1}: ({min_strength}+0.1*abs(sin(3.14159*t/{fps*duration/2}))), "
                strength_schedule += f"{int((seconds_elapsed+duration-1)*fps)}: ({min_strength}+0.1*abs(sin(3.14159*t/{fps*duration/2}))), "
            else:
                strength_schedule += f"{int(seconds_elapsed*fps)}:({min_strength}), "
                strength_schedule += f"{int(seconds_elapsed*fps)+1}: ({min_strength}+0.1*abs(sin(3.14159*t/{fps*duration}))), "
                strength_schedule += f"{int((seconds_elapsed+duration-1)*fps)}: ({min_strength}+0.1*abs(sin(3.14159*t/{fps*duration}))), "
            seconds_elapsed += duration
        
        self.anim_args["strength_schedule"] = strength_schedule[:-2]
        self._update_anim_args(**kwargs)

    def process_animation(self, animation:str, start_frame:int, n_frames:int=48, linear_transition:bool=True): 
        def _animations_params_dict():
            right = "translation_x", 1.5
            left =  "translation_x", -1.5
            up = "translation_y", 1.5
            down = "translation_y", -1.5
            spin_clockwise = "rotation_3d_z", 1.5
            spin_counterclockwise =  "rotation_3d_z", -1.5
            zoomin = "translation_z", 1.5
            zoomout = "translation_z", -1.5
            live = "translation_z", "sin(3.14159*t/90)*0.05"
            return locals()
        assert isinstance(start_frame, int) and isinstance(n_frames, int)
        
        # parse string to dictionary
        animations_params = _animations_params_dict()
        assert animation in animations_params, f"animation should be one of {list(animations_params.keys())}"
        
        if "spin" in animation:
            linear_transition = False
            
        animations_param, value = animations_params[animation]
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
    
