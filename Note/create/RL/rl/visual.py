import torch
import numpy as np
import tensorflow_docs.vis.embed as embed
from PIL import Image


class visual:
    def __init__(self,nn,env,max_step,rendering_step,mode='rgb_array',device='cuda'):
        self.nn=nn
        self.env=env
        self.mode=mode
        self.max_step=max_step
        self.rendering_step=rendering_step
        self.device=device
    
    
    def render_episode(self):
        screen=self.env.render(mode=self.mode)
        im=Image.fromarray(screen)
        images=[im]
        state=self.env.reset()
        for i in range(self.max_step):
            state=torch.tensor(np.expand_dims(state,0),dtype=torch.float)
            try:
                if self.nn.nn!=None:
                    pass
                try:
                    if self.nn.action!=None:
                        pass
                    action=self.nn.action(state.to(self.device))
                except AttributeError:
                    action_prob=self.nn.nn(state.to(self.device))
                    action=np.argmax(action_prob).numpy()
            except AttributeError:
                action=self.nn.actor(state.to(self.device))
                action=np.squeeze(action).numpy()
            state,_,done,_=self.env.step(action)
            state=state
            if i%self.rendering_step==0:
              screen=self.env.render(mode=self.mode)
              images.append(Image.fromarray(screen))
            if done:
                break
        return images
    
    
    def visualize_episode(self,images,file_name,save_all=True,append_images=None,loop=0,duration=1):
        image_file=file_name
        images[0].save(image_file,save_all,append_images,loop,duration)
        embed.embed_file(image_file)
        return
