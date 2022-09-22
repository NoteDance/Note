import numpy as np
import tensorflow_docs.vis.embed as embed
from PIL import Image


class visual:
    def __init__(self,agent,env,max_step,rendering_step,mode='rgb_array'):
        self.agent=agent
        self.env=env
        self.mode=mode
        self.max_step=max_step
        self.rendering_step=rendering_step
    
    
    def render_episode(self):
        screen=self.env.render(mode=self.mode)
        im=Image.fromarray(screen)
        images=[im]
        state=self.env.reset()
        for i in range(self.max_step):
            state=np.expand_dims(state,0)
            try:
                if self.agent.nn!=None:
                    pass
                try:
                    if self.agent.action!=None:
                        pass
                    action=self.agent.action(state)
                except AttributeError:
                    action_prob=self.agent.nn(state)
                    action=np.argmax(action_prob).numpy()
            except AttributeError:
                action=self.agent.actor(state)
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
