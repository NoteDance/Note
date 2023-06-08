import numpy as np
import tensorflow_docs.vis.embed as embed
from PIL import Image
import traceback


class visual:
    def __init__(self,agent,env,max_step,rendering_step,mode='rgb_array'):
        self.agent=agent
        self.env=env
        self.mode=mode
        self.max_step=max_step
        self.rendering_step=rendering_step
    
    
    def render_episode(self,seed=None):
        if seed==None:
            s=self.env.reset()
        else:
            s=self.env.reset(seed=seed)
        screen=self.env.render(mode=self.mode)
        im=Image.fromarray(screen)
        images=[im]
        for i in range(self.max_step):
            s=np.expand_dims(s,0)
            try:
                if self.agent.nn!=None:
                    pass
                try:
                    if self.agent.action!=None:
                        pass
                    a=self.agent.action(s).numpy()
                except Exception:
                    print(traceback.format_exc())
                    action_prob=self.agent.nn.fp(s).numpy()
                    a=np.argmax(action_prob)
            except Exception:
                print(traceback.format_exc())
                a=self.agent.actor.fp(s).numpy()
                a=np.squeeze(a)
            state,_,done,_=self.env.step(a)
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
