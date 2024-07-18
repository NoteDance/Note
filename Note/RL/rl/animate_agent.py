import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class animate_agent:
    def __init__(self,agent,env,platform='tf'):
        self.agent=agent
        self.env=env
        self.platform=platform
    
    
    def run_agent(self, max_steps):
        state_history = []

        state = self.env.reset()
        for step in range(max_steps):
            if self.platform=='tf':
                if not hasattr(self, 'noise'):
                    action = np.argmax(self.agent.nn.fp(state))
                else:
                    action = self.agent.actor.fp(state).numpy()
            elif self.platform=='pytorch':
                if not hasattr(self, 'noise'):
                    action = np.argmax(self.agent.nn.fp(state))
                else:
                    action = self.agent.actor.fp(state).detach().numpy()
            next_state, reward, done, _ = self.env.step(action)
            state_history.append(state)
            if done:
                break
            state = next_state
        
        return state_history
    
    
    def __call__(self, max_steps, mode='rgb_array', save_path=None, fps=None, writer='imagemagick'):
        state_history = self.run_agent(max_steps)
        
        fig = plt.figure()
        ax = fig.add_subplot()
        self.env.reset()
        img = ax.imshow(self.env.render(mode=mode))

        def update(frame):
            img.set_array(self.env.render(mode=mode))
            return [img]

        ani = animation.FuncAnimation(fig, update, frames=state_history, blit=True)
        plt.show()
        
        if save_path!=None:
            ani.save(save_path, writer=writer, fps=fps)
        return
