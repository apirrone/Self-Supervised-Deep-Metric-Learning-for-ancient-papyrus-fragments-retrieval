import numpy as np
import sys
import time

class Progress:
    def __init__(self):
        self.times = []
        self.buffer_size = 10000
        self.lastTime = None 
        self.displayLastTime = None
        self.displayTickTime = 100 # ms
        
    def computeHMS(self, s):

        hours   = (s/60)//60
        minutes = (s-hours*60*60)//60
        seconds = s-(minutes*60+hours*60*60)
        
        return hours, minutes, seconds

    
    def tick(self, step, total_nb_steps, additional_info=None):
        
        currentTime = int(round(time.time() * 1000))

        # first iteration
        if self.lastTime == None:
            self.lastTime = currentTime
            self.displayLastTime = currentTime
            return
        
        elapsed = currentTime - self.lastTime        
        self.lastTime = int(round(time.time() * 1000))
        
        self.times.append(elapsed)
        
        self.times = self.times[-self.buffer_size:]

        if (currentTime - self.displayLastTime) > self.displayTickTime:
            self.displayLastTime = int(round(time.time() * 1000))
        
            mean_step_time = np.mean(self.times)
            nb_steps_remaining = total_nb_steps - step
            time_remaining = mean_step_time*nb_steps_remaining
            percentage = round((step/total_nb_steps)*100, 2)
            s_remaining = int(time_remaining/1000)
            
            h, m, s = self.computeHMS(s_remaining)
            sys.stdout.write("%d/%d --  %d%% -- %dH - %dM - %dS \r" % (step, total_nb_steps, percentage, h, m, s))

            
        
            
