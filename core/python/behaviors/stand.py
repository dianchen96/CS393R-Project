"""Stand"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import memory
import pose
import commands
import cfgstiff
from task import Task
from state_machine import Node, C, T, StateMachine

class Playing(StateMachine):
    class Stand(Node):
        def run(self):
            commands.stand()
            commands.setHeadPanTilt(pan=0, tilt=0, time=0.1)

            # if self.getTime() > 5.0:
            #     memory.speech.say("playing stand complete")
            #     self.finish()

    def setup(self):
    	stand = self.Stand()
        self.trans(stand, T(0.5), stand)
