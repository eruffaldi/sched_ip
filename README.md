

Model:
- task
- lead
- effortlead
- others
- dependencies
- fixed


Modeling as IP:
- t_start_I
-     prob = LpProblem("The fantastic scheduler",LpMinimize)



Implementing as IP:
- http://web.emn.fr/x-info/cpaior-2012/uploads/slides%20master%20class/7-scheduling-Artigues.pdf
- https://blog.remix.com/an-intro-to-integer-programming-for-engineers-simplified-bus-scheduling-bd3d64895e92


Towards the Optimal Solution of the Multiprocessor Scheduling Problem with Communication Delays, Davidovic
  Using: Uang Cheung (2004), The berth allocation problem: models and solution methods. 

https://github.com/eruffaldi/taskschedule/blob/master/sched.py

import pulp

def makebin(name):
        return LpVariable(name,cat='Binary')
def makelinpos(name,maxv):
    return LpVariable(name,lowBound=0,upBound=maxv,cat='Continuous')
def makeint(name,maxv):
    return LpVariable(name,lowBound=1,upBound=maxv,cat='Integer')
