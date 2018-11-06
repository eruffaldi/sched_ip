from pulp import *

class TaskEdge:
    def __init__(self,source,delay=0):
        self.source = source
        self.delay = delay

class Task:
    def __init__(self,name,cost,deadline=None,fixed=False):
        self.parents = []
        self.name = name
        self.cost = cost
        self.fixed = fixed
        self.deadline = deadline 

def defmax(a,default):
    if len(a) == 0:
        return default
    else:
        return max(a)

def makebin(name):
    return LpVariable(name,cat='Binary')
def makelinpos(name,minv,maxv):
    return LpVariable(name,lowBound=minv,upBound=maxv,cat='Continuous')
def makeint(name,maxv):
    return LpVariable(name,lowBound=1,upBound=maxv,cat='Integer')

def xpulp(tasks,P,args):
    """
    Towards the Optimal Solution of the Multiprocessor Scheduling Problem with Communication Delays, Davidovic
    Using: Uang Cheung (2004), The berth allocation problem: models and solution methods. 
    """

    N = len(tasks)
    alli  = range(1,N+1)
    allp  = range(1,P+1)
    allij = list(itertools.product(range(1,N+1),range(1,N+1))) # product 1-based with i==j
    allhk = list(itertools.product(range(1,P+1),range(1,P+1))) # product 1-based with h==k
    allih = list(itertools.product(range(1,N+1),range(1,P+1))) # product 1-based
    #allijhk = itertools.product(range(1,N+1),range(1,N+1),range(1,P+1),range(1,P+1)) # product 1-based by 4

    print "allij",allij
    print "allhk",allhk
    print "allih",allih


    Wmax = sum([t.cost + defmax([te.delay for te in t.parents],0) for t in tasks])
    print "Wmax is",Wmax
    makenormalized = lambda name: makelinpos(name,0,1)
    makeproc = lambda name: makeint(name,P)
    maketask = lambda name: makeint(name,N)
    maketime = lambda name: makelinpos(name,0,Wmax)

    # we embed the constraint in the variable type definition for the time
    t = [makelinpos("t_%d" % i,0, Wmax if tasks[i-1].deadline is None else min(Wmax,tasks[i-1].deadline)) for i in alli] #t[i:task]:time TODO add here the timelimit


    #

    # ALTERNATIVELY use dicts of LpVariable
    p = [makeproc("p_%d" % i) for i in alli] # p[i:task]:proc
    x = dict([(ih,makebin("x_%d_%d" % ih)) for ih in allih]) # x[i:task,h:proc]:binary if i runs on h <=> p[i:task]

    # in the following we create but not use i== j
    si = dict([(ij,makebin("si_%d_%d" % ij)) for ij in allij]) # si[i,j:task]:binary
    eps = dict([(ij,makebin("eps_%d_%d" % ij)) for ij in allij]) # eps[i,j:task]:binary

    # we avoid creating z explicitly
    #z = dict([(ijhk,makebin("z_%d_%d_%d" % ijhk)) for allijhk]) # z[i,j:task,h,k:proc]:normalized
    z = dict()
    W = maketime("W") # the objective time function 0..Wmax

    prob = LpProblem("The fantastic scheduler",LpMinimize)

    #Not needed
    prob += W >= 0, "Enforce positivity"
    for tt in t:
        prob += tt >= 0, "Enforce positivity of " + tt.name

    prob += W, "Span"    

    # build task object to index 1-based
    # build cost array
    inv = {}
    L = []
    for i,tt in enumerate(tasks):
        inv[tt] = i+1
        L.append(tt.cost)
    print "costs",L

    # constraints
    for i in range(1,N+1):
        prob += t[i-1] + L[i-1] <= W # execution smaller than maxspan
        prob += sum([k*x[(i,k)] for k in allp]) == p[i-1]  # match task of processor wiht processor of task
        prob += sum([x[(i,k)] for k in allp]) == 1 # only one processor

    # external fixed are exact
    for i in alli:
        if tasks[i-1].fixed:
            prob += t[i-1] == tasks[i-1].deadline

    # this holds for all i != j
    for i in range(1,N+1):
        for j in range(1,N+1):
            if i == j:
                continue
            sij = si[(i,j)]
            epsij = eps[(i,j)]
            prob += t[j-1] - t[i-1] - L[i-1] - (sij-1) * Wmax >= 0  # ?
            prob += p[j-1] - p[i-1] - 1 - (epsij-1) * P >= 0 # ?
            if i > j:
                # symmetric enforcement
                sji = si[(j,i)]
                epsji = eps[(j,i)]
                prob += sij+sji+epsij+epsji >= 1
                prob += sij + sji <= 1
                prob += epsij + epsji <= 1

    for j in range(1,N+1): # all task 1-based
        tt = tasks[j-1]
        for pt in tt.parents: # all parent tasks objects
            i = inv[pt.source] # 1-based
            prob += si[(i,j)] == 1 # dependency

            if args.quadratic:
                prob += t[i-1] + L[i-1] + sum([ (h != k and pt.delay or 0) * x[(i,h)] * x[(j,k)] for h,k in allhk]) <= t[j-1] # time
            else:
                # z[ij hk] := x[ij hk] x[ji kh]
                for h,k in allhk:
                    q = (i,j,h,k)
                    z[q] = makebin("z_%d_%d_%d_%d" % q)
                    # NOTE z[(j,i,k,h)] = z[q]

                if not args.usecompact:
                    for h,k in allhk:
                        q = (i,j,h,k)
                        prob += x[(i,h)] >= z[q]
                        prob += x[(j,k)] >= z[q]
                        prob += x[(i,h)]+x[(j,k)]-1 <= z[q]
                else:
                    # eq. 32 
                    for k in allp:
                        prob += sum([z[(i,j,h,k)] for h in allp]) == x[(j,k)]   
                    # eq. 33 symmetric: z[ijhk] == z[jikh] BUT we never create z[ji**] being DAG
                    # BUT we do not USE z[]
                    # prob += z[(i,j,h,k)] == z[(j,i,k,h)]

                # the model employed is DELAY, this is a heavy constraint noting that z(i,j,h,k) is mutually exclusive in (h,k)
                prob += t[i-1] + L[i-1] + sum([ (h != k and pt.delay or 0) * z[(i,j,h,k)] for h,k in allhk]) <= t[j-1] # time


    prob.writeLP("pulp.lp")
    prob.solve()#pulp.GLPK_CMD())
    print("Status:", LpStatus[prob.status])
    print("Objective:", value(prob.objective))
    print("maxspan",W.varValue)
    for i,v in enumerate(t):
        print(v.name, "=", v.varValue)
        tasks[i].setOneRun(v.varValue)
    for i,v in enumerate(p):
        print(v.name, "=", v.varValue)
    for i,v in enumerate(z.values()):
        print(v.name, "=", v.varValue)

    stasks = tasks[:]
    stasks.sort(key=lambda k: k.earlieststart)
    schedule = [Proc(h) for h in range(1,P+1)]

    # build the ProcTask by scanning the tasks ordered
    for tt in stasks:
        i = inv[tt]
        proc = schedule[int(p[i-1].varValue)-1]
        proc.addTask(tt,tt.earlieststart,tt.endtime)

    schedule = [p for p in schedule if len(p.tasks) > 0]

    return dict(T=W.varValue,schedule=schedule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read')
    parser.add_argument('input',nargs="+")
    parser.add_argument('--split',help="split folder")
    parser.add_argument('--types',action="store_true")
    parser.add_argument('--subvis',action="store_true") 
    parser.add_argument('--json-dump',help="emit types as json dump for decoding")


    tasks = []
    args = {}
    P=4 # people
    print(xpulp(tasks,4,args))
