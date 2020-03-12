#!usr/bin/python3
from aima3.agents import *
from misio.aima import * 

number_of_no_op = 0 
no_op_limit = 9 
oper = 0 
model = {loc_A: None, loc_B: None} 

def MyAgent(test = False): 
    def program(percept): 
        location, status = percept 
        global number_of_no_op 
        global oper 
        global no_op_limit 
        oper+=1 
        model[location] = status 
        
        if model[loc_A] == model[loc_B] == 'Clean' and number_of_no_op < no_op_limit: 
            number_of_no_op += 1
            return 'NoOp' 
        
        elif status == 'Dirty':
            return 'Suck' 
        elif oper <= no_op_limit:
            number_of_no_op += 1
            return 'NoOp' 
        
        # elif oper % (n-1) and 'Clean':
            # return 'NoOp' 
        elif location == loc_A:
            number_of_no_op = 0
            return 'Right'
        elif location == loc_B:
            number_of_no_op = 0
            return 'Left' 
    
    if not test:
        return program
    else:
        return Agent(program) 

def parse_line(line):
    first = int(line[2])
    second = int(line[5])
    state = line[10:line.index('\')')]
    return ((first, second), state) 

n,s = [int(x) for x in input().split()]
for _ in range(n*s):
    percept = parse_line(input())
    result = MyAgent()
    print(result(percept))