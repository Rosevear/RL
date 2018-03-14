#!/usr/bin/env python

"""
 Copyright (C) 2017, Adam White, Mohammad M. Ajallooeian


"""

from importlib import import_module

environment = "w1_env"
agent = "w1_agent"

last_action = None # last_action: NumPy array
total_reward = None # total_reward: floating point
num_steps = None # num_steps: integer
num_episodes = None # num_episodes: integer

def RLGlue(env_name, agent_name): # env_name: string, agent_name: string
    global environment, agent

    environment = import_module(env_name)
    agent = import_module(agent_name)

def RL_init():
    global total_reward, num_steps, num_episodes#, environment, agent
    environment.env_init()
    agent.agent_init()

    total_reward = 0.0
    num_steps = 0
    num_episodes = 0

def RL_start(): # returns (NumPy array, NumPy array)
    global last_action, total_reward, num_steps#, environment, agent
    total_reward = 0.0;
    num_steps = 1;

    last_state = environment.env_start()
    last_action = agent.agent_start(last_state)

    observation = (last_state, last_action)

    return observation

def RL_agent_start(observation): # returns NumPy array, observation: NumPy array
    #global agent

    return agent.agent_start(observation)

def RL_agent_step(reward, observation): # returns NumPy array, reward: floating point, observation: NumPy array
    #global agent

    return agent.agent_step(reward, observation)

def RL_agent_end(reward): # reward: floating point
    #global agent

    agent.agent_end(reward)

def RL_env_start():
    global total_reward, num_steps#, environment
    total_reward = 0.0
    num_steps = 1

    thisObservation = environment.env_start()

    return thisObservation

def RL_env_step(action): # returns (floating point, NumPy array, Boolean), action: NumPy array
    global total_reward, num_steps, num_episodes#, environment
    ro = environment.env_step(action)
    (this_reward, _, terminal) = ro

    total_reward += this_reward

    if terminal == True:
        num_episodes += 1
    else:
        num_steps += 1

    return ro

def RL_step(): # returns (floating point, NumPy array, NumPy array, Boolean)
    global last_action, total_reward, num_steps, num_episodes#, environment, agent
    (this_reward, last_state, terminal) = environment.env_step(last_action)

    total_reward += this_reward;

    if terminal == True:
        num_episodes += 1
        agent.agent_end(this_reward)
        roa = (this_reward, last_state, None, terminal)
    else:
        num_steps += 1
        last_action = agent.agent_step(this_reward, last_state)
        roa = (this_reward, last_state, last_action, terminal)

    return roa

def RL_cleanup():
    #global environment, agent

    environment.env_cleanup()
    agent.agent_cleanup()

def RL_agent_message(message): # returns string, message: string
    #global agent

    if message is None:
        messageToSend = ""
    else:
        messageToSend = message

    theAgentResponse = agent.agent_message(messageToSend)
    if theAgentResponse is None:
        return ""

    return theAgentResponse

def RL_env_message(message):  # returns string, message: string
    #global environment

    if message is None:
        messageToSend = ""
    else:
        messageToSend = message

    theEnvResponse = environment.env_message(messageToSend)
    if theEnvResponse is None:
        return ""

    return theEnvResponse

def RL_episode(maxStepsThisEpisode): # returns Boolean, # maxStepsThisEpisode: integer
    #global num_steps
    isTerminal = False

    RL_start()

    while (not isTerminal) and ((maxStepsThisEpisode == 0) or (num_steps < maxStepsThisEpisode)):
        rl_step_result = RL_step()
        isTerminal = rl_step_result[3]

    return isTerminal


def RL_return(): # returns floating point
    #global total_reward
    return total_reward

def RL_num_steps(): # returns integer
    #global num_steps
    return num_steps

def RL_num_episodes(): # returns integer
    #global num_episodes
    return num_episodes
