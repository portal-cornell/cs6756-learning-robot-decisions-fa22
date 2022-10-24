"""Initialise agents/."""

from gym_simplifiedtetris.agents.dellacherie import DellacherieAgent
from gym_simplifiedtetris.agents.q_learning import QLearningAgent
from gym_simplifiedtetris.agents.uniform import UniformAgent

__all__ = ["DellacherieAgent", "QLearningAgent", "UniformAgent"]
