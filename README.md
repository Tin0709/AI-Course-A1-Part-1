Q1 – Depth-First Search
We wrote a graph-search DFS in search.py using a Stack, a visited set, and pushing successors in reverse order so that the first successor is expanded first.

Q2 – Breadth-First Search
We implemented graph-search BFS with a Queue, marking states as visited when enqueued, and returning the first path that reaches the goal.

Q3 – Uniform-Cost Search
We used a PriorityQueue keyed by cumulative path‐cost, kept a best_cost map to skip outdated entries, and returned the lowest‐cost path to the goal.

Q4 – A* Search
We extended UCS by adding a heuristic: the priority is g(n)+h(n), skipping entries whose stored g exceeds the best known cost, so we get an optimal, efficient A*.

Cost-of-Actions Override
We overrode getCostOfActions in search.py to walk the actions sequence, summing step costs (or returning a large cost for illegal moves), which the autograder uses to verify heuristic bounds.

Q5 – CornersProblem
In searchAgents.py we defined every state as (position, visitedCorners)—a 4-tuple of booleans—wrote getStartState, isGoalState (all four corners visited), and getSuccessors (updating the visited-corners tuple).

Q6 – Corners Heuristic
We added a greedy “nearest-corner tour” heuristic: repeatedly walk to the closest unvisited corner (by Manhattan distance), summing those legs—admissible and fast.

Q7 – Food Heuristic
We wrote an admissible & consistent heuristic for the food-search problem by computing a minimal‐spanning‐tree cost over just the food dots (Prim’s algorithm) plus the distance from Pacman to the nearest food.

Q8 – Closest-Dot Search
We filled in AnyFoodSearchProblem.isGoalState to return true whenever Pacman’s position contains food, and used simple BFS (search.bfs) in findPathToClosestDot to stop at the nearest dot.
