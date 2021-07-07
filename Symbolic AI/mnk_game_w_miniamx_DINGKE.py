import numpy as np
import time
from copy import deepcopy

class Game(object):
    BOARD_TO_ARRAY_MAPPING = {0:0, 'X':1, 'O':2}  #this allows board mapping into numpy arrray
    
    def __init__(self, m, n, k):
        self.board = Game.initialize_game(m=m, n=n)
        self.length = len(self.board)
        self.width = len(self.board[0])
        self.consecutive = k
        self.terminal = False
        self.player = None
    
    @staticmethod
    def initialize_game(m, n):
        #initialize board as a nested list with 0 as place holders
        board = [[0 for j in range(n)] for i in range(m)]
        return board
    
    def draw_board(self, board):
        for i, row in enumerate(board):
            if i == 0:
                print('--'*(2*self.width-1))
            row_format = '{} | '*(self.width-1) + '{}'
            print(row_format.format(*row))
            print('--'*(2*self.width-1))
    
    @staticmethod
    def to_array(sequence):
        #maps given sequence into a numpy array(1D)
        array_list = [Game.BOARD_TO_ARRAY_MAPPING[element] for element in sequence]
        return np.array(array_list)
    
    @staticmethod
    def has_consecutive_reached(array, number, number_consecutive):
        #checks if a number appears consecutive amount of times in an given array(1D)
        has_reached = np.any(np.convolve(array == number, np.ones(number_consecutive), mode='valid') == number_consecutive)
        return has_reached

    def evaluate_board(self, board):
        """Called to check whether the given board produces a winner or not, this is a powerful method, 
           in which full flexibility(accept any board shape and number of consecutive) is implemented, 
           via numpy functionalities(mapping to numpy arrays first).
           returns 'Tie' if and only if the board is full; 'X' if Max has won; 'O' if Min has won,
           else returns None
        """
        #get attributes
        number_Max = Game.BOARD_TO_ARRAY_MAPPING['X']
        number_Min = Game.BOARD_TO_ARRAY_MAPPING['O']
        number_consecutive = self.consecutive
        
        #checks if any row produces a winner
        for row_idx in range(self.length):
            row_array = Game.to_array(board[row_idx][:])
            #check if the player's number appears consecutive amount of times along that row array
            #check if Max wins
            has_won = Game.has_consecutive_reached(row_array, number_Max, number_consecutive)
            if has_won:  
                return 'X'
            #check if Min wins
            has_won = Game.has_consecutive_reached(row_array, number_Min, number_consecutive)
            if has_won:  
                return 'O'
        
        #checks if any column produces a winner
        for col_idx in range(self.width):
            col_list = [board[i][col_idx] for i in range(self.length)]
            col_array = Game.to_array(col_list)
            #check if the player's number appears consecutive amount of times along that column array
            #check if Max wins
            has_won = Game.has_consecutive_reached(col_array, number_Max, number_consecutive)
            if has_won:  
                return 'X'
            #check if Min wins
            has_won = Game.has_consecutive_reached(col_array, number_Min, number_consecutive)
            if has_won:  
                return 'O'
            
        #check if any diagonol produces a winner
        #converts board into a numpy matrix
        matrix = Game.to_array(board[0][:]).reshape(1,-1)
        for row_idx in range(1,self.length):
            matrix = np.concatenate((matrix, Game.to_array(board[row_idx][:]).reshape(1,-1)), axis=0)
        #get all diagonals in positive direction, stored as a list of arrays
        pos_range = range(-matrix.shape[0]+1, matrix.shape[1])
        pos_diags = [matrix.diagonal(i) for i in pos_range]
        #get all diagonals in negative direction, stored as a list of arrays
        neg_range = range(matrix.shape[1]-1, -matrix.shape[0], -1)
        neg_diags = [matrix[::-1,:].diagonal(i) for i in neg_range]
        #get all diagonals
        all_diags = pos_diags + neg_diags
        for diag in all_diags:
            if len(diag) >= number_consecutive:
                #check if Max wins
                has_won = Game.has_consecutive_reached(diag.reshape(-1,), number_Max, number_consecutive)
                if has_won:  
                    return 'X'
                #check if Min wins
                has_won = Game.has_consecutive_reached(diag.reshape(-1,), number_Min, number_consecutive)
                if has_won:  
                    return 'O'
        
        #finally checks if board is full
        #the check must be done in the end to avoid early tie call when the last move actually produces a winner
        is_tie = True
        for i in range(self.length):
            for j in range(self.width):
                if board[i][j] == 0:
                    is_tie = False
        if is_tie:
            return 'Tie'
        
        return None
    
    def is_terminal(self, board):
        #returns whether the game has terminated and who won(or tie)
        is_terminal = False
        result = self.evaluate_board(board)
        if result is not None:
            is_terminal = True
        return is_terminal, result
    
    def get_next_player(self, current_player):
        #change player for the next turn
        if current_player == 'X':
            next_player = 'O'
        else:
            next_player = 'X'
        return next_player
    
    def get_all_children(self, board, player):
        #get all possible next moves(children) as list of boards for the player from current board
        children = []
        for i in range(self.length):
            for j in range(self.width):
                if board[i][j] == 0:
                    child = deepcopy(board)
                    child[i][j] = player
                    children.append(child)
        return children
    
    def get_utility(self, board, winner):
        #utility function to calculate minimax value for leaf node(terminal)
        #the utility function takes the form of +1*(1+ empty spots left) for Max, -1*(1+ empty spots left) for Min
        score = 1
        if winner == 'X':
            #get number of remaining blank spots on the board
            num_blank = 0
            for i in range(self.length):
                for j in range(self.width):
                    if board[i][j] == 0:
                        num_blank += 1
            return score*(num_blank+1)
        elif winner == 'O':
            num_blank = 0
            for i in range(self.length):
                for j in range(self.width):
                    if board[i][j] == 0:
                        num_blank += 1
            return -score*(num_blank+1)
        else:
            return 0   #tie game
    
    def maximize(self, board, player, depth, nodes_visited):
        #the max method, dual recursive call to the min method
        terminal, winner = self.is_terminal(board)
        if terminal:
            #return nextboard-score-visitednodes tuple
            return (None, self.get_utility(board,winner), nodes_visited)
        depth += 1
        max_utility = float("-inf")
        next_board = None
        
        next_possible_boards = self.get_all_children(board, player)
        for child in next_possible_boards:
            nodes_visited += 1
            _, utility, nodes_visited = self.minimize(child, self.get_next_player(player), depth, nodes_visited)
            if utility > max_utility:
                next_board = child
                max_utility = utility
                
        return (next_board, max_utility, nodes_visited)
    
    def minimize(self, board, player, depth, nodes_visited):
        #the min method, dual recursive call to the max method
        terminal, winner = self.is_terminal(board)
        if terminal:
            #return nextboard-score-visitednodes tuple
            return (None, self.get_utility(board,winner), nodes_visited)
        depth += 1
        min_utility = float("inf")
        next_board = None
        
        next_possible_boards = self.get_all_children(board, player)
        for child in next_possible_boards:
            nodes_visited += 1
            _, utility, nodes_visited = self.maximize(child, self.get_next_player(player), depth, nodes_visited)
            if utility < min_utility:
                next_board = child
                min_utility = utility
                
        return (next_board, min_utility, nodes_visited)
        
    def get_minimax(self, board, player):
        #get minimax value and best action(coordinates)
        if player == 'X':
            next_board, score, nodes_visited = self.maximize(board, player, 0, 0)
        else:
            next_board, score, nodes_visited = self.minimize(board, player, 0, 0)
        best_action = self.get_best_action(board, next_board)
        return best_action, score, nodes_visited
    
    def get_best_action(self, board, next_board):
        if next_board is None:
            return None
        for i in range(self.length):
                for j in range(self.width):
                    if board[i][j] != next_board[i][j]:
                        return (i,j)
    
    def alpha_beta_maximizer(self, board, player, depth, nodes_visited, alpha, beta):
        #max method with alpha beta pruning, dual recursive call to the min method
        terminal, winner = self.is_terminal(board)
        if terminal:
            #return nextboard-score-visitednodes tuple
            return (None, self.get_utility(board,winner), nodes_visited)
        depth += 1
        max_utility = float("-inf")
        next_board = None
        
        next_possible_boards = self.get_all_children(board, player)
        for child in next_possible_boards:
            nodes_visited += 1
            _, utility, nodes_visited = self.alpha_beta_minimizer(child, self.get_next_player(player), depth, nodes_visited, alpha, beta)
            if utility > max_utility:
                next_board = child
                max_utility = utility
            #enable pruning of other unsearched(guaranteed less optimal) nodes via loop exiting
            if max_utility >= beta:
                break
            #restrict alpha search range with optimal utility
            alpha = max(alpha, max_utility)
                       
        return (next_board, max_utility, nodes_visited)
    
    def alpha_beta_minimizer(self, board, player, depth, nodes_visited, alpha, beta):
        #min method with alpha beta pruning, dual recursive call to the max method
        terminal, winner = self.is_terminal(board)
        if terminal:
            #return nextboard-score-visitednodes tuple
            return (None, self.get_utility(board,winner), nodes_visited)
        depth += 1
        min_utility = float("inf")
        next_board = None
        
        next_possible_boards = self.get_all_children(board, player)
        for child in next_possible_boards:
            nodes_visited += 1
            _, utility, nodes_visited = self.alpha_beta_maximizer(child, self.get_next_player(player), depth, nodes_visited, alpha, beta)
            if utility < min_utility:
                next_board = child
                min_utility = utility
            #enable pruning of other unsearched(guaranteed less optimal) nodes via loop exiting
            if min_utility <= alpha:
                break
            #restrict beta search range with optimal utility
            beta = min(beta, min_utility)
                       
        return (next_board, min_utility, nodes_visited)
    
    def get_alpha_beta_minimax(self, board, player):
        alpha = float("-inf")
        beta = float("inf")
        #get minimax value and best action(coordinates)
        if player == 'X':
            next_board, score, nodes_visited = self.alpha_beta_maximizer(board, player, 0, 0, alpha, beta)
        else:
            next_board, score, nodes_visited = self.alpha_beta_minimizer(board, player, 0, 0, alpha, beta)
        
        best_action = self.get_best_action(board, next_board)
        return best_action, score, nodes_visited
        
    
    def is_valid(self, move):
        #a game/instance method, check if the input move(in coordinates) is valid
        is_valid = True
        if (move[0] not in list(range(self.length))) or (move[1] not in list(range(self.width))) or (self.board[move[0]][move[1]]!=0):
            is_valid = False
        return is_valid
    
    def play(self):
        """User as Max(with maximizer's advice), computer as Min(assume minimizer with optimal strategy),
           User starts first selecting any arbitrary spot. User key: 'X', computer opponent key: 'O'.
           Note: user input coordinate has a strict format of "row_index, column_index"
           Note: only this method updates the instance variables as it actually plays the game
        """
        #let user first move at every start of the game
        num_blank = 0
        for i in range(self.length):
            for j in range(self.width):
                if self.board[i][j] == 0:
                    num_blank += 1
        if num_blank == (self.length*self.width):
            print('User is the Max player! The board looks like this:')
            self.draw_board(self.board)
            while True:
                user_selection = input("Please start the game by entering a first coordinate(without brackets) to place 'X':")
                first_coord = tuple(map(int, user_selection.split(', ')))
                if self.is_valid(first_coord):
                    break
                else:
                    print('Sorry, invalid move!')
            #make the first move
            self.board[first_coord[0]][first_coord[1]] = 'X'
            self.player = 'X'
            self.draw_board(self.board)
        
        self.terminal, _ = self.is_terminal(self.board)
        while not self.terminal:
            #next_coord = self.get_minimax(self.board, self.get_next_player(self.player))[0]
            next_coord = self.get_alpha_beta_minimax(self.board, self.get_next_player(self.player))[0]
            self.player = self.get_next_player(self.player)
            if self.player == 'X':  #allow user minimax recommendation
                print(f"It is advised to make a move at {next_coord}!")
                while True:
                    user_selection = input('Please enter coordinate(without brackets) for the next move:')
                    next_coord = tuple(map(int, user_selection.split(', ')))
                    if self.is_valid(next_coord):
                        break
                    else:
                        print('Sorry, invalid move!')
                print('User now makes the next move:')
                self.board[next_coord[0]][next_coord[1]] = 'X'
                self.draw_board(self.board)
            else:  #automatic action selection according to minimax for computer player
                print('Computer(Min) now makes the next move:')
                self.board[next_coord[0]][next_coord[1]] = 'O'
                self.draw_board(self.board)
            
            #check and update the board status
            self.terminal, result = self.is_terminal(self.board)
        
        if result == 'X':
            print("Wow, User has just won the game!")
        elif result == 'O':
            print("Shame, the computer has just won it.")
        else:
            print("It's a tie game!")
        
        
    
    def experiments(self, minimax):
        """called to perform time-measurement experiments for action selection for each step of the game,
           arg: minimax can take 'normal' or 'pruned'
        """
        #first initialize the board with a Max move, this board will then be fixed for all experiments
        self.board[0][0] = 'X'
        self.player = 'X'
        
        data_list = []  #record all time-nodes pair at each step
        self.terminal, _ = self.is_terminal(self.board)
        while not self.terminal:
            if minimax == 'normal':
                begin = time.time()
                next_coord, _, visited_nodes = self.get_minimax(self.board, self.get_next_player(self.player))
                end = time.time()
                elasped_time = end - begin
                data_list.append((elasped_time, visited_nodes))
                self.player = self.get_next_player(self.player)
            elif minimax == 'pruned':
                begin = time.time()
                next_coord, _, visited_nodes = self.get_alpha_beta_minimax(self.board, self.get_next_player(self.player))
                end = time.time()
                elasped_time = end - begin
                data_list.append((elasped_time, visited_nodes))
                self.player = self.get_next_player(self.player)
            
            self.board[next_coord[0]][next_coord[1]] = self.player
            self.terminal, _ = self.is_terminal(self.board)
        
        return data_list
    
    


def _test(m, n, k):
    game_exp = Game(m,n,k)
    data_normal = game_exp.experiments('normal')
    print(f'for a {m}-{n}-{k} game unpruned:')
    for idx, tup in enumerate(data_normal):
        print(f"at step {idx+1}, the action selection time taken is {np.round_(tup[0],5)} seconds, a total of {tup[1]} states visited.")
    game_exp = Game(m,n,k)
    data_pruned = game_exp.experiments('pruned')
    print(f'\nfor a {m}-{n}-{k} game pruned:')
    for idx, tup in enumerate(data_pruned):
        print(f"at step {idx+1}, the action selection time taken is {np.round_(tup[0],5)} seconds, a total of {tup[1]} states visited.")


if __name__ == '__main__':
    ticktacktoe = Game(3, 3, 3)
    ticktacktoe.play()
    #_test(3, 3, 3)  #uncomment this line to perform experiments on different games by varying the m, n and k
