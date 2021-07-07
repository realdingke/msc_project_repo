import random
from typing import Tuple

from battleship.board import Board, BoardAutomatic
from battleship.ship import Ship
from battleship.convert import get_tuple_coordinates_from_str, get_str_coordinates_from_tuple


class Player(object):
    """
    Class representing the player
    - chooses where to perform an attack
    """
    index_player = 0

    def __init__(self,
                 board: Board,
                 name_player: str = None,
                 ):
        Player.index_player += 1

        self.board = board
        
        self.is_ship_hit = None
        
        self.has_ship_sunk = None

        if name_player is None:
            self.name_player = "player_" + str(self.index_player)
        else:
            self.name_player = name_player

    def __str__(self):
        return self.name_player

    def attacks(self,
                opponent) -> Tuple[bool, bool]:
        """
        :param opponent: object of class Player representing the person to attack
        :return: a tuple of bool variables (is_ship_hit, has_ship_sunk) where:
                    - is_ship_hit is True if and only if the attack was performed at a set of coordinates where an
                    opponent's ship is.
                    - has_ship_sunk is True if and only if that attack made the ship sink.
        """

        assert isinstance(opponent, Player)

        print(f"Here is the current state of {opponent}'s board before {self}'s attack:\n")
        opponent.print_board_without_ships()

        coord_x, coord_y = self.select_coordinates_to_attack(opponent)

        print(f"{self} attacks {opponent} "
              f"at position {get_str_coordinates_from_tuple(coord_x, coord_y)}")

        is_ship_hit, has_ship_sunk = opponent.is_attacked_at(coord_x, coord_y)

        if has_ship_sunk:
            print(f"\nA ship of {opponent} HAS SUNK. {self} can play another time.")
        elif is_ship_hit:
            print(f"\nA ship of {opponent} HAS BEEN HIT. {self} can play another time.")
        else:
            print("\nMissed".upper())
        
        self.is_ship_hit = is_ship_hit
        self.has_ship_sunk = has_ship_sunk

        return is_ship_hit, has_ship_sunk

    def is_attacked_at(self,
                       coord_x: int,
                       coord_y: int
                       ) -> Tuple[bool, bool]:
        """
        :param coord_x: integer representing the projection of a coordinate on the x-axis
        :param coord_y: integer representing the projection of a coordinate on the y-axis
        :return: a tuple of bool variables (is_ship_hit, has_ship_sunk) where:
                    - is_ship_hit is True if and only if the attack was performed at a set of coordinates where a
                    ship is (on the board owned by the player).
                    - has_ship_sunk is True if and only if that attack made the ship sink.
        """
        # TODO
        return self.board.is_attacked_at(coord_x, coord_y)

    def select_coordinates_to_attack(self, opponent) -> Tuple[int, int]:
        """
        Abstract method, for choosing where to perform the attack
        :param opponent: object of class Player representing the player under attack
        :return: a tuple of coordinates (coord_x, coord_y) at which the next attack will be performed
        """
        raise NotImplementedError

    def has_lost(self) -> bool:
        """
        :return: True if and only if all the ships of the player have sunk
        """
        # TODO
        if self.board.has_no_ships_left():
            return True
        else:
            return False
            

    def print_board_with_ships(self):
        self.board.print_board_with_ships_positions()

    def print_board_without_ships(self):
        self.board.print_board_without_ships_positions()


class PlayerUser(Player):
    """
    Player representing a user playing manually
    """

    def select_coordinates_to_attack(self, opponent: Player) -> Tuple[int, int]:
        """
        Overrides the abstract method of the parent class.
        :param opponent: object of class Player representing the player under attack
        :return: a tuple of coordinates (coord_x, coord_y) at which the next attack will be performed
        """
        print(f"It is now {self}'s turn.")

        while True:
            try:
                coord_str = input('coordinates target = ')
                coord_x, coord_y = get_tuple_coordinates_from_str(coord_str)
                return coord_x, coord_y
            except ValueError as value_error:
                print(value_error)


class PlayerAutomatic(Player):
    """
    Player playing automatically using a strategy.
    """

    def __init__(self, name_player: str = None):
        board = BoardAutomatic()
        self.list_opponent_ships = []
        self.set_positions_previously_attacked = set()
        self.last_attact_coord = None

        super().__init__(board, name_player)
    
    def get_random_coordinates(self):
        """
        return: a tuple of randomly generated coordinates within the board
        """
        #
        return (random.randint(1, self.board.SIZE_X), random.randint(1, self.board.SIZE_Y))
    
    def is_out_of_bound(self, coordinates):
        """
        return: a boolean that is True if and only if the input coordinates are not within the boundaries
        """
        #
        if (coordinates[0] in list(range(1, Board.SIZE_X+1))) and (coordinates[1] in list(range(1, Board.SIZE_Y+1))):
            return False
        else:
            return True
    
    def _is_position_near_previously_sunk_ship(self, coordinates):
        """
        return: True if and only if the given coordinates are near a opponent's sunk ship
        """
        for ship_opponent in self.list_opponent_ships:
            if ship_opponent.has_sunk() and ship_opponent.is_near_coordinate(*coordinates):
                return True
        return False
    
    def prepare_random_coordinates_to_attack(self):
        """
        return: a randomly generated coordinate that is reasonbale to attack
        """
        has_position_been_previously_attacked = True
        is_position_near_previously_sunk_ship = True
        coord_random = None
        while has_position_been_previously_attacked or is_position_near_previously_sunk_ship:
            coord_random = self.get_random_coordinates()
            has_position_been_previously_attacked = coord_random in self.set_positions_previously_attacked
            is_position_near_previously_sunk_ship = self._is_position_near_previously_sunk_ship(coord_random)
        return coord_random
        

    def select_coordinates_to_attack(self, opponent: Player) -> tuple:
        """
        Overrides the abstract method of the parent class.
        :param opponent: object of class Player representing the player under attack
        :return: a tuple of coordinates (coord_x, coord_y) at which the next attack will be performed
        """
        # TODO
        #get the list of opponent's ship before the attack
        self.list_opponent_ships = opponent.board.list_ships
        
        if self.last_attact_coord is None:
            coord_random = self.get_random_coordinates()
            self.last_attact_coord = coord_random
            self.set_positions_previously_attacked.add(coord_random)
            return coord_random    #randomly selects the first attack
        
        #check the last attack
        is_ship_hit = self.is_ship_hit
        has_ship_sunk = self.has_ship_sunk
        
        if is_ship_hit and (not has_ship_sunk):    #if last attack hits a ship
            last_coord = self.last_attact_coord
            #selects the next coordinate for attack to be nearby the previous attacked coordinate
            #as it follows that ship can only be vertical or horizontal
            new_coord_list = [(last_coord[0]+1, last_coord[1]), (last_coord[0]-1, last_coord[1]),
                              (last_coord[0], last_coord[1]+1), (last_coord[0], last_coord[1]-1)]
            has_position_been_attacked = True
            bound_detect = True
            idx = 0
            while has_position_been_attacked or bound_detect:
                new_coord = random.choice(new_coord_list)
                has_position_been_attacked = new_coord in self.set_positions_previously_attacked
                bound_detect = self.is_out_of_bound(new_coord)
                if idx > 10000:     # a threshold value is set to enable loop exiting
                    #randomly selects valid coordinate when the four choices in new_coord_list are all invalid
                    new_coord = self.prepare_random_coordinates_to_attack()
                    break
                idx += 1
            self.last_attact_coord = new_coord
            self.set_positions_previously_attacked.add(new_coord)
            return new_coord
        else:
            #when the previous attack did not hit anything or the ship has already sunk
            #randomly select a valid coordinate for the next attack
            new_coord = self.prepare_random_coordinates_to_attack()
            self.last_attact_coord = new_coord
            self.set_positions_previously_attacked.add(new_coord)
            return new_coord
            
        
        
        
        
class PlayerRandom(Player):
    def __init__(self, name_player: str = None):
        board = BoardAutomatic()
        self.set_positions_previously_attacked = set()
        self.last_attack_coord = None
        self.list_ships_opponent_previously_sunk = []

        super().__init__(board, name_player)

    def select_coordinates_to_attack(self, opponent: Player) -> tuple:
        position_to_attack = self.select_random_coordinates_to_attack()

        self.set_positions_previously_attacked.add(position_to_attack)
        self.last_attack_coord = position_to_attack
        return position_to_attack

    def select_random_coordinates_to_attack(self) -> tuple:
        has_position_been_previously_attacked = True
        is_position_near_previously_sunk_ship = True
        coord_random = None

        while has_position_been_previously_attacked or is_position_near_previously_sunk_ship:
            coord_random = self._get_random_coordinates()

            has_position_been_previously_attacked = coord_random in self.set_positions_previously_attacked
            is_position_near_previously_sunk_ship = self._is_position_near_previously_sunk_ship(coord_random)

        return coord_random

    def _get_random_coordinates(self) -> tuple:
        coord_random_x = random.randint(1, self.board.SIZE_X)
        coord_random_y = random.randint(1, self.board.SIZE_Y)

        coord_random = (coord_random_x, coord_random_y)

        return coord_random

    def _is_position_near_previously_sunk_ship(self, coord: tuple) -> bool:
        for ship_opponent in self.list_ships_opponent_previously_sunk:  # type: Ship
            if ship_opponent.has_sunk() and ship_opponent.is_near_coordinate(*coord):
                return True
        return False

if __name__ == '__main__':
    # SANDBOX for you to play and test your functions

    list_ships = [
        Ship(coord_start=(1, 1), coord_end=(1, 1)),
        Ship(coord_start=(3, 3), coord_end=(3, 4)),
        Ship(coord_start=(5, 3), coord_end=(5, 5)),
        Ship(coord_start=(7, 1), coord_end=(7, 4)),
        Ship(coord_start=(9, 3), coord_end=(9, 7)),
    ]

    #board = Board(list_ships)
    player = PlayerAutomatic()
    player2 = PlayerAutomatic()
    print(player2.select_coordinates_to_attack(player))
    print(player.select_coordinates_to_attack(player2))
    print(player.has_lost(),player2.has_lost())
