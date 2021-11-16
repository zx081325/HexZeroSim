import numpy as np
from game import GameEnv


class Env:
    """
    Doudizhu multi-agent wrapper
    """
    def __init__(self, chess_num, board=None):
        if board is None:
            board = []
        self.chess_num = chess_num
        self._env = GameEnv(self.chess_num)
        self._env.board = board

    def reset(self):
        self._env.reset()
        return self.get_obs()

    def step(self, action):
        assert action in self._legal_actions
        self._env.step(action)
        done = False
        reward = 0.0
        if self._game_over:
            done = True
            reward = self._get_reward()
        obs = self.get_obs()
        return obs, reward, done

    def _get_reward(self):
        """
        This function is called in the end of each
        game. It returns either 1/-1 for win/loss,
        or ADP, i.e., every bomb will double the score.
        """
        winner = self._game_winner
        if winner == 'white':
            return 1.0
        else:
            return -1.0

    @property
    def _legal_actions(self):
        """ A string of landlord/peasants
        """
        return self._env.get_legal_actions()

    @property
    def _board(self):
        """ A string of landlord/peasants
        """
        return self._env.board

    @property
    def _game_winner(self):
        """ A string of landlord/peasants
        """
        return self._env.winner

    @property
    def _acting_player_position(self):
        """
        The player that is active. It can be landlord,
        landlod_down, or landlord_up.
        """
        return self._env.acting_player_position

    @property
    def _game_over(self):
        """ Returns a Boolean
        """
        return self._env.game_over

    def get_obs(self):
        legal_actions = self._legal_actions
        num_legal_actions = len(legal_actions)
        x_no_action = _board2array(self._board)
        x_batch = np.repeat(x_no_action[np.newaxis, :],
                                       num_legal_actions, axis=0)
        for index, action in enumerate(legal_actions):
            x_batch[index, action * 3] = 0
            if self._acting_player_position == "white":
                x_batch[index, action * 3 + 1] = 1
            else:
                x_batch[index, action * 3 + 2] = 1
        obs = {
            'position': self._acting_player_position,
            'x_batch': x_batch.astype(np.float32),
            'legal_actions': self._legal_actions,
            'x_no_action': x_no_action.astype(np.int8)
        }
        return obs


def _board2array(board):
    board_array = []
    for pos in board:
        if pos == 0:
            board_array.extend([1, 0, 0])
        if pos == 1:
            board_array.extend([0, 1, 0])
        if pos == 2:
            board_array.extend([0, 0, 1])
    return np.array(board_array)