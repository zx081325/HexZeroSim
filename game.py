class GameEnv:
    def __init__(self, chess_num):
        self.chess_num = chess_num
        self.board = []
        self.game_over = False
        self.acting_player_position = "white"
        self.winner = None
        self.fa_white, self.fa_black = [], []
        self.board_size = 11
        self.reset()

    # 初始化棋局
    def reset(self):
        self.board = [0 for i in range(self.chess_num)]
        self.board_size = int(self.chess_num ** 0.5)
        self.game_over = False
        self.acting_player_position = "white"
        self.winner = None
        self.fa_white = [i for i in range(self.chess_num + 2)]
        self.fa_black = [i for i in range(self.chess_num + 2)]
        for i in range(self.board_size):
            self.fa_white[i] = self.chess_num
        for i in range(self.chess_num - self.board_size, self.chess_num):
            self.fa_white[i] = self.chess_num + 1
        for i in range(0, self.chess_num, self.board_size):
            self.fa_black[i] = self.chess_num
        for i in range(self.board_size - 1, self.chess_num, self.board_size):
            self.fa_black[i] = self.chess_num + 1

    def get_legal_actions(self):
        moves = [i for i in range(self.chess_num) if self.board[i] == 0]
        return moves

    def game_done(self):
        if self.acting_player_position == "white":
            u, v = self.findest_w(self.chess_num), self.findest_w(self.chess_num + 1)
            if u == v:
                self.game_over = True
                self.winner = "white"
        if self.acting_player_position == "black":
            u, v = self.findest_b(self.chess_num), self.findest_b(self.chess_num + 1)
            if u == v:
                self.game_over = True
                self.winner = "black"

    def findest_w(self, x):
        if x == self.fa_white[x]:
            return x
        else:
            return self.findest_w(self.fa_white[x])

    def findest_b(self, x):
        if x == self.fa_black[x]:
            return x
        else:
            return self.findest_b(self.fa_black[x])

    def step(self, action):
        self.board[action] = 1 if self.acting_player_position == "white" else 2
        if action % self.board_size != 0 and self.board[action] == self.board[action - 1]:
            if self.acting_player_position == "white":
                u, v = self.findest_w(action), self.findest_w(action - 1)
                self.fa_white[u] = v
            else:
                u, v = self.findest_b(action), self.findest_b(action - 1)
                self.fa_black[u] = v
        if action % self.board_size != self.board_size - 1 and self.board[action] == self.board[action + 1]:
            if self.acting_player_position == "white":
                u, v = self.findest_w(action), self.findest_w(action + 1)
                self.fa_white[u] = v
            else:
                u, v = self.findest_b(action), self.findest_b(action + 1)
                self.fa_black[u] = v
        if action >= self.board_size and self.board[action] == self.board[action - self.board_size]:
            if self.acting_player_position == "white":
                u, v = self.findest_w(action), self.findest_w(action - self.board_size)
                self.fa_white[u] = v
            else:
                u, v = self.findest_b(action), self.findest_b(action - self.board_size)
                self.fa_black[u] = v
        if action < self.chess_num - self.board_size and self.board[action] == self.board[action + self.board_size]:
            if self.acting_player_position == "white":
                u, v = self.findest_w(action), self.findest_w(action + self.board_size)
                self.fa_white[u] = v
            else:
                u, v = self.findest_b(action), self.findest_b(action + self.board_size)
                self.fa_black[u] = v
        if action >= self.board_size and action % self.board_size != self.board_size - 1 \
                and self.board[action] == self.board[action - self.board_size + 1]:
            if self.acting_player_position == "white":
                u, v = self.findest_w(action), self.findest_w(action - self.board_size + 1)
                self.fa_white[u] = v
            else:
                u, v = self.findest_b(action), self.findest_b(action - self.board_size + 1)
                self.fa_black[u] = v
        if action < self.chess_num - self.board_size and action % self.board_size != 0 and \
                self.board[action] == self.board[action + self.board_size - 1]:
            if self.acting_player_position == "white":
                u, v = self.findest_w(action), self.findest_w(action + self.board_size - 1)
                self.fa_white[u] = v
            else:
                u, v = self.findest_b(action), self.findest_b(action + self.board_size - 1)
                self.fa_black[u] = v
        self.game_done()
        if not self.game_over:
            self.get_acting_player_position()

    def get_acting_player_position(self):
        if self.acting_player_position == 'white':
            self.acting_player_position = 'black'
        else:
            self.acting_player_position = 'white'
        return self.acting_player_position
